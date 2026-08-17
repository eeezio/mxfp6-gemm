// End-to-end test of libmxfp6gemm through its PUBLIC API (mxfp6/gemm.hpp): choose_tile
// routing + host tiled-scale (mxfp6/preprocess.hpp) + mxfp6::gemm() launch. Correctness
// (fresh-alloc + CPU ref) on both tile paths, then an indicative perf sweep over 12 shapes.
// (For precise numbers let the GPU reach steady state first; this is a functional gate.)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "mxfp6/gemm.hpp"
#include "mxfp6/preprocess.hpp"
#include "mxfp6/types.hpp"
#include "reference.hpp"  // CPU reference GEMM (test-only, lives in tests/)
using namespace mxfp6;
static constexpr int KT = K_TILE;

// gemm_force_tile(): UNSAFE test-only helper (bypasses choose_tile routing; the caller must keep
// TileChoice / scale-tiling / shape divisibility in sync — a mismatch is silent wrong output). It
// is intentionally NOT part of the public API (not declared in mxfp6/gemm.hpp); forward-declared
// here so only the test can reach the library symbol. Used to exercise the 128x128 path on small
// shapes where choose_tile would route elsewhere.
namespace mxfp6 {
void gemm_force_tile(OutType ot, int M, int N, int Kp, TileChoice tc, const void* dA,
                     const void* dBsh, const uint8_t* dsA, const uint8_t* dsB, void* dD,
                     int A_row_bytes, int B_row_bytes, int K_real = 0);
}  // namespace mxfp6

template <class F>
static double bench(F run) {
    for (int i = 0; i < 8; i++) run();
    hipDeviceSynchronize();
    double best = 1e30;
    for (int r = 0; r < 4; r++) {
        hipEvent_t a, b;
        hipEventCreate(&a);
        hipEventCreate(&b);
        hipEventRecord(a);
        for (int i = 0; i < 20; i++) run();
        hipEventRecord(b);
        hipDeviceSynchronize();
        float ms = 0;
        hipEventElapsedTime(&ms, a, b);
        hipEventDestroy(a);
        hipEventDestroy(b);
        best = fmin(best, ms / 20.0);
    }
    return best;
}
static double tf(int M, int N, int K, double ms) {
    return 2.0 * M * N * K / (ms * 1e-3) / 1e12;
}

// build device inputs for (M,N,K) using the chosen tile's MPW/NPW for scale tiling
struct Dev {
    void *dA, *dBsh;
    uint8_t *dsA, *dsB;
    int Ar, Br, Kp;
    QuantizedMatrix Aq, Bq;
};
static Dev setup(int M, int N, int K, TileChoice tc) {
    int Kp = ((K + KT - 1) / KT) * KT;
    std::mt19937 rng(M * 3 + N * 7 + K);
    std::uniform_real_distribution<float> d(-1, 1);
    std::vector<float> Af((size_t)M * Kp, 0.f), Bf((size_t)Kp * N, 0.f);
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++) Af[(size_t)i * Kp + k] = d(rng);
    for (int k = 0; k < K; k++)
        for (int j = 0; j < N; j++) Bf[(size_t)k * N + j] = d(rng);
    Dev x;
    x.Aq = quantize_to_mxfp6(Af.data(), M, Kp);
    x.Bq = preprocess_B(Bf.data(), Kp, N);
    auto Bsh = preshuffle_B(x.Bq);
    auto saC = tile_scale(preprocess_scale(x.Aq.scales.data(), M, Kp), tc.MPW, KT / 64);
    auto sbC = tile_scale(preprocess_scale(x.Bq.scales.data(), N, Kp), tc.NPW, KT / 64);
    hipMalloc(&x.dA, x.Aq.packed_data.size());
    hipMalloc(&x.dBsh, Bsh.data.size());
    hipMalloc(&x.dsA, saC.data.size());
    hipMalloc(&x.dsB, sbC.data.size());
    hipMemcpy(x.dA, x.Aq.packed_data.data(), x.Aq.packed_data.size(), hipMemcpyHostToDevice);
    hipMemcpy(x.dBsh, Bsh.data.data(), Bsh.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(x.dsA, saC.data.data(), saC.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(x.dsB, sbC.data.data(), sbC.data.size(), hipMemcpyHostToDevice);
    x.Ar = x.Aq.packed_row_bytes;
    x.Br = x.Bq.packed_row_bytes;
    x.Kp = Kp;
    return x;
}
static void teardown(Dev& x) {
    hipFree(x.dA);
    hipFree(x.dBsh);
    hipFree(x.dsA);
    hipFree(x.dsB);
}

// force_ws: hand gemm() a split-K workspace even when gemm_workspace_size() says none is needed,
// so a route that must not split is checked under the same call shape a splitting caller uses.
static bool verify(int M, int N, int K, bool force_ws = false) {
    int Kp_est = ((K + K_TILE - 1) / K_TILE) * K_TILE;
    TileChoice tc = choose_tile(M, N, Kp_est);
    Dev x = setup(M, N, K, tc);
    std::vector<float> Dref((size_t)M * N);
    mxfp6_gemm_ref(x.Aq, x.Bq, Dref.data(), M, x.Kp, N);
    float* dD;
    hipMalloc(&dD, (size_t)M * N * 4);
    hipMemset(dD, 0x5A, (size_t)M * N * 4);
    void* ws = nullptr;
    size_t wsb = gemm_workspace_size(M, N, x.Kp);
    if (!wsb && force_ws) wsb = (size_t)4 * M * N * sizeof(float);
    if (wsb) hipMalloc(&ws, wsb);
    gemm(OutType::F32, M, N, x.Kp, x.dA, x.dBsh, x.dsA, x.dsB, dD, x.Ar, x.Br, ws, wsb, K);
    hipDeviceSynchronize();
    std::vector<float> Dg((size_t)M * N);
    hipMemcpy(Dg.data(), dD, (size_t)M * N * 4, hipMemcpyDeviceToHost);
    if (ws) hipFree(ws);
    hipFree(dD);
    float er = 0, mx = 0;
    for (size_t i = 0; i < (size_t)M * N; i++) {
        er = fmaxf(er, fabsf(Dg[i] - Dref[i]));
        mx = fmaxf(mx, fabsf(Dref[i]));
    }
    bool ok = er < 2e-2f * fmaxf(1.f, mx);
    printf("  %4dx%4dx%4d -> %dx%d%s : %s\n", M, N, K, tc.MT, tc.NT, force_ws ? " +ws" : "",
           ok ? "OK" : "FAIL<<<");
    teardown(x);
    return ok;
}
// "pad-B-only / compact-A" recipe (mxfp6/preprocess.hpp): A in its natural COMPACT layout
// (row stride = packed(K), A_row_bytes = packed(K), buffer + a_compact_end_pad), B padded, A's
// scales kpad-extended with a non-NaN tail (pad_scales_k). Compared to the correct GEMM (a
// zero-Kp-padded A reference). Exercises the inter-row K-tail overlap nulled by B's zero tail.
static bool verify_compact(int M, int N, int K) {
    int Kp = kpad(K);
    TileChoice tc = choose_tile(M, N, Kp);
    std::mt19937 rng(M * 5 + N * 11 + K * 2);
    std::uniform_real_distribution<float> d(-1, 1);
    std::vector<float> Areal((size_t)M * K), Bf((size_t)Kp * N, 0.f);
    for (size_t i = 0; i < (size_t)M * K; i++) Areal[i] = d(rng);
    for (int k = 0; k < K; k++)
        for (int j = 0; j < N; j++) Bf[(size_t)k * N + j] = d(rng);
    auto Aq = quantize_to_mxfp6(Areal.data(), M, K);  // compact: packed_row_bytes = packed(K)
    auto Bq = preprocess_B(Bf.data(), Kp, N);         // B K-tail = 0

    // reference: zero-Kp-padded A (same real blocks) -> the correct GEMM
    std::vector<float> Apad((size_t)M * Kp, 0.f);
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++) Apad[(size_t)i * Kp + k] = Areal[(size_t)i * K + k];
    auto AqP = quantize_to_mxfp6(Apad.data(), M, Kp);
    std::vector<float> Dref((size_t)M * N);
    mxfp6_gemm_ref(AqP, Bq, Dref.data(), M, Kp, N);

    // compact A device buffer + end pad (last row's K-tail overread), poisoned tail
    void* dA;
    size_t Abytes = (size_t)M * Aq.packed_row_bytes + a_compact_end_pad(K);
    hipMalloc(&dA, Abytes);
    hipMemset(dA, 0x5A, Abytes);
    hipMemcpy(dA, Aq.packed_data.data(), (size_t)M * Aq.packed_row_bytes, hipMemcpyHostToDevice);
    auto Bsh = preshuffle_B(Bq);
    auto saC = tile_scale(preprocess_scale(pad_scales_k(Aq.scales.data(), M, K).data(), M, Kp),
                          tc.MPW, KT / 64);
    auto sbC = tile_scale(preprocess_scale(Bq.scales.data(), N, Kp), tc.NPW, KT / 64);
    void* dBsh;
    uint8_t *dsA, *dsB;
    hipMalloc(&dBsh, Bsh.data.size());
    hipMalloc(&dsA, saC.data.size());
    hipMalloc(&dsB, sbC.data.size());
    hipMemcpy(dBsh, Bsh.data.data(), Bsh.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsA, saC.data.data(), saC.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsB, sbC.data.data(), sbC.data.size(), hipMemcpyHostToDevice);
    float* dD;
    hipMalloc(&dD, (size_t)M * N * 4);
    hipMemset(dD, 0x5A, (size_t)M * N * 4);
    void* ws = nullptr;
    size_t wsb = gemm_workspace_size(M, N, Kp);
    if (wsb) hipMalloc(&ws, wsb);
    gemm(OutType::F32, M, N, Kp, dA, dBsh, dsA, dsB, dD, Aq.packed_row_bytes, Bq.packed_row_bytes,
         ws, wsb);
    hipDeviceSynchronize();
    std::vector<float> Dg((size_t)M * N);
    hipMemcpy(Dg.data(), dD, (size_t)M * N * 4, hipMemcpyDeviceToHost);
    if (ws) hipFree(ws);
    float er = 0, mx = 0;
    size_t nans = 0;
    for (size_t i = 0; i < (size_t)M * N; i++) {
        if (std::isnan(Dg[i])) nans++;
        er = fmaxf(er, fabsf(Dg[i] - Dref[i]));
        mx = fmaxf(mx, fabsf(Dref[i]));
    }
    bool ok = nans == 0 && er < 2e-2f * fmaxf(1.f, mx);
    printf("  compact %4dx%4dx%4d (Kp=%d) -> %dx%d : NaNs=%zu %s\n", M, N, K, Kp, tc.MT, tc.NT,
           nans, ok ? "OK" : "FAIL<<<");
    hipFree(dA);
    hipFree(dBsh);
    hipFree(dsA);
    hipFree(dsB);
    hipFree(dD);
    return ok;
}
// Correctness check for the 128x128 kernel path on SMALL shapes (fast CPU ref, ~5e7 MACs).
// Forces MPW=2/NPW=2 scale tiling and calls gemm_force_tile() to bypass choose_tile() routing,
// so the 128x128 kernel is exercised even when Kp < the large-K threshold. M must be a multiple
// of 128; N must be a multiple of 128.
static bool verify_128x128(int M, int N, int K) {
    constexpr TileChoice tc = {128, 128, 2, 2};
    Dev x = setup(M, N, K, tc);
    std::vector<float> Dref((size_t)M * N);
    mxfp6_gemm_ref(x.Aq, x.Bq, Dref.data(), M, x.Kp, N);
    float* dD;
    hipMalloc(&dD, (size_t)M * N * 4);
    hipMemset(dD, 0x5A, (size_t)M * N * 4);
    gemm_force_tile(OutType::F32, M, N, x.Kp, tc, x.dA, x.dBsh, x.dsA, x.dsB, dD, x.Ar, x.Br);
    hipDeviceSynchronize();
    std::vector<float> Dg((size_t)M * N);
    hipMemcpy(Dg.data(), dD, (size_t)M * N * 4, hipMemcpyDeviceToHost);
    hipFree(dD);
    float er = 0, mx = 0;
    for (size_t i = 0; i < (size_t)M * N; i++) {
        er = fmaxf(er, fabsf(Dg[i] - Dref[i]));
        mx = fmaxf(mx, fabsf(Dref[i]));
    }
    bool ok = er < 2e-2f * fmaxf(1.f, mx);
    printf("  128x128 %4dx%4dx%4d : %s\n", M, N, K, ok ? "OK" : "FAIL<<<");
    teardown(x);
    return ok;
}

// Correctness check for the 256x256 SHORT-TAIL path on SMALL shapes. The routed short tail only
// fires at N >= BRING8_MIN_N (105728-class), which no CPU reference can check, so force the tile
// and hand it the true K. Pick K with (K/64)%3 == 1 to make the last of the k-tiles carry one real
// 64-K sub-slab and two of pure padding. M and N must be multiples of 256.
static bool verify_256x256_tail(int M, int N, int K) {
    constexpr TileChoice tc = {256, 256, 4, 4};
    Dev x = setup(M, N, K, tc);
    std::vector<float> Dref((size_t)M * N);
    mxfp6_gemm_ref(x.Aq, x.Bq, Dref.data(), M, x.Kp, N);
    float* dD;
    hipMalloc(&dD, (size_t)M * N * 4);
    hipMemset(dD, 0x5A, (size_t)M * N * 4);
    gemm_force_tile(OutType::F32, M, N, x.Kp, tc, x.dA, x.dBsh, x.dsA, x.dsB, dD, x.Ar, x.Br, K);
    hipDeviceSynchronize();
    std::vector<float> Dg((size_t)M * N);
    hipMemcpy(Dg.data(), dD, (size_t)M * N * 4, hipMemcpyDeviceToHost);
    hipFree(dD);
    float er = 0, mx = 0;
    for (size_t i = 0; i < (size_t)M * N; i++) {
        er = fmaxf(er, fabsf(Dg[i] - Dref[i]));
        mx = fmaxf(mx, fabsf(Dref[i]));
    }
    bool ok = er < 2e-2f * fmaxf(1.f, mx);
    printf("  256x256 tail %4dx%4dx%4d (Kp=%d) : %s\n", M, N, K, x.Kp, ok ? "OK" : "FAIL<<<");
    teardown(x);
    return ok;
}

// Correctness check for the 128x384 kernel path on SMALL shapes (fast CPU ref).
// Uses WAVES_M=1, WAVES_N=4: MPW=4, NPW=3. M must be a multiple of 128; N of 384.
static bool verify_128x384(int M, int N, int K) {
    constexpr TileChoice tc = {128, 384, 4, 3};
    Dev x = setup(M, N, K, tc);
    std::vector<float> Dref((size_t)M * N);
    mxfp6_gemm_ref(x.Aq, x.Bq, Dref.data(), M, x.Kp, N);
    float* dD;
    hipMalloc(&dD, (size_t)M * N * 4);
    hipMemset(dD, 0x5A, (size_t)M * N * 4);
    gemm_force_tile(OutType::F32, M, N, x.Kp, tc, x.dA, x.dBsh, x.dsA, x.dsB, dD, x.Ar, x.Br);
    hipDeviceSynchronize();
    std::vector<float> Dg((size_t)M * N);
    hipMemcpy(Dg.data(), dD, (size_t)M * N * 4, hipMemcpyDeviceToHost);
    hipFree(dD);
    float er = 0, mx = 0;
    for (size_t i = 0; i < (size_t)M * N; i++) {
        er = fmaxf(er, fabsf(Dg[i] - Dref[i]));
        mx = fmaxf(mx, fabsf(Dref[i]));
    }
    bool ok = er < 2e-2f * fmaxf(1.f, mx);
    printf("  128x384 %4dx%4dx%4d : %s\n", M, N, K, ok ? "OK" : "FAIL<<<");
    teardown(x);
    return ok;
}

static bool check_routing() {
    struct Case {
        int M, N, K, MT, NT;
        const char* why;
    };
    // clang-format off
    static const Case cases[] = {
        {2048,  6144, 16128, 128, 384, "wave 2->1, measured 1.59x"},
        {2048,  6144,   512, 128, 384, "same, small K"},
        {4096,  3072, 16128, 128, 384, "wg384=256"},
        {8192,  1536, 16128, 128, 384, "wg384=256"},
        {16384,  768, 16128, 128, 384, "wg384=256"},
        {1024, 12288, 16128, 128, 384, "wg384=256"},
        {2048,  6912, 16128, 128, 256, "wg384=288 -> 2 waves, no saving"},
        {2048,  7680, 16128, 128, 256, "wg384=320 -> 2 waves, no saving"},
        {4096,  3840, 16128, 128, 256, "wg384=320 -> 2 waves, no saving"},
        {1024, 15360, 16128, 128, 256, "wg384=320 -> 2 waves, no saving"},
        {2048,  6528, 16128, 128, 384, "N%256!=0, only non-truncating route"},
        {2048,  7296, 16128, 128, 384, "N%256!=0, only non-truncating route"},
        // N%256!=0 outside every perf arm's gate: the divisibility floor has to catch these, since
        // the wave-saving arm needs wg384>=CU, the 128x128 arm needs wg128>=CU, and neither is
        // reachable here. All four used to fall through to 256x256 and drop 128 columns.
        {  256,   384, 32768, 128, 384, "wg384=2, wg128=6: only the floor covers N"},
        {  128,   384, 32768, 128, 384, "same, and 256x256 would launch a 0-wide grid"},
        { 8192,  6528,  4096, 128, 384, "wg256=800>=CU, so wide384 is false: floor covers N"},
        { 4096,  7296, 32768, 128, 384, "CU-filling and large K at once"},
        // M an odd multiple of 128 on a CU-filling grid: 256x256 would drop the last 128 rows.
        {  896, 22272,  4096, 128, 256, "M%256!=0 falls back to a tile that divides M"},
        // N%128==0 with no 256-wide divisor: 256x256 over a ceil grid beats the 128x128 that
        // divides (measured 1435 vs 1273 TFLOPs on the first of these).
        {2048,102272,  1024, 256, 256, "partial last N-tile beats the 4-acc tile that divides"},
        {2048,  1408,  4096, 256, 256, "same, small N"},
        {2048, 12672,  1024, 128, 384, "N%384==0 keeps the exact route over the masked one"},
        { 896,  1408,  4096, 128, 128, "M%256!=0 cannot mask N: needs a tile that divides both"},
        {2048,  6144, 105728, 128, 384, "wave-saving 128x384 is not K-gated (measured +44%)"},
        { 512, 24576, 105728, 128, 384, "same wg384=256 family, large K"},
        {2048,  4096, 105728, 128, 128, "large-K wide-N, N%384!=0 -> 128x128"},
        {2048,  2688, 105728, 128, 128, "N%256!=0 but wg384<CU: 128x128 wins before the floor"},
        {2048,  1024,  16128, 128, 256, "narrow N"},
        {2048,  4096,   4096, 128, 256, "N%384!=0"},
        {2048, 16384,   1024, 256, 256, "wg256 >= CU"},
        {2048, 20480,   6144, 256, 256, "wg256 >= CU"},
        {4096,  4096,    768, 256, 256, "wg256 >= CU"},
    };
    // clang-format on
    int bad = 0;
    for (const Case& c : cases) {
        TileChoice tc = choose_tile(c.M, c.N, kpad(c.K));
        bool ok = tc.MT == c.MT && tc.NT == c.NT;
        if (!ok) {
            printf("  route %5dx%6dx%6d : got %3dx%3d want %3dx%3d  FAIL<<<  (%s)\n", c.M, c.N,
                   c.K, tc.MT, tc.NT, c.MT, c.NT, c.why);
            bad++;
        }
    }
    printf("  choose_tile routing: %d/%d %s\n", (int)(sizeof(cases) / sizeof(*cases)) - bad,
           (int)(sizeof(cases) / sizeof(*cases)), bad ? "FAIL<<<" : "OK");

    // Coverage sweep: the launched grid must reach every output element. M is always by division
    // (there is no M-remainder handling), N either by division or via the 256-wide tile's ceil
    // grid + store mask. Anything else silently drops rows/columns. The four tiles plus that mask
    // cover every M%128==0, N%128==0 shape, so the whole grid up to 16384x24576 must route
    // cleanly, on both sides of the large-K threshold.
    static const int sweep_K[] = {512, 4096, 16128, 32768, 40000, 105728};
    long div_bad = 0, div_tot = 0;
    for (int M = 128; M <= 16384; M += 128)
        for (int N = 128; N <= 24576; N += 128) {
            for (int K : sweep_K) {
                TileChoice tc = choose_tile(M, N, kpad(K));
                div_tot++;
                bool n_ok = N % tc.NT == 0 || (tc.NT == 256 && N % 128 == 0);
                if (M % tc.MT == 0 && n_ok) continue;
                if (div_bad < 5)
                    printf("  covers %5dx%6dx%6d : %3dx%3d reaches only %dx%d  FAIL<<<\n", M, N, K,
                           tc.MT, tc.NT, (M / tc.MT) * tc.MT, (N / tc.NT) * tc.NT);
                div_bad++;
            }
        }
    printf("  tile covers M/N: %ld/%ld %s\n", div_tot - div_bad, div_tot,
           div_bad ? "FAIL<<<" : "OK");

    // The split path only implements 128x256/256x256, so a shape routed to any other tile must
    // come back with no workspace. These three all split before the guard: the 2688/5760 pair
    // faulted on the GPU, the 128x384 one returned garbage.
    struct NoSplit {
        int M, N, K;
    };
    static const NoSplit ns[] = {{128, 384, 3072}, {2048, 2688, 24576}, {2048, 5760, 16128}};
    int sbad = 0;
    for (const NoSplit& c : ns) {
        int Kp = kpad(c.K);
        TileChoice tc = choose_tile(c.M, c.N, Kp);
        if ((tc.NT == 384 || tc.NT == 128) && gemm_workspace_size(c.M, c.N, Kp) != 0) {
            printf("  split %5dx%6dx%6d : %dx%d must not split  FAIL<<<\n", c.M, c.N, c.K, tc.MT,
                   tc.NT);
            sbad++;
        }
    }
    printf("  split-K tile guard: %d/%d %s\n", (int)(sizeof(ns) / sizeof(*ns)) - sbad,
           (int)(sizeof(ns) / sizeof(*ns)), sbad ? "FAIL<<<" : "OK");

    // Shapes smaller than the tile they fall through to: (M/tc.MT)*(N/tc.NT) == 0, which splitk_S
    // used to divide by. The divisibility floor covers everything that is a multiple of 128, so
    // reaching zero now takes an M or N that no tile divides -- 192 and 64 below. Their output is
    // truncated regardless, but a query has no business killing the process. Nothing to assert
    // beyond "it returns": an unguarded build dies here with SIGFPE and takes the suite with it.
    static const NoSplit deg[] = {{192, 256, 32832}, {256, 192, 3072}, {64, 64, 3072}};
    for (const NoSplit& c : deg) (void)gemm_workspace_size(c.M, c.N, kpad(c.K));
    printf("  workspace query on sub-tile shapes: %d/%d OK (no SIGFPE)\n",
           (int)(sizeof(deg) / sizeof(*deg)), (int)(sizeof(deg) / sizeof(*deg)));
    return bad == 0 && div_bad == 0 && sbad == 0;
}

static void perf(int M, int N, int K) {
    int Kp_est = ((K + K_TILE - 1) / K_TILE) * K_TILE;
    TileChoice tc = choose_tile(M, N, Kp_est);
    Dev x = setup(M, N, K, tc);
    __half* dD;
    hipMalloc(&dD, (size_t)M * N * 2);
    // Allocate split-K workspace once and reuse across the bench loop (the intended usage).
    void* ws = nullptr;
    size_t wsb = gemm_workspace_size(M, N, x.Kp);
    if (wsb) hipMalloc(&ws, wsb);
    auto run = [&] {
        gemm(OutType::F16, M, N, x.Kp, x.dA, x.dBsh, x.dsA, x.dsB, dD, x.Ar, x.Br, ws, wsb, K);
    };
    double ms = bench(run);
    int base_wg = (M / tc.MT) * (N / tc.NT);
    int S = wsb ? (int)(wsb / ((size_t)M * N * sizeof(float))) : 1;  // S from workspace size
    int wg = base_wg * S;
    printf("  %5dx%6dx%6d wg=%5d S=%d -> %3dx%3d : %9.6f ms  %8.3f TFLOPs\n", M, N, K, wg, S,
           tc.MT, tc.NT, ms, tf(M, N, K, ms));
    if (ws) hipFree(ws);
    hipFree(dD);
    teardown(x);
}
int main(int argc, char** argv) {
    if (argc >= 2 && strcmp(argv[1], "--routing") == 0) return check_routing() ? 0 : 1;
    if (argc >= 4) {
        int M = atoi(argv[1]), N = atoi(argv[2]), K = atoi(argv[3]);
        perf(M, N, K);
        return 0;
    }
    printf("=== libmxfp6gemm correctness (end-to-end, CPU ref) ===\n");
    int f = 0;
    f += !check_routing();
    f += !verify(512, 512, 768);    // -> 128x256 path (wg256<CU)
    f += !verify(768, 1280, 960);   // -> 128x256, non-square
    f += !verify(4096, 4096, 768);  // -> 256x256 path (wg256=256), small K for fast ref
    // Short-tail path (TAIL_SUBS): the last k-tile is 1 real 64-K sub-slab + 2 of pure padding.
    // Both buffer parities matter -- the tile count decides whether the short tile lands in the
    // first or the second LDS buffer, and an early version of the peel got one of them wrong.
    f += !verify_256x256_tail(256, 256, 1024);  // Kp=1152, 6 k-tiles -> short tile in buffer 1
    f += !verify_256x256_tail(256, 512, 1600);  // Kp=1728, 9 k-tiles -> short tile in buffer 0
    // Two padding sub-slabs instead of one ((K/64)%3 == 2), again both parities.
    f += !verify_256x256_tail(256, 256, 320);   // Kp=384,  2 k-tiles -> short tile in buffer 1
    f += !verify_256x256_tail(256, 512, 512);   // Kp=576,  3 k-tiles -> short tile in buffer 0
    // K % 64 == 32: the last real sub-slab is only HALF real, which the old `K_real % 64 == 0`
    // guard refused to touch at all. Grouped by what a floor rounding (instead of ceil) does to
    // them -- measured with floor patched in, not assumed. r = K % K_TILE.
    //   r == 32: floor gives 0, which merely DISABLES the tail. Correct either way, so these two
    //   are coverage of the r=32 -> tail=1 mapping; they do not discriminate the rounding.
    f += !verify_256x256_tail(256, 256, 992);   // Kp=1152, 6 k-tiles, tail 1 -> buffer 1
    f += !verify_256x256_tail(256, 512, 800);   // Kp= 960, 5 k-tiles, tail 1 -> buffer 0
    //   r == 96: floor runs 1 sub-slab where 2 are real, so these DO fail under floor.
    f += !verify_256x256_tail(256, 256, 1056);  // Kp=1152, 6 k-tiles, tail 2 -> buffer 1
    f += !verify_256x256_tail(256, 512, 864);   // Kp= 960, 5 k-tiles, tail 2 -> buffer 0
    //   r == 160: the whole last tile is real, so tail_subs_for() normalizes 3 down to 0 and this
    //   runs the plain kernel. Also fails under floor (2 where 3 are real). It does NOT cover the
    //   normalize itself -- an un-normalized 3 reaches the same plain kernel, since dispatch wires
    //   no TAIL_SUBS == 3 branch.
    f += !verify_256x256_tail(256, 256, 1120);  // Kp=1152, nothing to skip
    // pad-B-only / compact-A recipe. K = multiple of 32 (MX block) but NOT of K_TILE, so a real
    // K-tail exists (exercises the inter-row overlap + end pad + NaN-safe scale tail):
    f += !verify_compact(4096, 4096, 224);  // -> 256x256, 2 k_tiles
    f += !verify_compact(768, 1280, 224);   // -> 128x256, non-square, 2 k_tiles
    f += !verify_compact(512, 512, 160);    // -> 128x256, k_tiles==1 (odd tail), compact
    f += !verify_compact(2048, 1024, 12288);  // -> split-K (S=4): base_wg=64<256, k_tiles=64
    f += !verify_compact(2048, 1024, 3808);   // -> split-K (S=2) WITH real K-tail: K%192=160,
                                              //    k_tiles=20, last segment includes the pad tile
    f += !verify_compact(2048, 1024, 6304);
    // 128x128 kernel path: forced via gemm_force_tile() on small shapes (fast CPU ref ~5e7 MACs).
    // 256x256x768: grid 2x2 WGs, k_tiles=4. 384x512x1536: grid 3x4 WGs, k_tiles=8.
    f += !verify_128x128(256, 256, 768);
    f += !verify_128x128(384, 512, 1536);
    // 128x384 kernel path (WAVES_M=1, WAVES_N=4, MPW=4, NPW=3): small shapes, fast CPU ref.
    // 256x384x768: grid 2x1 WGs, k_tiles=4. 384x768x960: grid 3x2 WGs, k_tiles=5.
    f += !verify_128x384(256, 384, 768);
    f += !verify_128x384(384, 768, 960);
    // Same path, but a grid of 10x3 = 30 workgroups. The 128x384 route is XCD-swizzled, and the
    // remap only has to distribute a remainder when the grid exceeds NXCC=8: below that,
    // per=0/rem=total collapses it to the identity, which is what the two shapes above hit. At 30
    // it is per=3, rem=6, so both arms of the bijection run. A remainder bug there does not fault
    // -- it leaves an output tile claimed by no workgroup, which only a value check catches.
    f += !verify_128x384(1280, 1152, 384);
    // Routed (not forced) 128x384 on a WG-starved grid deep enough to tempt split-K: wg384=1,
    // k_tiles=16. Before the splitk_S guard this ran a 128x256 kernel against NPW=3 scales and
    // wrote only 256 of the 384 columns. Run twice: once as a normal caller, once handing over a
    // workspace anyway, which is the call shape that triggered the corruption.
    f += !verify(128, 384, 3072);
    f += !verify(128, 384, 3072, /*force_ws=*/true);
    // Routed 128x384 at Kp >= LARGEK_THRESH (kpad(32768)=32832). The 128x384 arm is gated on
    // Kp < the threshold and the 128x128 arm on wg128 >= CU (here 6), so this used to fall
    // through to 256x256 and write only columns 0..255 on a 1x1 grid.
    f += !verify(256, 384, 32768);
    // Routed 256x256 over a ceil(N/256) grid: N=640 is 2.5 N-tiles, so the last one is half
    // outside the matrix. Checks the store mask AND that the clamped B / B-scale reads on that
    // tile stay in bounds -- an unclamped build faults or reads a neighbouring allocation.
    f += !verify(512, 640, 960);
    f += !verify(256, 1408, 384);
    if (f) {
        printf("  CORRECTNESS FAILED\n");
        return 1;
    }
    printf("  all OK\n");
    printf("\n=== first-tier shapes (M=2048), FP16 ===\n");
    int sh[][3] = {{2048, 1024, 12288},   // tier-1: narrow N + large K
                   {2048, 1024, 16128},   // tier-1: narrow N + larger K
                   {2048, 1024, 105728},
                   {2048, 6144, 512},     // wide N small K: 128x384 path
                   {2048, 6144, 4096},    // wide N medium K: 128x384 path
                   {2048, 6144, 16128},   // wide N larger K: 128x384 path
                   {2048, 6144, 105728}}; // wide N + large K: 128x128 path
    for (auto& s : sh) perf(s[0], s[1], s[2]);
    return 0;
}
