// Track 4 — 128x128 B-through-LDS path test.
// Tests the lds_gemm_bLDS<128,128,...> kernel (drip v2) via dispatch_gemm_blds128().
// Scale tiling uses TileChoice{128,128,MPW=2,NPW=2} — different from the 128x256 path.
// CPU reference is identical (mxfp6_gemm_ref takes the compact B_q, shape-agnostic).
//
// Usage:
//   ./test_gemm128               — correctness suite + perf sweep
//   ./test_gemm128 M N K         — original perf (existing public API, unchanged)
//   ./test_gemm128 blds M N K    — 128x128 bLDS drip perf for (M,N,K)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "mxfp6/gemm.hpp"
#include "mxfp6/preprocess.hpp"
#include "mxfp6/types.hpp"
#include "reference.hpp"
#include "../src/dispatch_blds128.hpp"  // dispatch_gemm_blds128, use_blds128_path, blds128_tile
using namespace mxfp6;
static constexpr int KT = K_TILE;

// ---- timing helpers (identical to test_gemm.cpp) ----
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

// ---- device setup for 128x128 bLDS path ----
// B: compact B^T (preprocess_B output, NOT preshuffled).
// Scales: tiled with MPW=2, NPW=2 (blds128_tile()).
struct DevBlds128 {
    void *dA, *dBc;
    uint8_t *dsA, *dsB;
    int Ar, Bc_row, Kp;
    QuantizedMatrix Aq, Bq;
};

static DevBlds128 setup128(int M, int N, int K) {
    TileChoice tc = detail::blds128_tile();  // {128,128,2,2}
    int Kp = ((K + KT - 1) / KT) * KT;
    std::mt19937 rng(M * 3 + N * 7 + K + 99);
    std::uniform_real_distribution<float> d(-1, 1);
    std::vector<float> Af((size_t)M * Kp, 0.f), Bf((size_t)Kp * N, 0.f);
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++) Af[(size_t)i * Kp + k] = d(rng);
    for (int k = 0; k < K; k++)
        for (int j = 0; j < N; j++) Bf[(size_t)k * N + j] = d(rng);
    DevBlds128 x;
    x.Aq = quantize_to_mxfp6(Af.data(), M, Kp);
    x.Bq = preprocess_B(Bf.data(), Kp, N);  // compact B^T [N][Kp]
    // Scale tiling with MPW=2, NPW=2 (128x128 path).
    auto saC = tile_scale(preprocess_scale(x.Aq.scales.data(), M, Kp), tc.MPW, KT / 64);
    auto sbC = tile_scale(preprocess_scale(x.Bq.scales.data(), N, Kp), tc.NPW, KT / 64);
    hipMalloc(&x.dA, x.Aq.packed_data.size());
    hipMalloc(&x.dBc, x.Bq.packed_data.size());
    hipMalloc(&x.dsA, saC.data.size());
    hipMalloc(&x.dsB, sbC.data.size());
    hipMemcpy(x.dA, x.Aq.packed_data.data(), x.Aq.packed_data.size(), hipMemcpyHostToDevice);
    hipMemcpy(x.dBc, x.Bq.packed_data.data(), x.Bq.packed_data.size(), hipMemcpyHostToDevice);
    hipMemcpy(x.dsA, saC.data.data(), saC.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(x.dsB, sbC.data.data(), sbC.data.size(), hipMemcpyHostToDevice);
    x.Ar = x.Aq.packed_row_bytes;
    x.Bc_row = x.Bq.packed_row_bytes;
    x.Kp = Kp;
    return x;
}
static void teardown128(DevBlds128& x) {
    hipFree(x.dA);
    hipFree(x.dBc);
    hipFree(x.dsA);
    hipFree(x.dsB);
}

// Correctness: run 128x128 bLDS, compare to CPU reference.
static bool verify128(int M, int N, int K) {
    int Kp = ((K + KT - 1) / KT) * KT;
    if (!detail::use_blds128_path(M, N, Kp)) {
        printf("  bLDS128 %4dx%4dx%4d -> SKIP (not in 128x128 bLDS regime)\n", M, N, K);
        return true;
    }
    DevBlds128 x = setup128(M, N, K);
    std::vector<float> Dref((size_t)M * N);
    mxfp6_gemm_ref(x.Aq, x.Bq, Dref.data(), M, x.Kp, N);
    float* dD;
    hipMalloc(&dD, (size_t)M * N * 4);
    hipMemset(dD, 0x5A, (size_t)M * N * 4);
    detail::dispatch_gemm_blds128(M, N, x.Kp, x.dA, x.dBc, x.dsA, x.dsB, dD,
                                  x.Ar, x.Bc_row);
    hipDeviceSynchronize();
    std::vector<float> Dg((size_t)M * N);
    hipMemcpy(Dg.data(), dD, (size_t)M * N * 4, hipMemcpyDeviceToHost);
    hipFree(dD);
    float er = 0, mx = 0;
    size_t nans = 0;
    for (size_t i = 0; i < (size_t)M * N; i++) {
        if (std::isnan(Dg[i])) nans++;
        er = fmaxf(er, fabsf(Dg[i] - Dref[i]));
        mx = fmaxf(mx, fabsf(Dref[i]));
    }
    bool ok = nans == 0 && er < 2e-2f * fmaxf(1.f, mx);
    printf("  bLDS128 %4dx%4dx%4d -> 128x128 : NaNs=%zu er=%.4f %s\n",
           M, N, K, nans, er, ok ? "OK" : "FAIL<<<");
    teardown128(x);
    return ok;
}

// Perf: run 128x128 bLDS, report TFLOPS (FP16 output).
static void perf128(int M, int N, int K) {
    int Kp = ((K + KT - 1) / KT) * KT;
    if (!detail::use_blds128_path(M, N, Kp)) {
        printf("  bLDS128 %5dx%5dx%5d -> SKIP\n", M, N, K);
        return;
    }
    DevBlds128 x = setup128(M, N, K);
    __half* dD;
    hipMalloc(&dD, (size_t)M * N * 2);
    auto run = [&] {
        detail::dispatch_gemm_blds128(M, N, x.Kp, x.dA, x.dBc, x.dsA, x.dsB, dD,
                                      x.Ar, x.Bc_row);
    };
    double ms = bench(run);
    int wg = (M / 128) * (N / 128);
    printf("  bLDS128 %5dx%5dx%5d wg=%4d -> 128x128 : %.0f TFLOPs\n",
           M, N, K, wg, tf(M, N, K, ms));
    hipFree(dD);
    teardown128(x);
}

int main(int argc, char** argv) {
    // argv >= 4, first arg not "blds": original public API perf (for baseline comparison)
    if (argc >= 4 && argv[1][0] != 'b') {
        // Run the ORIGINAL kernel via public gemm() for comparison.
        // We set up standard preshuffled B for the original path.
        int M = atoi(argv[1]), N = atoi(argv[2]), K = atoi(argv[3]);
        printf("=== original path perf (public API): %dx%dx%d ===\n", M, N, K);
        TileChoice tc = choose_tile(M, N);
        int Kp = ((K + KT - 1) / KT) * KT;
        std::mt19937 rng(M * 3 + N * 7 + K);
        std::uniform_real_distribution<float> d(-1, 1);
        std::vector<float> Af((size_t)M * Kp, 0.f), Bf((size_t)Kp * N, 0.f);
        for (int i = 0; i < M; i++)
            for (int k = 0; k < K; k++) Af[(size_t)i * Kp + k] = d(rng);
        for (int k = 0; k < K; k++)
            for (int j = 0; j < N; j++) Bf[(size_t)k * N + j] = d(rng);
        auto Aq = quantize_to_mxfp6(Af.data(), M, Kp);
        auto Bq = preprocess_B(Bf.data(), Kp, N);
        auto Bsh = preshuffle_B(Bq);
        auto saC = tile_scale(preprocess_scale(Aq.scales.data(), M, Kp), tc.MPW, KT / 64);
        auto sbC = tile_scale(preprocess_scale(Bq.scales.data(), N, Kp), tc.NPW, KT / 64);
        void *dA, *dBsh; uint8_t *dsA, *dsB;
        hipMalloc(&dA, Aq.packed_data.size());
        hipMalloc(&dBsh, Bsh.data.size());
        hipMalloc(&dsA, saC.data.size());
        hipMalloc(&dsB, sbC.data.size());
        hipMemcpy(dA, Aq.packed_data.data(), Aq.packed_data.size(), hipMemcpyHostToDevice);
        hipMemcpy(dBsh, Bsh.data.data(), Bsh.data.size(), hipMemcpyHostToDevice);
        hipMemcpy(dsA, saC.data.data(), saC.data.size(), hipMemcpyHostToDevice);
        hipMemcpy(dsB, sbC.data.data(), sbC.data.size(), hipMemcpyHostToDevice);
        __half* dD;
        hipMalloc(&dD, (size_t)M * N * 2);
        void* ws = nullptr; size_t wsb = gemm_workspace_size(M, N, Kp);
        if (wsb) hipMalloc(&ws, wsb);
        auto run = [&] {
            gemm(OutType::F16, M, N, Kp, dA, dBsh, dsA, dsB, dD, Aq.packed_row_bytes,
                 Bq.packed_row_bytes, ws, wsb);
        };
        double ms = bench(run);
        printf("  %5dx%5dx%5d -> %3dx%3d : %.0f TFLOPs\n",
               M, N, K, tc.MT, tc.NT, tf(M, N, K, ms));
        if (ws) hipFree(ws);
        hipFree(dD); hipFree(dA); hipFree(dBsh); hipFree(dsA); hipFree(dsB);
        return 0;
    }
    // argv == "blds M N K": 128x128 bLDS perf
    if (argc >= 5 && argv[1][0] == 'b') {
        int M = atoi(argv[2]), N = atoi(argv[3]), K = atoi(argv[4]);
        printf("=== 128x128 bLDS perf: %dx%dx%d ===\n", M, N, K);
        perf128(M, N, K);
        return 0;
    }

    // No args: correctness + perf suite.
    printf("=== 128x128 bLDS correctness ===\n");
    int fb = 0;
    // 256x256x57344 -> Kp=57408>50K, M%128=0, N%128=0 -> bLDS128 path ✓
    // CPU ref: 256*256*57344 ≈ 3.7G ops (fast)
    fb += !verify128(256, 256, 57344);
    // 512x512x57344 -> same guard ✓
    fb += !verify128(512, 512, 57344);
    // 256x512x57344 -> M%128=0, N%128=0 ✓ (non-square)
    fb += !verify128(256, 512, 57344);
    // Small K -> should SKIP
    fb += !verify128(256, 256, 4096);
    if (fb) {
        printf("  CORRECTNESS FAILED\n");
        return 1;
    }
    printf("  all OK\n");

    printf("\n=== 128x128 bLDS perf (FP16) ===\n");
    perf128(2048, 6144, 16128);
    perf128(2048, 6144, 105728);
    return 0;
}
