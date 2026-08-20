// libmxfp6gemm implementation (HIP). Defines the public type-erased entry points and
// instantiates the internal templated launcher for each output type.
#include "mxfp6/gemm.hpp"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#include "dispatch.hpp"  // mxfp6::detail::dispatch_gemm<OutT>

namespace mxfp6 {

TileChoice choose_tile(int M, int N, int Kp) {
    constexpr int CU = 256;  // MI350X (gfx950)
    int wg256 = (M / 256) * (N / 256);
    // 128x128 path: the large-K wide-N shapes 128x384 cannot take. It beats 128x256 by dropping
    // the half-empty second CU wave, NOT by fitting B in L2 (a 128-col B/N-slice is 10.16 MB
    // against a 4 MB L2). Threshold chosen to activate at the 105728-padded shapes at N=6144.
    constexpr int LARGEK_THRESH = 32768;
    // 128x384 path: 12 acc per wave, taken when it saves a whole CU wave over 128x256.
    // Requires WAVES_M=1, WAVES_N=4 so N_PW=3 satisfies the scale static_assert (N_PW<=4).
    // Compute per B-slot = 1152/9 = 128 cyc (vs 64 for 128x256), so PFD=5 covers 640 cyc.
    bool wide384 = wg256 < CU && (M % 128) == 0 && (N % 384) == 0;
    bool saves_a_wave = false;
    if (wide384) {
        int wg384 = (M / 128) * (N / 384);
        int wg256_grid = (M / 128) * (N / 256);
        saves_a_wave =
            wg384 >= CU && 3 * ((wg384 + CU - 1) / CU) < 2 * ((wg256_grid + CU - 1) / CU);
    }
    // Deliberately NOT K-gated: measured 35-41% faster than 128x128 at K=32768..105728 (2026-08-05,
    // M=2048 N=6144). A is re-read N/NT times, so the wider tile moves 1.5x fewer bytes overall
    // even though its B/N-slice grows past L2.
    if (saves_a_wave)
        return {128, 384, 4, 3};  // WAVES_M=1, WAVES_N=4, MPW=4, NPW=3
    if (wg256 < CU && (M % 128) == 0 && (N % 128) == 0 && Kp >= LARGEK_THRESH) {
        // Only when wg128 fills the CUs (halving NT doubles WGs) — that is what buys the win:
        // 3 full CU waves instead of 128x256's 2 half-empty ones.
        int wg128 = (M / 128) * (N / 128);
        if (wg128 >= CU)
            return {128, 128, 2, 2};  // large-K wide-N that 128x384 cannot take
    }
    if (wg256 < CU && (M % 128) == 0 && (N % 256) == 0)
        return {128, 256, 2, 4};  // WG-starved small-M: fill CUs
    // Coverage floor. Every arm above is a PERF choice gated on CU fill, and a shape they all
    // decline used to land on 256x256 unconditionally -- with an integer-division grid, so a 256
    // that does not divide N silently dropped the remainder (M=256 N=384: 1x1 grid, columns
    // 256..383 never written). Pick the widest IMPLEMENTED tile that reaches the whole shape,
    // preferring the better-measured route on ties.
    if ((M % 256) == 0 && (N % 256) == 0) return {256, 256, 4, 4};  // workhorse: 16-acc sweet spot
    if ((M % 128) == 0 && (N % 256) == 0) return {128, 256, 2, 4};
    if ((M % 128) == 0 && (N % 384) == 0) return {128, 384, 4, 3};  // only route when N%256!=0
    // N%128==0 with no 256-wide divisor. 128x128 divides it, but a 4-acc tile amortizes shallow-K
    // fixed cost over a quarter of the work: 1274 TFLOPs against 1455 on 2048x102272x1024. So run
    // 256x256 over a ceil(N/256) grid and mask the last N-tile at the store instead. Needs
    // (N/32)%NPW == 0, which for NPW=4 is exactly N%128==0; M must divide 256 (no M-remainder).
    // Any N reaching here is N%256==128, so the overshoot is one 128-column half-tile and the
    // waste is 128/N. N>=256 demands a full N-tile to amortize it; N=128 has none, masks half the
    // tile, and measured 0.49x against the 128x128 below (10 K values, ranges disjoint).
    if ((M % 256) == 0 && (N % 128) == 0 && N >= 256) return {256, 256, 4, 4};
    if ((M % 128) == 0 && (N % 128) == 0) return {128, 128, 2, 2};
    return {256, 256, 4, 4};  // no implemented tile covers this shape — caller must pad M/N
}

size_t gemm_workspace_size(int M, int N, int Kp) {
    return detail::splitk_workspace_bytes(M, N, Kp);
}

// Pre-K_real entry point, kept so objects compiled against the older header still link. Same
// behaviour as passing K_real=0: every sub-slab of the last tile runs, including the all-pad ones.
void gemm(OutType ot, int M, int N, int Kp, const void* dA, const void* dBsh, const uint8_t* dsA,
          const uint8_t* dsB, void* dD, int A_row_bytes, int B_row_bytes, void* ws,
          size_t ws_bytes) {
    gemm(ot, M, N, Kp, dA, dBsh, dsA, dsB, dD, A_row_bytes, B_row_bytes, ws, ws_bytes, 0);
}

void gemm(OutType ot, int M, int N, int Kp, const void* dA, const void* dBsh, const uint8_t* dsA,
          const uint8_t* dsB, void* dD, int A_row_bytes, int B_row_bytes, void* ws,
          size_t ws_bytes, int K_real) {
    // A K_real that does not pad to this Kp would make the kernel skip REAL work. Drop, not trust.
    if (K_real > 0 && kpad(K_real) != Kp) K_real = 0;
    switch (ot) {
        case OutType::F32:
            detail::dispatch_gemm<float>(M, N, Kp, dA, dBsh, dsA, dsB, static_cast<float*>(dD),
                                         A_row_bytes, B_row_bytes, ws, ws_bytes, K_real);
            break;
        case OutType::F16:
            detail::dispatch_gemm<__half>(M, N, Kp, dA, dBsh, dsA, dsB, static_cast<__half*>(dD),
                                          A_row_bytes, B_row_bytes, ws, ws_bytes, K_real);
            break;
        case OutType::BF16:
            detail::dispatch_gemm<__hip_bfloat16>(M, N, Kp, dA, dBsh, dsA, dsB,
                                                  static_cast<__hip_bfloat16*>(dD), A_row_bytes,
                                                  B_row_bytes, ws, ws_bytes, K_real);
            break;
    }
}

// UNSAFE test-only helper: bypasses choose_tile() routing to force a specific tile path on small
// shapes. Intentionally NOT declared in the public header (mxfp6/gemm.hpp) — tests/test_gemm.cpp
// forward-declares it. Caller must keep tc / scale-tiling / M%MT / N%NT in sync (mismatch = silent
// wrong output). No split-K.
void gemm_force_tile(OutType ot, int M, int N, int Kp, TileChoice tc, const void* dA,
                     const void* dBsh, const uint8_t* dsA, const uint8_t* dsB, void* dD,
                     int A_row_bytes, int B_row_bytes, int K_real) {
    if (K_real > 0 && kpad(K_real) != Kp) K_real = 0;
    switch (ot) {
        case OutType::F32:
            detail::dispatch_gemm_force_tile<float>(M, N, Kp, tc, dA, dBsh, dsA, dsB,
                                                    static_cast<float*>(dD), A_row_bytes, B_row_bytes, K_real);
            break;
        case OutType::F16:
            detail::dispatch_gemm_force_tile<__half>(M, N, Kp, tc, dA, dBsh, dsA, dsB,
                                                     static_cast<__half*>(dD), A_row_bytes, B_row_bytes, K_real);
            break;
        case OutType::BF16:
            detail::dispatch_gemm_force_tile<__hip_bfloat16>(M, N, Kp, tc, dA, dBsh, dsA, dsB,
                                                              static_cast<__hip_bfloat16*>(dD),
                                                              A_row_bytes, B_row_bytes, K_real);
            break;
    }
}

}  // namespace mxfp6
