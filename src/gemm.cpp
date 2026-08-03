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
    // 128x128 path: large-K wide-N where the 128x256 tile's per-WG B working-set
    // (256 cols * Kp * 0.75 B) exceeds L2, but halving NT to 128 may improve residency.
    // Guard: shape must be 128x128-grid-valid AND Kp is known AND large enough.
    // Threshold: Kp >= 32768 (= K > ~22K fp6-effective; chosen to activate at the
    // 105728-padded shapes observed to be L2-miss dominated at N=6144).
    constexpr int LARGEK_THRESH = 32768;
    if (wg256 < CU && (M % 128) == 0 && (N % 128) == 0 && Kp >= LARGEK_THRESH) {
        // Large-K wide-N: try 128x128 to halve B working-set per pass.
        // Only when wg128 fills the CUs (halving NT doubles WGs).
        int wg128 = (M / 128) * (N / 128);
        if (wg128 >= CU)
            return {128, 128, 2, 2};  // large-K wide-N: halve B working-set per WG
    }
    // 128x384 path: 12 acc per wave, taken when it saves a whole CU wave over 128x256.
    // Requires WAVES_M=1, WAVES_N=4 so N_PW=3 satisfies the scale static_assert (N_PW<=4).
    // Compute per B-slot = 1152/9 = 128 cyc (vs 64 for 128x256), so PFD=5 covers 640 cyc.
    // Only activates for moderate K (B-slice = 384*Kp*0.75; at LARGEK_THRESH=32768 this is
    // 9.4MB, past L2 — so we gate Kp < LARGEK_THRESH and let 128x128 handle the rest).
    if (wg256 < CU && (M % 128) == 0 && (N % 384) == 0 && Kp < LARGEK_THRESH) {
        int wg384 = (M / 128) * (N / 384);
        int wg256_grid = (M / 128) * (N / 256);
        bool saves_a_wave =
            wg384 >= CU && 3 * ((wg384 + CU - 1) / CU) < 2 * ((wg256_grid + CU - 1) / CU);
        bool only_exact_route = (N % 256) != 0;
        if (saves_a_wave || only_exact_route)
            return {128, 384, 4, 3};  // WAVES_M=1, WAVES_N=4, MPW=4, NPW=3
    }
    if (wg256 < CU && (M % 128) == 0 && (N % 256) == 0)
        return {128, 256, 2, 4};  // WG-starved small-M: fill CUs
    return {256, 256, 4, 4};      // workhorse: 16-acc sweet spot
}

size_t gemm_workspace_size(int M, int N, int Kp) {
    return detail::splitk_workspace_bytes(M, N, Kp);
}

void gemm(OutType ot, int M, int N, int Kp, const void* dA, const void* dBsh, const uint8_t* dsA,
          const uint8_t* dsB, void* dD, int A_row_bytes, int B_row_bytes, void* ws,
          size_t ws_bytes) {
    switch (ot) {
        case OutType::F32:
            detail::dispatch_gemm<float>(M, N, Kp, dA, dBsh, dsA, dsB, static_cast<float*>(dD),
                                         A_row_bytes, B_row_bytes, ws, ws_bytes);
            break;
        case OutType::F16:
            detail::dispatch_gemm<__half>(M, N, Kp, dA, dBsh, dsA, dsB, static_cast<__half*>(dD),
                                          A_row_bytes, B_row_bytes, ws, ws_bytes);
            break;
        case OutType::BF16:
            detail::dispatch_gemm<__hip_bfloat16>(M, N, Kp, dA, dBsh, dsA, dsB,
                                                  static_cast<__hip_bfloat16*>(dD), A_row_bytes,
                                                  B_row_bytes, ws, ws_bytes);
            break;
    }
}

// UNSAFE test-only helper: bypasses choose_tile() routing to force a specific tile path on small
// shapes. Intentionally NOT declared in the public header (mxfp6/gemm.hpp) — tests/test_gemm.cpp
// forward-declares it. Caller must keep tc / scale-tiling / M%MT / N%NT in sync (mismatch = silent
// wrong output). No split-K.
void gemm_force_tile(OutType ot, int M, int N, int Kp, TileChoice tc, const void* dA,
                     const void* dBsh, const uint8_t* dsA, const uint8_t* dsB, void* dD,
                     int A_row_bytes, int B_row_bytes) {
    switch (ot) {
        case OutType::F32:
            detail::dispatch_gemm_force_tile<float>(M, N, Kp, tc, dA, dBsh, dsA, dsB,
                                                    static_cast<float*>(dD), A_row_bytes, B_row_bytes);
            break;
        case OutType::F16:
            detail::dispatch_gemm_force_tile<__half>(M, N, Kp, tc, dA, dBsh, dsA, dsB,
                                                     static_cast<__half*>(dD), A_row_bytes, B_row_bytes);
            break;
        case OutType::BF16:
            detail::dispatch_gemm_force_tile<__hip_bfloat16>(M, N, Kp, tc, dA, dBsh, dsA, dsB,
                                                              static_cast<__hip_bfloat16*>(dD),
                                                              A_row_bytes, B_row_bytes);
            break;
    }
}

}  // namespace mxfp6
