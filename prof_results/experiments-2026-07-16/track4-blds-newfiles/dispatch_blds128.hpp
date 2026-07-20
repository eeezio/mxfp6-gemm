#pragma once
// Track 4 — 128x128 B-through-LDS dispatch.
// Same lds_gemm_bLDS template as the 128x256 path (kernel_blds.hpp), instantiated with
// M_TILE=128, N_TILE=128, WAVES_M=2, WAVES_N=2 => M_PW=2, N_PW=2, NB=6.
//
// Motivation vs 128x256 bLDS:
//   B working set per WG: 128 N-cols × Kp bytes ≈ 10.2MB  (vs 20.3MB for 128x256)
//   10.2MB is more likely to get L2 reuse across the K-loop OR across WGs on the same CU,
//   reducing effective HBM bytes. This is why Track 3's 128x128 subtile variant gained +16.6%.
//
// Drip schedule (NB=6 quartets):
//   ISSUES_A = ceil(128*9/256) = 5   A chunks
//   ISSUES_B = ceil(128*9/256) = 5   B chunks
//   static_assert: 1+5-1=5 < 6 ✓   last B chunk at p=5, 1 quartet margin
//
// Scale tiling: MPW=2, NPW=2 (TileChoice{128,128,2,2}) — caller must tile scales with these.
//
// LDS: 2*(128*144) + 2*(128*144) = 73728 = 72KB < 160KB ✓
// Grid for 2048x6144: 16x48 = 768 WGs (same WG count as 128x256, half B per WG).
#include "kernel_blds.hpp"
#include "mxfp6/gemm.hpp"
namespace mxfp6 {
namespace detail {

// TileChoice for the 128x128 bLDS path (NOT choose_tile — forced 128x128).
inline TileChoice blds128_tile() { return {128, 128, 2, 2}; }

// Returns true if the 128x128 bLDS path should be used.
// Guard: Kp > 50000 && M%128==0 && N%128==0.
// (We don't require N%256==0 — this path works on any N that is a multiple of 128.)
inline bool use_blds128_path(int M, int N, int Kp) {
    return (Kp > 50000) && (M % 128 == 0) && (N % 128 == 0);
}

// Launch the 128x128 bLDS kernel.
// dBc: compact B^T [N_col rows × Kp_bytes], row stride = B_compact_row_bytes.
// dsA, dsB: tiled scales built with MPW=2, NPW=2 (blds128_tile()).
template <typename OutT>
inline void dispatch_gemm_blds128(int M, int N, int Kp, const void* dA, const void* dBc,
                                  const uint8_t* dsA, const uint8_t* dsB, OutT* dD,
                                  int A_row_bytes, int B_compact_row_bytes) {
    constexpr int KT = K_TILE;
    constexpr int KT_BYTES = KT * 6 / 8;
    dim3 blk(256);
    dim3 g(M / 128, N / 128);
    int kit = Kp / 64;
    // LDS: 2*(128*KT_BYTES) A + 2*(128*KT_BYTES) B = 4*18432 = 73728 bytes
    int lds = 4 * (128 * KT_BYTES);
    lds_gemm_bLDS<128, 128, KT, 2, 2, 1, OutT>
        <<<g, blk, lds>>>(dA, dBc, dsA, dsB, dD, N, kit, A_row_bytes, B_compact_row_bytes);
}

}  // namespace detail
}  // namespace mxfp6
