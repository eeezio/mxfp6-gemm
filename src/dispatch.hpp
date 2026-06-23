#pragma once
#include <algorithm>
// INTERNAL: templated GEMM launch for the unified hybrid drip-A kernel. The public,
// type-erased entry points (mxfp6::gemm / mxfp6::choose_tile) live in mxfp6/gemm.hpp and
// are defined in src/gemm.cpp, which instantiates this template for F32/F16/BF16.
//
// ONE kernel paradigm (hybrid drip-A: A staged deep-K in LDS double-buffered + B streamed
// coalesced HBM->VGPR ring + dripped A loads + RDB barrier + tiled-scale), routed by shape:
//   * 256x256 (16-acc sweet spot) — workhorse for CU-filling shapes.
//   * 128x256 (8-acc)             — WG-starved small-M, to fill idle CUs.
// Same lds_gemm_hybrid_dripA kernel for both (tile-general); only the tile args differ.
#include "kernel.hpp"
#include "mxfp6/gemm.hpp"  // TileChoice, choose_tile (declarations)
namespace mxfp6 {
namespace detail {

// Split-K partial-sum workspace, cached across calls. A per-call hipMalloc/hipFree of the
// S*M*N FP32 buffer costs ~1.3 ms (both calls device-synchronize) — ~14x the entire tier-1
// GEMM, which would make split-K a net loss in any repeated-call workload. So we keep one
// grow-on-demand buffer for the process lifetime instead. Single-stream library (matches the
// no-locking convention here); the OS reclaims the buffer at exit.
inline float* splitk_workspace(size_t bytes) {
    static float* ptr = nullptr;
    static size_t cap = 0;
    if (bytes > cap) {
        if (ptr) hipFree(ptr);
        hipMalloc(&ptr, bytes);
        cap = bytes;
    }
    return ptr;
}

// Kp = K padded to a multiple of K_TILE(=192). Caller must have tiled the scales with
// choose_tile(...).MPW/NPW and preshuffled B (preshuffle_B).
template <typename OutT>
inline void dispatch_gemm(int M, int N, int Kp, const void* dA, const void* dBsh,
                          const uint8_t* dsA, const uint8_t* dsB, OutT* dD, int A_row_bytes,
                          int B_row_bytes) {
    constexpr int KT = K_TILE;
    constexpr int CU = 256;                 // MI350X (gfx950)
    constexpr int MIN_TILES_PER_SEG = 8;    // min deep tiles per segment (amortize prologue + small-K gate)
    TileChoice tc = choose_tile(M, N);
    dim3 blk(256);
    int kit = Kp / 64;
    int k_tiles = kit / (KT / 64);          // total deep tiles for full K = Kp/192
    int base_wg = (M / tc.MT) * (N / 256);
    int S = 1;
    if (base_wg < CU && k_tiles >= 2 * MIN_TILES_PER_SEG) {
        S = (CU + base_wg - 1) / base_wg;             // ceil(CU/base_wg): min segments to fill CUs
        int s_cap = k_tiles / MIN_TILES_PER_SEG;      // each segment not shorter than the floor
        if (S > s_cap) S = s_cap;
        if (S < 1) S = 1;
        while (S > 1 && (k_tiles % S) != 0) S--;       // divisibility: equal-length segments, kernel uses blockIdx.z
    }
    if (S > 1) {
        int seg_tiles = k_tiles / S;                  // exact (Step 1 guaranteed divisibility)
        size_t total = (size_t)M * N;
        float* Dpart = splitk_workspace((size_t)S * total * sizeof(float));
        if (tc.MT == 128) {
            dim3 g(M / 128, N / 256, S);
            int lds = 2 * (128 * (KT * 6 / 8));
            lds_gemm_hybrid_dripA<128, 256, KT, 2, 2, 1, 0, OutT, true>
                <<<g, blk, lds>>>(dA, dBsh, dsA, dsB, dD, N, kit, A_row_bytes, B_row_bytes,
                                  0, seg_tiles, Dpart);
        } else {
            dim3 g(M / 256, N / 256, S);
            int lds = 2 * (256 * (KT * 6 / 8));
            lds_gemm_hybrid_dripA<256, 256, KT, 2, 2, 1, 0, OutT, true>
                <<<g, blk, lds>>>(dA, dBsh, dsA, dsB, dD, N, kit, A_row_bytes, B_row_bytes,
                                  0, seg_tiles, Dpart);
        }
        int threads = 256;
        long red_blocks = ((long)total + threads - 1) / threads;
        reduce_splitk<OutT><<<dim3((unsigned)red_blocks), dim3(threads)>>>(Dpart, dD, (long)total, S);
        return;
    }
    if (tc.MT == 128) {
        // occ1. (The 8-acc 128x256 tile CAN be forced to occ2 via MIN_OCC=2 — 251 VGPR,
        // 0 spill — but in STEADY STATE that gives ~0% on small-M: a 256-WG shape has no
        // 2nd WG to fill the occ2 slot. An earlier +6~10% reading was a warm-up artifact
        // of always benchmarking occ1 first on a still-ramping GPU. Kept at occ1.)
        dim3 g(M / 128, N / 256);
        int lds = 2 * (128 * (KT * 6 / 8));
        lds_gemm_hybrid_dripA<128, 256, KT, 2, 2, 1, 0, OutT, false>
            <<<g, blk, lds>>>(dA, dBsh, dsA, dsB, dD, N, kit, A_row_bytes, B_row_bytes,
                              0, k_tiles, nullptr);
    } else {
        dim3 g(M / 256, N / 256);
        int lds = 2 * (256 * (KT * 6 / 8));
        lds_gemm_hybrid_dripA<256, 256, KT, 2, 2, 1, 0, OutT, false>
            <<<g, blk, lds>>>(dA, dBsh, dsA, dsB, dD, N, kit, A_row_bytes, B_row_bytes,
                              0, k_tiles, nullptr);
    }
}

}  // namespace detail
}  // namespace mxfp6
