#pragma once
#include <algorithm>
#include <cstdio>
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

// Split factor S for (M,N,Kp): the number of K-segments to split across independent workgroups
// for a WG-starved shape (S==1 means "do not split"). SINGLE SOURCE OF TRUTH — both the public
// gemm_workspace_size() query and dispatch_gemm() call this so the workspace a caller sizes always
// matches the workspace dispatch will use.
inline int splitk_S(int M, int N, int Kp) {
    constexpr int KT = K_TILE;
    constexpr int CU = 256;                 // MI350X (gfx950)
    constexpr int MIN_TILES_PER_SEG = 8;    // min deep tiles per segment (amortize prologue + small-K gate)
    TileChoice tc = choose_tile(M, N);
    int k_tiles = (Kp / 64) / (KT / 64);    // total deep tiles for full K = Kp/192
    int base_wg = (M / tc.MT) * (N / 256);
    int S = 1;
    if (base_wg < CU && k_tiles >= 2 * MIN_TILES_PER_SEG) {
        S = (CU + base_wg - 1) / base_wg;             // ceil(CU/base_wg): min segments to fill CUs
        int s_cap = k_tiles / MIN_TILES_PER_SEG;      // each segment not shorter than the floor
        if (S > s_cap) S = s_cap;
        if (S < 1) S = 1;
    }
    return S;
}

// Bytes of split-K partial-sum workspace dispatch_gemm would use for (M,N,Kp). 0 if no split.
inline size_t splitk_workspace_bytes(int M, int N, int Kp) {
    int S = splitk_S(M, N, Kp);
    return S > 1 ? (size_t)S * (size_t)M * N * sizeof(float) : 0;
}

// Kp = K padded to a multiple of K_TILE(=192). Caller must have tiled the scales with
// choose_tile(...).MPW/NPW and preshuffled B (preshuffle_B).
// ws/ws_bytes: caller-provided split-K workspace (see gemm_workspace_size). If a WG-starved shape
// would split but ws is null or too small, split is skipped (the result is still correct, just
// without the speedup) and a warning is logged to stderr.
template <typename OutT>
inline void dispatch_gemm(int M, int N, int Kp, const void* dA, const void* dBsh,
                          const uint8_t* dsA, const uint8_t* dsB, OutT* dD, int A_row_bytes,
                          int B_row_bytes, void* ws, size_t ws_bytes) {
    constexpr int KT = K_TILE;
    dim3 blk(256);
    int kit = Kp / 64;
    int k_tiles = kit / (KT / 64);          // total deep tiles for full K = Kp/192
    TileChoice tc = choose_tile(M, N);
    int S = splitk_S(M, N, Kp);
    if (S > 1) {
        size_t need = (size_t)S * (size_t)M * N * sizeof(float);
        if (!ws || ws_bytes < need) {
            // Caller wanted a WG-starved shape split but gave no/insufficient workspace. Fall back
            // to the single-kernel path (correct, just slower) and warn so the miss is visible.
            fprintf(stderr,
                    "[mxfp6] split-K skipped for %dx%dx%d: workspace %zu bytes, need %zu — "
                    "running unsplit (slower). Size it with gemm_workspace_size().\n",
                    M, N, Kp, ws_bytes, need);
            S = 1;
        }
    }
    if (S > 1) {
        int seg_floor = k_tiles / S, seg_rem = k_tiles % S;
        size_t total = (size_t)M * N;
        float* Dpart = static_cast<float*>(ws);
        if (tc.MT == 128) {
            dim3 g(M / 128, N / 256, S);
            int lds = 2 * (128 * (KT * 6 / 8));
            lds_gemm_hybrid_dripA<128, 256, KT, 2, 2, 1, 0, OutT, true>
                <<<g, blk, lds>>>(dA, dBsh, dsA, dsB, dD, N, kit, A_row_bytes, B_row_bytes,
                                  seg_floor, seg_rem, Dpart);
        } else {
            dim3 g(M / 256, N / 256, S);
            int lds = 2 * (256 * (KT * 6 / 8));
            lds_gemm_hybrid_dripA<256, 256, KT, 2, 2, 1, 0, OutT, true>
                <<<g, blk, lds>>>(dA, dBsh, dsA, dsB, dD, N, kit, A_row_bytes, B_row_bytes,
                                  seg_floor, seg_rem, Dpart);
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
