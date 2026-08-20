#pragma once
#include <algorithm>
#include <cstdio>
#include <type_traits>
// INTERNAL: templated GEMM launch for the unified hybrid drip-A kernel. The public,
// type-erased entry points (mxfp6::gemm / mxfp6::choose_tile) live in mxfp6/gemm.hpp and
// are defined in src/gemm.cpp, which instantiates this template for F32/F16/BF16.
//
// ONE kernel paradigm (hybrid drip-A: A staged deep-K in LDS double-buffered + B streamed
// coalesced HBM->VGPR ring + dripped A loads + RDB barrier + tiled-scale), routed by shape:
//   * 256x256 (16-acc sweet spot) — workhorse for CU-filling shapes.
//   * 128x384 (12-acc, WAVES_M=1/WAVES_N=4) — wide-N, perfect CU fill when N%384==0.
//   * 128x256 (8-acc)             — WG-starved small-M, to fill idle CUs.
//   * 128x128 (4-acc)             — large-K wide-N that 128x384 cannot take; wins on wave count.
// Same lds_gemm_hybrid_dripA kernel for all (tile-general); only the tile args differ.
#include "kernel.hpp"
#include "mxfp6/gemm.hpp"  // TileChoice, choose_tile (declarations)
namespace mxfp6 {
namespace detail {

// XCD-aware workgroup remap, for lds_gemm_hybrid_dripA's SWZ parameter. Measured +4.6%/+3.8% on
// the 256x256 large-N shallow-K shapes and +3.2% on 128x384, but LOSES 3.5% on 128x128 (its
// A-slices are 10 MB at large K, so grouping by wg_n costs more A locality than it buys in B).
// 128x128 and the unmeasured 128x256 therefore stay unswizzled.
constexpr int SWZ_XCD = -1;

// B-ring depth on the 256x256 route: 8 at or above this N, the default 5 below. The gate is
// required -- the deeper ring wins +5.5~6.7% here and loses below (-1.05% at N=16128, -3.5% at
// 4096^3). Empirical: the lowest measured win, no mechanism established. Re-measure on any other part.
constexpr int BRING8_MIN_N = 81920;

// Split factor S for (M,N,Kp): the number of K-segments to split across independent workgroups
// for a WG-starved shape (S==1 means "do not split"). SINGLE SOURCE OF TRUTH — both the public
// gemm_workspace_size() query and dispatch_gemm() call this so the workspace a caller sizes always
// matches the workspace dispatch will use.
inline int splitk_S(int M, int N, int Kp) {
    constexpr int KT = K_TILE;
    constexpr int CU = 256;                 // MI350X (gfx950)
    constexpr int MIN_TILES_PER_SEG = 8;    // min deep tiles per segment (amortize prologue + small-K gate)
    TileChoice tc = choose_tile(M, N, Kp);
    // The split path only implements the 128x256 and 256x256 kernels. Splitting any other tile
    // launches a kernel whose NPW disagrees with the scales the caller tiled from choose_tile(),
    // which is silently wrong at small K and a memory fault at large K.
    if (tc.NT == 384 || tc.NT == 128) return 1;
    // Nor one with a partial last N-tile: base_wg below would floor the remainder tile away.
    if (N % tc.NT != 0) return 1;
    int k_tiles = (Kp / 64) / (KT / 64);    // total deep tiles for full K = Kp/192
    int base_wg = (M / tc.MT) * (N / tc.NT);
    int S = 1;
    // base_wg == 0 when the shape is smaller than the tile it fell through to (M<256 or N<256
    // against 256x256). Dividing by it is a SIGFPE in what is a host-side *query* function.
    if (base_wg > 0 && base_wg < CU && k_tiles >= 2 * MIN_TILES_PER_SEG) {
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

// 64-K sub-slabs of the LAST k-tile that still carry real data. The kernel skips the rest, both
// their MFMA and their B loads. 0 = run the whole tile.
//
// Rounds UP: MFMA consumes K=64 atomically, so a partly-real sub-slab must run in full, and its
// padding contributes 0 anyway. Over-running is wasteful, under-running is wrong -- which is why
// no divisibility guard on K_real is needed. An entirely real last tile normalizes back to 0
// rather than paying a barrier and a scale re-read to save nothing.
inline int tail_subs_for(int K_real) {
    if (K_real <= 0) return 0;
    const int run = (K_real % K_TILE + 63) / 64;
    return run == K_TILE / 64 ? 0 : run;
}

// The one place runtime tail_subs becomes compile-time TAIL_SUBS. Every 256x256 launch goes
// through here so the template argument lists cannot drift apart -- identical arguments are the
// only reason the force-tile short-tail arms cost no extra kernels. TAIL_ARMS must be a template
// parameter: a runtime 0 would still instantiate the unused TAIL_SUBS 1 and 2 kernels.
template <typename OutT, bool KGE2, bool NMASK, int PFD, bool TAIL_ARMS>
inline void launch_256x256(int tail_subs, dim3 g, int lds, const void* dA, const void* dBsh,
                           const uint8_t* dsA, const uint8_t* dsB, OutT* dD, int N, int kit,
                           int A_row_bytes, int B_row_bytes, int k_tiles) {
    static_assert(K_TILE / 64 == 3, "launch_256x256 enumerates tail_subs 1..K_TILE/64-1 by hand");
    auto go = [&](auto TS) {
        lds_gemm_hybrid_dripA<256, 256, K_TILE, 2, 2, 1, SWZ_XCD, OutT, false, KGE2, PFD, true, 1,
                              1, 1, 0, NMASK, decltype(TS)::value>
            <<<g, dim3(256), lds>>>(dA, dBsh, dsA, dsB, dD, N, kit, A_row_bytes, B_row_bytes, 0,
                                    k_tiles, nullptr);
    };
    if constexpr (KGE2 && TAIL_ARMS) {
        if (tail_subs == 1) return go(std::integral_constant<int, 1>{});
        if (tail_subs == 2) return go(std::integral_constant<int, 2>{});
    }
    go(std::integral_constant<int, 0>{});
}

// Kp = K padded to a multiple of K_TILE(=192). Caller must have tiled the scales with
// choose_tile(...).MPW/NPW and preshuffled B (preshuffle_B).
// ws/ws_bytes: caller-provided split-K workspace (see gemm_workspace_size). If a WG-starved shape
// would split but ws is null or too small, split is skipped (the result is still correct, just
// without the speedup) and a warning is logged to stderr.
template <typename OutT, bool KGE2>
inline void dispatch_gemm_kge(int M, int N, int Kp, const void* dA, const void* dBsh,
                              const uint8_t* dsA, const uint8_t* dsB, OutT* dD, int A_row_bytes,
                              int B_row_bytes, void* ws, size_t ws_bytes, int K_real = 0) {
    // Only the 256x256 route at N >= BRING8_MIN_N acts on this; other routes ignore K_real.
    // Deliberate: the short tail measured -0.52% at N=16128.
    const int tail_subs = tail_subs_for(K_real);
    constexpr int KT = K_TILE;
    dim3 blk(256);
    int kit = Kp / 64;
    int k_tiles = kit / (KT / 64);          // total deep tiles for full K = Kp/192
    TileChoice tc = choose_tile(M, N, Kp);
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
    // split-K only exists in the KGE2 arm: splitk_S floors every segment at 8 tiles, so any
    // shape that splits already has k_tiles_seg >= 8.
    if constexpr (KGE2) {
        if (S > 1) {
            int seg_floor = k_tiles / S, seg_rem = k_tiles % S;
            size_t total = (size_t)M * N;
            float* Dpart = static_cast<float*>(ws);
            // Only the 128x256 and 256x256 tiles are implemented here; splitk_S() returns S=1 for
            // every other tile so this branch cannot be reached with one.
            if (tc.MT == 128) {
                dim3 g(M / 128, N / 256, S);
                int lds = 2 * (128 * (KT * 6 / 8));
                lds_gemm_hybrid_dripA<128, 256, KT, 2, 2, 1, 0, OutT, true, KGE2>
                    <<<g, blk, lds>>>(dA, dBsh, dsA, dsB, dD, N, kit, A_row_bytes, B_row_bytes,
                                      seg_floor, seg_rem, Dpart);
            } else {
                dim3 g(M / 256, N / 256, S);
                int lds = 2 * (256 * (KT * 6 / 8));
                lds_gemm_hybrid_dripA<256, 256, KT, 2, 2, 1, 0, OutT, true, KGE2>
                    <<<g, blk, lds>>>(dA, dBsh, dsA, dsB, dD, N, kit, A_row_bytes, B_row_bytes,
                                      seg_floor, seg_rem, Dpart);
            }
            int threads = 256;
            long red_blocks = ((long)total + threads - 1) / threads;
            reduce_splitk<OutT><<<dim3((unsigned)red_blocks), dim3(threads)>>>(Dpart, dD, (long)total, S);
            return;
        }
    }
    if (tc.MT == 128 && tc.NT == 128) {
        // 128x128 tile: large-K wide-N path. Beats 128x256 by dropping the half-empty second CU
        // wave (measured 1274 -> 1767 on 2048x6144x105728), not by fitting B in L2 — its 128-col
        // B/N-slice is 10.16 MB against a 4 MB L2. NB=6 (vs 12 for 128x256).
        dim3 g(M / 128, N / 128);
        int lds = 2 * (128 * (KT * 6 / 8));
        lds_gemm_hybrid_dripA<128, 128, KT, 2, 2, 1, 0, OutT, false, KGE2>
            <<<g, blk, lds>>>(dA, dBsh, dsA, dsB, dD, N, kit, A_row_bytes, B_row_bytes,
                              0, k_tiles, nullptr);
    } else if (tc.MT == 128 && tc.NT == 384) {
        // 128x384 tile: wide-N path — perfect CU fill (256 WGs), 12 acc/wave.
        // WAVES_M=1, WAVES_N=4 → N_PW=3 ≤ 4 (satisfies scale static_assert).
        // NB=9 but compute/B-slot = 1152/9 = 128 cyc (vs 64 for 128x256) → deeper look-ahead.
        dim3 g(M / 128, N / 384);
        int lds = 2 * (128 * (KT * 6 / 8));
        lds_gemm_hybrid_dripA<128, 384, KT, 1, 4, 1, SWZ_XCD, OutT, false, KGE2>
            <<<g, blk, lds>>>(dA, dBsh, dsA, dsB, dD, N, kit, A_row_bytes, B_row_bytes,
                              0, k_tiles, nullptr);
    } else if (tc.MT == 128) {
        // occ1. (The 8-acc 128x256 tile CAN be forced to occ2 via MIN_OCC=2 — 251 VGPR,
        // 0 spill — but in STEADY STATE that gives ~0% on small-M: a 256-WG shape has no
        // 2nd WG to fill the occ2 slot. An earlier +6~10% reading was a warm-up artifact
        // of always benchmarking occ1 first on a still-ramping GPU. Kept at occ1.)
        dim3 g(M / 128, N / 256);
        int lds = 2 * (128 * (KT * 6 / 8));
        // K-gated B-ring depth for the residual 128x256 large-K shapes (those too small for
        // the 128x128 wide-N path to fill CUs): in the very-large-K regime (Kp>50000, where
        // the 256-col B/N-slice exceeds L2) a deeper ring (PFD=7, measured Scratch=0) recovers
        // ~2-3% by covering more HBM-miss latency; below that PFD=5 stays best (deeper ring
        // costs ~1-2% in the L2-hit regime).
        if (Kp > 50000) {
            lds_gemm_hybrid_dripA<128, 256, KT, 2, 2, 1, 0, OutT, false, KGE2, 7>
                <<<g, blk, lds>>>(dA, dBsh, dsA, dsB, dD, N, kit, A_row_bytes, B_row_bytes,
                                  0, k_tiles, nullptr);
        } else {
            lds_gemm_hybrid_dripA<128, 256, KT, 2, 2, 1, 0, OutT, false, KGE2, 5>
                <<<g, blk, lds>>>(dA, dBsh, dsA, dsB, dD, N, kit, A_row_bytes, B_row_bytes,
                                  0, k_tiles, nullptr);
        }
    } else if (N % 256 != 0) {
        // NMASK: 256x256 over a ceil(N/256) grid, last N-tile masked at the store. For N%128==0,
        // N%256!=0, where the only tile that divides N is 128x128 -- and a 4-acc tile amortizes
        // the shallow-K fixed cost over a quarter of the work, which costs far more than wasting
        // half of one N-tile (measured 1274 vs 1455 TFLOPs on 2048x102272x1024).
        dim3 g(M / 256, (N + 255) / 256);
        int lds = 2 * (256 * (KT * 6 / 8));
        if (N >= BRING8_MIN_N)
            launch_256x256<OutT, KGE2, true, 8, true>(tail_subs, g, lds, dA, dBsh, dsA, dsB, dD, N,
                                                      kit, A_row_bytes, B_row_bytes, k_tiles);
        else
            launch_256x256<OutT, KGE2, true, 5, false>(tail_subs, g, lds, dA, dBsh, dsA, dsB, dD, N,
                                                       kit, A_row_bytes, B_row_bytes, k_tiles);
    } else {
        dim3 g(M / 256, N / 256);
        int lds = 2 * (256 * (KT * 6 / 8));
        if (N >= BRING8_MIN_N)
            launch_256x256<OutT, KGE2, false, 8, true>(tail_subs, g, lds, dA, dBsh, dsA, dsB, dD, N,
                                                       kit, A_row_bytes, B_row_bytes, k_tiles);
        else
            launch_256x256<OutT, KGE2, false, 5, false>(tail_subs, g, lds, dA, dBsh, dsA, dsB, dD,
                                                        N, kit, A_row_bytes, B_row_bytes, k_tiles);
    }
}

// k_tiles_seg < 2 only happens at Kp == K_TILE, so the KGE2 arm covers every other shape and lets
// the compiler drop the loop guard -- which is what stops it emitting the acc zero-init twice.
template <typename OutT>
inline void dispatch_gemm(int M, int N, int Kp, const void* dA, const void* dBsh,
                          const uint8_t* dsA, const uint8_t* dsB, OutT* dD, int A_row_bytes,
                          int B_row_bytes, void* ws, size_t ws_bytes, int K_real = 0) {
    if (Kp >= 2 * K_TILE)
        dispatch_gemm_kge<OutT, true>(M, N, Kp, dA, dBsh, dsA, dsB, dD, A_row_bytes, B_row_bytes,
                                      ws, ws_bytes, K_real);
    else
        dispatch_gemm_kge<OutT, false>(M, N, Kp, dA, dBsh, dsA, dsB, dD, A_row_bytes, B_row_bytes,
                                       ws, ws_bytes, K_real);
}

// Like dispatch_gemm but uses a caller-supplied TileChoice instead of choose_tile(). No split-K.
// Used by gemm_force_tile() to exercise a specific kernel path on small shapes for testing.
template <typename OutT, bool KGE2>
inline void dispatch_gemm_force_tile_kge(int M, int N, int Kp, TileChoice tc, const void* dA,
                                         const void* dBsh, const uint8_t* dsA, const uint8_t* dsB,
                                         OutT* dD, int A_row_bytes, int B_row_bytes, int K_real = 0) {
    const int tail_subs = tail_subs_for(K_real);
    constexpr int KT = K_TILE;
    dim3 blk(256);
    int kit = Kp / 64;
    int k_tiles = kit / (KT / 64);
    if (tc.MT == 128 && tc.NT == 128) {
        dim3 g(M / 128, N / 128);
        int lds = 2 * (128 * (KT * 6 / 8));
        lds_gemm_hybrid_dripA<128, 128, KT, 2, 2, 1, 0, OutT, false, KGE2>
            <<<g, blk, lds>>>(dA, dBsh, dsA, dsB, dD, N, kit, A_row_bytes, B_row_bytes,
                              0, k_tiles, nullptr);
    } else if (tc.MT == 128 && tc.NT == 384) {
        dim3 g(M / 128, N / 384);
        int lds = 2 * (128 * (KT * 6 / 8));
        lds_gemm_hybrid_dripA<128, 384, KT, 1, 4, 1, SWZ_XCD, OutT, false, KGE2>
            <<<g, blk, lds>>>(dA, dBsh, dsA, dsB, dD, N, kit, A_row_bytes, B_row_bytes,
                              0, k_tiles, nullptr);
    } else if (tc.MT == 128) {
        dim3 g(M / 128, N / 256);
        int lds = 2 * (128 * (KT * 6 / 8));
        lds_gemm_hybrid_dripA<128, 256, KT, 2, 2, 1, 0, OutT, false, KGE2>
            <<<g, blk, lds>>>(dA, dBsh, dsA, dsB, dD, N, kit, A_row_bytes, B_row_bytes,
                              0, k_tiles, nullptr);
    } else if (N % 256 != 0) {
        dim3 g(M / 256, (N + 255) / 256);  // NMASK, as in dispatch_gemm_kge above
        int lds = 2 * (256 * (KT * 6 / 8));
        // PFD asymmetry is deliberate: tail arms must be PFD=8 to dedup against the routed kernel,
        // the plain arm PFD=5 to match the routed small-N one. See the else branch for why at all.
        if constexpr (KGE2) {
            if (tail_subs) {
                launch_256x256<OutT, KGE2, true, 8, true>(tail_subs, g, lds, dA, dBsh, dsA, dsB, dD,
                                                          N, kit, A_row_bytes, B_row_bytes, k_tiles);
                return;
            }
        }
        launch_256x256<OutT, KGE2, true, 5, false>(tail_subs, g, lds, dA, dBsh, dsA, dsB, dD, N, kit,
                                                   A_row_bytes, B_row_bytes, k_tiles);
    } else {
        dim3 g(M / 256, N / 256);
        int lds = 2 * (256 * (KT * 6 / 8));
        // The short-tail arm is wired here only because the routed one needs N >= BRING8_MIN_N,
        // which no CPU reference can check. Same template args, so it adds no kernels.
        if constexpr (KGE2) {
            if (tail_subs) {
                launch_256x256<OutT, KGE2, false, 8, true>(tail_subs, g, lds, dA, dBsh, dsA, dsB,
                                                           dD, N, kit, A_row_bytes, B_row_bytes,
                                                           k_tiles);
                return;
            }
        }
        launch_256x256<OutT, KGE2, false, 5, false>(tail_subs, g, lds, dA, dBsh, dsA, dsB, dD, N, kit,
                                                    A_row_bytes, B_row_bytes, k_tiles);
    }
}

template <typename OutT>
inline void dispatch_gemm_force_tile(int M, int N, int Kp, TileChoice tc, const void* dA,
                                     const void* dBsh, const uint8_t* dsA, const uint8_t* dsB,
                                     OutT* dD, int A_row_bytes, int B_row_bytes, int K_real = 0) {
    if (Kp >= 2 * K_TILE)
        dispatch_gemm_force_tile_kge<OutT, true>(M, N, Kp, tc, dA, dBsh, dsA, dsB, dD, A_row_bytes,
                                                 B_row_bytes, K_real);
    else
        dispatch_gemm_force_tile_kge<OutT, false>(M, N, Kp, tc, dA, dBsh, dsA, dsB, dD, A_row_bytes,
                                                  B_row_bytes, K_real);
}

}  // namespace detail
}  // namespace mxfp6
