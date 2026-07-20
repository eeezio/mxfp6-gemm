#pragma once
// Public API for the unified MXFP6 GEMM (libmxfp6gemm).
//
// One kernel paradigm (hybrid drip-A: A staged deep-K in LDS double-buffered + B streamed
// coalesced HBM->VGPR ring + dripped A loads + RDB barrier + tiled-scale), shape-routed:
//   * 256x256 tile (16-acc) for shapes whose grid fills the machine (WG >= #CU)
//   * 128x256 tile (8-acc)  for WG-starved small-M shapes (fills idle CUs)
//
// This header is device-free (plain declarations) — a host translation unit can include it
// and link libmxfp6gemm without a HIP compiler. Inputs are pre-quantized DEVICE buffers:
// quantize A / preprocess+preshuffle B / tile the scales with mxfp6/preprocess.hpp first,
// using choose_tile(M,N).{MPW,NPW} for the scale grouping. Kp = K padded up to a multiple
// of the K-tile (192).
#include <cstddef>
#include <cstdint>

namespace mxfp6 {

enum class OutType { F32, F16, BF16 };

// K is processed in deep tiles of K_TILE; callers pad K up to kpad(K) (Kp) and pass that as the
// kernel's K. The padded K-tail must be zero on at least one operand (B in the pad-B-only recipe)
// so its contribution is nulled — see mxfp6/preprocess.hpp for the host-side padding helpers.
constexpr int K_TILE = 192;
inline int kpad(int K) {
    return (K + K_TILE - 1) / K_TILE * K_TILE;
}

// Tile chosen for (M,N[,Kp]). MPW/NPW = per-wave 32-block counts (2x2 waves); the host scale
// tiling (tile_scale) must use these for A and B respectively.
// Kp (optional, default 0 = unknown): when provided, enables K-aware routing (e.g. 128x128 for
// large-K wide-N shapes). Callers that know Kp (dispatch_gemm, perf benchmarks) should pass it so
// the tile choice matches the kernel that will actually run. Callers that don't know K (e.g. scale
// pre-processing at quantization time) pass the default 0 and get the M/N-only heuristic.
struct TileChoice {
    int MT, NT, MPW, NPW;
};
TileChoice choose_tile(int M, int N, int Kp = 0);

// Bytes of split-K workspace required for (M, N, Kp) — pass the same Kp you pass to gemm().
// Returns 0 for shapes that don't split (most shapes); for WG-starved narrow-N/large-K shapes it
// returns the size of the FP32 partial-sum buffer to allocate and hand to gemm() as `ws`. The
// caller owns this device buffer (alloc once, reuse across calls); sizing it from this function
// guarantees a match with what gemm() needs.
size_t gemm_workspace_size(int M, int N, int Kp);

// D[M,N] = A[M,K] * B[K,N], MXFP6 E2M3 inputs + per-32-block E8M0 scales.
//   dA   : packed MXFP6 A (row-major), A_row_bytes per row
//   dBsh : preshuffled MXFP6 B (preshuffle_B layout)
//   dsA  : tiled A scales (tile_scale with MPW)   dsB: tiled B scales (tile_scale with NPW)
//   dD   : output buffer of element type `ot` (float / __half / __hip_bfloat16)
//   ws/ws_bytes : caller-provided split-K workspace (device). Size with gemm_workspace_size().
//     Pass (nullptr, 0) to never split. If a shape would split but ws is null/too small, gemm()
//     runs unsplit (still correct, just without the speedup) and logs a warning to stderr.
void gemm(OutType ot, int M, int N, int Kp, const void* dA, const void* dBsh, const uint8_t* dsA,
          const uint8_t* dsB, void* dD, int A_row_bytes, int B_row_bytes, void* ws, size_t ws_bytes);

// Like gemm() but uses the caller-supplied TileChoice instead of calling choose_tile(). Intended
// for testing: forces a specific tile path (e.g. 128x128) on small shapes where choose_tile would
// return a different tile. The caller MUST have tiled the scales with tc.MPW/tc.NPW.
// Split-K is not applied (pass ws=nullptr/ws_bytes=0); M,N must be exact multiples of tc.MT/tc.NT.
void gemm_force_tile(OutType ot, int M, int N, int Kp, TileChoice tc, const void* dA,
                     const void* dBsh, const uint8_t* dsA, const uint8_t* dsB, void* dD,
                     int A_row_bytes, int B_row_bytes);

}  // namespace mxfp6
