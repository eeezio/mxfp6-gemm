#pragma once
// B-through-LDS kernel v2: both A and B DRIPPED into LDS double-buffers.
//
// v1 mistake: B was BURST at the start of compute() (all 9 chunks issued before the MFMA
// loop). wait_vmcnt(0) at the next barrier stalled immediately (HBM latency not hidden).
// Performance was BELOW baseline (1197 vs 1262 TFLOPS) because we added LDS overhead
// without hiding B's HBM latency.
//
// v2 fix: DRIP both A and B across the MFMA quartets, exactly mirroring A's proven mechanism.
// During compute of tile k:
//   - Issue A(k+1) chunk[i] at MFMA quartet p = ADRIP_START + i (one per quartet)
//   - Issue B(k+1) chunk[j] at MFMA quartet p = BDRIP_START + j (one per quartet)
// Both sets of buffer_load_lds are asm-invisible; wait_vmcnt(0) before the next barrier
// drains both A and B outstanding loads together (VMCNT is one shared counter).
// B loads issued early in the compute window → by the barrier they are ~NB*32 = 384 cycles
// deeper into the HBM pipeline → overlap reduces effective stall vs burst.
//
// Drip schedule (NB=12 quartets for 128x256):
//   ISSUES_A = ceil(M_TILE * ROW_CHUNKS / 256) = ceil(128*9/256) = 5
//   ISSUES_B = ceil(N_TILE * ROW_CHUNKS / 256) = ceil(256*9/256) = 9
//   Combined: 5+9=14 operations across NB=12 quartets → co-issue 1A+1B per quartet for
//   first 5 quartets, then 1B/quartet for quartets 6-10 (all fits in NB=12 with margin).
//
// B SOURCE LAYOUT: compact row-major packed B^T [N_col][K_byte], row stride = KT_BYTES per K-tile
// (B_q.packed_data from preprocess_B, NOT preshuffled).
//
// LDS layout: [A_buf0 | A_buf1 | B_buf0 | B_buf1]
//   A_buf: M_TILE * KT_BYTES bytes, same verified BC-free row-major layout
//   B_buf: N_TILE * KT_BYTES bytes, same row-major layout → read_op reusable
//   128x256: 2*(128*144) + 2*(256*144) = 36864 + 73728 = 110592 = 108KB < 160KB ✓
#include "device_ops.hpp"
#include "kernel.hpp"   // issue_A_chunks (used in prologue + drip), read_op, etc.
namespace mxfp6 {

// Issue B cooperative buffer_load_lds chunks [i0,i1) for ONE K-tile into LDS buffer at lds_base.
// Exact mirror of issue_A_chunks with B's compact row-major B^T layout:
//   chunk = i*256 + wave*64 + lane
//   n_col = chunk / CPR   (CPR = KT_BYTES/16, which N-column row in the WG's N-slice)
//   ck    = chunk % CPR   (which 16B chunk within that N-col row's KT_BYTES)
//   voff  = n_col * row_stride + kt_byte + ck*16
//     (row_stride = B_compact_row_bytes = fp6_packed_bytes(Kp), the full per-N-row stride)
// M0 is set per issue-group (wave-aligned) so consecutive lanes map to consecutive LDS dests.
// NCH guard: skips lanes whose chunk >= NCH (for partial last group, same as issue_A_chunks).
template <int CPR, int NCH = (1 << 30)>
__device__ __forceinline__ void issue_B_chunks(uint32_t lds_base, int row_stride, int kt_byte,
                                               int wave, int lane, const v4i& brsrc, int i0,
                                               int i1) {
    for (int i = i0; i < i1; i++) {
        int chunk = i * 256 + wave * 64 + lane;
        set_m0(__builtin_amdgcn_readfirstlane(lds_base + (uint32_t)((i * 256 + wave * 64) * 16)));
        if (chunk < NCH) {
            int n = chunk / CPR, ck = chunk % CPR;
            uint32_t voff = (uint32_t)(n * row_stride + kt_byte + ck * 16);
            asm volatile("s_nop 0\n buffer_load_dwordx4 %0, %1, 0 offen lds"
                         :
                         : "v"(voff), "s"(brsrc)
                         : "memory");
        }
    }
}

// Both-A-and-B-through-LDS GEMM kernel (drip version).
// A drip: 1 chunk per MFMA quartet starting at ADRIP_START=1 (5 chunks total for 128x256).
// B drip: 1 chunk per MFMA quartet starting at BDRIP_START (9 chunks total for 128x256).
//   BDRIP_START set so B drip starts early (maximizes latency hiding) but doesn't race A.
//   With NB=12 quartets, A occupies p=1..5, B can start at p=1 too (co-issue, different LDS).
//   => for p in [1..5]: issue 1 A chunk + 1 B chunk; for p in [6..9]: issue 1 B chunk only.
//   The 9 B chunks are all issued by p=9, leaving p=10,11 as latency margin.
// Prologue: burst both A and B tile-0 (no prior compute window to drip into).
// Scales: identical tiled-scale layout as the original lds_gemm_hybrid_dripA kernel.
template <int M_TILE, int N_TILE, int K_TILE, int WAVES_M, int WAVES_N, int MIN_OCC = 1,
          typename OutT = float>
__global__ void __launch_bounds__(256, MIN_OCC)
    lds_gemm_bLDS(const void* __restrict__ A, const void* __restrict__ Bc,
                  const uint8_t* __restrict__ sA, const uint8_t* __restrict__ sB,
                  OutT* __restrict__ D, int N, int k_iters, int A_row_bytes,
                  int B_compact_row_bytes) {
    constexpr int KT_BYTES = K_TILE * 6 / 8;
    constexpr int ROW_CHUNKS = KT_BYTES / 16;
    constexpr int K64_PER_TILE = K_TILE / 64;
    constexpr int M_BLKS = M_TILE / 32, N_BLKS = N_TILE / 32;
    constexpr int M_PW = M_BLKS / WAVES_M, N_PW = N_BLKS / WAVES_N;
    constexpr int A_BYTES = M_TILE * KT_BYTES;
    constexpr int B_BYTES = N_TILE * KT_BYTES;
    constexpr int NB = K64_PER_TILE * N_PW;   // MFMA quartet count per tile
    // ceil(total_chunks / 256) drip iterations for A and B respectively
    constexpr int ISSUES_A = (M_TILE * ROW_CHUNKS + 255) / 256;
    constexpr int ISSUES_B = (N_TILE * ROW_CHUNKS + 255) / 256;
    // A drip: p=1..ISSUES_A (skip p=0 sub-head stall, matching original ADRIP_START=1)
    // B drip: start at p=1 (co-issue with A for first ISSUES_A quartets, then B alone)
    //   B chunk j issued at p = 1 + j  (j=0..ISSUES_B-1, all fit since 1+ISSUES_B-1 < NB)
    static_assert(1 + ISSUES_B - 1 < NB, "B drip does not fit in compute window");

    // LDS layout: [A_buf0 | A_buf1 | B_buf0 | B_buf1]
    constexpr uint32_t A_BUF0 = 0;
    constexpr uint32_t A_BUF1 = A_BYTES;
    constexpr uint32_t B_BUF0 = 2 * A_BYTES;
    constexpr uint32_t B_BUF1 = 2 * A_BYTES + B_BYTES;

    extern __shared__ char smem[];
    int tid = threadIdx.x, wave = tid / 64, lane = tid % 64;
    int wm = wave / WAVES_N, wn = wave % WAVES_N;
    int wg_m = blockIdx.x, wg_n = blockIdx.y;

    const char* Ag = reinterpret_cast<const char*>(A) + (size_t)(wg_m * M_TILE) * A_row_bytes;
    // Bc = compact B^T[N_total][Kp_bytes]; WG's N-slice starts at wg_n*N_TILE rows.
    const char* Bg = reinterpret_cast<const char*>(Bc) +
                     (size_t)(wg_n * N_TILE) * B_compact_row_bytes;

    AccTile acc[M_PW][N_PW];
#pragma unroll
    for (int mi = 0; mi < M_PW; mi++)
#pragma unroll
        for (int ni = 0; ni < N_PW; ni++) clear_acc(acc[mi][ni]);

    constexpr int SA_PAD = ((M_PW + 3) / 4) * 4, SB_PAD = ((N_PW + 3) / 4) * 4;
    constexpr int NDA = SA_PAD / 4, NDB = SB_PAD / 4;
    static_assert(NDA == 1 && NDB == 1, "tiled-scale path assumes <=4 blocks/wave");
    int sa_grp = wg_m * WAVES_M + wm, sb_grp = wg_n * WAVES_N + wn;
    int k_tiles = k_iters / K64_PER_TILE;

    // A buffer descriptor (base = WG's M-row start)
    uint64_t ab = reinterpret_cast<uint64_t>(Ag);
    v4i arsrc{(int)(uint32_t)ab, (int)((uint32_t)(ab >> 32) & 0xFFFF), (int)0x7FFFFFFF,
              (int)0x00020000};
    // B buffer descriptor (base = WG's N-slice start in compact B^T)
    uint64_t bb = reinterpret_cast<uint64_t>(Bg);
    v4i brsrc{(int)(uint32_t)bb, (int)((uint32_t)(bb >> 32) & 0xFFFF), (int)0x7FFFFFFF,
              (int)0x00020000};

    auto load_scales = [&](int kt, int (*sa)[NDA], int (*sb)[NDB]) {
        const char* pa = reinterpret_cast<const char*>(sA) +
                         (size_t)((sa_grp * k_tiles + kt) * 64 + lane) * K64_PER_TILE * SA_PAD;
        const char* pb = reinterpret_cast<const char*>(sB) +
                         (size_t)((sb_grp * k_tiles + kt) * 64 + lane) * K64_PER_TILE * SB_PAD;
        int ta[K64_PER_TILE], tb[K64_PER_TILE];
        asm_load_dwordxN_nowait(ta, pa, K64_PER_TILE);
        asm_load_dwordxN_nowait(tb, pb, K64_PER_TILE);
#pragma unroll
        for (int sub = 0; sub < K64_PER_TILE; sub++) {
            sa[sub][0] = ta[sub];
            sb[sub][0] = tb[sub];
        }
    };

    // Compute tile (cur_a, cur_b from LDS), DRIP A(nxt) and B(nxt) into (nxt_a, nxt_b).
    // A drip: 1 chunk/quartet at p=1,2,...,ISSUES_A  (A loads start p=1, skip p=0 sub-head)
    // B drip: 1 chunk/quartet at p=1,2,...,ISSUES_B  (co-issue with A when p<=ISSUES_A)
    // Both A and B use asm buffer_load_lds => both counted by vmcnt (one shared counter).
    // wait_vmcnt(0) before next barrier drains all outstanding A+B loads together.
    auto compute = [&](uint32_t cur_a, uint32_t cur_b, uint32_t nxt_a, uint32_t nxt_b,
                       int kt_nxt, bool do_drip,
                       const int (*sa)[NDA], const int (*sb)[NDB]) {
        int kb_nxt = kt_nxt * KT_BYTES;

        v6i a[M_PW];
        int sav[M_PW];

#pragma unroll
        for (int p = 0; p < NB; p++) {
            int sub = p / N_PW, ni = p % N_PW;
            if (ni == 0) {
#pragma unroll
                for (int mi = 0; mi < M_PW; mi++) {
                    int blk = wm * M_PW + mi;
                    a[mi] = read_op<KT_BYTES>(smem, cur_a, blk, sub, lane);
                    sav[mi] = (sa[sub][mi / 4] >> (8 * (mi % 4))) & 0xff;
                }
            }
            // Read B from LDS (same read_op, same BC-free layout, blk_n = N-col block index).
            int blk_n = wn * N_PW + ni;
            v6i b = read_op<KT_BYTES>(smem, cur_b, blk_n, sub, lane);
            int sbv = (sb[sub][ni / 4] >> (8 * (ni % 4))) & 0xff;

            if (do_drip) {
                // A drip: issue chunk p-1 at MFMA quartet p (start at p=1, skip p=0 sub-head).
                // p=1 -> A chunk 0, p=2 -> A chunk 1, ..., p=ISSUES_A -> A chunk ISSUES_A-1.
                if (p >= 1) {
                    int a_slot = p - 1;  // chunk index in [0, ISSUES_A)
                    if (a_slot < ISSUES_A)
                        issue_A_chunks<ROW_CHUNKS, M_TILE * ROW_CHUNKS>(
                            nxt_a, A_row_bytes, kb_nxt, wave, lane, arsrc, a_slot, a_slot + 1);
                }
                // B drip: issue chunk p-1 at MFMA quartet p (same schedule as A).
                // p=1 -> B chunk 0, p=2 -> B chunk 1, ..., p=ISSUES_B -> B chunk ISSUES_B-1.
                if (p >= 1) {
                    int b_slot = p - 1;  // chunk index in [0, ISSUES_B)
                    if (b_slot < ISSUES_B)
                        issue_B_chunks<ROW_CHUNKS, N_TILE * ROW_CHUNKS>(
                            nxt_b, B_compact_row_bytes, kb_nxt, wave, lane, brsrc,
                            b_slot, b_slot + 1);
                }
            }

#pragma unroll
            for (int mi = 0; mi < M_PW; mi++)
                mfma_scale_f32_32x32x64_fp6_swapC(acc[mi][ni], a[mi], b, sav[mi], sbv);
        }
    };

    int sa0[K64_PER_TILE][NDA], sa1[K64_PER_TILE][NDA];
    int sb0[K64_PER_TILE][NDB], sb1[K64_PER_TILE][NDB];

    // Prologue: burst A+B tile 0 into buf0. No prior compute window, so burst all at once.
    load_scales(0, sa0, sb0);
    issue_A_chunks<ROW_CHUNKS, M_TILE * ROW_CHUNKS>(A_BUF0, A_row_bytes, 0, wave, lane, arsrc,
                                                    0, ISSUES_A);
    issue_B_chunks<ROW_CHUNKS, N_TILE * ROW_CHUNKS>(B_BUF0, B_compact_row_bytes, 0, wave, lane,
                                                    brsrc, 0, ISSUES_B);

    int kt = 0;
    for (; kt + 1 < k_tiles; kt += 2) {
        wait_vmcnt(0);
        __syncthreads();
        load_scales(kt + 1, sa1, sb1);
        // Compute tile kt from BUF0; drip A+B tile kt+1 into BUF1.
        compute(A_BUF0, B_BUF0, A_BUF1, B_BUF1, kt + 1, true, sa0, sb0);

        bool pf = (kt + 2 < k_tiles);
        wait_vmcnt(0);
        __syncthreads();
        if (pf) load_scales(kt + 2, sa0, sb0);
        // Compute tile kt+1 from BUF1; drip A+B tile kt+2 into BUF0 (if pf).
        compute(A_BUF1, B_BUF1, A_BUF0, B_BUF0, kt + 2, pf, sa1, sb1);
    }
    // Odd tail: one tile remains in BUF0, no drip.
    if (kt < k_tiles) {
        wait_vmcnt(0);
        __syncthreads();
        compute(A_BUF0, B_BUF0, 0, 0, 0, false, sa0, sb0);
    }

    // EPILOGUE: coalesced store (swapped-operand MFMA => acc = C^T, lane = N-column).
#pragma unroll
    for (int mi = 0; mi < M_PW; mi++)
#pragma unroll
        for (int ni = 0; ni < N_PW; ni++) {
            int n = wg_n * N_TILE + (wn * N_PW + ni) * 32 + (lane & 31);
            int nh = lane >> 5;
            int mb = wg_m * M_TILE + (wm * M_PW + mi) * 32;
            const v16f& a_acc = acc[mi][ni].vec;
#pragma unroll
            for (int g = 0; g < 4; g++) {
                int m0 = mb + g * 8 + nh * 4;
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    D[(size_t)(m0 + j) * N + n] = (OutT)a_acc[g * 4 + j];
                }
            }
        }
}

}  // namespace mxfp6
