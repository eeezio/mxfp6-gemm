#pragma once
// The MXFP6 GEMM kernel: A staged deep-K in LDS (double-buffered) + B streamed DIRECT from
// HBM -> VGPR through a compile-time register ring of depth PFD. Bypassing B's LDS round-trip
// removes the ds-read wall; the per-tile MFMA window (48 MFMA x 32cyc = 1536cyc) hides the
// direct global_load latency as long as the ring stays ahead. The ring depth is bounded by
// the VGPR budget (16 acc = 256 AccVGPR); PFD=5 is the deepest that keeps spill == 0.
//
// B's operand global address mirrors the LDS layout read_op uses for A: for a given
// (blk, sub, lane) the same 24 contiguous FP6 bytes live in global at
//   Bg + (blk*32 + lane%32)*B_row_bytes + kt*KT_BYTES + sub*48 + (lane>>5)*24
// (load_b_shuf reads them from the preshuffle_B layout instead, coalesced).
#include "device_ops.hpp"
namespace mxfp6 {

// Coalesced B operand load from a PRESHUFFLED B (preshuffle_B layout): per 32x64 tile,
// section0 = lane*16 (dwordx4), section1 = 1024 + lane*8 (dwordx2). Consecutive lanes read
// consecutive bytes -> fully coalesced.
__device__ __forceinline__ v6i load_b_shuf(const char* Bsh, int k64iters, int bg, int ki,
                                           int lane) {
    using v4i_a = int __attribute__((__vector_size__(16), __aligned__(4)));
    using v2i_a = int __attribute__((__vector_size__(8), __aligned__(4)));
    const char* base = Bsh + (size_t)(bg * k64iters + ki) * 1536;
    v4i_a lo = *reinterpret_cast<const v4i_a*>(base + lane * 16);
    v2i_a hi = *reinterpret_cast<const v2i_a*>(base + 1024 + lane * 8);
    return v6i{lo[0], lo[1], lo[2], lo[3], hi[0], hi[1]};
}

// Issue A cooperative buffer_load_lds chunks [i0,i1) for ONE tile (MUBUF, M0-implicit).
// Cooperative buffer_load_lds, split so the hybrid can DRIP A's 9 loads across the compute window
// (1 per MFMA quartet) instead of bursting them after the barrier — the burst is the top
// ATT stall (318 cyc/hit issue backpressure). s_nop guards the SALU-writes-M0 -> load(lds)
// 1-wait-state hazard (drip places set_m0 + load adjacent, unlike the burst the compiler
// spread out). Caller manages vmcnt (these loads are compiler-invisible, M0-implicit asm).
// NCH = total chunks in the tile (M_TILE*ROW_CHUNKS). When 256 does not divide NCH (e.g.
// M_TILE<256), the last issuing iteration is partial: lanes whose chunk>=NCH skip the load
// (their LDS slots lie past the tile and are never read). Default NCH huge => no guard, so
// the divisible 256x256 path is unchanged.
template <int CPR, int NCH = (1 << 30)>
__device__ __forceinline__ void issue_A_chunks(uint32_t lds_base, int row_stride, int kt_byte,
                                               int wave, int lane, const v4i& rsrc, int i0,
                                               int i1) {
    for (int i = i0; i < i1; i++) {
        int chunk = i * 256 + wave * 64 + lane;
        set_m0(__builtin_amdgcn_readfirstlane(lds_base + (uint32_t)((i * 256 + wave * 64) * 16)));
        if (chunk < NCH) {
            int m = chunk / CPR, ck = chunk % CPR;
            uint32_t voff = (uint32_t)(m * row_stride + kt_byte + ck * 16);
            asm volatile("s_nop 0\n buffer_load_dwordx4 %0, %1, 0 offen lds"
                         :
                         : "v"(voff), "s"(rsrc)
                         : "memory");
        }
    }
}

// DRIP-A kernel (A staged deep-K in LDS + B-direct coalesced ring): A's cooperative
// buffer_loads for the NEXT tile are dripped across THIS tile's MFMA quartets instead of
// bursted. HARD_WAIT=true puts wait_vmcnt(0) before each double-buffer barrier (hard RAW guarantee for
// the dripped A; cheap because A is issued early).
// A-drip schedule knob (LINEAR): which MFMA quartet issues which A chunks.
//   ADRIP_START : first quartet that issues A chunks (skip the sub-head-stall quartets)
//   ADRIP_STRIDE: quartets between successive issuing quartets (1 = consecutive)
//   ADRIP_PER   : A chunks issued per issuing-quartet (>=2 finishes A earlier => more RAW
//                 margin, but MORE issue backpressure/point -- measured NET LOSS, keep =1)
//   ADRIP_STOP  : last quartet (exclusive) allowed to issue (<=0 => NB; tail protection)
// issue_A_chunks(...,a0,a1) with a0==a1 emits nothing; all predicates fold to constants
// (p is the #pragma-unroll index) => branch-free, zero runtime/register cost.
//
// DEFAULTS are the tuned best @8192^3 (~2290 TFLOPs, +24% vs pure-LDS baseline):
//   PFD=5 (B-ring sweet spot), ADRIP_START=1 (skip the p=0 sub-head 302cyc/hit hotspot),
//   STRIDE=1, PER=1, STOP=NB. ADRIP_START=0 reproduces the original front-loaded schedule;
//   batching (PER>=2) was swept and lost. swz0 stays best for drip-A (swz32 did not help).
template <int M_TILE, int N_TILE, int K_TILE, int WAVES_M, int WAVES_N, int MIN_OCC = 1,
          int SWZ = 0, typename OutT = float, bool SPLITK = false, bool KGE2 = false,
          int PFD = 5, bool HARD_WAIT = true, int ADRIP_START = 1, int ADRIP_STRIDE = 1,
          int ADRIP_PER = 1, int ADRIP_STOP = 0, bool NMASK = false>
__global__ void __launch_bounds__(256, MIN_OCC)
    lds_gemm_hybrid_dripA(const void* __restrict__ A, const void* __restrict__ B,
                          const uint8_t* __restrict__ sA, const uint8_t* __restrict__ sB,
                          OutT* __restrict__ D, int N, int k_iters, int A_row_bytes,
                          int B_row_bytes, int kt_base, int k_tiles_seg, float* __restrict__ Dpart) {
    constexpr int KT_BYTES = K_TILE * 6 / 8;
    constexpr int ROW_CHUNKS = KT_BYTES / 16;
    constexpr int K64_PER_TILE = K_TILE / 64;
    constexpr int M_BLKS = M_TILE / 32, N_BLKS = N_TILE / 32;
    constexpr int M_PW = M_BLKS / WAVES_M, N_PW = N_BLKS / WAVES_N;
    constexpr int A_BYTES = M_TILE * KT_BYTES;
    constexpr int NB = K64_PER_TILE * N_PW;                      // 12 b-stream positions / tile
    constexpr int ISSUES_A = (M_TILE * ROW_CHUNKS + 255) / 256;  // 9 A loads / tile

    // The drip schedule hangs A's loads off the MFMA quartets, so the quartet count is a hard
    // budget: a tile needs ISSUES_A chunks and only gets ADRIP_PER per issuing quartet. Run out
    // and the trailing chunks are never issued at all -- the loop below just clamps a0/a1 to
    // ISSUES_A and moves on, so the tail of the A tile stays whatever the previous tile left in
    // LDS. That is silent wrong output, not a fault, and it is invisible in the ISA unless you
    // count buffer_load_lds against M_TILE. Wide-M/narrow-N tiles are where it bites: 256x128 has
    // ISSUES_A=9 but only NB-1=5 issuing quartets, so it needs ADRIP_PER=2.
    constexpr int ADRIP_STOP_EFF = (ADRIP_STOP > 0 && ADRIP_STOP <= NB) ? ADRIP_STOP : NB;
    constexpr int ADRIP_QUARTETS =
        ADRIP_STOP_EFF > ADRIP_START
            ? (ADRIP_STOP_EFF - ADRIP_START + ADRIP_STRIDE - 1) / ADRIP_STRIDE
            : 0;
    static_assert(ADRIP_QUARTETS * ADRIP_PER >= ISSUES_A,
                  "A-drip schedule cannot issue every A chunk for this tile: raise ADRIP_PER, "
                  "lower ADRIP_START, or widen N_TILE (more MFMA quartets to drip into)");

    extern __shared__ char smem[];
    int tid = threadIdx.x, wave = tid / 64, lane = tid % 64;
    int wm = wave / WAVES_N, wn = wave % WAVES_N;
    int wg_m, wg_n;
    if constexpr (SWZ < 0) {
        // XCD-aware remap (SWZ_XCD). The hardware hands workgroup pid to XCD pid%NXCC, so
        // consecutive pids land on DIFFERENT XCDs and the 32 workgroups resident on one XCD end up
        // holding 32 distinct wg_n -> B's 8x reuse across XCDs is never realised. Handing each XCD
        // a contiguous run of linear tile ids instead drops that to ~5 distinct wg_n, so its
        // resident workgroups share B slices. Bijective for any grid: the first `rem` XCDs take
        // one extra tile. A non-bijective formula would silently drop output tiles.
        static_assert(!SPLITK, "XCD remap assumes a 2D grid");
        constexpr int NXCC = 8;
        int mb = gridDim.x, total = mb * (int)gridDim.y;
        int pid = blockIdx.y * mb + blockIdx.x;
        int per = total / NXCC, rem = total % NXCC;
        int x = pid % NXCC, s = pid / NXCC;
        int L = x < rem ? x * (per + 1) + s : rem + x * per + s;
        wg_m = L % mb;
        wg_n = L / mb;
    } else if constexpr (SWZ > 0) {
        int mb = gridDim.x, nb = gridDim.y, pid = blockIdx.y * mb + blockIdx.x;
        const int G = SWZ, span = G * mb;
        int grp = pid / span, fn = grp * G, gs = (nb - fn) < G ? (nb - fn) : G, r = pid % span;
        wg_m = r / gs;
        wg_n = fn + r % gs;
    } else {
        wg_m = blockIdx.x;
        wg_n = blockIdx.y;
    }
    const char* Ag = reinterpret_cast<const char*>(A) + (size_t)(wg_m * M_TILE) * A_row_bytes;

    AccTile acc[M_PW][N_PW];
#pragma unroll
    for (int mi = 0; mi < M_PW; mi++)
#pragma unroll
        for (int ni = 0; ni < N_PW; ni++) clear_acc(acc[mi][ni]);

    constexpr int SA_PAD = ((M_PW + 3) / 4) * 4, SB_PAD = ((N_PW + 3) / 4) * 4;
    constexpr int NDA = SA_PAD / 4, NDB = SB_PAD / 4;
    static_assert(NDA == 1 && NDB == 1, "tiled-scale path assumes <=4 blocks/wave");
    int sa_grp = wg_m * WAVES_M + wm, sb_grp = wg_n * WAVES_N + wn;
    // NMASK: N is not a multiple of N_TILE, so the grid is ceil(N/N_TILE) and the LAST N-tile is
    // only partly inside the matrix. Its out-of-range columns are dropped at the store (below);
    // the work is still done, it just goes nowhere. What must not happen is the READ side running
    // off the end of B or of B's scales -- those buffers are sized to the true N, so an
    // out-of-range block would fault or read a neighbouring allocation. Clamp both indices to the
    // last valid entry: the operands are then wrong for the masked columns, which is fine because
    // their accumulators are never stored, and clamping is branch-free (a predicated load would
    // sit in the MFMA-feeding path). Requires (N/32) % N_PW == 0 so the scale grouping still
    // covers every real block -- dispatch only takes this path when it does.
    int b_blk_max = 0, sb_grp_max = 0;
    if constexpr (NMASK) {
        b_blk_max = N / 32 - 1;
        sb_grp_max = (N / 32) / N_PW - 1;
        sb_grp = sb_grp < sb_grp_max ? sb_grp : sb_grp_max;
    }
    int k_tiles = k_iters / K64_PER_TILE;
    if constexpr (SPLITK) {
        int z = blockIdx.z, seg_floor = kt_base, seg_rem = k_tiles_seg;
        kt_base = z * seg_floor + (z < seg_rem ? z : seg_rem);
        k_tiles_seg = seg_floor + (z < seg_rem ? 1 : 0);
    }

    // A buffer descriptor (Ag constant across the kernel)
    uint64_t ab = reinterpret_cast<uint64_t>(Ag);
    v4i arsrc{(int)(uint32_t)ab, (int)((uint32_t)(ab >> 32) & 0xFFFF), (int)0x7FFFFFFF,
              (int)0x00020000};

    auto load_scales = [&](int kt, int (*sa)[NDA], int (*sb)[NDB]) {
        const char* pa = reinterpret_cast<const char*>(sA) +
                         (size_t)((sa_grp * k_tiles + kt_base + kt) * 64 + lane) * K64_PER_TILE * SA_PAD;
        const char* pb = reinterpret_cast<const char*>(sB) +
                         (size_t)((sb_grp * k_tiles + kt_base + kt) * 64 + lane) * K64_PER_TILE * SB_PAD;
        int ta[K64_PER_TILE], tb[K64_PER_TILE];
        asm_load_dwordxN_nowait(ta, pa, K64_PER_TILE);
        asm_load_dwordxN_nowait(tb, pb, K64_PER_TILE);
#pragma unroll
        for (int sub = 0; sub < K64_PER_TILE; sub++) {
            sa[sub][0] = ta[sub];
            sb[sub][0] = tb[sub];
        }
    };

    // compute tile kt_cur from buffer `cur`; if adrip, drip A(kt_nxt) into buffer nxt_base.
    auto compute = [&](uint32_t cur, uint32_t nxt_base, int kt_cur, int kt_nxt, bool adrip,
                       const int (*sa)[NDA], const int (*sb)[NDB]) {
        int kb_nxt = (kt_base + kt_nxt) * KT_BYTES;
        v6i bring[PFD];
        int sbring[PFD];
#pragma unroll
        for (int q = 0; q < PFD; q++)
            if (q < NB) {
                int s = q / N_PW, n = q % N_PW, blk = wn * N_PW + n;
                int bg = wg_n * N_BLKS + blk;
                if constexpr (NMASK) bg = bg < b_blk_max ? bg : b_blk_max;
                bring[q] = load_b_shuf(reinterpret_cast<const char*>(B), k_iters,
                                       bg, (kt_base + kt_cur) * K64_PER_TILE + s, lane);
                sbring[q] = (sb[s][n / 4] >> (8 * (n % 4))) & 0xff;
            }
        v6i a[M_PW];
        int sav[M_PW];
#pragma unroll
        for (int p = 0; p < NB; p++) {
            int sub = p / N_PW, ni = p % N_PW;
            if (ni == 0) {
#pragma unroll
                for (int mi = 0; mi < M_PW; mi++) {
                    int blk = wm * M_PW + mi;
                    a[mi] = read_op<KT_BYTES>(smem, cur, blk, sub, lane);
                    sav[mi] = (sa[sub][mi / 4] >> (8 * (mi % 4))) & 0xff;
                }
            }
            v6i b_cur = bring[p % PFD];
            int sbv_cur = sbring[p % PFD];
            if (p + PFD < NB) {
                int np = p + PFD, ns = np / N_PW, nn = np % N_PW, nblk = wn * N_PW + nn;
                int nbg = wg_n * N_BLKS + nblk;
                if constexpr (NMASK) nbg = nbg < b_blk_max ? nbg : b_blk_max;
                bring[(p + PFD) % PFD] =
                    load_b_shuf(reinterpret_cast<const char*>(B), k_iters, nbg,
                                (kt_base + kt_cur) * K64_PER_TILE + ns, lane);
                sbring[(p + PFD) % PFD] = (sb[ns][nn / 4] >> (8 * (nn % 4))) & 0xff;
            }
            // DRIP A (tunable LINEAR schedule; default = 1 chunk/quartet front-loaded)
            if (adrip) {
                constexpr int STOP = (ADRIP_STOP > 0 && ADRIP_STOP <= NB) ? ADRIP_STOP : NB;
                if (p >= ADRIP_START && p < STOP && ((p - ADRIP_START) % ADRIP_STRIDE) == 0) {
                    int slot = (p - ADRIP_START) / ADRIP_STRIDE;  // 0,1,2,... issuing-quartet idx
                    int a0 = slot * ADRIP_PER, a1 = a0 + ADRIP_PER;
                    a0 = a0 < ISSUES_A ? a0 : ISSUES_A;  // clamp to chunk count (folds)
                    a1 = a1 < ISSUES_A ? a1 : ISSUES_A;
                    if (a0 < a1)
                        issue_A_chunks<ROW_CHUNKS, M_TILE * ROW_CHUNKS>(
                            nxt_base, A_row_bytes, kb_nxt, wave, lane, arsrc, a0, a1);
                }
            }
#pragma unroll
            for (int mi = 0; mi < M_PW; mi++)
                mfma_scale_f32_32x32x64_fp6_swapC(acc[mi][ni], a[mi], b_cur, sav[mi], sbv_cur);
        }
    };

    int sa0[K64_PER_TILE][NDA], sa1[K64_PER_TILE][NDA], sb0[K64_PER_TILE][NDB],
        sb1[K64_PER_TILE][NDB];
    // prologue: tile 0 A bursted into buf0 (no compute to drip into yet) + its scales.
    // A's global K-byte offset is kt_base*KT_BYTES (0 for non-split; the segment's first tile
    // under split-K) — must match the drip path's (kt_base+kt_nxt)*KT_BYTES.
    load_scales(0, sa0, sb0);
    issue_A_chunks<ROW_CHUNKS, M_TILE * ROW_CHUNKS>(0, A_row_bytes, kt_base * KT_BYTES, wave, lane,
                                                    arsrc, 0, ISSUES_A);
    int kt = 0;
    if constexpr (KGE2) __builtin_assume(k_tiles_seg >= 2);
    for (; kt + 1 < k_tiles_seg; kt += 2) {
        if (HARD_WAIT) wait_vmcnt(0);
        __syncthreads();
        load_scales(kt + 1, sa1, sb1);
        compute(0, A_BYTES, kt, kt + 1, true, sa0, sb0);  // compute buf0, drip A(kt+1)->buf1
        bool pf = (kt + 2 < k_tiles_seg);
        if (HARD_WAIT) wait_vmcnt(0);
        __syncthreads();
        if (pf) load_scales(kt + 2, sa0, sb0);
        compute(A_BYTES, 0, kt + 1, kt + 2, pf, sa1, sb1);  // compute buf1, drip A(kt+2)->buf0
    }
    if (kt < k_tiles_seg) {
        wait_vmcnt(0);
        __syncthreads();
        compute(0, 0, kt, 0, false, sa0, sb0);
    }

    // EPILOGUE — naturally COALESCED, no transpose. The swapped-operand MFMA above made acc hold
    // C^T (lane = N-column), so storing row-major D[m][n] with n = base + lane%32 means consecutive
    // lanes write consecutive N = consecutive addresses (each store instruction = one 64B-coalesced
    // cache-line transaction across the 32 N-lanes). No LDS, no barrier, works for any OutT.
    // (The MFMA layout was the root cause of store backpressure; fixing it at the source beats an
    // LDS-transpose epilogue — same +2.7% but simpler and FP32/BF16 get it too.)
#pragma unroll
    for (int mi = 0; mi < M_PW; mi++)
#pragma unroll
        for (int ni = 0; ni < N_PW; ni++) {
            int n = wg_n * N_TILE + (wn * N_PW + ni) * 32 + (lane & 31);
            if constexpr (NMASK) {
                if (n >= N) continue;  // partial last N-tile: this column is past the matrix
            }
            int nh = lane >> 5;
            int mb = wg_m * M_TILE + (wm * M_PW + mi) * 32;
            const v16f& a = acc[mi][ni].vec;
#pragma unroll
            for (int g = 0; g < 4; g++) {
                int m0 = mb + g * 8 + nh * 4;
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    if constexpr (SPLITK) {
                        size_t Mtot = (size_t)gridDim.x * M_TILE;
                        size_t seg_off = (size_t)blockIdx.z * Mtot * N;
                        Dpart[seg_off + (size_t)(m0 + j) * N + n] = a[g * 4 + j];
                    } else {
                        D[(size_t)(m0 + j) * N + n] = (OutT)a[g * 4 + j];
                    }
                }
            }
        }
}

// Split-K stage 2: sum Dpart[S][M][N] (FP32 partial sums) in fixed order s=0..S-1, convert to
// OutT, write D. One thread per output element; adjacent threads read consecutive idx ->
// coalesced; fixed add order -> bit-reproducible.
template <typename OutT>
__global__ void reduce_splitk(const float* __restrict__ Dpart, OutT* __restrict__ D,
                              long total, int S) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    float acc = 0.f;
    for (int s = 0; s < S; s++) acc += Dpart[(size_t)s * total + idx];
    D[idx] = (OutT)acc;
}

}  // namespace mxfp6
