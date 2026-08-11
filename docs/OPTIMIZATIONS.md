# libmxfp6gemm — Optimization Notes

This document records the optimizations used in `libmxfp6gemm` (MXFP6 GEMM, targeting
AMD CDNA4 / gfx950), together with **the motivation behind each one**. 

> This is a summary of design intent. The actual implementation lives in `src/kernel.hpp`,
> `src/device_ops.hpp`, `src/dispatch.hpp`, and `include/mxfp6/preprocess.hpp`, where the
> long inline comments capture the finer, measured details.

---

## 1. Large register tile + occ1 (the dominant lever)

**What**: A Volkov-style large register accumulator tile (256×256, 16 accumulators per wave),
with occupancy deliberately fixed at 1 and the register budget pinned via `__launch_bounds__`.

**Why**: The larger the tile along N, the more B values each A tile is reused against, which
lowers A's traffic-per-FLOP and raises arithmetic intensity. Low occupancy is not a problem in a
latency-bound regime that has ample instruction-level parallelism — each wave holds a large
accumulator tile and hides latency through the independent FMA pipeline in registers, rather than
by oversubscribing waves.

---

## 2. MFMA instruction choice: 32×32×64

**What**: The core instruction is fixed to `v_mfma_scale_f32_32x32x64_f8f6f4` (32 cycles each).

**Why**: Compared with 16×16×128, the 32×32×64 form is the right density on this hardware — more
FLOPs per instruction and lower operand-bandwidth-per-FLOP, which better suits a latency- and
operand-bound workload.

---

## 3. Data path: hybrid (A through deep-K LDS, B read directly into registers)

This is the heart of the current kernel (`lds_gemm_hybrid_dripA`): A and B take different paths,
each playing to its strengths.

### 3.1 A: deep-K staging in LDS + double buffering + dripped loads

**What**: A is cooperatively loaded into LDS in deep-K tiles (`K_TILE=192`), double-buffered
(~72 KB; only A occupies LDS — B is read directly into registers and uses none), and the loads are
**dripped** across the MFMA compute window — one `buffer_load_lds` per quartet — rather than
bursted all at once after the barrier.

**Why**:
- **Deep K enlarges the MFMA window.** Making "the amount of compute backing one global load" far
  exceed the memory latency lets the latency be absorbed by the math. LDS has enough capacity to
  sustain this depth (registers cannot hold deep-K operands).
- **Dripping smooths issue.** Spreading the loads out one at a time across the whole compute window
  keeps them from piling up back-to-back right after the barrier. Back-to-back memory issues
  saturate the memory queue and create backpressure, which stalls subsequent instructions from
  issuing; dripping keeps headroom in the queue so issue is never blocked.

### 3.2 B: bypass LDS, read HBM → VGPR directly (register prefetch ring)

**What**: B never enters LDS; it streams straight from HBM into VGPRs through a compile-time
register ring (`PFD=5`, deepest that keeps spill=0). B is first laid out coalesced by
`preshuffle_B`. For the residual 128×256 shapes at very large K (`Kp>50000`, where the 256-col
B/N-slice exceeds L2), the ring is K-gated one step deeper (`PFD=7`, still spill-free) to cover
more of the HBM-miss latency — worth ~2–3% there, while `PFD=5` stays best in the L2-hit regime.

**Why**: LDS is a limited resource and is already fully taken by A's deep-K double buffer; B, with
low in-loop reuse and large volume, is a poor fit for LDS reuse anyway — staging it would only
waste capacity and add a ds_write/ds_read round-trip. Reading B directly into registers is the
natural division of labor. With A on the LDS path (gated by `lgkmcnt`) and B on the VMEM path
(gated by `vmcnt`), the two memory streams land on different hardware paths and different wait
counters, so they can be tracked independently and issued off-peak — spreading issue pressure and
reducing the stalls that come from memory instructions bunching up. The preshuffle keeps the direct
reads contiguous and coalesced, avoiding scattered accesses.

### 3.3 Swapped-operand MFMA: the accumulator comes out as Cᵀ

**What**: The MFMA operand order is taken as `src0=A, src1=B`
(`mfma_scale_f32_32x32x64_fp6_swapC`), so the accumulator in registers is directly Cᵀ
(lane ↔ N-column).

**Why**: This makes consecutive lanes correspond to consecutive N-columns, so storing row-major D
is **naturally coalesced** — no LDS transpose, no barrier, and it holds for every output type
(FP32/FP16/BF16). The two operands are symmetric (each lane holds 32 K-values), so swapping them is
numerically correct. This is what makes the store epilogue coalesce efficiently.

---

## 4. Scale loads: manual vmcnt + tile-grouped wide loads

**What**: The shared E8M0 scales are repacked on the host by `tile_scale()` into a layout where
"all the scales a wave needs are contiguous," and the device reads them with **inline-asm and
hand-managed vmcnt** in a single wide `dwordx3` load (just 2 VMEM ops per K_TILE).

**Why**: Scale traffic has to overlap with A's LDS prefetch. Managing vmcnt by hand keeps the scale
loads from disturbing the in-flight prefetch, and the tile-grouped contiguous layout lets one wide
load fetch all of a wave's scales for an entire K_TILE — minimizing the number of scale VMEM ops.

> Constraint: the tiled-scale path requires ≤4 scale blocks per wave (the `dwordx4` limit ⇒
> `K_TILE≤256`), and the `tile_scale` grouping must match the tile (`MPW` for A, `NPW` for B).

---

## 5. Loop order and swizzle

- **N-major MFMA loop** (`for ni: for mi`): a zero-register-cost loop-order choice that improves
  the AGPR access pattern.
- **Workgroup swizzle**: the kernel's `SWZ` parameter remaps `(blockIdx.x, blockIdx.y)` to
  `(wg_m, wg_n)`. Positive `SWZ` groups `G` N-blocks so `wg_n` varies fastest, which optimizes A
  reuse — that was swept for drip-A and **lost** (`swz32` did not beat `swz0`), so it is unused.
  Negative `SWZ` selects the XCD-aware remap of §11, which is what ships on two of the four routes.

---

## 6. Shape routing: four-tile dispatch

**What**: `choose_tile(M,N,Kp)` selects between four tiles. It runs two stages: a *perf* stage of
three arms, each gated on `(M/256)*(N/256) < CU && M%128==0`, then a *correctness floor*.

- **128×384** (12 acc, `WAVES_M=1`/`WAVES_N=4`) — wide-N shapes where 128×256 wastes a CU wave.
  Taken whenever it saves a whole wave, at **any** K.
- **128×128** (4 acc) — the large-K wide-N shapes 128×384 cannot take (`Kp≥32768 &&
  (M/128)*(N/128) ≥ CU`). Its win over 128×256 is wave quantization, not L2 residency (see §9).
- **128×256** (8 acc) — small-M, WG-starved shapes, where a smaller tile spawns more workgroups to
  fill the CUs (`N%256==0`). This is the fall-through of the two above.
- **Coverage floor** — a shape all three arms decline picks the widest tile that *reaches* it:
  256×256 → 128×256 → 128×384 (`N%256!=0`) → **256×256 over a ceil grid** (`N%128==0`, masked last
  N-tile) → 128×128. 256×256 stays the default for the CU-filling `M%256==0 && N%256==0` case, so
  nothing that already routed well moves. The floor also subsumes the old `only_exact_route` arm:
  an `N%256!=0` shape that no perf arm takes lands on the 128×384 line here, so a separate K-gated
  arm for it would be dead code.

**Why**: At occ1, a large tile on small shapes leaves the grid with fewer workgroups than CUs,
idling half the machine. The small-M path raises the workgroup count with a smaller tile to fill
the CUs, while **reusing the entire hybrid machinery** (drip-A, B-direct, swapped-C) — only the
tile template parameters change; no new kernel is introduced. The 128×128 and 128×384 paths are that
same machinery again, chosen when something other than raw workgroup count binds: *wave
quantization* for both, and for 128×384 also the DRAM traffic it saves by re-reading A fewer times
(§9).

**The 128×384 cost model.** Filling the CUs is not a sufficient test for a wider tile, because a
128×384 workgroup does 1.5× the work of a 128×256 one — it only pays when it saves a whole CU wave.
So the gate compares wave-quantized cost (scaled by 2 to stay integer):

```
cost384 = 3 * ceil(wg384 / CU)      cost256 = 2 * ceil(wg256_grid / CU)
```

`M=2048, N=6144` gives 3·1 = 3 against 2·2 = 4, so it is taken (measured 1.59× on MI350X, more than
the 1.33× the model alone predicts — the rest is the 8→12 accumulators-per-wave ILP bonus).
`N=6912`/`7680` give 3·2 = 6 against 2·2 = 4 and are rejected: both tiles need two waves there, so
the bigger workgroup is pure loss. Ties reject, since the extra accumulators may well win in the tie
band but that is unmeasured.

**Why the correctness floor is separate from the cost model.** Every perf arm is gated on
`wg256 < CU` and on a workgroup count of its own (`wg384 ≥ CU`, `wg128 ≥ CU`), so any shape they all
decline used to land on 256×256 unconditionally — and the grid is launched with integer division, so
a 256 that does not divide M or N silently drops the remainder. Three families hit this:
`N%384==0 && N%256!=0` on a small grid (`M=256, N=384`: `wg384=2`, `wg128=6`, so `dim3(1,1)` and
columns 256..383 are never written), the same N on a CU-filling grid (`M=8192, N=6528` at any K,
since `wg256=800` makes `wide384` false), and M an odd multiple of 128 on a CU-filling grid
(`M=896`, last 128 rows dropped). Making divisibility a floor *below* the cost model — rather than
another clause *inside* it — fixes all three at once and cannot regress a tuned shape, because the
floor's first line returns exactly the 256×256 the old fall-through did whenever that tile is valid.
⚠️ It only reaches as far as the implemented tiles do: an M or N that is not a multiple of 128 still
truncates, and proper remainder handling is not implemented below that granularity.

**The masked last N-tile (`NMASK`).** Dividing is not always enough. `N = 128 * odd` (e.g.
`102272 = 128 x 799`) has no 256-wide divisor at all, so the floor's only dividing option is
128×128 — and a 4-acc tile amortizes shallow-K fixed cost over a quarter of the work: measured
**1273 TFLOPs against 1435** on `2048x102272x1024`. So for `M%256==0 && N%128==0` the router
returns 256×256 anyway and `dispatch_gemm` launches a `ceil(N/256)` grid with the kernel's `NMASK`
flag set. Three things change inside, all `if constexpr` so every other route keeps its exact
codegen:

1. the store drops columns with `n >= N` — this is the one that makes the output correct;
2. B's block index is clamped to the last real block;
3. B's scale-group index likewise.

(2) and (3) are bounds guards, not correctness guards: their columns are masked at the store, so
reading the wrong operand there changes nothing numerically. What they prevent is running off the
end of B — removing (2) faults with `Memory access fault by GPU node-1` on a large-K partial-N
shape, where the overshoot is ~10 MB. Removing (3) was **not** observed to fault or to change any
result; it is kept because the read is out of bounds by spec, not because a test catches it.
The route needs `(N/32) % NPW == 0`, which for NPW=4 is exactly `N%128==0`, so every real block
still has a scale group.

`choose_tile` is host code with no HIP dependency, so its decisions are gated by
`test_gemm --routing` (ctest target `routing`) on machines without a GPU.

---

## 7. K-padding recipe: "pad-B-only / compact-A"

**What**: The default recipe lets A stay in its natural compact layout (no per-row padding), with
only a small guard pad at the end of the buffer; B's K-tail is zero-padded; and A's K-tail scales
are extended to a non-NaN value via `pad_scales_k`.

**Why**:
- K is internally padded to a multiple of `K_TILE=192` to satisfy the deep-K tile's divisibility
  (zero-padding does not change the result).
- B's K-tail is exactly zero, so any K-tail over-read on A is nulled by B·0; A therefore needs no
  per-row padding, saving that overhead.
- The E8M0 scale byte `0xFF` decodes to NaN, and `0·NaN = NaN` would poison the entire output, so
  A's K-tail scales must be padded to a non-NaN value (127) to keep results correct.

---

## 8. Split-K for WG-starved narrow-N / large-K shapes

**What**: For shapes whose base grid underfills the machine, the K dimension is split into `S` equal
segments computed by independent workgroups (a third grid dimension). Each writes an FP32 partial
result to a workspace; a lightweight reduce kernel then sums the `S` layers and converts to the
output type. It is auto-selected in `dispatch_gemm` and invisible at the public API. Where section 6
fills the CUs along M/N, split-K is the complementary lever that fills them along K.

**Why**: A narrow-N + large-K shape (e.g. `M=2048, N=1024`) launches only `(M/128)·(N/256) = 64`
workgroups for 256 CUs, idling three-quarters of the machine, and the small tile cannot help (N is
already only 4 tiles wide). The one dimension left to harvest is K, and split-K turns it into `64×S`
workgroups — roughly doubling throughput on these shapes (≈555→1180 TFLOPs at S=4). The reduce runs
as a separate fixed-order pass (bit-reproducible, and it keeps the GEMM epilogue unchanged) rather
than via non-deterministic atomics. Its FP32 partial-sum buffer is **caller-provided** (sized via
`gemm_workspace_size`, allocated once and reused) rather than allocated internally per call — a
per-call `hipMalloc`/`hipFree` was measured at ~1.3 ms (~14× the GEMM) and would erase the gain;
caller ownership also lets a multi-stream caller give each stream its own buffer (no data race).

> `S = ceil(CU/base_wg)`, capped to keep ≥8 deep tiles per segment; shapes that already fill the CUs
> get `S=1` and the original path. Segments may be **uneven** — the first `k_tiles % S` take one extra
> tile, so `S` need not divide `k_tiles` (each WG derives its `(kt_base, length)` from `blockIdx.z`).
> This is what lets very-large-K shapes split even when `k_tiles` is prime-ish: `K=105728` →
> `k_tiles=551 (=19·29)`, which the old "reduce S until it divides evenly" rule collapsed to `S=1`;
> uneven keeps `S=4` (138/138/138/137) → **491→1437 TFLOPs**. Split-K reuses the full-K scale/B
> layouts (each segment just adds a `kt_base` offset), and the pad-B-only recipe (§7) still holds —
> only the last segment touches the zero K-tail.

**Reduce-cost-aware guard.** A split is not free: it adds an FP32 partial-sum reduce round-trip
(`~S·M·N·4` bytes), and it only buys GEMM parallelism up to a full CU wave. Naively taking
`S = ceil(CU/base_wg)` whenever `base_wg < CU` regresses two classes:
> 1. **Oversubscribed grids** — `base_wg` that doesn't divide `CU` (e.g. `base_wg=192`, `S=2` →
>    `384 > 256` WGs → 2 waves). Each WG still does `k_tiles/2` work over 2 waves = the unsplit wall
>    time, so the GEMM gains *nothing* while paying the full reduce. Measured: every `N=3072` `S=2`
>    shape lost (down to 0.45×).
> 2. **Shallow-K near-full grids** — `base_wg=128`, `S=2`, modest `K`: the reduce dominates the small
>    depth gain. Measured: `2048×2048×3490` regressed 931→838 (0.90×).

The guard models both. With `W = ceil(base_wg·S / CU)` (the CU-waves the split grid occupies), the
effective depth gain is `(S − W)/S` (not `1 − 1/S`), and the reduce cost scales as `S²·base_wg`.
Split only when the GEMM time saved clears it:

```
W = ceil(base_wg * S / CU)
split iff   Kp * (S - W)  >  ALPHA * base_wg * S²      (ALPHA = 10, gfx950)
```

`ALPHA` is a per-architecture compute/bandwidth balance constant, calibrated on a 30-shape
`base_wg × K` sweep (unsplit-vs-split speedup measured for each; the win/loss boundary lands at the
margin `Kp·(S−W)/(S²·base_wg) ≈ 8–10`, conservatively 10). Validated: it drops the `2048×2048×3490`
(→ `S=1`, 905) and `N=3072` (→ `S=1`, 1408) regressions, **keeps** the deep-K `S=2` win
`2048×1024×3490` (524 vs 503), and leaves every `S=4` win unchanged. See `benchmark_results.md`.

---

## 9. Large-K wide-N: the L2 sag, and what actually decides the tile

**Symptom**: on wide-N shapes that already fill the CUs along M/N (so split-K does not apply),
throughput *peaks* at moderate K and then *drops* at very large K. Measured on **MI350X** (the
primary gfx950 target), `2048×6144×K` (FP16, 128×256 tile): `K=16128 → 1588`, `K=105728 → 1260`
TFLOPS — a 21% fall, and CK's mxfp8 (`1413`) overtakes us there. (Same pattern on MI355X:
`1799 → 1323`.)

**Root cause of the sag** (ATT-confirmed on the MI355X reference node, see the wiki case study): B is
streamed HBM→VGPR (§3.2) and each 128×256 workgroup streams its `256 × K × 0.75`-byte B/N-slice. That
slice is shared across the M-row workgroups of an N-slice, so it *wants* to live in L2:
- `K=16128` → 3.1 MB/N-slice → fits the 4 MB per-XCD L2 → `buffer_load` avg stall ≈ 257 cyc.
- `K=105728` → 20.3 MB/N-slice → overflows it → every tile re-fetches from HBM → stall ≈ **2538
  cyc** (≈10×). The `PFD=5` ring covers only ~250 cyc, so the latency is exposed.

That is a real and sufficient account of **why throughput sags as K grows**. It is *not* an account of
**which tile wins** — an earlier version of this section conflated the two and concluded that the fix
was to shrink B's slice until it fit L2. Measurement refuted that; the corrected reasoning follows.

**What actually decides the tile.** Every workgroup reads its own A slice and its own B slice once,
so the totals over the whole GEMM are

```
A bytes = M · (N/NT) · Kp · 0.75      ∝ 1/NT     (A is re-read once per N-tile column)
B bytes = (M/MT) · N · Kp · 0.75      ∝ 1/MT
```

At `M=2048, N=6144, Kp=105792` (MI350X `bg-1w300-k2-3a`, clocks locked at 2200 MHz, n=5, tiles forced
through a test-only bypass of `choose_tile`, XCD swizzle off — so these are tile effects only):

| tile | A | B | total | WGs | CU waves | wave-quantized cost | measured |
|---|---:|---:|---:|---:|---:|---:|---:|
| 128×256 | 3.9 GB | 7.8 GB | 11.7 GB | 384 | 2 (second half-empty) | 2 × 32768 = 65536 | 1274 |
| 128×128 | 7.8 GB | 7.8 GB | 15.6 GB | 768 | 3 | 3 × 16384 = 49152 | 1767 |
| 128×384 | **2.6 GB** | 7.8 GB | **10.4 GB** | 256 | **1** | 1 × 49152 = 49152 | **2454** |

- **128×128 beats 128×256 while moving *more* bytes.** What it removes is the half-empty second CU
  wave (65536 → 49152). That is the mechanism of the original large-K fix — not L2 residency. A
  128-column B/N-slice is 10.16 MB against a 4 MB L2 and does not fit either.
- **128×384 beats 128×128 by 35–41%** at *identical* wave-quantized cost, purely by moving 1.5× fewer
  bytes (A is re-read 16 times instead of 48). Its B/N-slice is 30.5 MB — three times *larger* than
  the slice the L2-residency story said had to shrink. This is a direct disproof of B-slice capacity
  as the operative criterion for tile choice.

So the rule is **wave-quantized cost first, then total traffic** (§6). Un-K-gating the wave-saving
arm moves exactly **three `(M,N)` families** — those whose 128×384 grid is exactly 256 workgroups —
found by enumerating `choose_tile` old-vs-new over `M ≤ 4096`, `N ≤ 20480` (multiples of 128) and
`Kp` up to 105792. All three were measured, with CK MXFP8 in the same session:

| shape | CK MXFP8 | 128×128 | 128×384 | delta | vs CK |
|---|---:|---:|---:|---:|---:|
| `2048×6144×105728`  | 1463 | 1767 | **2557** | +44.7% | 1.21× → **1.75×** |
| `1024×12288×105728` | 1463 | 1649 | **2540** | +54.1% | 1.13× → **1.74×** |
| `4096×3072×105728`  | 1442 | 1760 | **2504** | +42.3% | 1.22× → **1.74×** |

At `K=32768` (the old gate boundary) the same three give +44.4% / +45.5% / +38.7%. The 128×128 tile
is what remains for the large-K
wide-N shapes where `N%384 != 0` puts 128×384 out of reach — where it is still worth **1260 → 1509
TFLOPS (+19.8%), 1.07× CK** over 128×256 on MI350X (cross-checked on the higher-clocked MI355X
reference node: **1318 → 1811, +37%, 1.19× CK**).

**Why not latency-hiding?** Two latency-oriented attempts were built and measured, and both *lost* —
which is the load-bearing lesson here:
- **B staged through an LDS double-buffer** (mirror of A's deep-K path, `issue_B_chunks` dripped
  across the MFMA window, one `wait_vmcnt(0)`/tile): correct (er=0.0000) but **1215 TFLOPS, −8%**.
  A single tile of look-ahead (~384 cyc of compute) cannot cover 2538 cyc of HBM latency, and the
  added ds-read traffic roughly cancels the partial gain. Staging through LDS hides latency but does
  **not reduce HBM bytes** — and at this shape the bottleneck is bytes/bandwidth, not latency.
- **Cross-tile B prefetch (BXPRE)** — issuing next-tile B loads at the tail of the current tile:
  ISA analysis showed it cannot help, because the hardware VMCNT is a single shared counter, so the
  `wait_vmcnt(0)` that guards the A double-buffer barrier necessarily drains the cross-tile B loads
  too (an earlier build measured −25%). No inline-asm trick escapes a shared counter.

**Takeaway**: three mechanisms live in this regime and they answer different questions. L2 capacity
explains **the sag against K**. Wave quantization and total DRAM traffic decide **which tile wins**.
Latency-hiding answers neither, because at large K the kernel is bandwidth-bound and neither LDS
staging nor a deeper ring reduces the bytes moved. The trap this section fell into was reading the
first as an answer to the second: B's slice does overflow L2, so shrinking it *looked* like the fix,
and the tile that shrank it *was* faster — for an unrelated reason. The wider tile, which overflows
L2 three times harder, is faster still.

---

## 10. Shallow-K fixed cost: emitting the accumulator zero-init once

**What**: `dispatch_gemm` picks one of two compile-time template arms on a host-known predicate,
`Kp >= 2*K_TILE`. Inside the `KGE2=true` arm the kernel asserts `__builtin_assume(k_tiles_seg >= 2)`,
which lets LLVM drop the `k_tiles_seg < 2` guard on the main k-loop.

**Why**: with the guard present the compiler emits the 256-instruction accumulator clear **twice** on
the main path — two blocks of `v_accvgpr_write_b32`, one on each side of the `s_cmp_lt_i32 s25, 2`
loop guard, both writing exactly `a0..a255`, with nothing touching AGPRs between them. Both execute.
The assumption is sound rather than wishful: `k_tiles_seg < 2` can only happen at `Kp == K_TILE`,
because `splitk_S()` floors every split segment at 8 deep tiles, so any shape that splits already has
`k_tiles_seg >= 8`.

The saving is a **fixed ~1024 cycles per wave**, so it is a shallow-K optimization by construction —
worth ~2.7% at `K=1024` and 0.009% at `K=105728`. Measured on MI350X: **+2.76%** on
`2048×105728×1024` and **+2.63%** on `2048×102272×1024`, neutral elsewhere.

**Cost**: kernel instantiations go 24 → 39 and `libmxfp6gemm.a` grows from 452 kB to ~700 kB. That is
the whole price — no register, occupancy or scratch change (all 39 kernels keep `ScratchSize 0`,
zero spill).

> **Every zero-cost formulation was tried and failed.** Wrapping the loop in a tautological
> `if (k_tiles_seg >= 2)`, manual loop rotation (`if` + `do-while`), `__builtin_amdgcn_sched_barrier(0)`,
> an `asm` tie on the accumulator, and sinking the `clear_acc` loop below the prologue all leave the
> instruction count unchanged — LLVM canonicalizes every one of them back. Removing the branch at
> *compile* time is the only mechanism that works, and paying for it with a template arm is the
> reason this is a dispatch-level change rather than a one-line kernel edit.

---

## 11. XCD-aware grid swizzle: making B's reuse across XCDs actually happen

**What**: on the 256×256 and 128×384 routes the kernel remaps the linear workgroup id so that each
XCD receives a **contiguous run** of tile ids (`SWZ_XCD`, the `SWZ < 0` branch in `kernel.hpp`):

```
x   = pid % NXCC            // XCD id -- fixed by hardware, cannot be changed
s   = pid / NXCC            // this XCD's s-th job
L   = x < rem ? x*(per+1) + s : rem + x*per + s     // per = total/NXCC, rem = total%NXCC
wg_m = L % gridDim.x        // gridDim.x consecutive L share one wg_n
wg_n = L / gridDim.x
```

**Why**: MI350X is 8 XCDs of 32 CUs, each XCD with its own 4 MB L2 behind a shared 256 MB LLC. The
dispatcher hands workgroup `pid` to XCD `pid % 8` — verified on hardware with a probe kernel reading
`hwreg(HW_REG_XCC_ID)`, which found `xcc == pid % 8` for 3304/3304 workgroups of the real grid.
Consecutive pids therefore land on *different* XCDs, and the 32 workgroups resident on one XCD end up
holding 32 **distinct** `wg_n` — so the B slice that M-blocks are supposed to share is fetched once
per workgroup instead of once per XCD. For `2048×105728×1024` that is a 221 KB B/N-slice × 32 =
7.1 MB of B live per XCD against a 4 MB L2; B's 8× reuse is never realized. The remap drops the
resident `wg_n` count to 4.88, i.e. 1.08 MB — back inside L2.

Measured on MI350X (interleaved A/B, deterministic clocks, n=10): **+4.56%** on `2048×105728×1024`,
**+3.77%** on `2048×102272×1024`, **+3.24%** on `2048×6144×16128`.

> **Do not predict an 8× effect.** All of B is 91 MB and the LLC is 256 MB, so most of the traffic
> the remap eliminates was LLC→L2, not HBM. What is saved is LLC round-trip latency and bandwidth.

**The gain really is XCD locality, not reordering.** A sham control was built: the *same* bijection
and the *same* runtime division cost, with only the final decomposition swapped
(`wg_m = L/nb, wg_n = L%nb`) so that the distinct-`wg_n`-per-XCD count returns to its baseline 32.
It measured −0.55% / −0.77% / −0.23% — all slightly negative, i.e. the division cost with none of
the benefit. Reordering per se buys nothing.

**It is a trade, so it is gated per route.** The remap groups `wg_n` by *scattering* `wg_m`:

| route | distinct `wg_m` / XCD | distinct `wg_n` / XCD | one slice |
|---|---|---|---|
| 256×256, `K=1024` | 1 → 8 | 32 → 4.88 | 221 KB |
| 128×128, `K=105728` | 2 → 16 | 16 → 2 | 10.16 MB |

At `K=1024` all of A is 1.77 MB, so scattering `wg_m` costs essentially nothing and the B win is
free. At `K=105728` both slices are 10.16 MB and the exchange is **numerically symmetric** — 18
slices before, 18 after — yet the route measures **−3.48%** (8/8 reps). Working-set size therefore
does not explain it; the asymmetry is in the two data paths. A is a cooperative `buffer_load_lds`
DMA with a `wait_vmcnt(0)` before every double-buffer barrier, so its latency sits on the tile-level
critical path, whereas B rides a `PFD=5` register ring built to absorb latency — scattering A hurts
more than gathering B helps. *(That attribution is a hypothesis; it has not been isolated by
experiment.)* The practical consequence is that **128×128 and 128×256 ship unswizzled**, and
split-K is excluded outright by `static_assert(!SPLITK)` because it uses a 3D grid where the
`pid % 8` premise does not hold.

**Bijectivity is a correctness requirement, not a nicety.** If the remap is not a bijection some
tile is claimed by no workgroup, and the output silently loses a block with no error anywhere. The
`rem` term exists precisely so the map stays bijective when `NXCC` does not divide the grid; it was
verified exhaustively over all 1600 grids in `1..40 × 1..40`, non-divisible cases included.

**Combined with §10** (both are shallow-K/large-N levers and they compose without interference), the
two changes measure **+8.12%** on `2048×105728×1024` and **+6.79%** on `2048×102272×1024`
(MI350X, deterministic clocks at 2200 MHz, interleaved A/B against the tip of the split-K guard
branch, median of 3). The unswizzled routes stay flat: `2048×4096×105728` (128×128) +0.43% and
`2048×1024×12288` (split-K) −0.12%.

**Known cost: a ~1% regression on a few 256×256 shapes.** Over the whole 50-shape benchmark table
the swizzle+§10 pair gives 16 shapes better than +2% (best +9.1%) and two shapes worse than −1%:
`2048×20480×6144` (−1.13%) and, from a follow-up sweep, `2048×12672×6144` (−1.03%). Both are
reproducible, not noise — every rep of the new build sits below every rep of the old one. There is
**no clean gate for them**, which is worth stating plainly rather than tuning around:

| shape | delta | | shape | delta |
|---|---:|---|---|---:|
| `2048×20480×1024` | +7.25% | | `2048×20480×12288` | −0.32% |
| `2048×20480×2048` | +4.04% | | `2048×20480×24576` | +2.16% |
| `2048×20480×4096` | +1.21% | | `2048×16128×6144` | **+2.02%** |
| `2048×20480×6144` | **−0.98%** | | `2048×12672×6144` | **−1.03%** |

`K=6144` loses at `N=20480` and `N=12672` but *wins* at `N=16128`, so the sign is not a function of
K alone, nor of N alone. That is consistent with the mechanism — the remap trades A locality for B
locality, and which side pays depends on the two slice sizes for that specific shape — but it means
any threshold would be fitted to ~1% effects on three data points. The pair ships ungated on the two
routes measured positive on aggregate, and these shapes are documented as the price.
