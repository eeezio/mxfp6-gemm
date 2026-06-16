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
register ring (`PFD=5`). B is first laid out coalesced by `preshuffle_B`.

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
- **Per-machine swizzle width**: swizzle affects the geometric distribution of L2 hits; the
  dispatcher selects the swizzle width that is best on this machine for each shape.

---

## 6. Shape routing: two-tile dispatch

**What**: `choose_tile(M,N)` selects between two tiles with a simple threshold — when the 256×256
workgroup count (`(M/256)*(N/256)`) is below the CU count (256) and `M%128==0 && N%256==0`, it
picks the small tile; otherwise the large tile:
- **256×256** (16 acc) — the workhorse for CU-filling shapes (default).
- **128×256** (8 acc) — small-M, WG-starved shapes, where a smaller tile spawns more workgroups to
  fill the CUs.

**Why**: At occ1, a large tile on small shapes leaves the grid with fewer workgroups than CUs,
idling half the machine. The small-M path raises the workgroup count with a smaller tile to fill
the CUs, while **reusing the entire hybrid machinery** (drip-A, B-direct, swapped-C) — only the
tile template parameters change; no new kernel is introduced.

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
