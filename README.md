# libmxfp6gemm — MXFP6 GEMM for AMD CDNA4 (gfx950 / MI350X)

A standalone HIP library that computes

```
D[M,N] = A[M,K] · B[K,N]
```

where **A and B are MXFP6** (FP6 E2M3 values + one shared E8M0 scale per 32-element K-block) and
**D is FP32, FP16, or BF16**. One kernel paradigm, routed by shape; the public header is
device-free so a host translation unit can drive the library without a HIP compiler.

---

## Performance vs CK MXFP6

FP16 output, K=8192, on MI350X (gfx950), ROCm 7.0.2.1. Same-precision MXFP6 performance vs CK's
`mx_flatmm` (`-mx_prec=fp6xfp6`), measured on the same machine in the same session.

| M × N (K=8192) | CK FP6 (TFLOPs) | libmxfp6gemm (TFLOPs) | speedup |
|---|---:|---:|---:|
| 2048 × 4096 | 1050 | 1706 | **1.63×** |
| 2048 × 8192 | 1059 | 2176 | **2.06×** |
| 4096 × 4096 | 1059 | 2329 | **2.20×** |
| 4096 × 8192 | 1112 | 2352 | **2.12×** |
| 8192 × 8192 | 1115 | 2418 | **2.17×** |

Absolute numbers vary run-to-run; the **ratio** is the stable takeaway.

### Narrow-N / large-K shapes (split-K)

Shapes with a small N and a large K launch too few workgroups to fill the machine — e.g.
`M=2048, N=1024` on the 128×256 path is only `(2048/128)·(1024/256) = 64` workgroups for 256 CUs
(25 % occupancy). For these the library automatically splits the K dimension across independent
workgroups (**split-K**, see [`docs/OPTIMIZATIONS.md`](docs/OPTIMIZATIONS.md) §8). FP16, MI350X:

| M × N × K | without split-K (TFLOPs) | with split-K (TFLOPs) | S | speedup | vs CK MXFP8 |
|---|---:|---:|---:|---:|---:|
| 2048 × 1024 × 12288  | 561 | **1109** | 4 | **1.98×** | 1.02× (CK 1087) |
| 2048 × 1024 × 16128  | 570 | **1255** | 4 | **2.20×** | 1.11× (CK 1135) |
| 2048 × 1024 × 105728 | 491 | **1437** | 4 | **2.93×** | **1.57× (CK 912)** |
| 2048 × 512 × 6144    | 277 | **483**  | 4 | **1.74×** | 0.91× (CK 533) |
| 2048 × 6144 × 16128  | 1588 | 1586 (no split) | 1 | — | control |

Re-measured on MI350X / ROCm 7.0.2 (2026-06-30). The S=1 (no-split) path is unchanged by the
split-K work — verified by an A/B build of the pre-split-K commit (<1% delta). The last row is a
control for split-K only and predates the 128×384 route; that shape now runs at ~2700 TFLOPs (see
[Wide-N shapes](#wide-n-shapes-128384-and-128128-tiles)).

Split-K is enabled inside `gemm()` for workgroup-starved shapes when you supply a scratch buffer
(sized via `gemm_workspace_size`, see below); shapes that already fill the CUs (e.g. wide N) skip it
and need no workspace.

### Wide-N shapes (128×384 and 128×128 tiles)

Wide-N shapes fill the CUs, so split-K does not apply, but the 128×256 tile is wrong for them in two
different ways depending on K, and each gets its own route (see
[`docs/OPTIMIZATIONS.md`](docs/OPTIMIZATIONS.md) §6, §9):

* **Wide N → 128×384.** `2048×6144` on 128×256 is 384 workgroups for 256 CUs — 1.5 CU waves, so the
  second wave runs half empty. A 384-wide tile makes it exactly 256 workgroups (one full wave) and
  raises accumulators per wave from 8 to 12. Taken only when the wider tile saves a whole CU wave,
  since its workgroup does 1.5× the work — but then at **any** K: at `K=105728` it is 44.7% faster
  than the 128×128 route, because a wider N-tile also re-reads A fewer times (2.6 GB vs 7.8 GB).
* **Very large K, `N%384 != 0` → 128×128.** What binds is still wave count, not the cache: 128×128
  runs 3 full waves against 128×256's 2 half-empty ones. Its B slice does overflow L2, but shrinking
  that slice is not why it wins — see §9.

Measured on **MI350X** `bg-1w300-k2-3a`, ours + CK in one session, interleaved, FP16, `2048 × 6144`
(median of 3 reps; raw:
[`prof_results/bench_mi350x_k2-3a_2026-07-29.txt`](prof_results/bench_mi350x_k2-3a_2026-07-29.txt)):

| M × N × K | CK MXFP8 | before (128×256) | after | route | after vs CK |
|---|---:|---:|---:|---|---:|
| 2048 × 6144 × 512    | 473  | 746  | **966**  | 128×384 | **2.04×** (was 1.58×) |
| 2048 × 6144 × 4096   | 1290 | 1427 | **2131** | 128×384 | **1.65×** (was 1.11×) |
| 2048 × 6144 × 16128  | 1654 | 1702 | **2707** | 128×384 | **1.64×** (was 1.03×) |
| 2048 × 6144 × 105728 | 1463 | 1762 | 1759     | 128×128 | **1.20×** (control: already routed to 128×128, unchanged) |

> ⚠️ The last row is a faithful record of the 2026-07-29 run, but its **route has since changed** —
> the wave-saving 128×384 arm is no longer K-gated. Re-measured below.

Removing that K-gate moves exactly three `(M,N)` families: the ones where a 128×384 grid lands on
exactly 256 workgroups. Re-measured on the same MI350X with CK MXFP8 in the same session
(2026-08-11, deterministic clocks at 2200 MHz, interleaved, median of 3):

| M × N × K | CK MXFP8 | before (128×128) | after (128×384) | delta | after vs CK |
|---|---:|---:|---:|---:|---:|
| 2048 × 6144 × 105728  | 1463 | 1767 | **2557** | **+44.7%** | **1.75×** (was 1.21×) |
| 1024 × 12288 × 105728 | 1463 | 1649 | **2540** | **+54.1%** | **1.74×** (was 1.13×) |
| 4096 × 3072 × 105728  | 1442 | 1760 | **2504** | **+42.3%** | **1.74×** (was 1.22×) |

The same three families at `K=32768`, the old gate boundary: +44.4% / +45.5% / +38.7%. CK in this
session reproduces its 2026-07-29 numbers to within 1% (K=16128: 1669 vs 1654; K=105728: 1463 vs
1463), which is the check that the two sessions are comparable.

Both routes are auto-selected in `choose_tile`; shapes that meet neither condition are
byte-for-byte unchanged. The 128×128 route was separately cross-checked on **MI355X**
(`smci355-ccs-aus-n03-05`, ours + CK in one session) where K=105728 went 1318 → 1811 against CK 1523,
i.e. 0.87× → 1.19×. Full per-track / per-machine data:
[`prof_results/experiments-2026-07-16/EXPERIMENT-LOG.md`](prof_results/experiments-2026-07-16/EXPERIMENT-LOG.md).

> Cross-machine absolute TFLOPs are not comparable — only same-session ratios and deltas are. The
> table above is one machine, one session. Earlier revisions of this section quoted a 128×256 →
> 128×128 A/B taken on a different node, which is why its K=16128 row read 1.02× rather than 1.64×.

---

## Build

```bash
cmake -S . -B build -DCMAKE_HIP_ARCHITECTURES=gfx950
cmake --build build -j
ctest --test-dir build          # end-to-end correctness gate (vs CPU reference)
```

This produces the static library `libmxfp6gemm.a` and the test `test_gemm`.

To use it in your project: include `<mxfp6/gemm.hpp>`, link `mxfp6gemm`. The header `gemm.hpp`
is device-free (plain declarations); the host preprocessing helpers are in `mxfp6/preprocess.hpp`.

---

## The shape of the data

`gemm()` does **not** quantize and **does not** take your raw float matrices. It takes
**pre-quantized, pre-laid-out DEVICE buffers**. The host-side preprocessing (in
`mxfp6/preprocess.hpp`) turns float matrices into those buffers. The flow is:

```
float A[M][K]  ─ quantize_to_mxfp6 ──────────────► packed A  + A scales ─ pad_scales_k ─ preprocess_scale ─ tile_scale ─► tiled A scales
float B[K][N]  ─ preprocess_B (transpose+quant) ─► packed Bᵀ + B scales ──────────────── preprocess_scale ─ tile_scale ─► tiled B scales
                                                             └─ preshuffle_B ─► preshuffled B
```

You then upload the four buffers (packed A, preshuffled B, tiled A scales, tiled B scales) to the
device and call `gemm()`.

The quick start below uses the **default, recommended layout**: B (weights) is K-padded once,
and A (activations) stays in its **natural compact layout** — no per-row padding, just a tiny
end pad via `a_compact_end_pad`. See [K-padding](#k-padding--why-a-stays-compact) for why.

---

## Quick start

```cpp
#include <mxfp6/gemm.hpp>
#include <mxfp6/preprocess.hpp>
using namespace mxfp6;

// Inputs (host): A is row-major [M][K] (activations), B is row-major [K][N] (weights).
// M and N must be multiples of 128; K a multiple of 32 (see Requirements).
int M = 8192, N = 8192, K = 8192;

int Kp = kpad(K);                        // pad K up to a multiple of K_TILE (=192); here 8256
TileChoice tc = choose_tile(M, N, Kp);   // picks the tile + scale grouping for this shape
                                         // (Kp is required: tile choice is K-aware — pass the SAME
                                         //  Kp you pass to gemm(), or the scale grouping won't match)

// --- 1. quantize (host) ---
// A: quantized in its NATURAL COMPACT K layout — no per-row padding, A_f32 is just M*K floats.
QuantizedMatrix Aq = quantize_to_mxfp6(A_f32, M, K);
// B (weights): padded to Kp with a ZERO K-tail. B_f32 is Kp*N floats with rows [K..Kp) = 0.
QuantizedMatrix Bq = preprocess_B(B_f32, Kp, N);        // transposes B[K][N]→Bᵀ[N][K], then quantizes

// --- 2. lay out for the kernel (host) ---
PreshuffledB Bsh = preshuffle_B(Bq);
// A scales: pad_scales_k extends them to Kp with a non-NaN tail (REQUIRED for compact A — see below).
TiledScale   sA  = tile_scale(preprocess_scale(pad_scales_k(Aq.scales.data(), M, K).data(), M, Kp),
                              tc.MPW, K_TILE / 64);
TiledScale   sB  = tile_scale(preprocess_scale(Bq.scales.data(), N, Kp), tc.NPW, K_TILE / 64);
//                                                                       ^^^^^^  ^^^^^^^^^^^^
//                                                          A uses tc.MPW, B uses tc.NPW; subs = K_TILE/64 = 3

// --- 3. upload to device ---
// A buffer = compact data + a small end pad for the last row's K-tail read (content irrelevant).
size_t Abytes = (size_t)M * Aq.packed_row_bytes + a_compact_end_pad(K);
void *dA, *dBsh; uint8_t *dsA, *dsB; void *dD;
hipMalloc(&dA,   Abytes);                     hipMemcpy(dA, Aq.packed_data.data(),
                                                        (size_t)M * Aq.packed_row_bytes, ...);
hipMalloc(&dBsh, Bsh.data.size());           hipMemcpy(dBsh, Bsh.data.data(), Bsh.data.size(), ...);
hipMalloc(&dsA,  sA.data.size());            hipMemcpy(dsA,  sA.data.data(),  sA.data.size(),  ...);
hipMalloc(&dsB,  sB.data.size());            hipMemcpy(dsB,  sB.data.data(),  sB.data.size(),  ...);
hipMalloc(&dD,   (size_t)M * N * sizeof(OutElem));   // OutElem = float / __half / __hip_bfloat16

// Split-K scratch: 0 for most shapes; non-zero for WG-starved narrow-N/large-K. Allocate once and
// reuse across calls (don't malloc/free per call). Pass (nullptr,0) to never split.
void* ws = nullptr;
size_t ws_bytes = gemm_workspace_size(M, N, Kp);
if (ws_bytes) hipMalloc(&ws, ws_bytes);

// --- 4. launch ---
gemm(OutType::F16, M, N, Kp,
     dA, dBsh, dsA, dsB, dD,
     Aq.packed_row_bytes,        // A_row_bytes = packed(K)  (compact stride)
     Bq.packed_row_bytes,        // B_row_bytes = packed(Kp)
     ws, ws_bytes);              // split-K workspace (nullptr,0 = never split)
hipDeviceSynchronize();
```

`dD` now holds the `M×N` row-major result in the requested output type. Note A is never padded
per-row — only `a_compact_end_pad(K)` bytes are added at the very end of the whole buffer.

---

## API reference

### `mxfp6/gemm.hpp` (public, device-free)

| Symbol | Purpose |
|---|---|
| `enum class OutType { F32, F16, BF16 }` | Output element type. |
| `constexpr int K_TILE` (=192) | The kernel's K-tile depth. K is padded to a multiple of it. |
| `int kpad(int K)` | K rounded up to a multiple of `K_TILE` → pass as the kernel's `Kp`. |
| `struct TileChoice { int MT, NT, MPW, NPW; }` | Tile (MT×NT) + per-wave 32-block counts (MPW/NPW) for scale grouping. |
| `TileChoice choose_tile(int M, int N, int Kp)` | Pick the tile/scale grouping for a shape. **Use its `MPW`/`NPW` for `tile_scale`.** `Kp` is required and must be the same padded `Kp` passed to `gemm()` (tile choice is K-aware). |
| `size_t gemm_workspace_size(int M, int N, int Kp)` | Bytes of split-K scratch to allocate for this shape; **0** if it doesn't split. |
| `void gemm(OutType, M, N, Kp, dA, dBsh, dsA, dsB, dD, A_row_bytes, B_row_bytes, ws, ws_bytes)` | Launch. All `d*`/`ws` are device pointers. `ws`/`ws_bytes` = split-K scratch (`gemm_workspace_size`); `(nullptr,0)` never splits. |

### `mxfp6/preprocess.hpp` (host)

| Symbol | Purpose |
|---|---|
| `quantize_to_mxfp6(const float* mat, rows, cols)` | float → `QuantizedMatrix` (packed FP6 + per-block E8M0 scales). |
| `preprocess_B(const float* B, K, N)` | Transpose `B[K][N]`→`Bᵀ[N][K]`, then quantize. |
| `preshuffle_B(const QuantizedMatrix& Bq)` | Re-pack Bᵀ for coalesced VMEM loads → `PreshuffledB`. |
| `preprocess_scale(const uint8_t* scales, dim, K)` | Re-order per-block scales to the MFMA lane layout. |
| `tile_scale(const PreprocessedScale&, group, subs)` | Group a wave's scales contiguous. `group = MPW` (A) / `NPW` (B); `subs = K_TILE/64`. |
| `a_compact_end_pad(int K)` | (compact-A recipe) bytes to over-allocate at the A-buffer end. |
| `pad_scales_k(const uint8_t* scales, dim, K)` | (compact-A recipe) extend scales to `kpad(K)` with a non-NaN tail. |
| `dequantize_mxfp6(const QuantizedMatrix&, float* out)` | MXFP6 → float (for debugging / reference). |

The data types and FP6/E8M0 conversions live in `mxfp6/types.hpp`; a CPU reference GEMM
(`mxfp6_gemm_ref`) lives in `tests/reference.hpp`.

---

## K-padding — why A stays compact

The kernel reads K in deep tiles, so K is padded to `kpad(K)` and the kernel computes over the
full `Kp`. The padded K-tail is harmless **as long as it is zero on B**, because `B[k]·anything = 0`.
The quick start above exploits this so you never pad your activations:

* **B (weights):** padded to `Kp` with a zero K-tail — done once, offline. `preprocess_B` over a
  zero-`Kp`-padded float already produces this.
* **A (activations):** stays in its natural compact layout (`A_row_bytes = packed(K)`). The kernel
  still reads `packed(Kp)` per row, so each row's K-tail read spills into the **next row's** real
  data — which is fine, because B's zero K-tail multiplies it away. Only two things are needed:
  1. **`a_compact_end_pad(K)`** extra bytes at the very end of the A buffer, so the *last* row's
     K-tail read stays in bounds (its content is irrelevant), and
  2. **`pad_scales_k`** to extend A's scales to `Kp` with a non-NaN tail — see the footgun below.

This is measured perf-neutral vs per-row padding and is gated by `verify_compact` in `test_gemm`.

> **Footgun — the one thing you must not skip.** The fp6 *data* tail is inert (`B·0 = 0`), but an
> E8M0 scale byte of `0xFF` decodes to **NaN**, and `0·NaN = NaN` poisons the *entire* output.
> Always build A's scale tail with `pad_scales_k` (or quantize A over a zero-`Kp`-padded float).

**Simpler alternative (if you don't mind padding activations):** quantize A over `Kp` directly
(`quantize_to_mxfp6(A_f32, M, Kp)` with the K-tail zeroed), pass `A_row_bytes = Aq.packed_row_bytes`
(now `packed(Kp)`), and skip both `a_compact_end_pad` and `pad_scales_k` (the zero-padded quant
gives a safe scale tail automatically). Same result, but A is padded per-row.

---

## Requirements & caveats

- **K must be a multiple of 32** (the MX block size). It is then padded internally to `kpad(K)`
  (a multiple of `K_TILE`=192) — pass `kpad(K)` as the kernel's `Kp`. ⚠️ A K that is not a
  multiple of 32 is silently mis-quantized in a release build (the `assert` is compiled out).
- **M and N must be multiples of 128.** Multiples of 256 hit the workhorse 256×256 tile; the
  128-wide routes cover the rest (N a multiple of 384 → 128×384, of 256 → 128×256, otherwise
  128×128). `choose_tile()` treats divisibility as a hard floor: if the perf arms — which are gated
  on CU fill and on `Kp` — all decline, it still returns the widest tile that *divides* the shape
  rather than the fastest one. ⚠️ Below that, the grid is launched with integer division and there
  is **no remainder handling**, so an M or N that is not a multiple of 128 silently drops the
  remainder rows/cols. Pad up to 128, or check `choose_tile(M,N,Kp).MT/.NT` divide your M/N.
- **B is transposed by `preprocess_B`.** Pass B in its natural `[K][N]` row-major layout; the
  helper produces `Bᵀ[N][K]` and quantizes it.
- **Scale grouping must match the tile.** Use `choose_tile(M,N,Kp).MPW` for A's `tile_scale` and
  `.NPW` for B's, passing the **same padded `Kp` you pass to `gemm()`** — tile choice is K-aware, so
  a different `Kp` here silently tiles the scales for the wrong kernel. Every route has its own
  `MPW`/`NPW` (256×256 → 4,4; 128×256 → 2,4; 128×128 → 2,2; 128×384 → 4,3), and mismatched grouping
  gives wrong results with no error. ⚠️ This also means scale layouts are **not portable across
  library versions**: if a shape's route changes, a cached layout produced by an older `choose_tile`
  will silently corrupt. Re-run the scale preprocessing whenever you update the library.
- **`A_row_bytes` / `B_row_bytes`** are the packed bytes per row (= `Quantized.packed_row_bytes`).
  For compact A this is `packed(K)`; for per-row-padded A it is `packed(Kp)`.
- **E8M0 scale `0xFF` is NaN** and will poison the **entire** output (`0·NaN = NaN`). This is the
  one real footgun of the pad-B-only/compact-A recipe — always extend A's scale tail with
  `pad_scales_k` (or quantize A over a zero-`Kp`-padded float). The fp6 *data* tail is inert.
- **Feed finite activations.** A `0xFF` scale only ever comes from uninitialized memory, never from
  `quantize_to_mxfp6` (it clamps to ≤254 and `std::max` ignores NaN). So a NaN/Inf in A's *real*
  data does **not** NaN-poison the output — but it is silently swallowed and **corrupts the
  affected 32-block's row** (wrong values, verified). The fp6 data encoding has no NaN/Inf, so A's
  data can never *introduce* NaN; only a `0xFF` scale can. Sanitize activations upstream.
- **The pad-B-only recipe requires B's K-tail to be exactly zero.** It does not work if you pad
  only A.
- **Inputs are device buffers.** `gemm()` only launches the kernel; you allocate, quantize on the
  host, and `hipMemcpy` the four buffers yourself. The output `dD` must be sized for the chosen
  `OutType`.
- **Split-K needs a caller-provided workspace.** For workgroup-starved shapes (narrow N + large K),
  `gemm()` splits K and writes FP32 partial sums to a scratch buffer you supply via `ws`/`ws_bytes`,
  then reduces them into `dD` (bit-reproducible — segments summed in a fixed order). Size the buffer
  with `gemm_workspace_size(M, N, Kp)` (returns 0 for shapes that don't split), allocate it once on
  the device, and reuse it across calls — do **not** `hipMalloc`/`hipFree` per call (measured ~1.3 ms,
  it would erase the speedup). Pass `(nullptr, 0)` to never split. If a shape would split but `ws` is
  null or too small, `gemm()` runs unsplit (still correct, just slower) and logs a warning to stderr.
  Allocating per-stream avoids data races. See [`docs/OPTIMIZATIONS.md`](docs/OPTIMIZATIONS.md) §8.
- **Targets gfx950 (CDNA4 / MI350X).** The kernel uses `v_mfma_scale_f32_32x32x64_f8f6f4`.

---

## Tuning — kernel template parameters

The public API (`gemm.hpp`) is device-free and exposes **no** tuning knobs; the only shape-level
decision it makes is the tile, via `choose_tile`. The performance-relevant knobs live on the
internal kernel template `lds_gemm_hybrid_dripA` (`src/kernel.hpp`), launched from
`src/dispatch.hpp`. To experiment, edit the template arguments at the two `dispatch.hpp` call
sites (and re-thread the scale grouping — see *coupling* below). The defaults below are the
**measured optima** on gfx950; the rationale for the main design choices is in
[`docs/OPTIMIZATIONS.md`](docs/OPTIMIZATIONS.md), and the per-knob tuning notes (what was
measured, what lost) live in the `src/kernel.hpp` comments.

```cpp
template <int M_TILE, int N_TILE, int K_TILE, int WAVES_M, int WAVES_N, int MIN_OCC = 1,
          int SWZ = 0, typename OutT = float, int PFD = 5, bool HARD_WAIT = true,
          int ADRIP_START = 1, int ADRIP_STRIDE = 1, int ADRIP_PER = 1, int ADRIP_STOP = 0>
```

| Param | Default | Meaning |
|---|---|---|
| `M_TILE` × `N_TILE` | 256×256 / 128×384 / 128×256 / 128×128 | Register accumulator tile per workgroup. Larger ⇒ higher arithmetic intensity (AI) but fewer workgroups. The live shape knob (`choose_tile`): 256×256 for CU-filling shapes, 128×256 for WG-starved small-M, 128×384 for wide-N (removes wave quantization, 12 acc/wave, and re-reads A fewer times), 128×128 for the large-K wide-N shapes 128×384 cannot take. |
| `K_TILE` | 192 | Deep-K tile depth = size of the MFMA window backing one load. Bigger window hides load latency. `subs = K_TILE/64`. |
| `WAVES_M` × `WAVES_N` | 2×2 (1×4 for 128×384) | How the tile is split across the block's 4 waves → per-wave block shape `MPW×NPW` = `(M_TILE/32/WAVES_M)×(N_TILE/32/WAVES_N)`. Sets the load-vs-MFMA ratio (perimeter vs area) and input VGPR. 128×384 must use 1×4: 2×2 would give `N_PW=6` and trip the `NPW≤4` scale assert. |
| `MIN_OCC` | 1 | Occupancy floor (`__launch_bounds__` 2nd arg). occ1 holds the largest tile; occ2 forces a smaller AI (see coupling). |
| `SWZ` | 0 | Workgroup-ID swizzle width for L2 locality (`0` = off). Functional but currently always `0` from the dispatcher (swz0 is best on this machine). |
| `OutT` | — | Output element type (`float` / `__half` / `__hip_bfloat16`), set by `gemm()`. |
| `PFD` | 5 | Depth of B's compile-time register prefetch ring = B tiles in flight. Costs VGPR; `5` is the deepest that keeps spill = 0 at 16-acc. |
| `HARD_WAIT` | true | Insert `wait_vmcnt(0)` before each double-buffer barrier — a hard RAW guard for the dripped (compiler-invisible) A loads. **`false` is a footgun** (risks a RAW race); leave it `true`. |
| `ADRIP_START` | 1 | First MFMA quartet that issues A chunks (skip the sub-head stall quartet). |
| `ADRIP_STRIDE` | 1 | Quartets between successive issuing quartets (`1` = consecutive). |
| `ADRIP_PER` | 1 | A chunks issued per issuing quartet (`≥2` finishes A earlier ⇒ more RAW margin but more issue backpressure — measured net loss). |
| `ADRIP_STOP` | 0 | Last quartet (exclusive) allowed to issue A; `≤0` ⇒ run to the end of the tile. |

### Coupling — what must move together

These knobs are not independent. Four clusters share the same hardware budgets, and changing one
member forces re-balancing the rest:

- **A — register budget** `{M_TILE, N_TILE, WAVES_M, WAVES_N, MIN_OCC, PFD}`. All draw from the
  merged 512-VGPR pool (gfx950 fuses arch + acc; occupancy gate = total ≤ `512/MIN_OCC`). Per-wave
  accumulators `MPW×NPW` consume AGPR (16 acc = 256 AGPR); the `PFD` ring + operands + addresses
  consume arch VGPR. Because 16 acc alone is the *entire* occ2 budget, you cannot raise `MIN_OCC`
  without shrinking the tile or adding waves (i.e. cutting AI). Change any one ⇒ re-check spill = 0.
- **B — deep-K / LDS / scale** `{K_TILE, M_TILE}`. LDS use = `2 × M_TILE × (K_TILE·6/8)` must fit
  160 KB. And `subs = K_TILE/64` is the scale load width, capped at `dwordx4` ⇒ **`K_TILE ≤ 256`**.
- **C — host↔device layout (correctness, not just perf)**. `MPW`/`NPW` (derived from the tile and
  wave counts) **must** be passed to the host `tile_scale` (`MPW` for A, `NPW` for B), and `K_TILE`
  must match `tile_scale`'s `subs` and `kpad`. The kernel also asserts `MPW ≤ 4 && NPW ≤ 4`
  (`NDA==1 && NDB==1`). Get this wrong and the output is silently incorrect — so any change to
  cluster A/B that moves `MPW`/`NPW`/`K_TILE` requires the matching host change.
- **D — drip schedule** `{ADRIP_START, ADRIP_STRIDE, ADRIP_PER, ADRIP_STOP}`. The schedule must fit
  `ISSUES_A` (≈ `M_TILE·ROW_CHUNKS/256`) A-load chunks into the available compute quartets
  `NB = (K_TILE/64)·N_PW`: `(STOP−START)/STRIDE × PER ≥ ISSUES_A`. Changing `M_TILE`, `N_TILE`,
  `WAVES_*`, or `K_TILE` changes `NB`/`ISSUES_A`, so the drip schedule must be re-checked.

`SWZ` and `HARD_WAIT` are effectively standalone (`SWZ` couples only to shape/L2; `HARD_WAIT` is a
correctness guard whose only safe value is `true`).

> The shipped defaults have been swept to their optimum and the kernel is occ1 latency-bound — in
> practice there is little headroom left in these knobs alone. See `docs/OPTIMIZATIONS.md` for the
> measured ceilings and the dead ends already ruled out.

---

## Testing

`ctest` runs two gates:

* **`gemm`** (`test_gemm`) — the correctness gate: fresh-allocated, `0x5A`-poisoned outputs compared
  against a CPU reference on all four tile paths (256×256, 128×384, 128×256, 128×128), including
  non-square / partial-grid / `k_tiles==1`, split-K, and the compact-A recipe (`verify_compact`).
  Needs a GPU. It also prints an indicative FP16 performance sweep (let it reach steady state before
  trusting the numbers).
* **`routing`** (`test_gemm --routing`) — checks `choose_tile`'s decisions against a table of 18
  shapes. This is **host-only and needs no GPU**, because the kernel-level checks above force a tile
  via `gemm_force_tile()` and so never exercise the routing decision itself.
