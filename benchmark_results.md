# Benchmark: libmxfp6gemm vs CK MXFP8 / CK MXFP6

FP16 output, M=2048, MI350X (gfx950).

Columns:
- `tile` / `S` = the tile `choose_tile()` routes this shape to, and the split-K factor
- `ours_ms` / `ours_TF` = this library's latency and effective TFLOPs (nominal `2·M·N·K` over the
  measured latency, so K-padding waste is charged to us)
- `ck_ms` / `ck_TF` = CK MXFP8 latency and effective TFLOPs
- `ours/CK` = `ck_ms / ours_ms` (>1.0 = this library wins)

**Measurement (2026-07-29):** MI350X (gfx950, 256 CU) `bg-1w300-k2-3a`, image `rocm/atom:latest`,
`HIP_VISIBLE_DEVICES=0`, one pass of `test_gemm M N K` per shape. CK is
`tile_example_mx_flatmm -mx_prec=fp8xfp8 -v=0 -warmup=20 -repeat=50` in the same session.
K is zero-padded to a multiple of `K_TILE=192` inside the harness and TFLOPs are charged against the
nominal K, so rows whose K is not a multiple of 32 (which `preprocess.hpp` requires of a real caller)
are measured as their padded equivalent — the padding waste shows up as a lower `ours_TF`.


| N | K | count | tile | S | ours_ms | ours_TF | ck_ms | ck_TF | ours/CK | source |
|---:|---:|---:|:---:|:---:|---:|---:|---:|---:|:---:|:---|
| 1 | 512 | 4 | — | — | 0.007884 | 0.3 | 0.004772 | 0.4 | 0.61× | not re-measured ‡ |
| 128 | 24 | 2 | — | — | 0.005060 | 2.5 | 0.004634 | 2.7 | 0.92× | not re-measured ‡ |
| 128 | 32 | 1 | — | — | 0.005066 | 3.3 | 0.004760 | 3.5 | 0.94× | not re-measured ‡ |
| 128 | 64 | 1 | — | — | 0.005050 | 6.6 | 0.004740 | 7.1 | 0.94× | not re-measured ‡ |
| 128 | 92 | 1 | — | — | 0.005034 | 9.6 | 0.004720 | 10.2 | 0.94× | not re-measured ‡ |
| 128 | 96 | 1 | — | — | 0.005052 | 10.0 | 0.004746 | 10.6 | 0.94× | not re-measured ‡ |
| 128 | 104 | 1 | — | — | 0.005072 | 10.8 | 0.004874 | 11.2 | 0.96× | not re-measured ‡ |
| 128 | 152 | 1 | — | — | 0.005058 | 15.8 | 0.004746 | 16.8 | 0.94× | not re-measured ‡ |
| 128 | 240 | 1 | — | — | 0.006800 | 18.5 | 0.004706 | 26.7 | 0.69× | not re-measured ‡ |
| 128 | 520 | 1 | — | — | 0.007864 | 34.7 | 0.005254 | 51.9 | 0.67× | not re-measured ‡ |
| 128 | 1840 | 1 | — | — | 0.016388 | 58.9 | 0.008052 | 119.8 | 0.49× | not re-measured ‡ |
| 256 | 64 | 23 | 128x256 | 1 | 0.005186 | **12.9** | 0.004768 | 14.1 | **0.92×** | CK: 2026-06 † |
| 256 | 80 | 22 | 128x256 | 1 | 0.005180 | **16.2** | 0.004794 | 17.5 | **0.93×** | CK: 2026-06 † |
| 256 | 96 | 38 | 128x256 | 1 | 0.005172 | **19.5** | 0.004892 | 20.6 | **0.95×** | CK: 2026-06 † |
| 256 | 128 | 21 | 128x256 | 1 | 0.005174 | **25.9** | 0.004789 | 28.0 | **0.93×** | CK: 2026-06 † |
| 256 | 144 | 2 | 128x256 | 1 | 0.005186 | **29.1** | 0.004728 | 31.9 | **0.91×** | CK: 2026-06 † |
| 256 | 160 | 2 | 128x256 | 1 | 0.005178 | **32.4** | 0.004788 | 35.0 | **0.92×** | CK: 2026-06 † |
| 256 | 256 | 18 | 128x256 | 1 | 0.006546 | **41.0** | 0.004702 | 57.1 | **0.72×** | CK: 2026-06 † |
| 256 | 512 | 1 | 128x256 | 1 | 0.007850 | **68.4** | 0.004768 | 112.6 | **0.61×** | CK: 2026-06 † |
| 256 | 2048 | 6 | 128x256 | 1 | 0.017514 | **122.6** | 0.008366 | 256.7 | **0.48×** | CK: 2026-06 † |
| 512 | 2560 | 8 | 128x256 | 1 | 0.021296 | **252.1** | 0.009712 | 552.8 | **0.46×** | CK: 2026-06 † |
| 512 | 6144 | 4 | 128x256 | 4 | 0.024156 | **533.4** | 0.024172 | 533.0 | **1.00×** | CK: 2026-06 † |
| 768 | 24 | 2 | 128x256 | 1 | 0.005448 | **13.9** | 0.004804 | 15.7 | **0.88×** | CK: 2026-06 † |
| 768 | 32 | 1 | 128x256 | 1 | 0.005472 | **18.4** | 0.004862 | 20.7 | **0.89×** | CK: 2026-06 † |
| 768 | 64 | 1 | 128x256 | 1 | 0.005442 | **37.0** | 0.004788 | 42.0 | **0.88×** | CK: 2026-06 † |
| 768 | 92 | 1 | 128x256 | 1 | 0.005460 | **53.0** | 0.004930 | 58.7 | **0.90×** | CK: 2026-06 † |
| 768 | 96 | 1 | 128x256 | 1 | 0.005456 | **55.4** | 0.004830 | 62.5 | **0.89×** | CK: 2026-06 † |
| 768 | 104 | 1 | 128x256 | 1 | 0.005478 | **59.7** | 0.004838 | 67.6 | **0.88×** | CK: 2026-06 † |
| 768 | 152 | 1 | 128x256 | 1 | 0.005462 | **87.5** | 0.004836 | 98.9 | **0.89×** | CK: 2026-06 † |
| 768 | 240 | 1 | 128x256 | 1 | 0.006854 | **110.2** | 0.004908 | 153.8 | **0.72×** | CK: 2026-06 † |
| 768 | 520 | 1 | 128x256 | 1 | 0.008190 | **199.7** | 0.005898 | 277.3 | **0.72×** | CK: 2026-06 † |
| 768 | 1840 | 1 | 128x256 | 1 | 0.016798 | **344.6** | 0.008860 | 653.3 | **0.53×** | CK: 2026-06 † |
| 1024 | 1024 | 10 | 128x256 | 1 | 0.011988 | **358.3** | 0.006670 | 643.9 | **0.56×** | CK: 2026-06 † |
| 1024 | 3490 | 1 | 128x256 | 2 | 0.027386 | **534.5** | 0.016516 | 886.3 | **0.60×** | CK: 2026-06 † |
| 1024 | 12288 | 20 | 128x256 | 4 | 0.042668 | **1207.9** | 0.047398 | 1087.4 | **1.11×** | CK: 2026-06 † |
| 1024 | 16128 | 9 | 128x256 | 4 | 0.049794 | **1358.5** | 0.059578 | 1135.4 | **1.20×** | CK: 2026-06 † |
| 1024 | 105728 | 1 | 128x256 | 4 | 0.277809 | **1596.3** | 0.486003 | 912.5 | **1.75×** | CK: 2026-06 † |
| 2048 | 3490 | 1 | 128x256 | 2 | 0.033978 | **861.6** | 0.020896 | 1401.0 | **0.61×** | CK: 2026-06 † |
| 3490 | 1024 | 1 | — | — | 0.014544 | 1006.5 | 0.016278 | 899.2 | 1.12× | not re-measured ‡ |
| 4096 | 4096 | 20 | 128x256 | 1 | 0.040990 | **1676.5** | 0.048385 | 1464.6 | **1.14×** | same-session |
| 6144 | 512 | 8 | 128x384 | 1 | 0.013384 | **962.7** | 0.028225 | 470.8 | **2.04×** | same-session |
| 6144 | 4096 | 20 | 128x384 | 1 | 0.048574 | **2122.1** | 0.083084 | 1279.4 | **1.66×** | same-session |
| 6144 | 16128 | 9 | 128x384 | 1 | 0.152826 | **2655.8** | 0.252273 | 1659.2 | **1.60×** | same-session |
| 6144 | 105728 | 1 | 128x128 | 1 | 1.513014 | **1758.6** | 1.875560 | 1463.0 | **1.20×** | same-session |
| 12672 | 1024 | 9 | — | — | 0.038762 | 1371.2 | 0.051960 | 1022.9 | 1.34× | not re-measured ‡ |
| 16128 | 1024 | 9 | 256x256 | 1 | 0.043622 | **1550.7** | 0.067595 | 1032.0 | **1.50×** | same-session |
| 16384 | 1024 | 10 | 256x256 | 1 | 0.043702 | **1572.5** | 0.068504 | 1034.5 | **1.52×** | same-session |
| 20480 | 6144 | 9 | 256x256 | 1 | 0.207322 | **2486.0** | 0.272072 | 1953.5 | **1.27×** | same-session |
| 102272 | 1024 | 1 | — | — | 0.311703 | 1376.2 | 0.472239 | 908.4 | 1.52× | not re-measured ‡ |
| 105728 | 1024 | 1 | 256x256 | 1 | 0.311644 | **1423.0** | 0.490322 | 932.7 | **1.53×** | same-session |

† = **`ours` is 2026-07-29, `ck` is carried over from the 2026-06 harness** — so the ratio on these
rows is *indicative, not same-session*. Reason: the CK binary available on this node
(`tile_example_mx_flatmm`) exposes a single kernel config with no tuning, and at N ≤ 1024 it lands at
**0.33–0.54×** of the tuned CK numbers the 2026-06 harness recorded (at N ≥ 4096 the two agree to
0.89–1.12×, which is why those rows are marked same-session). Publishing the untuned CK number here
would overstate our margin by 2–3×, so the previously tuned CK figure is kept instead. CK also
refuses `K % 256 != 0` outright (`KPerBlock`), which rules it out on most of the N=768 rows.

‡ = **not re-measured — our kernel cannot run these shapes.** The launch grid is
`dim3(M/MT, N/NT)` with integer division and no remainder handling, so N must be a multiple of the
tile's N. For N=1 and N=128 `floor(N/256)=0`, i.e. **zero workgroups launch** and the old row is
timing an empty dispatch; for N=3490 / 12672 / 102272 only `floor(N/256)·256` columns are computed
(162 / 128 / 128 columns short). Their numbers are left exactly as the old harness recorded them and
should not be read as measurements of this library. Supporting them needs N-remainder handling.

Split-K rows are identified by `S > 1` in the table (`S` = number of K segments); their `ours_ms`
includes the FP32 partial-sum reduce.

---

## Performance vs CK MXFP6 (same precision, K=8192)

| M × N | CK MXFP6 (TFLOPs) | libmxfp6gemm (TFLOPs) | speedup |
|---|---:|---:|---:|
| 2048 × 4096 | 1050 | 1706 | **1.63×** |
| 2048 × 8192 | 1059 | 2176 | **2.06×** |
| 4096 × 4096 | 1059 | 2329 | **2.20×** |
| 4096 × 8192 | 1112 | 2352 | **2.12×** |
| 8192 × 8192 | 1115 | 2418 | **2.17×** |

---

## Split-K results (FP16, MI350X/gfx950, ROCm 7.0.2)

Auto split-K for workgroup-starved shapes (`base_wg < 256` CUs and `k_tiles ≥ 2·MIN_TILES_PER_SEG`):
the K dimension is split into `S` segments computed by independent workgroups (grid z-dim), each
writing FP32 partial sums to a caller-provided workspace; a reduce kernel then sums + converts to
the output type. `S = ceil(CU/base_wg)`, capped so every segment keeps `≥ MIN_TILES_PER_SEG` tiles.
Segments may now be **uneven** (the first `k_tiles%S` take one extra tile), so `S` no longer has to
divide `k_tiles` — see the "uneven" fix note at the end.

### Measured (re-measured 2026-06-30, MI350X, ROCm 7.0.2, single-shape bench; TF_real)

| shape (M×N×K) | unsplit | split-K | S | vs unsplit | vs CK MXFP8 |
|---|---|---|---|---|---|
| 2048×1024×12288  | 561 | 1109 | 4 (even)   | 1.98× | 1.02× (CK 1087) |
| 2048×1024×16128  | 570 | 1255 | 4 (even)   | 2.20× | 1.11× (CK 1135) |
| 2048×1024×105728 | 491 | **1437** | 4 (uneven 138/138/138/137) | 2.93× | **1.57×** (CK 912) |
| 2048×512×6144    | 277 | 483  | 4 (even)   | 1.74× | 0.91× (CK 533) |
| 2048×6144×16128 (control) | 1588 | 1586 | 1 (no split) | ~1.0× | already fills CUs |

Run-to-run jitter ~1–3% (single-shape `bench_all`: latency = best of 50 single launches incl. reduce;
throughput = best back-to-back ×20). CK columns from the original vs-CK harness — kept as reference,
so the **ratios** are the reliable takeaway. The S=1 non-split path was A/B-verified unchanged by the
split-K work (pre-split-K commit built + benched identically, <1% delta).

**`3490`-K shapes** (`2048×1024×3490`, `2048×2048×3490`) are not direct library inputs (K not a
multiple of 32); the main table measures them as their zero-padded equivalent (`Kp=3648`), which is
why they appear there but not in this split-K table. The guard (below) keeps `2048×2048×3490` at
S=1, avoiding the regression that motivated it.

**⚠ Historical (pre-guard) regression that motivated the guard: `2048×2048×3490` 931→838 at S=2.**
`k_tiles=19` (odd); the OLD divisibility constraint forced `19%2≠0 → S=1` (931). Removing it for the
uneven fix made it split S=2, which lost: `base_wg=128` already half-fills the CUs (S=2 barely helps
occupancy) while `N=2048` makes the FP32 partial-sum + reduce round-trip expensive. The guard now
restores S=1 here.

### Threshold experiment (MIN_TILES_PER_SEG 8→4, lets borderline shapes split)

| shape (M×N×K) | MIN=8 | MIN=4 | Δ |
|---|---|---|---|
| 2048×512×6144  | 521 (S4) | 547 (S8) | +5% |
| 2048×1024×3490 | 521 (S2) | 577 (S4) | +11% |
| 2048×768×1840  | 334 (S1) | 270 (S2) | **−19% ⚠** |
| 2048×256×2048  | 120 (S1) | 129 (S2) | +7.5% |
| 2048×512×2560  | 247 (S1) | 246 (S3) | flat |
| 2048×1024×1024 | 345 (S1) | 345 (S1) | gated out (k_tiles=6 < 8) |

Mixed (helps some, costs `768×1840` −19%). Not adopted as a blanket change — but it motivated the
calibration below.

## Reduce-cost-aware guard (implemented)

A split costs an FP32 partial-sum reduce round-trip (`~S·M·N·4` bytes) and only buys GEMM
parallelism up to a full CU wave. Calibrated the decision on a 30-shape `base_wg × K` sweep
(M=2048, FP16, gfx950 — each shape timed **unsplit vs split**, MIN_TILES_PER_SEG=2 to expose every
boundary).

**Key finding — wave quantization.** `S = ceil(CU/base_wg)` oversubscribes when `base_wg` doesn't
divide `CU`. E.g. `base_wg=192, S=2 → 384 > 256` WGs = 2 waves: each WG still does `k_tiles/2` over
2 waves = the unsplit wall time → **zero GEMM gain, full reduce cost → always loses** (every `N=3072`
`S=2` shape lost, 0.45–0.90×). The raw margin `Kp·(S−1)/(base_wg·S²)` does NOT separate winners from
losers (overlap r=9–16). Correcting for waves does:

> `W = ceil(base_wg·S / CU)` → effective depth gain `(S−W)/S`. Margin `r' = Kp·(S−W)/(S²·base_wg)`.
> Win/loss boundary collapses to **r' ≈ 8–10 across all base_wg/S** → guard: split iff
> `Kp·(S−W) > ALPHA·base_wg·S²`, **ALPHA = 10**.

Calibration boundary (unsplit→split speedup vs r'):

| base_wg | S | crossover r' | example |
|---|---|---|---|
| 192 | 2 | never (W=2, S−W=0) | all loss, down to 0.45× |
| 128 | 2 | ~8 | r'=6 → 0.88× ; r'=9 → 1.03× |
| 96  | 3 | ~7 | r'=7.1 → 1.00× ; r'=10.7 → 1.08× |
| 64  | 4 | ~9 | r'=4.5 → 0.59× ; r'=9 → 1.02× |
| 32  | 8 | ~10 | r'=5.3 → 0.70× ; r'=10.5 → 1.03× |

**Validation (guard live, MIN=8, ALPHA=10):**

| shape | S | TFLOPs | outcome |
|---|---|---|---|
| 2048×2048×3490 | 1 | 905 | regression FIXED (was 838 at S=2) |
| 2048×3072×6144 | 1 | 1408 | oversubscribed bw192 correctly not split |
| 2048×1024×3490 | 2 | 524 | deep-enough S=2 win KEPT (vs 503 unsplit) |
| 2048×512×6144  | 4 | 523 | kept |
| 2048×1024×12288 | 4 | 1174 | kept |
| 2048×1024×16128 | 4 | 1315 | kept |
| 2048×1024×105728 | 4 | 1466 | kept |
| 2048×6144×16128 | 1 | 1582 | control unchanged |

The guard beats a blunt `S≥3` rule: it keeps `1024×3490` (S=2, deep K) which `S≥3` would wrongly
drop, and would correctly split a hypothetical `S=2 + huge-K` shape. `ALPHA` is per-architecture
(absorbs achieved-FLOPS/BW); re-calibrate when CU count, HBM BW, or peak change.

### The "uneven split" fix (K=105728)

`kpad(105728)=105792` → `k_tiles=551=19×29`. The original heuristic required `S` to divide
`k_tiles` evenly; 551 has no factor near the wanted `S=4`, so `S` collapsed to 1 and the shape ran
unsplit at 64/256 CU — exactly the case split-K targets. Fixed by allowing uneven segments
(`offset = z·floor + min(z,rem)`, `length = floor + (z<rem)`; reduces to the old equal-length path
when `k_tiles%S==0`). Verified vs CPU reference on the small uneven analog `2048×1024×6304`
(`k_tiles=33=3×11`, S=4 → 9/8/8/8, with a real K-tail in the shortest final segment).

---

## Wide-N (N=6144) — 128×256 → 128×384, same-machine A/B (MI350X, 2026-07-29)

**Machine:** MI350X (gfx950, 256 CU) `bg-1w300-k2-3a`, image `rocm/atom:latest`,
`HIP_VISIBLE_DEVICES=0`. **Method:** the `128×256` baseline (`origin/main`) and the `128×384` build
are compiled from the same source pair in one container invocation and run **interleaved, 3 reps**,
with CK measured in the same session. M=2048, N=6144, FP16.
Raw: `prof_results/bench_mi350x_k2-3a_2026-07-29.txt`.

> Why this run exists: every earlier "128×256 → 128×384" delta compared numbers taken on *different*
> machines. Absolute TFLOPs are not comparable across nodes — only same-session deltas and ratios.

| K | baseline (128×256) | **128×384** | Δ | CK mxfp8 | baseline/CK | **new/CK** |
|---:|---:|---:|:---:|---:|:---:|:---:|
| 512    | 746  | **966**  | +29.5% | 473  | 1.58× | **2.04×** |
| 4096   | 1427 | **2131** | +49.3% | 1290 | 1.11× | **1.65×** |
| 16128  | 1702 | **2707** | +59.0% | 1654 | 1.03× | **1.64×** |
| 105728 | 1762 | 1759     | ±0     | 1463 | 1.20× | **1.20×** (control: stays 128×128) |

Median of 3 reps; spread was ≤3% on every cell. These are the interleaved A/B medians, so they
differ by ≤2% from the single-pass numbers in the main table above (962.7 / 2122.1 / 2655.8).
K=105728 is the control — it is gated out of the
128×384 route (`Kp < LARGEK_THRESH`) because a 384-column B slice would be 30.5 MB and overflow L2,
so it keeps the 128×128 route and is byte-for-byte identical. `base/CK` at K=16128 = 1.03×
reproduces the historical 1.02× for this shape, which is the check that the baseline is sound.

The gain compounds three effects: wave quantization removed (384 WGs / 1.5 waves → 256 WGs /
1 wave), accumulators per wave 8 → 12, and compute per B ring slot 64 → 128 cyc so the PFD=5
prefetch ring covers 640 cyc instead of 320. Effects 2 and 3 are why the shallower K gained
proportionally more than the wave-quantization argument alone predicts.

Earlier cross-check of the **128×128** (large-K) fix on the **MI355X** reference node (`smci355-ccs-aus-n03-05`, 2026-07-20, same-session,
ROCm 7.0.2):

| K | ours baseline | ours integrated | CK mxfp8 | integrated/CK |
|---:|---:|---:|---:|:---:|
| 512    | 772  | 774  | 505  | **1.53×** |
| 4096   | 1509 | 1532 | 1397 | **1.10×** |
| 16128  | 1771 | 1770 | 1780 | **0.99×** |
| **105728** | **1318** | **1811** | **1523** | **1.19×** (was 0.87×, +37.4%) |

Raw: `prof_results/bench_n03-05_2026-07-20.txt`. The superseded 2026-07-17 MI350X-B run that this
section previously quoted is retained raw in `prof_results/bench_mi350x_10.7.191.60_2026-07-17.txt`.

---

## Remaining problems / gaps

1. **N-remainder shapes are unsupported** — `N=1`, `128`, `3490`, `12672`, `102272` cannot be run
   (see ‡ above): the grid is `dim3(M/MT, N/NT)` with no remainder handling. Their table rows are
   stale old-harness numbers, not measurements of this library.
2. **Borderline shapes gated OUT by `k_tiles ≥ 16`** — CU-starved shapes that just miss the K
   threshold get no split and lose hard to CK: `2048×1024×1024` (k_tiles=6, 0.56×),
   `2048×768×1840` (10, 0.53×), `2048×256×2048` (11, 0.48×), `2048×512×2560` (14, 0.46×).
3. **Mid-N moderate-K is the weakest band vs CK** — `256×2048` 0.48×, `512×2560` 0.46×,
   `2048×3490` 0.61×. (`512×6144` was in this list; the uneven split-K now takes it to 1.00×.)
4. **Small N + small K is overhead-bound** — N≤256 with small K sit at ~40 kernel TFLOPs and a
   ~5 µs latency floor. Fixed launch/prologue cost dominates; split-K cannot help.
5. ~~**Wide-N very-large-K L2 sag**~~ — **RESOLVED (PR #4).** `2048×6144×105728` was 1288 TF (0.90×
   CK). Root cause: B/N-slice (256 cols × K × 0.75 = 20.3 MB) overflows L2 → capacity/BW-bound (not
   latency-bound). Fixed by routing to a **128×128 tile** (halves B slice to 10.2 MB). MI350X result:
   **1.20× CK**; MI355X **1.19× CK**. See `docs/OPTIMIZATIONS.md §9` and `prof_results/bench_mi350x_*.txt`.
6. **Split-K's own ceiling** — the FP32 partial-sum write + reduce round-trip caps speedup well below the ideal `S×` (2.99× of an ideal 4× for 105728).
7. ~~**Wide-N moderate-K wave quantization**~~ — **RESOLVED.** `2048×6144×{512,4096,16128}` ran
   384 WGs on 256 CUs (1.5 waves). Routing `N%384==0` moderate-K shapes to a **128×384 tile** gives
   exactly 256 WGs; +29%/+49%/+59% and 1.64–2.04× CK. The N=6144 shapes that still trail are gone;
   remaining N=6144 work is the large-K `128×128` route at 1.20× CK.
