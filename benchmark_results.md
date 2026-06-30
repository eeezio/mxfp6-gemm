# Benchmark: libmxfp6gemm vs CK MXFP8 / CK MXFP6

FP16 output, M=2048, MI350X (gfx950), ROCm 7.0.2.1.

† = auto split-K active (S in parentheses); unsplit TFLOPs in brackets for reference.

Columns:
- `mxfp6_ms` = this library's latency (ms)
- `mxfp6_TF_real` / `mxfp6_TF_kernel` = this library's effective / kernel-only TFLOPs
- `mxfp8_best_config` = CK MXFP8 best config index
- `mxfp8_ms` = CK MXFP8 best-config latency (ms)
- `mxfp8_TF_real` / `mxfp8_TF_kernel` = CK MXFP8 effective / kernel-only TFLOPs
- `mxfp6_over_mxfp8_speedup_ms` = mxfp8_ms / mxfp6_ms (>1.0 = this library wins)

| N | K | count | M | mxfp6_ms | mxfp6_TF_real | mxfp6_TF_kernel | mxfp8_best_config | mxfp8_ms | mxfp8_TF_real | mxfp8_TF_kernel | mxfp6_over_mxfp8_speedup_ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 512 | 4 | 2048 | 0.007884 | 0.266 | 76.607 | 1 | 0.004772 | 0.439 | 56.252 | 0.605 |
| 128 | 24 | 2 | 2048 | 0.005060 | 2.487 | 39.788 | 1 | 0.004634 | 2.715 | 28.963 | 0.916 |
| 128 | 32 | 1 | 2048 | 0.005066 | 3.312 | 39.740 | 1 | 0.004760 | 3.525 | 28.197 | 0.940 |
| 128 | 64 | 1 | 2048 | 0.005050 | 6.644 | 39.866 | 4 | 0.004740 | 7.079 | 42.474 | 0.939 |
| 128 | 92 | 1 | 2048 | 0.005034 | 9.582 | 39.993 | 1 | 0.004720 | 10.219 | 28.436 | 0.938 |
| 128 | 96 | 1 | 2048 | 0.005052 | 9.963 | 39.850 | 1 | 0.004746 | 10.605 | 28.280 | 0.939 |
| 128 | 104 | 1 | 2048 | 0.005072 | 10.750 | 39.693 | 1 | 0.004874 | 11.187 | 27.537 | 0.961 |
| 128 | 152 | 1 | 2048 | 0.005058 | 15.755 | 39.803 | 4 | 0.004746 | 16.791 | 42.420 | 0.938 |
| 128 | 240 | 1 | 2048 | 0.006800 | 18.504 | 59.213 | 1 | 0.004706 | 26.738 | 28.521 | 0.692 |
| 128 | 520 | 1 | 2048 | 0.007864 | 34.668 | 76.803 | 1 | 0.005254 | 51.889 | 76.637 | 0.668 |
| 128 | 1840 | 1 | 2048 | 0.016388 | 58.865 | 122.849 | 1 | 0.008052 | 119.807 | 133.350 | 0.491 |
| 256 | 64 | 23 | 2048 | 0.005058 | 13.190 | 39.569 | 1 | 0.004768 | 14.075 | 56.299 | 0.937 |
| 256 | 80 | 22 | 2048 | 0.005074 | 16.519 | 39.646 | 1 | 0.004794 | 17.498 | 55.993 | 0.944 |
| 256 | 96 | 38 | 2048 | 0.005074 | 19.917 | 39.835 | 4 | 0.004892 | 20.577 | 82.308 | 0.968 |
| 256 | 128 | 21 | 2048 | 0.005074 | 26.452 | 39.678 | 1 | 0.004789 | 27.973 | 55.947 | 0.946 |
| 256 | 144 | 2 | 2048 | 0.005082 | 29.711 | 39.615 | 1 | 0.004728 | 31.936 | 56.776 | 0.930 |
| 256 | 160 | 2 | 2048 | 0.005056 | 33.182 | 39.819 | 1 | 0.004788 | 35.040 | 56.064 | 0.947 |
| 256 | 256 | 18 | 2048 | 0.006700 | 40.065 | 60.097 | 1 | 0.004702 | 57.089 | 57.089 | 0.702 |
| 256 | 512 | 1 | 2048 | 0.007860 | 68.304 | 76.842 | 1 | 0.004768 | 112.598 | 112.598 | 0.607 |
| 256 | 2048 | 6 | 2048 | 0.017556 | 122.321 | 126.143 | 1 | 0.008366 | 256.690 | 256.690 | 0.477 |
| 512 | 2560 | 8 | 2048 | 0.021368 | 251.248 | 263.811 | 1 | 0.009712 | 552.786 | 552.786 | 0.455 |
| 512 | 6144 | 4 | 2048 | 0.026680 | **483** † (S=4) [277] | 535 | 1 | 0.024172 | 533.046 | 533.046 | **0.906** |
| 768 | 24 | 2 | 2048 | 0.005332 | 14.159 | 113.273 | 1 | 0.004804 | 15.716 | 167.632 | 0.901 |
| 768 | 32 | 1 | 2048 | 0.005374 | 18.731 | 112.388 | 1 | 0.004862 | 20.704 | 165.631 | 0.905 |
| 768 | 64 | 1 | 2048 | 0.005300 | 37.986 | 113.957 | 1 | 0.004788 | 42.048 | 168.191 | 0.903 |
| 768 | 92 | 1 | 2048 | 0.005322 | 54.379 | 113.486 | 1 | 0.004930 | 58.703 | 163.346 | 0.926 |
| 768 | 96 | 1 | 2048 | 0.005306 | 56.915 | 113.830 | 1 | 0.004830 | 62.523 | 166.728 | 0.910 |
| 768 | 104 | 1 | 2048 | 0.005270 | 62.078 | 114.606 | 1 | 0.004838 | 67.621 | 166.453 | 0.918 |
| 768 | 152 | 1 | 2048 | 0.005294 | 90.319 | 114.087 | 1 | 0.004836 | 98.872 | 166.522 | 0.913 |
| 768 | 240 | 1 | 2048 | 0.006970 | 108.317 | 173.307 | 1 | 0.004908 | 153.824 | 164.079 | 0.704 |
| 768 | 520 | 1 | 2048 | 0.008160 | 200.462 | 222.050 | 1 | 0.005898 | 277.342 | 409.613 | 0.723 |
| 768 | 1840 | 1 | 2048 | 0.016802 | 344.489 | 359.467 | 1 | 0.008860 | 653.285 | 727.135 | 0.527 |
| 1024 | 1024 | 10 | 2048 | 0.011918 | 360.373 | 405.420 | 1 | 0.006670 | 643.918 | 643.918 | 0.560 |
| 1024 | 3490 | 1 | 2048 | 0.029070 | 503.543 | 526.340 | 1 | 0.016516 | 886.291 | 910.163 | 0.568 |
| 1024 | 12288 | 20 | 2048 | 0.046481 | **1109** † (S=4) [561] | 1087 | 1 | 0.047398 | 1087.370 | 1087.370 | **1.020** |
| 1024 | 16128 | 9 | 2048 | 0.053881 | **1255** † (S=4) [570] | 1200 | 1 | 0.059578 | 1135.400 | 1135.400 | **1.106** |
| 1024 | 105728 | 1 | 2048 | 0.308685 | **1437** † (S=4) [491] | 1462 | 1 | 0.486003 | 912.455 | 912.455 | **1.574** |
| 2048 | 3490 | 1 | 2048 | 0.031420 | 931.764 | 973.947 | 0 | 0.020896 | 1401.030 | 1438.770 | 0.665 |
| 3490 | 1024 | 1 | 2048 | 0.014544 | 1006.460 | 1162.770 | 0 | 0.016278 | 899.250 | 923.470 | 1.119 |
| 4096 | 4096 | 20 | 2048 | 0.053004 | 1944.730 | 1944.730 | 0 | 0.062535 | 1648.350 | 1648.350 | 1.180 |
| 6144 | 512 | 8 | 2048 | 0.017264 | 746.336 | 839.629 | 3 | 0.015872 | 811.793 | 811.793 | 0.919 |
| 6144 | 4096 | 20 | 2048 | 0.070471 | 1462.730 | 1508.440 | 0 | 0.068291 | 1509.420 | 1509.420 | 0.969 |
| 6144 | 16128 | 9 | 2048 | 0.238366 | 1702.730 | 1702.730 | 0 | 0.211298 | 1920.860 | 1920.860 | 0.886 |
| 6144 | 105728 | 1 | 2048 | 2.065000 | 1288.490 | 1289.270 | 0 | 1.854740 | 1434.560 | 1434.560 | 0.898 |
| 12672 | 1024 | 9 | 2048 | 0.038762 | 1371.180 | 1558.160 | 0 | 0.051960 | 1022.900 | 1022.900 | 1.340 |
| 16128 | 1024 | 9 | 2048 | 0.041494 | 1630.240 | 1834.020 | 0 | 0.060195 | 1123.790 | 1123.790 | 1.451 |
| 16384 | 1024 | 10 | 2048 | 0.042054 | 1634.060 | 1838.320 | 0 | 0.060787 | 1130.500 | 1130.500 | 1.445 |
| 20480 | 6144 | 9 | 2048 | 0.199770 | 2579.950 | 2579.950 | 0 | 0.294245 | 1751.590 | 1751.590 | 1.473 |
| 102272 | 1024 | 1 | 2048 | 0.311703 | 1376.180 | 1550.140 | 0 | 0.472239 | 908.354 | 908.354 | 1.515 |
| 105728 | 1024 | 1 | 2048 | 0.313101 | 1416.330 | 1593.380 | 0 | 0.486935 | 910.708 | 910.708 | 1.555 |

† = auto split-K active. These rows re-measured on MI350X / ROCm 7.0.2 (2026-06-30): `mxfp6_ms` =
best single-launch latency (incl. reduce); `mxfp6_TF_real` from that latency; `mxfp6_TF_kernel` =
best back-to-back throughput; `[brackets]` = unsplit TFLOPs for reference. All other (non-†) rows
are the original vs-CK harness snapshot (ROCm 7.0.2.1) — verified unchanged by the split-K work via
an A/B build of the pre-split-K commit (S=1 path identical within <1%), so they are left as-is.

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
multiple of 32) so they are not re-measured here. The guard (below) keeps `2048×2048×3490` at S=1,
avoiding the regression that motivated it.

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

## Remaining problems / gaps

1. **Data gaps** — `512×6144`, `1024×3490`, `2048×3490` now split but ms latency not re-measured; main table shows `—` for those cells.
2. **Borderline shapes gated OUT by `k_tiles ≥ 16`** — CU-starved shapes that just miss the K
   threshold get no split and lose hard to CK: `2048×1024×1024` (k_tiles=6, 0.56×),
   `2048×768×1840` (10, 0.53×), `2048×256×2048` (11, 0.48×), `2048×512×2560` (14, 0.46×).
3. **Mid-N moderate-K is the weakest band vs CK** — `256×2048` 0.48×, `512×2560` 0.46×,
   `512×6144` 0.52× (now splits → should improve once re-measured).
4. **Small N + small K is overhead-bound** — N≤256 with small K sit at ~40 kernel TFLOPs and a
   ~5 µs latency floor. Fixed launch/prologue cost dominates; split-K cannot help.
5. **Wide-N very-large-K L2 sag** — `2048×6144×105728` = 1288 TF; base_wg=384 already fills CUs so split-K doesn't apply; throughput drops as growing K erodes L2 reuse.
6. **Split-K's own ceiling** — the FP32 partial-sum write + reduce round-trip caps speedup well below the ideal `S×` (2.99× of an ideal 4× for 105728).
