# Benchmark: libmxfp6gemm vs CK MXFP8 / CK MXFP6

FP16 output, M=2048, MI350X (gfx950).

Columns:
- `tile` / `S` = the tile `choose_tile()` routes this shape to, and the split-K factor
- `ours_ms` / `ours_TF` = this library's latency and effective TFLOPs (nominal `2·M·N·K` over the
  measured latency, so K-padding waste is charged to us)
- `ck_ms` / `ck_TF` = CK MXFP8 latency and effective TFLOPs
- `ours/CK` = `ours_TF / ck_TF` (>1.0 = this library wins). **The two halves of this ratio come
  from different hosts** — see the provenance note below before quoting it.
- `vs main` = `main_ms / ours_ms` from an interleaved A/B in the same session (>1.0 = this branch is
  faster than `origin/main`). This is the only column where both numbers share a machine, a ROCm,
  and a session, so it is the one that supports a causal claim.

**`ours_*` re-measured 2026-08-18** on host `bg-1w300-g4-1` (gfx950, device `0x75a0`), binaries
built locally in `rocm/pytorch:rocm7.0.2_ubuntu24.04_py3.13_pytorch_release_2.9.1` (HIP 7.0.51831 —
same ROCm major/minor as the rows it replaces) and executed under a ROCm 7.2.3 container on the
host. `HIP_VISIBLE_DEVICES=0`, `rocm-smi --setperfdeterminism 2200`, median of 3 interleaved passes
(arms alternate on every shape and every rep). Raw:
[`prof_results/ab_pr7fix_vs_main_g4-1_2026-08-18.tsv`](prof_results/ab_pr7fix_vs_main_g4-1_2026-08-18.tsv).
The pre-fix sweep that surfaced the `N=128` regression, and a 9-rep recheck of its marginal shapes,
are in [`ab_pr7_vs_main_g4-1_...`](prof_results/ab_pr7_vs_main_g4-1_2026-08-18.tsv) and
[`..._recheck9_...`](prof_results/ab_pr7_vs_main_recheck9_g4-1_2026-08-18.tsv).

**`ck_*` are NOT from that run.** They are carried over unchanged from the 2026-08-11 measurement on
`bg-1w300-k2-3a` (image `rocm/pytorch:rocm7.0.2_ubuntu24.04_py3.12_pytorch_release_2.8.0`, CK
`tile_example_mx_flatmm -mx_prec=fp8xfp8 -v=0 -warmup=20 -repeat=50`, raw
[`prof_results/bench_mi350x_k2-3a_2026-08-11.txt`](prof_results/bench_mi350x_k2-3a_2026-08-11.txt)).
So `ours/CK` mixes two hosts and is indicative only. `g4-1` measured slower than `k2-3a` on shapes
both have seen (N=105728 K=1024: 0.3333 ms against 0.2875 ms), so the mixed ratios more likely
understate this library than flatter it; no correction has been applied. Same-machine CK for the
shapes that support it is in the section below.

K is zero-padded to a multiple of `K_TILE=192` inside the harness and TFLOPs are charged against the
nominal K, so rows whose K is not a multiple of 32 (which `preprocess.hpp` requires of a real caller)
are measured as their padded equivalent — the padding waste shows up as a lower `ours_TF`.

| N | K | count | tile | S | ours_ms | ours_TF | ck_ms | ck_TF | ours/CK | vs main |
|---:|---:|---:|:---:|:---:|---:|---:|---:|---:|:---:|:---|
| 1 | 512 | 4 | — | — | 0.007884 | 0.3 ‡ | 0.004772 | 0.4 | — ‡ | — ‡ |
| 128 | 24 | 2 | 128x128 | 1 | 0.003852 | **3.3** | 0.004634 | 2.7 | **1.21×** | 1.036× |
| 128 | 32 | 1 | 128x128 | 1 | 0.003858 | **4.3** | 0.004760 | 3.5 | **1.24×** | 1.037× |
| 128 | 64 | 1 | 128x128 | 1 | 0.003848 | **8.7** | 0.004740 | 7.1 | **1.23×** | 1.038× |
| 128 | 92 | 1 | 128x128 | 1 | 0.003856 | **12.5** | 0.004720 | 10.2 | **1.23×** | 1.035× |
| 128 | 96 | 1 | 128x128 | 1 | 0.003850 | **13.1** | 0.004746 | 10.6 | **1.23×** | 1.031× |
| 128 | 104 | 1 | 128x128 | 1 | 0.003854 | **14.1** | 0.004874 | 11.2 | **1.26×** | 1.042× |
| 128 | 152 | 1 | 128x128 | 1 | 0.003854 | **20.7** | 0.004746 | 16.8 | **1.23×** | 1.040× |
| 128 | 240 | 1 | 128x128 | 1 | 0.004856 | **25.9** | 0.004706 | 26.7 | **0.97×** | 1.059× |
| 128 | 520 | 1 | 128x128 | 1 | 0.005770 | **47.2** | 0.005254 | 51.9 | **0.91×** | 1.057× |
| 128 | 1840 | 1 | 128x128 | 1 | 0.012318 | **78.3** | 0.008052 | 119.8 | **0.65×** | 1.031× |
| 256 | 64 | 23 | 128x256 | 1 | 0.004934 | **13.6** | 0.004768 | 14.1 | **0.96×** | 1.060× |
| 256 | 80 | 22 | 128x256 | 1 | 0.004934 | **17.0** | 0.004794 | 17.5 | **0.97×** | 1.059× |
| 256 | 96 | 38 | 128x256 | 1 | 0.004922 | **20.5** | 0.004892 | 20.6 | **0.99×** | 1.063× |
| 256 | 128 | 21 | 128x256 | 1 | 0.004934 | **27.2** | 0.004789 | 28.0 | **0.97×** | 1.059× |
| 256 | 144 | 2 | 128x256 | 1 | 0.004936 | **30.6** | 0.004728 | 31.9 | **0.96×** | 1.060× |
| 256 | 160 | 2 | 128x256 | 1 | 0.004914 | **34.1** | 0.004788 | 35.0 | **0.98×** | 1.063× |
| 256 | 256 | 18 | 128x256 | 1 | 0.006228 | **43.1** | 0.004702 | 57.1 | **0.75×** | 1.090× |
| 256 | 512 | 1 | 128x256 | 1 | 0.007386 | **72.7** | 0.004768 | 112.6 | **0.65×** | 1.076× |
| 256 | 2048 | 6 | 128x256 | 1 | 0.017014 | **126.2** | 0.008366 | 256.7 | **0.49×** | 1.030× |
| 512 | 2560 | 8 | 128x256 | 1 | 0.020724 | **259.1** | 0.009712 | 552.8 | **0.47×** | 1.028× |
| 512 | 6144 | 4 | 128x256 | 4 | 0.023518 | **547.9** | 0.024172 | 533.0 | **1.03×** | 1.022× |
| 768 | 24 | 2 | 128x256 | 1 | 0.005238 | **14.4** | 0.004804 | 15.7 | **0.92×** | 1.055× |
| 768 | 32 | 1 | 128x256 | 1 | 0.005202 | **19.4** | 0.004862 | 20.7 | **0.93×** | 1.060× |
| 768 | 64 | 1 | 128x256 | 1 | 0.005188 | **38.8** | 0.004788 | 42.0 | **0.92×** | 1.062× |
| 768 | 92 | 1 | 128x256 | 1 | 0.005204 | **55.6** | 0.004930 | 58.7 | **0.95×** | 1.060× |
| 768 | 96 | 1 | 128x256 | 1 | 0.005200 | **58.1** | 0.004830 | 62.5 | **0.93×** | 1.059× |
| 768 | 104 | 1 | 128x256 | 1 | 0.005198 | **62.9** | 0.004838 | 67.6 | **0.93×** | 1.059× |
| 768 | 152 | 1 | 128x256 | 1 | 0.005198 | **92.0** | 0.004836 | 98.9 | **0.93×** | 1.059× |
| 768 | 240 | 1 | 128x256 | 1 | 0.006492 | **116.3** | 0.004908 | 153.8 | **0.76×** | 1.091× |
| 768 | 520 | 1 | 128x256 | 1 | 0.007654 | **213.7** | 0.005898 | 277.3 | **0.77×** | 1.075× |
| 768 | 1840 | 1 | 128x256 | 1 | 0.016166 | **358.0** | 0.008860 | 653.3 | **0.55×** | 1.036× |
| 1024 | 1024 | 10 | 128x256 | 1 | 0.011444 | **375.3** | 0.006670 | 643.9 | **0.58×** | 1.052× |
| 1024 | 3490 | 1 | 128x256 | 2 | 0.026714 | **548.0** | 0.016516 | 886.3 | **0.62×** | 1.019× |
| 1024 | 12288 | 20 | 128x256 | 4 | 0.045550 | **1131.5** | 0.047398 | 1087.4 | **1.04×** | 1.000× |
| 1024 | 16128 | 9 | 128x256 | 4 | 0.053524 | **1263.8** | 0.059578 | 1135.4 | **1.11×** | 0.984× |
| 1024 | 105728 | 1 | 128x256 | 4 | 0.317142 | **1398.3** | 0.486003 | 912.5 | **1.53×** | 1.001× |
| 2048 | 3490 | 1 | 128x256 | 2 | 0.035942 | **814.5** | 0.020896 | 1401.0 | **0.58×** | 1.001× |
| 3490 | 1024 | 1 | 256x256 | 1 | 0.019642 | 745.2 ‡ | 0.016278 | 899.2 | — ‡ | — ‡ |
| 4096 | 4096 | 20 | 128x256 | 1 | 0.041716 | **1647.3** | 0.048385 | 1464.6 | **1.12×** | 1.019× |
| 6144 | 512 | 8 | 128x384 | 1 | 0.018736 | **687.7** | 0.028225 | 470.8 | **1.46×** | 0.896× |
| 6144 | 4096 | 20 | 128x384 | 1 | 0.050602 | **2037.1** | 0.083084 | 1279.4 | **1.59×** | 1.148× |
| 6144 | 16128 | 9 | 128x384 | 1 | 0.198743 | **2042.2** | 0.252273 | 1659.2 | **1.23×** | 1.047× |
| 6144 | 105728 | 1 | 128x384 | 1 | 1.312899 | **2026.6** | 1.875560 | 1463.0 | **1.39×** | 1.443× |
| 12672 | 1024 | 9 | 128x384 | 1 | 0.050774 | **1046.8** | 0.051960 | 1022.9 | **1.02×** | 1.061× |
| 16128 | 1024 | 9 | 256x256 | 1 | 0.047862 | **1413.3** | 0.067595 | 1032.0 | **1.37×** | 1.154× |
| 16384 | 1024 | 10 | 256x256 | 1 | 0.048384 | **1420.3** | 0.068504 | 1034.5 | **1.37×** | 1.117× |
| 20480 | 6144 | 9 | 256x256 | 1 | 0.264430 | **1949.1** | 0.272072 | 1953.5 | **1.00×** | 1.047× |
| 102272 | 1024 | 1 | 256x256 | 1 | 0.329811 | **1300.6** | 0.472239 | 908.4 | **1.43×** | 1.290× |
| 105728 | 1024 | 1 | 256x256 | 1 | 0.332689 | **1332.9** | 0.490322 | 932.7 | **1.43×** | 1.141× |

No row is same-session across `ours` and `ck`: the `ck` column has not been re-run since 2026-08-11
and comes from a different host. The `vs main` column is same-session by construction.

† = **CK carried over from the 2026-06 harness**, so the ratio is *indicative*. The CK binary on
this node (`tile_example_mx_flatmm`) exposes a single untuned kernel config, and at N ≤ 1024 it
lands at **0.33–0.54×** of the tuned 2026-06 numbers — publishing it here would overstate our
margin by 2–3×, so the tuned figure is kept. CK also refuses `K % 256 != 0` outright (`KPerBlock`),
which rules it out on most of the N=768 rows.

‡ = **the harness cannot produce a valid number for this shape**, so its row is not a measurement of
this library and its ratios are blanked.

- `N=1` — not a multiple of 32, so the MX preprocessing has no valid tiling. On `main` the grid is
  `dim3(M/256, 0)`, i.e. zero workgroups, and the old row times an empty dispatch; on this branch the
  ceil grid launches and the shape **faults** (`GPU coredump`). `ours_ms` is left at the old value.
- `N=3490` — not a multiple of 128, so no implemented tile divides it and `choose_tile()` falls
  through to its last-resort `256x256`. Measured `wg=104` = `8 x floor(3490/256)`, i.e. **3328 of
  3490 columns computed, 162 never written**, on *both* arms. The latency is real but it is the
  latency of 95% of the problem, charged against 100% of the FLOPs. Pre-existing; this branch neither
  causes nor fixes it.

**Three of the five shapes the old harness could not run now do**: `N=128` and `N=102272` route to
256x256 over a ceil grid with the last N-tile masked, and `N=12672` (a multiple of 384) routes to
128x384. All three carry real numbers above — `N=102272` gains 1.29x from it, `N=128` loses 2x
(see the regression note under the table).

Split-K rows are identified by `S > 1` in the table (`S` = number of K segments); their `ours_ms`
includes the FP32 partial-sum reduce.

### What the `vs main` column says (2026-08-18, 50 shapes, interleaved)

48 rows carry a valid comparison: **42 disjoint wins, 0 disjoint losses, 6 overlapping (noise)**.
The other 2 rows are the ‡ shapes, which the harness cannot measure at all. "Disjoint" means the two
arms' min/max ranges do not overlap; overlapping differences are reported as noise regardless of how
the medians fell.

The first pass of this sweep found **10 disjoint losses, all of them `N=128`**, traced to the
coverage-floor ordering and fixed before these numbers were taken — see below.

**Wins.** Two route changes carry the large ones: `N=6144 K=105728` at **1.44×** (128x128 -> 128x384,
the removed `LARGEK_THRESH` gate) and `N=102272 K=1024` at **1.29×** (128x128 -> 256x256 + masked
last N-tile). Below those, `N=16128 K=1024` 1.15×, `N=6144 K=4096` 1.15×, `N=105728 K=1024` 1.14×,
`N=16384 K=1024` 1.12×, `N=12672 K=1024` 1.06×. A broad **1.02–1.09×** covers most of the small-N 128x256 rows, consistent
with the fp16 packed-convert epilogue and the one-shot accumulator init, both of which pay off most
where the epilogue is a large share of a short kernel.

**The `N=128` regression, found and fixed in this branch.** `main`'s floor ends at
`(M%128)==0 && (N%128)==0 -> 128x128`; `ba01d19` inserted `(M%256)==0 && (N%128)==0 -> 256x256`
above it, so `N=128` ran a 256-wide tile over a `ceil(N/256)=1` grid and masked half of it — 256
columns computed for a 128-column problem, measured **0.49–0.54×** across all ten `N=128` rows.
The same rule wins 1.29× on `N=102272`, where it wastes 0.1%.

Any N reaching that arm is `N%256==128`, so the overshoot is always one 128-column half-tile and the
waste is `128/N`; `N=128` is the only N with no full N-tile to amortize it. Fixed by adding
`N >= 256` to the arm. Enumerated over M 128..8192 x N 128..131072 x two Kp: coverage unchanged
(0 uncovered before and after) and `N=128` is the only N that re-routes. Those rows now measure
**1.03–1.06× against `main`**, and `N=102272` still measures 1.29×. Three `check_routing()` cases
pin it; removing the guard takes routing from 33/33 to 30/33.

**`N=1 K=512` faults on this branch** (`GPU coredump`) where `main` returns without computing
anything (zero-width grid). `N=1` is outside the contract — `preprocess.hpp` requires N%32==0 — so
neither behaviour is correct. Row left at its old numbers and marked ‡.

**`N=6144 K=512` is bimodal on this branch and unresolved.** Over 12 runs its median is 0.92× but its
ranges overlap `main`'s, because 3 runs land at 0.0143–0.0157 ms (faster than `main`'s best of
0.0153) and 9 land at 0.0176–0.0203. `main` is tight at 0.0153–0.0170 throughout. Both arms route to
128x384, which is one of the two routes carrying the XCD grid swizzle. Flagged, not diagnosed.

### Same-machine CK, 2026-08-18 (`g4-1`) — 18 of 50 shapes

The table above pairs `ours` from `g4-1` with `ck` from `k2-3a`. This section removes that mismatch
for the shapes where it can. Raw:
[`prof_results/ck_untuned_g4-1_2026-08-18.tsv`](prof_results/ck_untuned_g4-1_2026-08-18.tsv).

Two limits, both measured rather than assumed:

**CK ran 18 of the 50 shapes.** It rejects any shape with `N % 256 != 0` or `K % 256 != 0` outright,
which removes every `N=128` and `N=768` row, all six short-K `N=256` rows, both `K=3490` rows, and
`N=1 / 3490 / 12672 / 102272`.

**The binary available to us is the untuned single-config build**, and this run confirms the `†`
footnote's number rather than relying on it: against the tuned 2026-06 figures it lands at
**0.33–0.54× for N ≤ 1024** — the footnote says 0.33–0.54× — and at **0.82–0.98× for N ≥ 4096**.
So the small-N ratios below are inflated by roughly 2–3× and are shown only to document that; the
`N ≥ 4096` block is the part worth reading, where the untuned penalty is comparable to the ~10%
this host is slower overall.

| N | K | ours_ms | ours_TF | ck_ms | ck_TF | ours/CK |
|---:|---:|---:|---:|---:|---:|:---:|
| 4096 | 4096 | 0.041716 | **1647.3** | 0.056032 | 1264.8 | **1.30×** |
| 6144 | 512 | 0.018736 | **687.7** | 0.029423 | 451.6 | **1.52×** |
| 6144 | 4096 | 0.050602 | **2037.1** | 0.093033 | 1142.6 | **1.78×** |
| 6144 | 16128 | 0.198743 | **2042.2** | 0.282605 | 1481.1 | **1.38×** |
| 6144 | 105728 | 1.312899 | **2026.6** | 1.929890 | 1421.8 | **1.43×** |
| 16128 | 1024 | 0.047862 | **1413.3** | 0.076966 | 906.4 | **1.56×** |
| 16384 | 1024 | 0.048384 | **1420.3** | 0.079608 | 890.2 | **1.60×** |
| 20480 | 6144 | 0.264430 | **1949.1** | 0.333614 | 1593.2 | **1.22×** |
| 105728 | 1024 | 0.332689 | **1332.9** | 0.497931 | 918.4 | **1.45×** |

**Same host, same session: 1.22–1.78×, median 1.45×** across those 9 shapes. The mixed-host ratios
in the main table give 1.36× median on the same 9.

Small-N rows, **inflated — do not quote**:

| N | K | ours_TF | ck_TF (untuned) | ratio |
|---:|---:|---:|---:|:---:|
| 256 | 256 | **43.2** | 20.0 | 2.17× |
| 256 | 512 | **72.7** | 36.6 | 1.99× |
| 256 | 2048 | **126.3** | 93.3 | 1.35× |
| 512 | 2560 | **258.9** | 206.0 | 1.26× |
| 512 | 6144 | **548.2** | 232.5 | 2.36× |
| 1024 | 1024 | **374.8** | 231.8 | 1.62× |
| 1024 | 12288 | **1145.1** | 533.7 | 2.15× |
| 1024 | 16128 | **1270.4** | 545.3 | 2.33× |
| 1024 | 105728 | **1398.4** | 492.4 | 2.84× |

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

> The `2048×6144×16128` control row is scoped to this 2026-06-30 split-K experiment and predates the
> 128×384 route: it shows only that split-K left that shape alone. The shape now routes to 128×384
> and runs at ~2700 TFLOPs — see the main table and the wide-N A/B below.

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
| 2048×6144×16128 | 1 | 1582 | control unchanged (pre-128×384; that shape now routes to 128×384) |

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

The main table gives absolutes; this is the A/B that attributes them to the route change. The
`128×256` baseline and the `128×384` build were compiled from the same source pair in one container
invocation and run **interleaved, 3 reps**, with CK in the same session. M=2048, N=6144, FP16, on
`bg-1w300-k2-3a`. Raw:
[`prof_results/bench_mi350x_k2-3a_2026-07-29.txt`](prof_results/bench_mi350x_k2-3a_2026-07-29.txt).

| K | baseline (128×256) | **128×384** | Δ | CK mxfp8 | baseline/CK | **new/CK** |
|---:|---:|---:|:---:|---:|:---:|:---:|
| 512    | 746  | **966**  | +29.5% | 473  | 1.58× | **2.04×** |
| 4096   | 1427 | **2131** | +49.3% | 1290 | 1.11× | **1.65×** |
| 16128  | 1702 | **2707** | +59.0% | 1654 | 1.03× | **1.64×** |

`base/CK` at K=16128 = 1.03× reproduces the historical 1.02× for this shape, which is the check
that the baseline is sound. K=105728 was a control in this run and is omitted here: the 128×384
route was still K-gated then, so the shape kept 128×128 — a gate later shown backwards (item 5
below).

The gain compounds three effects: wave quantization removed (384 WGs / 1.5 waves → 256 WGs /
1 wave), accumulators per wave 8 → 12, and compute per B ring slot 64 → 128 cyc so the PFD=5
prefetch ring covers 640 cyc instead of 320. Effects 2 and 3 are why the shallower K gained
proportionally more than the wave-quantization argument alone predicts.

---

## Remaining problems / gaps

1. **N-remainder shapes** — mostly resolved. `N=128`, `12672`, `102272` now run correctly (masked
   last N-tile / 128×384; see ‡ above), and their table rows are the only stale ones left.
   `N=1` and `N=3490` are still unsupported, but the blocker is the MX block size — they are not
   multiples of 32, so the preprocessing rejects them. Sub-128 N-remainder and any M-remainder
   remain unimplemented.
2. **Borderline shapes gated OUT by `k_tiles ≥ 16`** — CU-starved shapes that just miss the K
   threshold get no split and lose hard to CK: `2048×1024×1024` (k_tiles=6, 0.56×),
   `2048×768×1840` (10, 0.53×), `2048×256×2048` (11, 0.48×), `2048×512×2560` (14, 0.46×).
3. **Mid-N moderate-K is the weakest band vs CK** — `256×2048` 0.48×, `512×2560` 0.46×,
   `2048×3490` 0.61×. (`512×6144` was in this list; the uneven split-K now takes it to 1.00×.)
4. **Small N + small K is overhead-bound** — N≤256 with small K sit at ~40 kernel TFLOPs and a
   ~5 µs latency floor. Fixed launch/prologue cost dominates; split-K cannot help.
5. ~~**Wide-N very-large-K L2 sag**~~ — **RESOLVED.** `2048×6144×105728` was 1288 TF (0.90× CK).
   PR #4 routed it to a **128×128 tile** → MI350X **1.20× CK**, MI355X **1.19× CK**. The stated
   reason (halving the B slice to fit L2) was **wrong**: 10.2 MB does not fit a 4 MB per-XCD L2
   either, and the win was the dropped half-empty CU wave. Removing the K-gate on the wave-saving
   128×384 arm then took the same shape to **2557 TF (+44.7%)** with a 3× *larger* B slice.
   See `docs/OPTIMIZATIONS.md §9` and
   [`prof_results/bench_mi350x_k2-3a_2026-08-11.txt`](prof_results/bench_mi350x_k2-3a_2026-08-11.txt).
6. **Split-K's own ceiling** — the FP32 partial-sum write + reduce round-trip caps speedup well below the ideal `S×` (2.99× of an ideal 4× for 105728).
7. ~~**Wide-N wave quantization**~~ — **RESOLVED.** `2048×6144×{512,4096,16128}` ran 384 WGs on
   256 CUs (1.5 waves). Routing `N%384==0` shapes to a **128×384 tile** gives exactly 256 WGs;
   +29%/+49%/+59% and 1.64–2.04× CK. The gate was later found to apply at large K too (item 5), so
   every N=6144 shape now takes this route.
8. **`2048×6144×512` is the one open regression risk on this branch.** The 50-shape A/B reports it
   as overlapping rather than a loss, but the margin is **0.22%**: `main` spans 0.015986–0.017128 ms
   and this branch 0.017090–0.019114, so the ranges touch by 38 ns. Best-against-best is **0.935×**,
   and over 12 runs the median is 0.92× with 3 runs beating `main`'s best and 9 clearly behind.
   Calling it noise is a verdict the disjointness test barely licensed, not one the data supports.
   Both arms route 128×384, one of the two routes carrying the XCD swizzle. Diagnosing it needs an
   ATT capture, since the bimodality is the whole signal. **Not started.**
9. **`N=1` faults on this branch** where `main` launched a zero-width grid and returned without
   computing. `N=1` is outside the contract (`preprocess.hpp` requires `N%32==0`) so neither is
   correct, but a GPU coredump is a worse failure mode than a silent no-op. A cheap fix is an early
   return with a stderr warning for `N%32!=0`, matching what `gemm()` already does when the split-K
   workspace is too small. **Not started, deliberately deferred.**
