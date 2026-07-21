# mxfp6-gemm — K=105728 regression investigation & fix (experiment log)

**Date:** 2026-07-16 → 07-17 · **Author:** Claude (subagent team) · **Repo:** `~/workspace/mxfp6-gemm` @ `main` (2744234)

## Objective
Fix the throughput regression at `2048×6144×K` for very large K (K=105728), where libmxfp6gemm
(mxfp6) fell behind CK (mxfp8): 0.87× CK on MI355X. Root cause (case-005): B's per-N-slice working
set (256 cols × K × 0.75 = 20.3 MB) overflows L2 → `buffer_load` stall spikes to ~2538 cyc (HBM),
and the PFD=5 register ring hides only ~250 cyc.

## Environment
- **Build:** container `ck_arliu` (ROCm 7.0.2) on the WSL host; `cmake -DCMAKE_HIP_ARCHITECTURES=gfx950`.
- **Run:** always inside a ROCm **7.0.2** container on the GPU node (`rocm/pytorch:rocm7.0.2_ubuntu24.04_py3.13_pytorch_release_2.9.1`), GPU pinned via `HIP_VISIBLE_DEVICES=0`. Never on bare metal.
- **Test machines:**
  - **MI355X** = `smci355-ccs-aus-m06-05.cs-aus.dcgpu` (the target-class machine; has the CK reference numbers).
  - **MI350X-A** = `10.7.189.38` (`bg-1w300-h3-2a`) — first dev node.
  - **MI350X-B** = `10.7.191.60` (`bg-1w300-h3-3`) — confirmation retest.
- All shapes below are **2048×6144×K**, FP16 out. `perf()` bench (steady state); numbers drift run-to-run ~±3%, so `=` marks a byte-identical (unchanged) code path.
- **Hardware fact corrected this session:** gfx950 LDS = **160 KB/CU** (rocminfo GROUP segment), not 64 KB.

---

## Results by machine (TFLOPS)

### MI355X (`smci355-ccs-aus-m06-05`) — target machine
| Track | K=512 | K=4096 | K=16128 | K=105728 | Δ @105728 |
|---|---|---|---|---|---|
| Baseline | 767 | 1479 | 1799 | 1323 | — |
| T1 PFD K-gate | = | = | = | 1361 | **+2.9%** |
| T2 asm BXPRE | — | — | — | (not viable, no build) | — |
| T3 128×128 | 763 | 1486 | 1797 | 1785 | **+34.9%** |
| T4 B-via-LDS (drip) | = | = | = | 1215 | **−8.2%** ✗ |
| **T5 integrated (shipped)** | 762 | 1480 | 1797 | **1788** | **+35.1%** |
| *CK mxfp8 (ref)* | 505 | 1371 | 1730 | 1523 | ours 0.87×→**1.17×** |

### MI350X-A (`10.7.189.38`)
| Track | K=512 | K=4096 | K=16128 | K=105728 | Δ @105728 |
|---|---|---|---|---|---|
| Baseline | 717 | 1284 | 1527 | 1262 | — |
| T1 PFD K-gate | = | = | 1529 | 1295 | **+2.6%** |
| T3 128×128 | 719 | 1251 | 1529 | 1472 | **+16.6%** |
| T4 B-via-LDS (v1 burst) | = | = | = | 1197 | **−5.2%** ✗ |

### MI350X-B (`10.7.191.60`) — confirmation retest (2026-07-17)
| Track | K=512 | K=4096 | K=16128 | K=105728 (×3 avg) | Δ @105728 |
|---|---|---|---|---|---|
| Baseline | 714 | 1355 | 1579 | 1256 (1252/1256/1259) | — |
| T1 PFD K-gate | 717 | 1313 | 1589 | 1295 | **+3.1%** |
| T3 128×128 | 715 | 1286 | 1586 | 1506 (1507/1505/1506) | **+19.9%** |
| T4 B-via-LDS (drip) | = | = | = | 1160 | **−7.6%** ✗ |
| **T5 integrated (shipped)** | 716 | 1335 | 1581 | 1507 (1504/1508/1508) | **+20.0%** |

Correctness (T5 final, MI350X-B): full no-arg suite **all OK**, incl. direct `128x128 256x256x768` and `384x512x1536` (er=0.0000).

---

## vs CK mxfp8 (same-precision-workload reference)

CK = `tile_example_mx_flatmm -mx_prec=fp8xfp8 -v=0 -warmup=20 -repeat=50`, `2048×6144×K`, FP16.
The K=105728 win is delivered entirely by **T3 (128×128 tile)** — that shape routes to `128x128`
(`wg256<CU && Kp≥32768 && wg128≥CU`); the T1 PFD-gate does not touch it.

### ★ AUTHORITATIVE: MI355X reference node `smci355-ccs-aus-n03-05` (= 10.235.26.75), same session, 2026-07-20
Baseline (1318) and CK (1523) reproduce the original case-005 reference **exactly** → clean apples-to-apples.
Raw: `prof_results/bench_n03-05_2026-07-20.txt`. Correctness: full suite all OK (incl. direct 128×128 tests).
| K | ours baseline | ours integrated (128×128) | CK mxfp8 | after/CK |
|---|---:|---:|---:|:---:|
| 512 | 772 | 774 | 505 | **1.53×** |
| 4096 | 1509 | 1532 | 1397 | **1.10×** |
| 16128 | 1771 | 1770 | 1780 | 0.99× (tied; unchanged 128×256) |
| **105728** | **1318** | **1811** | **1523** | **1.19×** (was 0.87×; **+37.4%** vs baseline) |

### MI350X-B (`10.7.191.60`) — ours + CK measured in the SAME GPU session (2026-07-17)
| K | CK mxfp8 | ours before | ours after (T5) | before/CK | after/CK |
|---|---:|---:|---:|:---:|:---:|
| 512 | 467 | 716 | 715 | 1.53× | **1.53×** |
| 4096 | 1195 | 1283 | 1332 | 1.07× | **1.11×** |
| 16128 | 1550 | 1588 | 1581 | 1.02× | **1.02×** |
| **105728** | **1413** | **1260** | **1509** | **0.89× ✗** | **1.07× ✓** |

### MI355X (`smci355-…-m06-05`) — CK from the 2026-07-15/16 reference run
| K | CK mxfp8 | ours before | ours after (T5) | before/CK | after/CK |
|---|---:|---:|---:|:---:|:---:|
| 512 | 505 | 767 | 762 | 1.52× | **1.51×** |
| 4096 | 1371 | 1479 | 1480 | 1.08× | **1.08×** |
| 16128 | 1730 | 1799 | 1797 | 1.04× | **1.04×** |
| **105728** | **1523** | **1323** | **1788** | **0.87× ✗** | **1.17× ✓** |

**Bottom line:** the fix turns K=105728 from a loss into a win on both GPUs — MI350X **0.89×→1.07×**,
MI355X **0.87×→1.17×** — and every other shape stays ≥ CK (1.02×–1.53×). (MI350X-A had no CK run.)

---

## Per-track summary

> The `track*.patch` files and `track4-blds-newfiles/` referenced below are **not committed to this
> repo** — the unshipped/duplicate experiment code was dropped from the PR to keep the tree clean and
> is preserved in a local research archive only. The shipped code lives in the normal source tree
> (`src/`, `include/`).

**T1 — PFD ring-depth K-gate.** Deeper B-ring (PFD=5→7, spill-free) covers more HBM-miss latency
at very large K but costs ~1-2% in the L2-hit regime → gated on `Kp>50000` for the 128×256 path.
+2.6–3.1% at K=105728, zero regression. Kept as a complement (mostly subsumed by T3). **Shipped.**
Code: in `track5-integrated-shipped.patch` (dispatch.hpp `if (Kp > 50000)`).

**T2 — asm_nowait cross-tile B prefetch (BXPRE).** NOT VIABLE (ISA-proven). Hardware VMCNT is a
single shared counter, so the `wait_vmcnt(0)` guarding the A double-buffer barrier necessarily
drains the cross-tile B loads too — asm-invisibility can't dodge it (earlier build measured −25%).
No code shipped. Reference code (BXPRE template path + argv): `track2-asm-bxpre.patch`.

**T3 — 128×128 N-subtiling (WINNER).** Route large-K wide-N (`wg256<CU && Kp≥32768 && wg128≥CU`)
to a 128×128 tile → halves B/N-slice per WG (20.3→10.2 MB) → restores L2 residency. Reuses the whole
tile-general kernel; only template args + a `choose_tile` branch + dispatch branch change. All gates
pass (G1 regression, G2 CPU-ref oracle, G3 negative-control confirmed, G4 er=0.0000, G6 VGPRs=100/
Scratch=0). **Shipped.** Code: `track3-128x128.patch` (also folded into `track5-integrated-shipped.patch`).

**T4 — B-via-LDS double-buffer.** Built and CORRECT (er=0.0000), but a net LOSS on every machine
(−5% to −8%). Staging B through an LDS double-buffer + dripped `issue_B_chunks` hides latency but
does NOT reduce HBM bytes; single-tile look-ahead (~384 cyc compute) can't cover 2538-cyc HBM
latency, and the added ds-reads roughly cancel the partial gain. The valuable NEGATIVE RESULT that
proves this shape is bandwidth/capacity-bound, not latency-bound → justifies T3. No code shipped.
Code: `track4-b-via-lds.patch` (dispatch/CMake/test edits) + `track4-blds-newfiles/` (the actual
kernel: `kernel_blds.hpp`, plus the untested 128×128 bLDS variant `dispatch_blds128.hpp` and
`test_gemm128.cpp`).

**T5 — Integration (shipped).** T3 + T1 merged into `main`, BXPRE dead code stripped (kernel.hpp
reverted to HEAD), `splitk_S(Kp)` reconciled. Validated on MI355X and MI350X-B. **Uncommitted** per
the project no-commit rule. Full diff (code + docs): `track5-integrated-shipped.patch`.

## Key lesson (→ wiki Pattern 020)
Latency-hiding is the tool when latency is the bottleneck; **working-set reduction is the tool when
bandwidth/cache-capacity is.** At large K on a wide tile the kernel is capacity-bound, so shrinking
B/N-slice (128×128) beats every attempt to hide its latency (LDS staging, cross-tile prefetch,
deeper ring). Diagnose `working_set vs L2` before reaching for LDS/prefetch.

---

## File manifest (this directory)
- `EXPERIMENT-LOG.md` — this file (the only committed artifact in this directory).

The per-track diffs (`track2-asm-bxpre.patch`, `track3-128x128.patch`, `track4-b-via-lds.patch`,
`track4-blds-newfiles/`, `track5-integrated-shipped.patch`) and the experiment worktrees are kept in
a **local research archive only** — they were removed from the PR to keep the tree clean, being
either unshipped dead-ends (T2/T4) or duplicates of the shipped source (T3/T5, now live in `src/`).

## Reproduce
```bash
# build (ck_arliu 7.0.2)
docker exec ck_arliu bash -lc 'cd /workspace/mxfp6-gemm && cmake -S . -B build -DCMAKE_HIP_ARCHITECTURES=gfx950 && cmake --build build -j'
# (the shipped 128x128 fix is already in the source tree — no patch to apply)
# run (ROCm 7.0.2 container on a gfx950 node, GPU pinned):
docker run --rm --privileged -e HIP_VISIBLE_DEVICES=0 -v $HOME:/home/arliu --device=/dev/kfd --device=/dev/dri \
  --ipc=host --group-add video rocm/pytorch:rocm7.0.2_ubuntu24.04_py3.13_pytorch_release_2.9.1 \
  bash -c '/home/arliu/test_gemm            # no-arg = correctness suite
           /home/arliu/test_gemm 2048 6144 105728   # single-shape perf'
```
