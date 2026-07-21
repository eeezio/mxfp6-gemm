#!/bin/bash
# Rebuild benchmark data on smci355-ccs-aus-n03-05 (MI355X, reference node).
docker run --rm --privileged -e HIP_VISIBLE_DEVICES=0 -v ${HOME}:/home/arliu \
  --device=/dev/kfd --device=/dev/dri --ipc=host --group-add video \
  rocm/pytorch:rocm7.0.2_ubuntu24.04_py3.13_pytorch_release_2.9.1 bash -c '
  CK=/home/arliu/prof_n6144/ck/tile_example_mx_flatmm
  M=2048; N=6144
  /home/arliu/test_gemm_main 2048 6144 16128 >/dev/null 2>&1   # warmup
  echo "===CORRECTNESS main==="; /home/arliu/test_gemm_main 2>/dev/null | grep -iE "128x128|FAILED|all OK"
  echo "===OURS baseline==="; for K in 512 4096 16128 105728; do /home/arliu/test_gemm_base 2048 6144 $K 2>/dev/null | tail -1; done
  echo "===OURS integrated (128x128 fix)==="; for K in 512 4096 16128 105728; do /home/arliu/test_gemm_main 2048 6144 $K 2>/dev/null | tail -1; done
  echo "===CK mxfp8==="; for K in 512 4096 16128 105728; do printf "ck %d %d %6d : " $M $N $K; $CK -m=$M -n=$N -k=$K -mx_prec=fp8xfp8 -v=0 -warmup=20 -repeat=50 2>&1 | grep -oiE "[0-9.]+ TFlops" | tail -1; done
'
