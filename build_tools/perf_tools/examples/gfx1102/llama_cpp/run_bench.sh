#!/usr/bin/env bash
B=~/rocm-env/llama-bench
declare -A SPD=( [rocm7]=~/rocm-env/strata-bench/rocm7/lib/python3.12/site-packages [rocm101]=~/rocm-env/.venv/lib/python3.12/site-packages )
mkdir -p $B/out
for round in 1 2 3; do
  if [ $((round % 2)) -eq 1 ]; then order="rocm7 rocm101"; else order="rocm101 rocm7"; fi
  for eng in $order; do
    SP=${SPD[$eng]}; LIBS=$(ls -d $SP/_rocm_sdk_libraries* | head -1)
    for m in Qwen2.5-3B-Instruct-Q4_K_M Qwen2.5-3B-Instruct-Q8_0; do
      LD_LIBRARY_PATH=$SP/_rocm_sdk_devel/lib:$LIBS/lib $B/build/$eng/bin/llama-bench -m $B/models/$m.gguf -ngl 99 -p 512,2048 -n 128 -fa 0,1 -r 5 -o json > $B/out/${eng}_${m}_r${round}.json 2> $B/out/${eng}_${m}_r${round}.err
      echo "$eng $m round $round rc=$?"
    done
  done
done
echo ALL_DONE
