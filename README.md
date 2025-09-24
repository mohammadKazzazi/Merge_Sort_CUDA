# Merge_Sort_CUDA

Parallel Merge Sort algorithm using CUDA.


Compile: 
```console
nvcc -O2 merge_main.cu merge.cu -o merge
```
Execute: 
```console
./merge M
```

Note that $N=2^M$ and $M = 18$, $19$, $20$ and $21$.
