cd /home/user/gpu4pyscf/benchmarks/dft/research/experiments_r9/agent
BPA=/home/user/gpu4pyscf/benchmarks/molecules/organic/033_Bisphenol_A.xyz
timeout 1200 python a1_incremental_x.py $BPA 0:fp64,0:fp32 > logs/a1_bpa.log 2>&1
timeout 1200 python a2_local_sparsity.py $BPA > logs/a2_bpa.log 2>&1
KSURR=def2-universal-jfit timeout 1000 python a1_incremental_x.py phenol.xyz 0:fp64 > logs/a1_phenol_weak.log 2>&1
NOUTER=3 timeout 1000 python a3_direct_raw.py phenol.xyz fp64:fp32,fp64:fp16 > logs/a3_phenol_b.log 2>&1
echo chain done
