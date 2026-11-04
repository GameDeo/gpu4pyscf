source /content/env.sh
B="python benchmarks/dft/bench_wb97mv.py --warmup --basis def2-tzvp --verbose 3 --mol 033_Bisphenol_A --mixed auto --scheme crt --grad --ref /content/res/bpa_tight.json"
$B --out /content/res/s36.json > /content/res/s36.log 2>&1
echo DONE > /content/res/done36
