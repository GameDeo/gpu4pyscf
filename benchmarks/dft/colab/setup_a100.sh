nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p /content/src && cd /content/src && rm -rf gpu4pyscf && tar xzf /content/repo3.tgz && cd /content/src && tar xzf /content/libs80.tgz
ls gpu4pyscf/gpu4pyscf/lib/*.so | wc -l
pip install -q pyscf==2.14.0 gpu4pyscf-libxc-cuda12x==0.9.0 cutensor-cu13 basis-set-exchange pyscf-dispersion geometric 2>&1 | tail -2
mkdir -p /content/res; cp /content/env.sh /content/env.sh 2>/dev/null; echo SETUP_DONE
