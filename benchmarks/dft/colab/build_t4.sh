set -x
mkdir -p /content/src && cd /content/src && rm -rf gpu4pyscf && tar xzf /content/repo.tgz
pip install -q pyscf==2.14.0 gpu4pyscf-libxc-cuda12x==0.9.0 cutensor-cu13 basis-set-exchange pyscf-dispersion geometric 2>&1 | tail -3
cd /content && rm -rf build && mkdir build && cd build
cmake /content/src/gpu4pyscf/gpu4pyscf/lib -DCUDA_ARCHITECTURES=75-real -DBUILD_LIBXC=OFF 2>&1 | tail -5
make -j4 gdft cupy_helper gvhf gecp gvhf_rys gvhf_rys_f32 gvhf_md gint solvent sem pbc mgrid_v2 mgrid_v3 2>&1 | tail -5
echo BUILD_DONE
