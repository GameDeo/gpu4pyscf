set -x
mkdir -p /content/src80 && cd /content/src80 && rm -rf gpu4pyscf && tar xzf /content/repo2.tgz
cd /content && rm -rf build80 && mkdir build80 && cd build80
cmake /content/src80/gpu4pyscf/gpu4pyscf/lib -DCUDA_ARCHITECTURES=80-real -DBUILD_LIBXC=OFF 2>&1 | tail -3
make -j3 gdft cupy_helper gvhf gecp gvhf_rys gvhf_md gint solvent sem pbc mgrid_v2 mgrid_v3 2>&1 | tail -3
cd /content/src80 && tar czf /content/libs80.tgz $(find gpu4pyscf/gpu4pyscf/lib -name '*.so')
echo BUILD80_DONE
