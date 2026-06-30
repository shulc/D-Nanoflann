# D-Nanoflann

Static D bindings for [jlblancoc/nanoflann](https://github.com/jlblancoc/nanoflann) —
a header-only C++ KD-tree library for fast nearest-point queries on 3-D float
point clouds.

Because nanoflann is a C++ template library, a thin `extern "C"` shim
(`csrc/nanoflann_c.cpp`) bridges the template instantiation to a plain-C API
that D can bind without any C++ dependency on the D side.

Upstream pinned: **v1.10.0** (`57d2d8c`).

## Layout

```
include/nanoflann_c.h      extern "C" API — opaque handle + build/knn/radius/free
csrc/nanoflann_c.cpp       shim implementation (PointCloud adaptor + KDTreeSingleIndexAdaptor)
CMakeLists.txt             builds static lib `nanoflann_c` (PIC, C++14, no shared)
source/nanoflann/c.d       D bindings — extern(C) mirror of the header
examples/nearest.d         build + correctness demo: KD-tree vs brute-force
extern/nanoflann/          git submodule, pinned to v1.10.0
dub.json                   dub package (library + nearest executable configs)
```

## First build

```bash
git clone --recurse-submodules <this-repo> D-Nanoflann
cd D-Nanoflann

# Build the C shim (also triggered automatically by dub preBuildCommands):
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target nanoflann_c

# Build and run the nearest-point example:
dub run --config=nearest
```

The `preBuildCommands` in `dub.json` run the cmake steps automatically, so a
plain `dub build` or `dub run --config=nearest` on a fresh clone is
self-contained.

## API

```d
import nanoflann.c;

// Build from tightly-packed float[3*N] array (library copies the data).
nfc_kdtree_t* tree = nfc_build(points_xyz.ptr, cast(int)N);
scope(exit) nfc_free(tree);

// k nearest neighbours — out_dist2 contains squared L2 distances.
uint[K]  idx;
float[K] dist2;
int found = nfc_knn(tree, query.ptr, K, idx.ptr, dist2.ptr);

// Radius search — radius is geometric (NOT squared); dist2 output IS squared.
uint[MAX]  ridx;
float[MAX] rdist2;
int cnt = nfc_radius(tree, query.ptr, radius, ridx.ptr, rdist2.ptr, MAX);
```

## License

BSD 2-Clause — see `LICENSE`.  nanoflann is also BSD 2-Clause (attribution in
`LICENSE`).
