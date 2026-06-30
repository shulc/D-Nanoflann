// D-Nanoflann — extern "C" shim around jlblancoc/nanoflann.
//
// nanoflann is a header-only C++ template library; its types cannot cross
// a C/D boundary directly.  This thin shim instantiates one concrete
// KDTreeSingleIndexAdaptor over a float[3] point cloud, hides all C++
// behind an opaque handle, and exposes a plain-C API so D can bind via
// extern(C) with no C++ dependency on the D side.
//
// All entry points are @nogc / nothrow-safe from the D perspective: they
// allocate internally but never throw (exceptions are caught and converted
// to NULL / -1 returns).
//
// Distance convention: knn and radius output squared L2 distances to match
// nanoflann's internal representation.  The `radius` parameter of
// nfc_radius() is the GEOMETRIC radius (NOT squared); the shim squares it
// before passing to nanoflann.

#ifndef NANOFLANN_C_H
#define NANOFLANN_C_H

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle to a built KD-tree.  Create via nfc_build, free via
/// nfc_free.  All query functions tolerate a NULL handle (returning 0 /
/// -1 as appropriate).
typedef struct nfc_kdtree nfc_kdtree_t;

/// Build a KD-tree from `num_points` 3-D float points.
/// `points_xyz` is a tightly-packed array of length 3 * num_points
/// (x0,y0,z0, x1,y1,z1, ...).  The function makes an internal copy —
/// the caller may free `points_xyz` immediately after this returns.
/// Returns NULL on out-of-memory or invalid arguments (num_points <= 0).
nfc_kdtree_t* nfc_build(const float* points_xyz, int num_points);

/// Find the `k` nearest neighbours of query point `q[3]`.
/// `out_idx`   — caller-allocated, length >= k; filled with 0-based point
///               indices in ascending distance order.
/// `out_dist2` — caller-allocated, length >= k; filled with squared L2
///               distances corresponding to out_idx.
/// Returns the number of neighbours actually found (may be < k when the
/// cloud has fewer than k points).  Returns -1 on NULL handle or k <= 0.
int nfc_knn(const nfc_kdtree_t* tree,
            const float         q[3],
            int                 k,
            unsigned int*       out_idx,
            float*              out_dist2);

/// Find all points within geometric radius `radius` of query point `q[3]`.
/// Results are written into the caller-provided arrays (both of length
/// `max_out`); at most `max_out` results are returned even if more exist.
/// `out_idx`   — 0-based point indices of matching points.
/// `out_dist2` — squared L2 distances of matching points.
/// Returns the number of results written (0..max_out), or -1 on NULL
/// handle / invalid arguments.  Results are NOT guaranteed to be sorted.
int nfc_radius(const nfc_kdtree_t* tree,
               const float         q[3],
               float               radius,
               unsigned int*       out_idx,
               float*              out_dist2,
               int                 max_out);

/// Destroy the KD-tree and free all associated memory.  Safe to call with
/// NULL (no-op).
void nfc_free(nfc_kdtree_t* tree);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // NANOFLANN_C_H
