/// D bindings for the D-Nanoflann C shim.
///
/// Mirrors include/nanoflann_c.h exactly — one opaque struct per handle,
/// one extern(C) declaration per entry point.  No D-side wrapping; higher-
/// level ergonomic layers (RAII handles, slice-based queries, range
/// iteration) belong in sibling modules that import this one.
///
/// Distance convention: knn and radius output are SQUARED L2 distances,
/// matching nanoflann's internal L2_Simple metric.
module nanoflann.c;

extern (C) @nogc nothrow:

/// Opaque handle to a built KD-tree.  Create via nfc_build, free via
/// nfc_free.  Passing null to query functions returns 0 / -1 gracefully.
struct nfc_kdtree_t;

/// Build a KD-tree from `num_points` 3-D float points.
/// `points_xyz` must be a tightly-packed array of length 3 * num_points
/// (x0,y0,z0, x1,y1,z1, ...).  The library makes an internal copy so the
/// caller may free the buffer immediately after return.
/// Returns null on out-of-memory or num_points <= 0.
nfc_kdtree_t* nfc_build(const(float)* points_xyz, int num_points);

/// k nearest neighbours of query point `q[3]`.
/// `out_idx`   — 0-based point indices, ascending distance, length >= k.
/// `out_dist2` — squared L2 distances corresponding to out_idx, length >= k.
/// Returns number of neighbours found (may be < k for small clouds).
/// Returns -1 on null handle or k <= 0.
int nfc_knn(const(nfc_kdtree_t)* tree,
            const(float)*         q,       // q[3]
            int                   k,
            uint*                 out_idx,
            float*                out_dist2);

/// All points within geometric radius `radius` of query point `q[3]`.
/// Results land in caller-allocated arrays of size `max_out`.
/// `out_idx`   — 0-based point indices of matching points.
/// `out_dist2` — squared L2 distances of matching points.
/// Returns the number of results written (0..max_out) or -1 on error.
/// Results are NOT sorted by distance.
int nfc_radius(const(nfc_kdtree_t)* tree,
               const(float)*         q,       // q[3]
               float                 radius,
               uint*                 out_idx,
               float*                out_dist2,
               int                   max_out);

/// Free the KD-tree.  Safe to call with null (no-op).
void nfc_free(nfc_kdtree_t* tree);
