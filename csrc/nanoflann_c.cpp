// D-Nanoflann — C shim implementation.
//
// Instantiates a concrete KDTreeSingleIndexAdaptor<L2_Simple_Adaptor<float>,
// ..., 3> over an owned copy of the caller's float[] point cloud.  All
// exceptions are caught here so the extern "C" surface is nothrow from
// the D side.

#include "nanoflann_c.h"
#include <nanoflann.hpp>

#include <cstring>    // memcpy
#include <cstdlib>    // malloc / free
#include <new>        // std::nothrow
#include <vector>
#include <algorithm>  // std::min
#include <limits>

// ---------------------------------------------------------------------------
// Point-cloud adaptor (nanoflann requirement).
// Owns a flat copy of the caller's float array.
// ---------------------------------------------------------------------------
struct FloatCloud3
{
    std::vector<float> pts;  // length = 3 * num_points, layout x,y,z,x,y,z,...
    int                num_points;

    FloatCloud3() : num_points(0) {}

    // nanoflann adaptor interface
    inline std::size_t kdtree_get_point_count() const
    {
        return static_cast<std::size_t>(num_points);
    }

    inline float kdtree_get_pt(const std::size_t idx, const std::size_t dim) const
    {
        return pts[3 * idx + dim];
    }

    // Optional bounding-box: return false to let nanoflann compute it.
    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const { return false; }
};

// ---------------------------------------------------------------------------
// Concrete KD-tree type (3-D, float, L2-simple metric, uint32 index).
// ---------------------------------------------------------------------------
using NfIndex = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, FloatCloud3>,
    FloatCloud3,
    3 /* DIM */,
    std::uint32_t
>;

// ---------------------------------------------------------------------------
// Opaque handle
// ---------------------------------------------------------------------------
struct nfc_kdtree
{
    FloatCloud3 cloud;
    NfIndex*    index;  // owning pointer

    nfc_kdtree() : index(nullptr) {}
    ~nfc_kdtree() { delete index; }
};

// ---------------------------------------------------------------------------
// API implementation
// ---------------------------------------------------------------------------

extern "C" nfc_kdtree_t* nfc_build(const float* points_xyz, int num_points)
{
    if (!points_xyz || num_points <= 0)
        return nullptr;

    nfc_kdtree* h = new (std::nothrow) nfc_kdtree;
    if (!h)
        return nullptr;

    try
    {
        // Copy the points.
        h->cloud.num_points = num_points;
        h->cloud.pts.resize(3 * static_cast<std::size_t>(num_points));
        std::memcpy(h->cloud.pts.data(), points_xyz,
                    3 * static_cast<std::size_t>(num_points) * sizeof(float));

        // Build the index (leaf size = 10, a reasonable default).
        h->index = new NfIndex(3 /*dim*/, h->cloud,
                               nanoflann::KDTreeSingleIndexAdaptorParams(10));
        // index constructor calls buildIndex() automatically.
    }
    catch (...)
    {
        delete h;
        return nullptr;
    }

    return h;
}

extern "C" int nfc_knn(const nfc_kdtree_t* tree,
                        const float         q[3],
                        int                 k,
                        unsigned int*       out_idx,
                        float*              out_dist2)
{
    if (!tree || k <= 0 || !out_idx || !out_dist2)
        return -1;

    try
    {
        // knnSearch returns the actual number found (may be < k).
        std::size_t n = static_cast<std::size_t>(k);
        std::size_t found = tree->index->knnSearch(
            q,
            n,
            reinterpret_cast<std::uint32_t*>(out_idx),
            out_dist2);
        return static_cast<int>(found);
    }
    catch (...)
    {
        return -1;
    }
}

extern "C" int nfc_radius(const nfc_kdtree_t* tree,
                           const float         q[3],
                           float               radius,
                           unsigned int*       out_idx,
                           float*              out_dist2,
                           int                 max_out)
{
    if (!tree || radius < 0.0f || max_out <= 0 || !out_idx || !out_dist2)
        return -1;

    try
    {
        // nanoflann's L2 radiusSearch takes radius² (it uses squared
        // distances internally); the user-visible API takes geometric radius.
        float radius2 = radius * radius;

        std::vector<nanoflann::ResultItem<std::uint32_t, float>> matches;
        matches.reserve(static_cast<std::size_t>(max_out));

        nanoflann::SearchParameters params;
        params.sorted = false;  // skip sort for speed; caller sorts if needed

        std::size_t n = tree->index->radiusSearch(q, radius2, matches, params);

        int cap = std::min(static_cast<int>(n), max_out);
        for (int i = 0; i < cap; ++i)
        {
            out_idx[i]   = static_cast<unsigned int>(matches[i].first);
            out_dist2[i] = matches[i].second;
        }
        return cap;
    }
    catch (...)
    {
        return -1;
    }
}

extern "C" void nfc_free(nfc_kdtree_t* tree)
{
    delete tree;
}
