/// D-Nanoflann nearest-point example and correctness check.
///
/// Builds a fixed 3-D point cloud (20 points), then for three query points
/// verifies:
///   - nearest(k=1) index and distance match O(n) brute-force
///   - knn(k=3)     indices and distances match brute-force top-3
///   - radius(r)    result set matches brute-force within-radius set
///
/// Query points are chosen so the top-3 distances are strictly ordered
/// (no equidistant ties at the k-th slot), which ensures unique answers
/// and makes index-exact comparison valid.
///
/// Statically linked: no libnanoflann*.so appears in `ldd`.
module nearest;

import std.stdio     : writefln, writeln;
import std.math      : sqrt, abs;
import std.conv      : to;
import std.algorithm : sort;
import nanoflann.c;

// ---------------------------------------------------------------------------
// Fixed 20-point cloud (deterministic, no randomness)
// ---------------------------------------------------------------------------
immutable float[60] CLOUD_XYZ = [
     0.0f,  0.0f,  0.0f,   //  0 — origin
     1.0f,  0.0f,  0.0f,   //  1
     0.0f,  1.0f,  0.0f,   //  2
     0.0f,  0.0f,  1.0f,   //  3
     1.0f,  1.0f,  0.0f,   //  4
     1.0f,  0.0f,  1.0f,   //  5
     0.0f,  1.0f,  1.0f,   //  6
     1.0f,  1.0f,  1.0f,   //  7
     0.5f,  0.5f,  0.5f,   //  8 — cube centre
     2.0f,  0.0f,  0.0f,   //  9
     0.0f,  2.0f,  0.0f,   // 10
     0.0f,  0.0f,  2.0f,   // 11
     2.0f,  2.0f,  2.0f,   // 12
    -1.0f,  0.0f,  0.0f,   // 13
     0.0f, -1.0f,  0.0f,   // 14
     0.0f,  0.0f, -1.0f,   // 15
     3.0f,  0.0f,  0.0f,   // 16
     0.0f,  3.0f,  0.0f,   // 17
     0.0f,  0.0f,  3.0f,   // 18
     4.0f,  4.0f,  4.0f,   // 19
];
enum int NUM_POINTS = 20;

// ---------------------------------------------------------------------------
// Brute-force helpers
// ---------------------------------------------------------------------------

float dist2pt(const(float)[] cloud, uint i, const float[3] q) pure @safe
{
    float dx = cloud[3*i+0] - q[0];
    float dy = cloud[3*i+1] - q[1];
    float dz = cloud[3*i+2] - q[2];
    return dx*dx + dy*dy + dz*dz;
}

struct Pair { uint idx; float d2; }

/// Brute-force k nearest neighbours, sorted ascending by d².
Pair[] bruteKNN(const(float)[] cloud, int n, const float[3] q, int k) pure @safe
{
    auto all = new Pair[n];
    foreach (i; 0 .. n)
        all[i] = Pair(cast(uint)i, dist2pt(cloud, cast(uint)i, q));
    all.sort!((a, b) => a.d2 < b.d2);
    return all[0 .. k].dup;
}

/// Brute-force radius search (within geometric radius r).
Pair[] bruteRadius(const(float)[] cloud, int n, const float[3] q, float r) pure @safe
{
    float r2 = r * r;
    Pair[] res;
    foreach (i; 0 .. n)
    {
        float d = dist2pt(cloud, cast(uint)i, q);
        if (d <= r2)
            res ~= Pair(cast(uint)i, d);
    }
    return res;
}

// ---------------------------------------------------------------------------
// Assertion helpers
// ---------------------------------------------------------------------------

void assertClose(float a, float b, float tol, string label) @safe
{
    float err = abs(a - b);
    assert(err <= tol,
        label ~ ": |" ~ a.to!string ~ " - " ~ b.to!string
              ~ "| = " ~ err.to!string ~ " > tol=" ~ tol.to!string);
}

/// Compare knn results. Requires no d² ties at the k-th slot — see
/// query-point selection note at the top of this file.
void assertKNNMatch(const(uint)[] gotIdx, const(float)[] gotD2,
                    const Pair[]  want,   string label) @safe
{
    assert(gotIdx.length == want.length,
        label ~ ": length got=" ~ gotIdx.length.to!string
              ~ " want=" ~ want.length.to!string);
    foreach (i; 0 .. want.length)
    {
        assert(gotIdx[i] == want[i].idx,
            label ~ "[" ~ i.to!string ~ "]: idx got=" ~ gotIdx[i].to!string
                  ~ " want=" ~ want[i].idx.to!string);
        assertClose(gotD2[i], want[i].d2, 1e-5f,
                    label ~ "[" ~ i.to!string ~ "].d2");
    }
}

/// Compare radius-search results (order is unspecified — set equality).
void assertRadiusMatch(const(uint)[] gotIdx, const(float)[] gotD2,
                       const Pair[]  want,   string label) @safe
{
    assert(gotIdx.length == want.length,
        label ~ ": set size got=" ~ gotIdx.length.to!string
              ~ " want=" ~ want.length.to!string);
    foreach (i; 0 .. gotIdx.length)
    {
        bool found = false;
        foreach (w; want)
        {
            if (w.idx == gotIdx[i]) {
                assertClose(gotD2[i], w.d2, 1e-5f,
                            label ~ " d2 for idx=" ~ gotIdx[i].to!string);
                found = true;
                break;
            }
        }
        assert(found,
            label ~ ": unexpected idx=" ~ gotIdx[i].to!string ~ " in result");
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

// Three query points carefully chosen so the top-3 nearest neighbours all
// have strictly distinct distances (no equidistant tie at the 3rd slot).
//
//   Q0 = (0.4, 0.3, 0.2) — top3: #8(0.14), #0(0.29), #1(0.49)
//   Q1 = (0.7, 0.2, 0.8) — top3: #5(0.17), #8(0.22), #3(0.57)
//   Q2 = (2.2, 0.3, 0.4) — top3: #9(0.29), #16(0.89), #1(1.69)
immutable float[3][3] QUERIES = [
    [0.4f, 0.3f, 0.2f],
    [0.7f, 0.2f, 0.8f],
    [2.2f, 0.3f, 0.4f],
];

int main()
{
    // ---- Build -----------------------------------------------------------------
    auto tree = nfc_build(CLOUD_XYZ.ptr, NUM_POINTS);
    assert(tree !is null, "nfc_build returned null");
    scope(exit) nfc_free(tree);

    writeln("KD-tree built from ", NUM_POINTS, " points.");

    // ---- Per-query checks ------------------------------------------------------
    foreach (qi, q; QUERIES)
    {
        writefln("\n--- Query %d: (%.2f, %.2f, %.2f) ---", qi, q[0], q[1], q[2]);

        // knn k=1 — nearest single point
        {
            uint[1]  idx;
            float[1] d2;
            int found = nfc_knn(tree, q.ptr, 1, idx.ptr, d2.ptr);
            assert(found == 1,
                "knn(1) q" ~ qi.to!string ~ ": returned " ~ found.to!string);

            auto want = bruteKNN(CLOUD_XYZ[], NUM_POINTS, q, 1);
            assertKNNMatch(idx[], d2[], want, "knn(1) q" ~ qi.to!string);

            writefln("  nearest: idx=%d dist=%.5f  (bf idx=%d dist=%.5f)",
                     idx[0], sqrt(d2[0]), want[0].idx, sqrt(want[0].d2));
        }

        // knn k=3 — top 3 (unique distances, see QUERIES table above)
        {
            uint[3]  idx;
            float[3] d2;
            int found = nfc_knn(tree, q.ptr, 3, idx.ptr, d2.ptr);
            assert(found == 3,
                "knn(3) q" ~ qi.to!string ~ ": returned " ~ found.to!string);

            auto want = bruteKNN(CLOUD_XYZ[], NUM_POINTS, q, 3);
            assertKNNMatch(idx[], d2[], want, "knn(3) q" ~ qi.to!string);

            writefln("  knn(3): indices=%s  d²=%s", idx[], d2[]);
        }

        // radius r=1.2 — all points within geometric radius 1.2
        {
            immutable float r = 1.2f;
            uint[NUM_POINTS]  idx;
            float[NUM_POINTS] d2;
            int found = nfc_radius(tree, q.ptr, r, idx.ptr, d2.ptr, NUM_POINTS);
            assert(found >= 0,
                "nfc_radius q" ~ qi.to!string ~ ": returned error");

            auto want = bruteRadius(CLOUD_XYZ[], NUM_POINTS, q, r);
            assertRadiusMatch(idx[0..found], d2[0..found], want,
                              "radius(1.2) q" ~ qi.to!string);

            writefln("  radius(%.1f): %d points  (bf: %d)", r, found, want.length);
        }
    }

    // ---- Edge-case: k > cloud size --------------------------------------------
    {
        float[3] q = [0.5f, 0.5f, 0.5f];
        int bigK   = NUM_POINTS + 10;
        auto idx   = new uint[bigK];
        auto d2    = new float[bigK];
        int found  = nfc_knn(tree, q.ptr, bigK, idx.ptr, d2.ptr);
        assert(found == NUM_POINTS,
            "knn(k>N) expected=" ~ NUM_POINTS.to!string
                                 ~ " got=" ~ found.to!string);
        writefln("\n--- k>N edge case: requested=%d, got=%d ---", bigK, found);
    }

    // ---- Edge-case: tiny radius isolates the exact-coincident point only -------
    // Point #8 is exactly at (0.5, 0.5, 0.5).  With radius=0.001 the only
    // cloud point inside the sphere is #8 (d²=0).  nanoflann's RadiusResultSet
    // uses strict `dist < radius²`, so radius=0 never admits anything; we use
    // a small but positive epsilon here to test "almost exact match" semantics.
    {
        float[3] q       = [0.5f, 0.5f, 0.5f];
        float    epsilon = 1e-3f;  // geometric radius; epsilon² ≈ 1e-6 >> 0 = d²(#8)
        uint[NUM_POINTS]  idx;
        float[NUM_POINTS] d2;
        int found = nfc_radius(tree, q.ptr, epsilon, idx.ptr, d2.ptr, NUM_POINTS);
        assert(found == 1,
            "radius(epsilon) expected=1 got=" ~ found.to!string);
        assert(idx[0] == 8,
            "radius(epsilon) expected idx=8 got=" ~ idx[0].to!string);
        assertClose(d2[0], 0.0f, 1e-5f, "radius(epsilon) d2 for #8");
        writefln("\n--- radius(%.3f) edge case: idx=%d d2=%f ---",
                 epsilon, idx[0], d2[0]);
    }

    writeln("\nAll assertions passed — KD-tree results match brute-force.");
    return 0;
}
