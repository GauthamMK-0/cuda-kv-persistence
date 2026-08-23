#pragma once

// Swappable tile-selection interface (plan Section 3.2). Implementations
// decide WHICH spatial tiles deserve persisting-L2 residency for the current
// step; the memory machinery underneath is unaffected (Phase 5 swaps
// variants without touching KVTileManager).

class TileGatingPolicy {
public:
    virtual ~TileGatingPolicy() {}
    // Pick up to max_tiles candidate IDs (0..num_tiles-1) given the current
    // frame's per-tile motion scores. Writes at most max_tiles IDs and the
    // actual count.
    virtual void select(const float* motion_scores, int num_tiles,
                        int max_tiles, long* out_ids, int* out_count) = 0;
    virtual const char* name() const = 0;
};

// Control: persist nothing (pure streaming baseline).
class NoPersistencePolicy : public TileGatingPolicy {
public:
    void select(const float*, int, int, long*, int* out_count) override;
    const char* name() const override;
};

// Naive prior-method proxy: a fixed recency-style window regardless of
// motion (always the leading tile IDs).
class UniformWindowPolicy : public TileGatingPolicy {
public:
    void select(const float*, int, int max_tiles, long* out_ids,
                int* out_count) override;
    const char* name() const override;
};

// Naive prior-method proxy under cross-frame context: newest-frames-first
// recency window. Candidates are frame-major flattened ids g*num_tiles+t;
// selects the LAST max_tiles entries.
class RecencyWindowPolicy : public TileGatingPolicy {
public:
    void select(const float*, int num_candidates, int max_tiles,
                long* out_ids, int* out_count) override;
    const char* name() const override;
};

// Proposed (Part 6): persistence value under streaming semantics.
// priority = remaining_sweeps(source frame) * (1 - normalized change rate).
// Early stable tiles are swept by every later query and almost never
// rewritten; recent volatile tiles have few future sweeps and churn.
class ReuseWeightedPolicy : public TileGatingPolicy {
public:
    ReuseWeightedPolicy(int tiles_per_frame, int total_frames);
    void select(const float* scores_flat, int num_candidates, int max_tiles,
                long* out_ids, int* out_count) override;
    const char* name() const override;

private:
    int tpf_;
    int total_frames_;
};

// Proposed: rank by motion score ASCENDING (low motion = redundant across
// frames = high persistence value); ties broken by tile ID for determinism.
class MotionGatedPolicy : public TileGatingPolicy {
public:
    void select(const float* motion_scores, int num_tiles, int max_tiles,
                long* out_ids, int* out_count) override;
    const char* name() const override;
};
