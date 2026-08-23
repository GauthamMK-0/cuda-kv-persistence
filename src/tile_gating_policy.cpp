#include "tile_gating_policy.h"

#include <algorithm>
#include <vector>

void NoPersistencePolicy::select(const float*, int, int, long*,
                                 int* out_count) {
    *out_count = 0;
}

const char* NoPersistencePolicy::name() const { return "none"; }

void UniformWindowPolicy::select(const float*, int, int max_tiles,
                                 long* out_ids, int* out_count) {
    for (int i = 0; i < max_tiles; ++i) out_ids[i] = i;
    *out_count = max_tiles;
}

const char* UniformWindowPolicy::name() const { return "uniform"; }

void RecencyWindowPolicy::select(const float*, int num_candidates,
                                 int max_tiles, long* out_ids, int* out_count) {
    int n = 0;
    for (int i = std::max(0, num_candidates - max_tiles);
         i < num_candidates; ++i)
        out_ids[n++] = i;
    *out_count = n;
}

const char* RecencyWindowPolicy::name() const { return "recency"; }

ReuseWeightedPolicy::ReuseWeightedPolicy(int tiles_per_frame, int total_frames)
    : tpf_(tiles_per_frame), total_frames_(total_frames) {}

void ReuseWeightedPolicy::select(const float* scores_flat, int num_candidates,
                                 int max_tiles, long* out_ids, int* out_count) {
    // priority = remaining_sweeps(source frame) * (1 - normalized rate),
    // with the rate normalized by the max within the candidate prefix.
    float max_rate = 1e-6f;
    for (int i = 0; i < num_candidates; ++i)
        max_rate = std::max(max_rate, scores_flat[i]);
    const double max_rate_d = (double)max_rate;
    struct Cand {
        long id;
        double prio;
    };
    std::vector<Cand> cands;
    cands.reserve((size_t)num_candidates);
    for (int i = 0; i < num_candidates; ++i) {
        const long gid = i;
        const long src_f = (long)(gid / tpf_);
        const double remaining = (double)(total_frames_ - src_f);
        const double rate = std::min(1.0, scores_flat[i] / max_rate_d);
        cands.push_back({gid, remaining * (1.0 - rate)});
    }
    std::stable_sort(cands.begin(), cands.end(),
                     [](const Cand& a, const Cand& b) {
                         if (a.prio != b.prio) return a.prio > b.prio;
                         return a.id < b.id;
                     });
    int n = 0;
    for (int i = 0; i < num_candidates && n < max_tiles; ++i) {
        out_ids[n++] = cands[i].id;
    }
    *out_count = n;
}

const char* ReuseWeightedPolicy::name() const { return "reuse_weighted"; }

void MotionGatedPolicy::select(const float* motion_scores, int num_tiles,
                               int max_tiles, long* out_ids, int* out_count) {
    std::vector<int> order(num_tiles);
    for (int i = 0; i < num_tiles; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) {
                         return motion_scores[a] < motion_scores[b];
                     });
    const int n = std::min(max_tiles, num_tiles);
    for (int i = 0; i < n; ++i) out_ids[i] = order[i];
    *out_count = n;
}

const char* MotionGatedPolicy::name() const { return "motion"; }
