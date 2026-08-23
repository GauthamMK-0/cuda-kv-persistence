#pragma once

// KVTileManager (plan Section 3.3): owns persisting-L2 budget accounting and
// translates per-tile admit/release decisions into real access-property
// windows on the default stream.
//
// Phase 3 scope: correct accounting + admission control only. No gating
// policy lives here — callers decide WHICH tiles to propose; the manager
// refuses anything that would exceed the set-aside.

#include <cstddef>
#include <unordered_map>
#include <vector>

enum KvtmResult {
    KVTM_ADMITTED = 1,
    KVTM_REFUSED_BUDGET = 0,
    KVTM_INVALID = -1,  // bad args, duplicate admit, or release of unknown tile
};

class KVTileManager {
public:
    KVTileManager(size_t budget_bytes, size_t max_window_bytes);
    ~KVTileManager();

    bool init();  // applies cudaDeviceSetLimit(persisting L2 carve-out)

    // Apply a persisting window over [ptr, ptr+bytes) if the budget allows.
    KvtmResult admit(long tile_id, void* ptr, size_t bytes);
    bool release(long tile_id);  // demote range back to normal property
    void clear();                // layer teardown: reset device persisting state

    size_t persisted_bytes() const { return used_bytes_; }
    int persisted_tiles() const;
    bool is_persisted(long tile_id) const;
    size_t query_device_limit() const;  // cudaDeviceGetLimit read-back

    // Arena mode (Phase 4): a stream supports exactly ONE access-policy
    // window, so scattered per-tile windows overwrite each other. Instead the
    // step's admitted tiles are compacted into one contiguous staging buffer
    // and a single window covers it. Accounting still flows through
    // budget_bytes_; row layout is [K half | V half] of row_stride_floats.
    bool arena_init(size_t capacity_bytes, size_t row_stride_floats);
    void arena_begin_step();
    // D2D-copy one tile's K/V halves into the next arena slot. Returns false
    // if this would exceed either the budget or the arena capacity.
    bool arena_stage(const void* k_dev, size_t k_bytes, const void* v_dev,
                     size_t v_bytes, long* out_row);
    bool arena_apply_window();  // persisting window over staged bytes
    void arena_end_step();      // demote + reset device persisting state
    size_t arena_used_bytes() const { return arena_used_; }
    void* arena_device_ptr() const { return arena_; }

    // Streaming mode (Part 6): slots are keyed by logical row id and persist
    // while their content stays valid. stage copies ONLY on first admission
    // or when the caller reports the content changed since last staging —
    // sticky hits are pure map lookups. Restage volume is the policy-sensitive
    // quantity this mode exists to measure.
    bool streaming_begin();   // clear slot table; requires arena_init first
    bool streaming_stage(long row_id, const void* k_dev, size_t k_bytes,
                         const void* v_dev, size_t v_bytes, bool changed,
                         long* out_slot, bool* restaged);
    bool streaming_release(long row_id);  // free a slot for reuse
    bool streaming_apply_window();        // persisting window over slot span
    void streaming_end();                 // final teardown: demote + reset
    size_t streaming_staged_bytes() const { return staged_bytes_; }

private:
    struct Entry {
        void* ptr;
        size_t bytes;
    };

    std::unordered_map<long, Entry> active_;
    size_t budget_bytes_;
    size_t max_window_bytes_;
    size_t used_bytes_;
    bool initialized_;

    float* arena_;
    size_t arena_cap_;
    size_t arena_row_stride_floats_;
    size_t arena_next_row_;
    size_t arena_used_;

    std::unordered_map<long, long> row_slot_;
    std::vector<long> free_slots_;
    long slot_high_water_;
    size_t staged_bytes_;
};
