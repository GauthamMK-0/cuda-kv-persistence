// KVTileManager implementation (plan Section 3.3) + extern "C" facade for
// ctypes-driven tests. Phase 3: accounting + admission control; windows are
// applied over caller-provided contiguous device ranges (Phase 4 will pass
// repacked K/V pointers).

#include "kv_tile_manager.h"
#include "tile_layout.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

KVTileManager::KVTileManager(size_t budget_bytes, size_t max_window_bytes)
    : budget_bytes_(budget_bytes), max_window_bytes_(max_window_bytes),
      used_bytes_(0), initialized_(false), arena_(nullptr), arena_cap_(0),
      arena_row_stride_floats_(0), arena_next_row_(0), arena_used_(0),
      slot_high_water_(-1), staged_bytes_(0) {}

KVTileManager::~KVTileManager() {
    clear();
    if (arena_) cudaFree(arena_);
}

bool KVTileManager::init() {
    // Grounding constraint: never request more than the device's max
    // set-aside (2,162,688 B on this GA106). The caller passes exactly that.
    cudaError_t e = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,
                                       budget_bytes_);
    if (e != cudaSuccess) return false;
    initialized_ = true;
    return true;
}

size_t KVTileManager::query_device_limit() const {
    size_t v = 0;
    cudaDeviceGetLimit(&v, cudaLimitPersistingL2CacheSize);
    return v;
}

static void apply_window(void* ptr, size_t bytes, cudaAccessProperty prop) {
    cudaStreamAttrValue attr{};
    attr.accessPolicyWindow.base_ptr = ptr;
    attr.accessPolicyWindow.num_bytes = bytes;
    attr.accessPolicyWindow.hitRatio = 1.0f;
    attr.accessPolicyWindow.hitProp = prop;
    attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &attr);
}

KvtmResult KVTileManager::admit(long tile_id, void* ptr, size_t bytes) {
    if (!initialized_ || tile_id < 0 || ptr == nullptr || bytes == 0)
        return KVTM_INVALID;
    if (active_.count(tile_id)) return KVTM_INVALID;
    if (bytes > max_window_bytes_) return KVTM_INVALID;
    if (used_bytes_ + bytes > budget_bytes_) return KVTM_REFUSED_BUDGET;

    apply_window(ptr, bytes, cudaAccessPropertyPersisting);
    active_[tile_id] = {ptr, bytes};
    used_bytes_ += bytes;
    return KVTM_ADMITTED;
}

bool KVTileManager::release(long tile_id) {
    auto it = active_.find(tile_id);
    if (it == active_.end()) return false;
    apply_window(it->second.ptr, it->second.bytes, cudaAccessPropertyNormal);
    used_bytes_ -= it->second.bytes;
    active_.erase(it);
    return true;
}

void KVTileManager::clear() {
    if (active_.empty()) return;
    active_.clear();
    used_bytes_ = 0;
    cudaCtxResetPersistingL2Cache();
}

int KVTileManager::persisted_tiles() const {
    return static_cast<int>(active_.size());
}

bool KVTileManager::is_persisted(long tile_id) const {
    return active_.count(tile_id) != 0;
}

// ---- arena mode (Phase 4) ----

bool KVTileManager::arena_init(size_t capacity_bytes,
                               size_t row_stride_floats) {
    if (arena_) return false;
    if (cudaMalloc(&arena_, capacity_bytes) != cudaSuccess) return false;
    arena_cap_ = capacity_bytes;
    arena_row_stride_floats_ = row_stride_floats;
    arena_next_row_ = 0;
    arena_used_ = 0;
    return true;
}

void KVTileManager::arena_begin_step() {
    arena_next_row_ = 0;
    arena_used_ = 0;
}

bool KVTileManager::arena_stage(const void* k_dev, size_t k_bytes,
                                const void* v_dev, size_t v_bytes,
                                long* out_row) {
    const size_t total = k_bytes + v_bytes;
    if (arena_used_ + total > budget_bytes_) return false;  // admission gate
    if (arena_used_ + total > arena_cap_) return false;
    float* dst = arena_ + (size_t)arena_next_row_ * arena_row_stride_floats_;
    if (cudaMemcpyAsync(dst, k_dev, k_bytes, cudaMemcpyDeviceToDevice) !=
        cudaSuccess)
        return false;
    if (cudaMemcpyAsync((char*)dst + k_bytes, v_dev, v_bytes,
                        cudaMemcpyDeviceToDevice) != cudaSuccess)
        return false;
    *out_row = (long)arena_next_row_;
    ++arena_next_row_;
    arena_used_ += total;
    return true;
}

bool KVTileManager::arena_apply_window() {
    if (!initialized_ || arena_used_ == 0 || arena_used_ > max_window_bytes_)
        return false;
    apply_window(arena_, arena_used_, cudaAccessPropertyPersisting);
    return true;
}

void KVTileManager::arena_end_step() {
    if (arena_used_ > 0) {
        apply_window(arena_, arena_used_, cudaAccessPropertyNormal);
        cudaCtxResetPersistingL2Cache();
    }
    arena_next_row_ = 0;
    arena_used_ = 0;
}

// ---- streaming mode (Part 6) ----

bool KVTileManager::streaming_begin() {
    if (!arena_) return false;
    row_slot_.clear();
    free_slots_.clear();
    const long cap_rows =
        (long)(arena_cap_ / (arena_row_stride_floats_ * sizeof(float)));
    for (long s = cap_rows - 1; s >= 0; --s) free_slots_.push_back(s);
    slot_high_water_ = -1;
    staged_bytes_ = 0;
    return true;
}

bool KVTileManager::streaming_stage(long row_id, const void* k_dev,
                                    size_t k_bytes, const void* v_dev,
                                    size_t v_bytes, bool changed,
                                    long* out_slot, bool* restaged) {
    auto it = row_slot_.find(row_id);
    if (it != row_slot_.end()) {
        *out_slot = it->second;
        if (!changed) {
            *restaged = false;
            return true;  // sticky hit: content already valid in slot
        }
        float* dst =
            arena_ + (size_t)it->second * arena_row_stride_floats_;
        if (cudaMemcpyAsync(dst, k_dev, k_bytes,
                            cudaMemcpyDeviceToDevice) != cudaSuccess)
            return false;
        if (cudaMemcpyAsync((char*)dst + k_bytes, v_dev, v_bytes,
                            cudaMemcpyDeviceToDevice) != cudaSuccess)
            return false;
        staged_bytes_ += k_bytes + v_bytes;
        *restaged = true;
        return true;
    }
    if (free_slots_.empty()) return false;  // capacity gate
    const long slot = free_slots_.back();
    free_slots_.pop_back();
    float* dst = arena_ + (size_t)slot * arena_row_stride_floats_;
    if (cudaMemcpyAsync(dst, k_dev, k_bytes,
                        cudaMemcpyDeviceToDevice) != cudaSuccess)
        return false;
    if (cudaMemcpyAsync((char*)dst + k_bytes, v_dev, v_bytes,
                        cudaMemcpyDeviceToDevice) != cudaSuccess)
        return false;
    row_slot_[row_id] = slot;
    staged_bytes_ += k_bytes + v_bytes;
    if (slot > slot_high_water_) slot_high_water_ = slot;
    *out_slot = slot;
    *restaged = true;
    return true;
}

bool KVTileManager::streaming_release(long row_id) {
    auto it = row_slot_.find(row_id);
    if (it == row_slot_.end()) return false;
    free_slots_.push_back(it->second);
    row_slot_.erase(it);
    return true;
}

bool KVTileManager::streaming_apply_window() {
    if (!initialized_ || slot_high_water_ < 0) return false;
    const size_t span =
        ((size_t)slot_high_water_ + 1) * arena_row_stride_floats_ *
        sizeof(float);
    if (span > max_window_bytes_) return false;
    apply_window(arena_, span, cudaAccessPropertyPersisting);
    return true;
}

void KVTileManager::streaming_end() {
    streaming_apply_window();
    apply_window(arena_, 1, cudaAccessPropertyNormal);
    cudaCtxResetPersistingL2Cache();
    row_slot_.clear();
    free_slots_.clear();
    slot_high_water_ = -1;
}

// ---- extern "C" facade (ctypes contract) ----

extern "C" {

void* kvtm_create(long budget_bytes, long max_window_bytes) {
    try {
        return new KVTileManager((size_t)budget_bytes,
                                 (size_t)max_window_bytes);
    } catch (...) {
        return nullptr;
    }
}

void kvtm_destroy(void* m) { delete static_cast<KVTileManager*>(m); }

int kvtm_init(void* m) {
    return static_cast<KVTileManager*>(m)->init() ? 1 : 0;
}

long kvtm_query_limit(void* m) {
    return (long)static_cast<KVTileManager*>(m)->query_device_limit();
}

int kvtm_admit(void* m, long tile_id, void* ptr, long bytes) {
    return (int)static_cast<KVTileManager*>(m)->admit(tile_id, ptr,
                                                      (size_t)bytes);
}

int kvtm_release(void* m, long tile_id) {
    return static_cast<KVTileManager*>(m)->release(tile_id) ? 1 : 0;
}

void kvtm_clear(void* m) { static_cast<KVTileManager*>(m)->clear(); }

long kvtm_persisted_bytes(void* m) {
    return (long)static_cast<KVTileManager*>(m)->persisted_bytes();
}

int kvtm_persisted_tiles(void* m) {
    return static_cast<KVTileManager*>(m)->persisted_tiles();
}

// Device scratch allocation so tests can aim windows at real device memory.
void* kvtm_dev_alloc(long bytes) {
    void* p = nullptr;
    if (cudaMalloc(&p, (size_t)bytes) != cudaSuccess) return nullptr;
    return p;
}

void kvtm_dev_free(void* p) { cudaFree(p); }

// ---- arena-mode facade (Phase 4) ----

int kvtm_arena_init(void* m, long cap_bytes, long row_stride_floats) {
    return static_cast<KVTileManager*>(m)->arena_init((size_t)cap_bytes,
                                                      (size_t)row_stride_floats)
               ? 1
               : 0;
}
void kvtm_arena_begin(void* m) { static_cast<KVTileManager*>(m)->arena_begin_step(); }
int kvtm_arena_stage(void* m, const void* k_dev, long k_bytes,
                     const void* v_dev, long v_bytes, long* out_row) {
    return static_cast<KVTileManager*>(m)->arena_stage(k_dev, (size_t)k_bytes,
                                                       v_dev, (size_t)v_bytes,
                                                       out_row)
               ? 1
               : 0;
}
int kvtm_arena_window(void* m) {
    return static_cast<KVTileManager*>(m)->arena_apply_window() ? 1 : 0;
}
void kvtm_arena_end(void* m) { static_cast<KVTileManager*>(m)->arena_end_step(); }
long kvtm_arena_used(void* m) {
    return (long)static_cast<KVTileManager*>(m)->arena_used_bytes();
}
void* kvtm_arena_ptr(void* m) {
    return static_cast<KVTileManager*>(m)->arena_device_ptr();
}

// ---- streaming facade (Part 6) ----

int kvtm_streaming_begin(void* m) {
    return static_cast<KVTileManager*>(m)->streaming_begin() ? 1 : 0;
}
int kvtm_streaming_stage(void* m, long row_id, const void* k_dev,
                         long k_bytes, const void* v_dev, long v_bytes,
                         int changed, long* out_slot, int* restaged) {
    bool r = false;
    long slot = -1;
    int ok = static_cast<KVTileManager*>(m)->streaming_stage(
        row_id, k_dev, (size_t)k_bytes, v_dev, (size_t)v_bytes,
        changed != 0, &slot, &r);
    if (out_slot) *out_slot = slot;
    if (restaged) *restaged = r ? 1 : 0;
    return ok;
}
int kvtm_streaming_release(void* m, long row_id) {
    return static_cast<KVTileManager*>(m)->streaming_release(row_id) ? 1 : 0;
}
int kvtm_streaming_window(void* m) {
    return static_cast<KVTileManager*>(m)->streaming_apply_window() ? 1 : 0;
}
void kvtm_streaming_end(void* m) {
    static_cast<KVTileManager*>(m)->streaming_end();
}
long kvtm_streaming_staged_bytes(void* m) {
    return (long)static_cast<KVTileManager*>(m)->streaming_staged_bytes();
}

// Tile-layout cross-check helpers (mirrors src/tile_layout.h).
long kvtm_layout_tiles_per_frame(int tokens_per_frame, int tile_tokens) {
    return TileLayout(tokens_per_frame, tile_tokens).tiles_per_frame;
}
long kvtm_layout_tile_start(int tpf, int tt, long id) {
    return TileLayout(tpf, tt).tile_start(id);
}
int kvtm_layout_tile_count(int tpf, int tt, long id) {
    return TileLayout(tpf, tt).tile_count(id);
}

}  // extern "C"
