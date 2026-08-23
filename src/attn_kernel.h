#pragma once

// Shared declaration of the Phase-2 tiled attention kernel so other
// translation units can link against it.
//
// KVMap/Arena (Phase 4): optional row-remapping for persistence-arena reads.
// KVMap[mbase + j] >= 0 means logical key row j lives at arena row KVMap
// (each arena row = K row || V row, stride 2*D); negative means read from
// the original K/V buffers. Nullpointers restore plain Phase-2 behavior.
//
// cross (Phase 4b): causal temporal prefix — query frame f attends to key
// rows [0, (f+1)*T) of the FULL K/V buffers. Per-frame launches pass
// pre-offset Q/Out pointers, full K/V/map bases, and frame_idx = f.

void attn_forward_launcher(const float* d_q, const float* d_k,
                           const float* d_v, const float* d_arena,
                           const long* d_kvmap, float* d_out,
                           int frames, int T, int D, int heads,
                           bool remap);

// One-frame launch. d_q_f/d_out_f pre-offset to the frame. intra (cross=0):
// d_k_f/d_v_f/d_kvmap_f pre-offset too, span = T. cross (cross=1): d_k/d_v/
// d_kvmap at FULL buffer bases, keys span [(0)..frame_idx]*T..+T, i.e.
// [0, (frame_idx+1)*T).
void attn_forward_single_frame(const float* d_q_f, const float* d_k,
                               const float* d_v, const float* d_arena,
                               const long* d_kvmap_f, float* d_out_f,
                               int T, int D, int heads, int cross,
                               int frame_idx);

static const int ATTENTION_KTILE = 64;
static const int ATTENTION_WARPS = 8;
