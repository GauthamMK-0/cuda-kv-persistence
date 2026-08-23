#pragma once

// Spatial tile partitioning (plan Section 1): fixed-size row-major chunks of
// the latent token grid, stable tile IDs across frames. Mirrors the Python
// side (traces/gen_synthetic_motion.py) and the grounding config's
// tiles_per_frame derivation: 1560/16 -> 97 full tiles + 1 partial (8 tokens).
//
// NOTE for later phases: these tiles are LOGICAL spatial units. The access
// policy windows in kv_tile_manager require physically contiguous ranges, so
// Phase 4 will consume K/V repacked as [F][H][T][dh]; this header stays the
// single definition of the logical mapping either way.

struct TileLayout {
    int tokens_per_frame;
    int tile_tokens;
    int tiles_per_frame;

    TileLayout(int tpf, int tt)
        : tokens_per_frame(tpf), tile_tokens(tt),
          tiles_per_frame((tpf + tt - 1) / tt) {}

    long tile_start(long tile_id) const { return tile_id * tile_tokens; }

    int tile_count(long tile_id) const {
        long s = tile_start(tile_id);
        long rem = tokens_per_frame - s;
        return static_cast<int>(rem < tile_tokens ? rem : tile_tokens);
    }

    long tile_of_token(long token) const { return token / tile_tokens; }
};
