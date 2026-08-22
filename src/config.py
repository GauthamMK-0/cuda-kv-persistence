from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class BackboneConfig:
    num_layers: int = 30
    hidden_dim: int = 1536
    num_heads: int = 12
    head_dim: int = 128
    latent_channels: int = 16
    tokens_per_frame: int = 1560
    latent_grid_rows: int = 30
    latent_grid_cols: int = 52
    tile_size_tokens: int = 16

    def __post_init__(self):
        if self.hidden_dim != self.num_heads * self.head_dim:
            raise ValueError(
                f"hidden_dim {self.hidden_dim} != num_heads({self.num_heads}) * head_dim({self.head_dim})"
            )
        if self.latent_grid_rows * self.latent_grid_cols != self.tokens_per_frame:
            raise ValueError(
                f"latent grid {self.latent_grid_rows}x{self.latent_grid_cols} "
                f"!= tokens_per_frame {self.tokens_per_frame}"
            )

    @classmethod
    def from_grounding(cls, path="configs/grounding_config.json"):
        m = json.loads(Path(path).read_text())["model"]
        return cls(
            num_layers=m["layers"],
            hidden_dim=m["hidden_dim"],
            num_heads=m["heads"],
            head_dim=m["head_dim"],
            tokens_per_frame=m["tokens_per_frame"],
            latent_grid_rows=m["latent_grid"][0],
            latent_grid_cols=m["latent_grid"][1],
            tile_size_tokens=m["tile_size_tokens"],
        )
