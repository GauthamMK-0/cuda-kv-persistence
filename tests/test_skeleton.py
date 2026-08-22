import pytest
import torch

from src.backbone import DiTSelfAttentionBlock, SelfAttentionBackbone
from src.config import BackboneConfig


def expected_param_count(cfg: BackboneConfig) -> int:
    qkv_w = cfg.hidden_dim * (3 * cfg.hidden_dim)
    qkv_b = 3 * cfg.hidden_dim
    out_w = cfg.hidden_dim * cfg.hidden_dim
    out_b = cfg.hidden_dim
    ln = 2 * cfg.hidden_dim
    per_block = qkv_w + qkv_b + out_w + out_b + ln
    embed = cfg.latent_channels * cfg.hidden_dim + cfg.hidden_dim
    return cfg.num_layers * per_block + embed + ln


def test_config_invariants():
    cfg = BackboneConfig()
    assert cfg.num_heads * cfg.head_dim == cfg.hidden_dim == 1536
    assert cfg.latent_grid_rows * cfg.latent_grid_cols == cfg.tokens_per_frame == 1560
    assert cfg.num_layers == 30
    with pytest.raises(ValueError):
        BackboneConfig(hidden_dim=1000, num_heads=12, head_dim=128)


def test_param_count_matches_closed_form():
    cfg = BackboneConfig()
    model = SelfAttentionBackbone(cfg, seed=0)
    actual = sum(p.numel() for p in model.parameters())
    assert actual == expected_param_count(cfg) == 283_421_184


def test_block_output_and_kv_shapes_cpu():
    cfg = BackboneConfig()
    torch.manual_seed(0)
    block = DiTSelfAttentionBlock(cfg)
    x = torch.randn(2, cfg.tokens_per_frame, cfg.hidden_dim)
    out, k, v = block(x)
    assert out.shape == x.shape
    assert k.shape == v.shape == (2, cfg.num_heads, cfg.tokens_per_frame, cfg.head_dim)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA device")
def test_backbone_forward_kv_per_layer_gpu():
    cfg = BackboneConfig()
    model = SelfAttentionBackbone(cfg, seed=0).cuda()
    latents = torch.randn(1, cfg.latent_channels, cfg.latent_grid_rows, cfg.latent_grid_cols,
                          device="cuda")
    hidden, kv = model(latents, timestep=torch.zeros(1), context=None)
    assert hidden.shape == (1, cfg.tokens_per_frame, cfg.hidden_dim)
    assert len(kv) == cfg.num_layers
    for k, v in kv:
        assert k.shape == v.shape == (1, cfg.num_heads, cfg.tokens_per_frame, cfg.head_dim)
        assert k.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA device")
def test_random_init_is_reproducible_with_seed():
    a = SelfAttentionBackbone(seed=123)
    b = SelfAttentionBackbone(seed=123)
    pa = next(a.parameters())
    pb = next(b.parameters())
    assert torch.equal(pa, pb)
    assert not torch.allclose(pa, torch.zeros_like(pa))


def test_pretrained_flag_not_implemented():
    with pytest.raises(NotImplementedError):
        SelfAttentionBackbone(init="pretrained")
