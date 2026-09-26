import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from groundiff.diffusion import DiffusionConfig, GrounDiff, gating
from groundiff.losses import LossConfig, groundiff_loss
from groundiff.metrics import classification_errors, dtm_metrics, surface_roughness_deg
from groundiff.models.resdepth import build_resdepth
from groundiff.models.unet import UNet
from groundiff.normalise import denormalise, normalise, tile_range
from groundiff.schedule import build_schedule


def test_unet_matches_palette_size():
    m = UNet(in_channel=3, out_channel=2)
    n = sum(p.numel() for p in m.parameters())
    assert abs(n - 62.639e6) < 5e3          # GrounDiff §8.1 reports 62.6M


def test_unet_checkpointing_same_output_and_grads():
    torch.manual_seed(0)
    a = UNet(in_channel=3, inner_channel=32, channel_mults=(1, 2), dropout=0.0)
    b = UNet(in_channel=3, inner_channel=32, channel_mults=(1, 2), dropout=0.0, use_checkpoint=True)
    for p in a.parameters():
        p.data.normal_(0, 0.05)
    b.load_state_dict(a.state_dict())
    x, g = torch.randn(2, 3, 32, 32), torch.rand(2)
    ya, yb = a(x, g), b(x, g)
    assert torch.allclose(ya, yb, atol=1e-6)
    ya.square().mean().backward()
    yb.square().mean().backward()
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.allclose(pa.grad, pb.grad, atol=1e-6)


def test_resdepth_size_and_residual():
    m = build_resdepth(4)
    assert abs(sum(p.numel() for p in m.parameters()) - 12.63e6) < 0.02e6
    m.eval()
    with torch.no_grad():
        for p in m.last_layer.parameters():
            p.zero_()
        x = torch.randn(1, 4, 64, 64)
        assert torch.allclose(m(x), x[:, :1])      # output = channel 0 + learned correction


def test_schedules():
    c = build_schedule("cosine", 10)
    assert math.isclose(c.alphas_bar[-1], 2.409e-5, rel_tol=1e-2)
    r = build_schedule("cosine_range", 10)
    assert math.isclose(r.alphas_bar[-1], 0.9037, rel_tol=1e-3)
    assert r.betas.min() >= 1e-4 - 1e-12 and r.betas.max() <= 2e-2 + 1e-12
    for s in (c, r, build_schedule("linear", 10)):
        assert s.posterior_var[0] == 0.0 and (s.posterior_var[1:] > 0).all()
        # posterior of q(x_{t-1}|x_t,x0) is exact when x_t = sqrt(alpha_t) x_{t-1} with no noise:
        # mean must equal x_{t-1} when x0 is consistent with a noiseless chain
        x0 = 0.7
        for t in range(2, s.T + 1):
            xtm1 = math.sqrt(s.alphas_bar_prev[t - 1]) * x0
            xt = math.sqrt(s.alphas[t - 1]) * xtm1
            mean = s.coef_x0[t - 1] * x0 + s.coef_xt[t - 1] * xt
            assert math.isclose(mean, xtm1, rel_tol=1e-9, abs_tol=1e-12)


def test_gating_limits():
    s, r = torch.full((1, 1, 2, 2), 3.0), torch.full((1, 1, 2, 2), 1.0)
    assert torch.allclose(gating(r, torch.full_like(s, 50.0), s), s)
    assert torch.allclose(gating(r, torch.full_like(s, -50.0), s), s - r)


class OracleDenoiser(nn.Module):
    """Returns r = s - g_true and a confident 'non-ground' logit."""

    def __init__(self, g_true):
        super().__init__()
        self.g_true = g_true

    def forward(self, x, gamma):
        s = x[:, 1:2]
        return torch.cat([s - self.g_true, torch.full_like(s, -30.0)], dim=1)


@pytest.mark.parametrize("schedule", ["cosine", "cosine_range"])
@pytest.mark.parametrize("init", ["dsm_noise", "noise", "dsm", "prior", "prior_noise"])
def test_sampler_recovers_oracle(schedule, init):
    torch.manual_seed(0)
    g = torch.rand(2, 1, 16, 16) * 0.5 - 0.8
    cond = torch.cat([g + 0.5, g], dim=1)                     # [dsm_max, dsm_min]
    gd = GrounDiff(OracleDenoiser(g), DiffusionConfig(schedule=schedule), ["dsm_max", "dsm_min"])
    prior = g + 0.1 * torch.randn_like(g)
    out, logit = gd.sample(cond, init=init, prior=prior if init.startswith("prior") else None)
    assert torch.allclose(out, g, atol=1e-5)
    out2, _ = gd.sample(cond, init=init, prior=prior, t_start=4)
    assert torch.allclose(out2, g, atol=1e-5)


def test_initial_state_semantics():
    gd = GrounDiff(OracleDenoiser(torch.zeros(1)), DiffusionConfig(), ["dsm_max"])
    s = torch.full((4000, 1, 4, 4), 0.5)
    x = gd.initial_state(s, "dsm_noise", None, gd.T)
    assert abs(x.mean().item() - 0.5) < 0.02 and abs(x.std().item() - 1.0) < 0.02   # N(s, I)
    assert torch.equal(gd.initial_state(s, "dsm", None, gd.T), s)
    p = torch.full_like(s, -0.3)
    assert torch.equal(gd.initial_state(s, "prior", p, gd.T), p)
    x5 = gd.initial_state(s, "prior", p, 5)                       # q(g_5 | prior)
    ab = gd.schedule.alphas_bar[4]
    assert abs(x5.mean().item() - math.sqrt(ab) * -0.3) < 0.02


def test_loss_zero_when_perfect_and_masks():
    g = torch.randn(2, 1, 8, 8)
    valid = torch.ones_like(g)
    m = (torch.rand_like(g) > 0.5).float()
    logit = (m * 2 - 1) * 40
    out = groundiff_loss(g.clone(), logit, g, m, valid, LossConfig())
    assert out["l1"] < 1e-6 and out["l2"] < 1e-9 and out["grad"] < 1e-5 and out["conf"] < 1e-6
    bad = g.clone()
    bad[:, :, :4] += 100.0
    valid2 = valid.clone()
    valid2[:, :, :5] = 0          # also hides the gradient row touching the corrupted block
    out2 = groundiff_loss(bad, logit, g, m, valid2, LossConfig())
    assert out2["l1"] < 1e-6 and out2["grad"] < 1e-5


def test_normalise_roundtrip_and_flat_tiles():
    rng = np.random.default_rng(0)
    stack = rng.normal(50, 5, (2, 16, 16))
    valid = np.ones((16, 16), bool)
    lo, sc = tile_range(stack, valid)
    xn = normalise(stack, lo, sc)
    assert np.isclose(xn.min(), -1) and np.isclose(xn.max(), 1)
    assert np.allclose(denormalise(xn, lo, sc), stack)
    flat = np.full((1, 4, 4), 10.0)
    lo, sc = tile_range(flat, np.ones((4, 4), bool), min_range=2.0)
    assert sc == 2.0 and np.isclose(normalise(10.0, lo, sc), 0.0)


def test_classification_errors_definitions():
    gt = np.array([True, True, False, False])
    pred = np.array([True, False, True, False])
    e = classification_errors(pred, gt, np.ones(4, bool))
    assert e["type1_pct"] == 50.0 and e["type2_pct"] == 50.0 and e["total_pct"] == 50.0


def test_dtm_metrics_basic():
    gt = np.zeros((10, 10))
    pred = gt + 0.1
    dsm = gt.copy()
    dsm[:5] = 5.0
    m = dtm_metrics(pred, gt, np.ones_like(gt, bool), dsm, alpha=0.2, gsd=1.0)
    assert math.isclose(m["rmse"], 0.1) and math.isclose(m["mae"], 0.1)
    assert m["type1_pct"] == 0.0 and m["type2_pct"] == 0.0
    assert surface_roughness_deg(np.zeros((5, 5)), np.ones((5, 5), bool), 1.0) == 0.0
