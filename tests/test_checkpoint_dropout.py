"""Activation checkpointing with dropout must give the gradients of the plain
forward pass even when torch does not restore the device RNG (as on MPS)."""
import torch

from groundiff.models import unet as unet_mod
from groundiff.models.unet import UNet


def _grads(use_ckpt: bool, bypass: bool, monkeypatch):
    if bypass:   # run the checkpointed function directly: same masks, no recomputation
        monkeypatch.setattr(unet_mod, "checkpoint", lambda fn, *a, **k: fn(*a))
    torch.manual_seed(0)
    net = UNet(3, 2, inner_channel=32, channel_mults=(1, 2), res_blocks=1, attn_res=(), dropout=0.5,
               num_head_channels=8, use_checkpoint=use_ckpt)
    net.train()
    x = torch.randn(2, 3, 16, 16)
    g = torch.rand(2)
    torch.manual_seed(1)
    net(x, g).square().mean().backward()
    return [p.grad.clone() for p in net.parameters() if p.grad is not None]


def test_checkpoint_dropout_gradients_match(monkeypatch):
    ref = _grads(True, True, monkeypatch)
    monkeypatch.undo()
    got = _grads(True, False, monkeypatch)
    assert len(ref) == len(got)
    for a, b in zip(ref, got):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-5)


def test_dropout_active_in_train_mode():
    torch.manual_seed(0)
    net = UNet(3, 2, inner_channel=32, channel_mults=(1, 2), res_blocks=1, attn_res=(), dropout=0.5,
               num_head_channels=8, use_checkpoint=True)
    for m in net.modules():   # make the zero-initialised output convs non-zero so dropout shows
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.normal_(m.weight, std=0.1)
    net.train()
    x, g = torch.randn(1, 3, 16, 16), torch.rand(1)
    a, b = net(x, g), net(x, g)
    assert not torch.allclose(a, b)
    net.eval()
    assert torch.allclose(net(x, g), net(x, g))
