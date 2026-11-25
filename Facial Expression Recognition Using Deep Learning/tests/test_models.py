import pytest
import torch

from fer.models import build_model, count_parameters


@pytest.mark.parametrize("cfg,ch,size", [
    ({"name": "custom_cnn"}, 1, 48),
    ({"name": "custom_cnn", "widths": [16, 32], "convs_per_block": 1}, 3, 48),
    ({"name": "resnet18", "pretrained": False}, 3, 48),
    ({"name": "resnet18", "pretrained": False, "small_input": True}, 1, 48),
    ({"name": "mobilenetv2", "pretrained": False}, 3, 64),
    ({"name": "mobilenetv2", "pretrained": False, "small_input": True}, 1, 48),
])
def test_forward_shape(cfg, ch, size):
    m = build_model(cfg, ch).eval()
    assert m(torch.randn(2, ch, size, size)).shape == (2, 7)


def test_resnet_freezing_levels():
    fc = count_parameters(build_model({"name": "resnet18", "pretrained": False, "trainable_from": "fc"}, 3))
    l4 = count_parameters(build_model({"name": "resnet18", "pretrained": False, "trainable_from": "layer4"}, 3))
    al = count_parameters(build_model({"name": "resnet18", "pretrained": False, "trainable_from": "all"}, 3))
    assert fc["trainable"] == 512 * 7 + 7
    assert fc["trainable"] < l4["trainable"] < al["trainable"] == al["total"]


def test_mobilenet_freezing_levels():
    n = lambda k: count_parameters(build_model({"name": "mobilenetv2", "pretrained": False, "freeze_blocks": k}, 3))
    assert n(19)["trainable"] == 1280 * 7 + 7
    assert n(19)["trainable"] < n(10)["trainable"] < n(0)["trainable"] == n(0)["total"]


def test_frozen_batchnorm_stats_do_not_update():
    m = build_model({"name": "resnet18", "pretrained": False, "trainable_from": "fc"}, 3)
    before = m.net.layer1[0].bn1.running_mean.clone()
    m.train()
    m(torch.randn(8, 3, 48, 48) * 5 + 3)
    assert torch.equal(before, m.net.layer1[0].bn1.running_mean)


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        build_model({"name": "vgg"}, 1)
