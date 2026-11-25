import numpy as np
import pytest
import torch
from PIL import Image

from conftest import make_data
from fer.data.dataset import build_dataloaders
from fer.data.splits import build_splits
from fer.data.transforms import build_transform
from fer.explain.gradcam import GradCAM, border_mass, randomized_copy
from fer.models import build_model
from fer.utils.config import apply_overrides, load_config
from fer.utils.seed import seed_everything


@pytest.mark.parametrize("ch,size,norm", [(1, 48, "simple"), (3, 48, "imagenet"), (3, 96, "imagenet")])
@pytest.mark.parametrize("aug", ["none", "basic", "strong"])
def test_transform_shapes(ch, size, norm, aug):
    d = {"image_size": size, "channels": ch, "normalize": norm, "augmentation": aug}
    x = build_transform(d, train=True)(Image.fromarray(np.random.randint(0, 255, (48, 48), dtype=np.uint8), mode="L"))
    assert x.shape == (ch, size, size)


def test_eval_transform_is_deterministic():
    d = {"image_size": 48, "channels": 1, "augmentation": "strong"}
    tf = build_transform(d, train=False)
    img = Image.fromarray(np.random.randint(0, 255, (48, 48), dtype=np.uint8), mode="L")
    assert torch.equal(tf(img), tf(img))


def test_loaders_and_seed_reproducibility():
    data = make_data()
    sp = build_splits(data)
    d = {"image_size": 48, "channels": 1, "augmentation": "basic", "batch_size": 32, "num_workers": 0}
    def first_batch():
        seed_everything(7)
        loaders, counts = build_dataloaders(d, data, sp.indices, 7)
        return next(iter(loaders["train"]))
    (x1, y1), (x2, y2) = first_batch(), first_batch()
    assert torch.equal(y1, y2) and torch.allclose(x1, x2)


def test_test_split_not_loaded_unless_requested():
    data = make_data(); sp = build_splits(data)
    loaders, _ = build_dataloaders({"batch_size": 8}, data, sp.indices, 0)
    assert set(loaders) == {"train", "val"}


@pytest.mark.parametrize("cfg", [{"name": "custom_cnn"}, {"name": "resnet18", "pretrained": False, "trainable_from": "fc"},
                                 {"name": "mobilenetv2", "pretrained": False, "freeze_blocks": 19}])
def test_gradcam_shape_range_and_frozen_models(cfg):
    m = build_model(cfg, 1).eval()
    gc = GradCAM(m, m.gradcam_layer())
    cam, logits, idx = gc(torch.randn(3, 1, 48, 48))
    assert cam.shape == (3, 48, 48) and cam.min() >= 0 and cam.max() <= 1 + 1e-6
    assert logits.shape == (3, 7) and idx.shape == (3,)
    gc.remove()


def test_randomized_copy_changes_weights_not_original():
    m = build_model({"name": "custom_cnn"}, 1)
    w0 = m.features[0].convs[0].weight.clone()
    r = randomized_copy(m)
    assert torch.equal(w0, m.features[0].convs[0].weight)
    assert not torch.equal(w0, r.features[0].convs[0].weight)


def test_border_mass_uniform_equals_expected():
    obs, exp = border_mass(np.ones((48, 48)))
    assert abs(obs - exp) < 0.05


def test_config_inheritance_and_overrides():
    cfg = load_config("configs/resnet18_finetune.yaml", ["train.lr=0.5", "data.augmentation=strong"])
    assert cfg["train"]["lr"] == 0.5 and cfg["data"]["augmentation"] == "strong"
    assert cfg["train"]["optimizer"] == "adamw" and cfg["model"]["name"] == "resnet18"
    assert apply_overrides({"a": {"b": 1}}, {"a.c": 2}) == {"a": {"b": 1, "c": 2}}


def test_all_shipped_configs_build_a_model():
    import glob
    for p in glob.glob("configs/*.yaml"):
        if p.endswith("base.yaml"):
            continue
        cfg = load_config(p)
        m = build_model({**cfg["model"], "pretrained": False}, cfg["data"]["channels"])
        size = cfg["data"]["image_size"]
        assert m.eval()(torch.randn(1, cfg["data"]["channels"], size, size)).shape == (1, 7), p
