import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from cnn_env import CNNEnv
from feature_extractor import UTTTFeatureExtractor


def make_extractor(features_dim=256, n_filters=64):
    obs_space = CNNEnv().observation_space
    return UTTTFeatureExtractor(obs_space, features_dim=features_dim, n_filters=n_filters)


def random_batch(batch_size=4):
    """Random binary obs batch matching the (6,9,9) space."""
    return torch.randint(0, 2, (batch_size, 6, 9, 9)).float()


def test_output_shape_default():
    """Default settings produce (batch, 256) output."""
    extractor = make_extractor()
    out = extractor(random_batch(4))
    assert out.shape == (4, 256), f"Expected (4, 256), got {out.shape}"
    print("PASS test_output_shape_default")


def test_output_shape_custom_features_dim():
    """features_dim argument controls output width."""
    extractor = make_extractor(features_dim=128)
    out = extractor(random_batch(4))
    assert out.shape == (4, 128), f"Expected (4, 128), got {out.shape}"
    print("PASS test_output_shape_custom_features_dim")


def test_output_shape_custom_n_filters():
    """n_filters argument changes internal channel count but not output shape."""
    extractor = make_extractor(n_filters=32)
    out = extractor(random_batch(4))
    assert out.shape == (4, 256), f"Expected (4, 256), got {out.shape}"
    print("PASS test_output_shape_custom_n_filters")


def test_n_filters_changes_parameter_count():
    """Doubling n_filters should produce more parameters than the default."""
    params_64  = sum(p.numel() for p in make_extractor(n_filters=64).parameters())
    params_128 = sum(p.numel() for p in make_extractor(n_filters=128).parameters())
    assert params_128 > params_64, "More filters should mean more parameters"
    print(f"PASS test_n_filters_changes_parameter_count  (64->{params_64}, 128->{params_128})")


def test_no_nans_or_infs():
    """Output contains no NaN or Inf values on typical binary input."""
    extractor = make_extractor()
    out = extractor(random_batch(8))
    assert not torch.isnan(out).any(), "Output contains NaN"
    assert not torch.isinf(out).any(), "Output contains Inf"
    print("PASS test_no_nans_or_infs")


def test_output_dtype():
    """Output tensor is float32."""
    extractor = make_extractor()
    out = extractor(random_batch(2))
    assert out.dtype == torch.float32, f"Expected float32, got {out.dtype}"
    print("PASS test_output_dtype")


def test_single_sample():
    """Batch size of 1 works correctly."""
    extractor = make_extractor()
    out = extractor(random_batch(1))
    assert out.shape == (1, 256), f"Expected (1, 256), got {out.shape}"
    print("PASS test_single_sample")


def test_real_obs_from_env():
    """Extractor runs without error on a real observation from CNNEnv."""
    env = CNNEnv()
    obs, _ = env.reset()
    extractor = make_extractor()
    tensor = torch.tensor(obs).unsqueeze(0)  # add batch dim
    out = extractor(tensor)
    assert out.shape == (1, 256)
    assert not torch.isnan(out).any()
    print("PASS test_real_obs_from_env")


if __name__ == "__main__":
    test_output_shape_default()
    test_output_shape_custom_features_dim()
    test_output_shape_custom_n_filters()
    test_n_filters_changes_parameter_count()
    test_no_nans_or_infs()
    test_output_dtype()
    test_single_sample()
    test_real_obs_from_env()
    print("\nAll tests passed.")
