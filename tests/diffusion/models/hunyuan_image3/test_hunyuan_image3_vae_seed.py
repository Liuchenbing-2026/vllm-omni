# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts for request-local HunyuanImage3 VAE sampling.

Use the real processor, multimodal grouping, model methods and posterior sampler.
Only the weight-dependent VAE/ViT/projection forwards use tiny deterministic
fixtures; this does not test pretrained-model image quality or a real cache.
"""

import pytest
import torch
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from transformers import PretrainedConfig
from vllm.multimodal.inputs import MultiModalKwargsItem
from vllm.multimodal.utils import group_and_batch_mm_kwargs

from vllm_omni.model_executor.models.hunyuan_image3.autoencoder_kl_3d import DiagonalGaussianDistribution
from vllm_omni.model_executor.models.hunyuan_image3.hunyuan_image3 import (
    HunyuanImage3ForConditionalGeneration,
    HunyuanImage3MultiModalProcessor,
    HunyuanImage3ProcessingInfo,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Posterior(DiagonalGaussianDistribution):
    def __init__(self, parameters: torch.Tensor, seeds: list[int | None]):
        super().__init__(parameters)
        self.seeds = seeds

    def sample(self, generator: torch.Generator | None = None):
        self.seeds.append(None if generator is None else generator.initial_seed())
        return super().sample(generator)


class _TinyVAE(torch.nn.Module):
    dtype = torch.float32

    def __init__(self):
        super().__init__()
        self.config = PretrainedConfig()
        self.seeds: list[int | None] = []

    def encode(self, images: torch.Tensor):
        mean = images[:, :1]
        posterior = _Posterior(torch.cat([mean, torch.zeros_like(mean)], dim=1), self.seeds)
        return AutoencoderKLOutput(latent_dist=posterior)


class _TokenEmbedding(torch.nn.Module):
    def embed_input_ids(self, ids: torch.Tensor):
        return torch.zeros((ids.numel(), 1))


class _Info(HunyuanImage3ProcessingInfo):
    def __init__(self, pixels: torch.Tensor):
        self.pixels = pixels

    def get_image_processor(self, **kwargs):
        def process(prompt, images, **tok_kwargs):
            count = len(images)
            return {
                "vit_pixel_values": torch.zeros((count, 1, 1)),
                "vit_pixel_attention_mask": torch.ones((count, 1), dtype=torch.bool),
                "vit_spatial_shapes": torch.ones((count, 2), dtype=torch.long),
                "vae_pixel_values": self.pixels.flatten(),
                "vae_pixel_size": torch.full((count,), 12, dtype=torch.long),
                "vae_token_grid_hw": torch.full((count, 2), 2, dtype=torch.long),
            }

        return process


def _item(marker: float, seed: int | None, count: int = 1):
    processor = object.__new__(HunyuanImage3MultiModalProcessor)
    processor.info = _Info(torch.full((count, 3, 2, 2), marker))
    kwargs = {} if seed is None else {"vae_generator_seed": seed}
    features = processor._call_hf_processor("", {"images": [object()] * count}, kwargs, {})
    configs = processor._get_mm_fields_config(features, kwargs)
    return [
        (
            "image",
            MultiModalKwargsItem({key: config.build_elems(key, features[key])[i] for key, config in configs.items()}),
        )
        for i in range(count)
    ]


def _model():
    model = object.__new__(HunyuanImage3ForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.config = PretrainedConfig(vae_downsample_factor=(1, 1), patch_size=1)
    model.vae = _TinyVAE()
    model.model = _TokenEmbedding()
    model._mrope_joint_img_sep_token_id = 0
    model._vit_encode = lambda pixels, mask, shapes: torch.zeros((pixels.shape[0], 1, 1))
    model.time_embed = lambda t: torch.zeros((1, 1))
    model.patch_embed = lambda latents, t: (latents.flatten(2).transpose(1, 2), None, None)
    return model


def _batch(requests: list[tuple[float, int | None]]):
    items = [item for marker, seed in requests for item in _item(marker, seed)]
    outputs, used_seeds = [], []
    groups = list(group_and_batch_mm_kwargs(items, device="cpu", pin_memory=False))
    for _, count, kwargs in groups:
        model = _model()
        outputs.extend(model.embed_multimodal(**kwargs))
        used_seeds.extend(model.vae.seeds)
        assert len(model.vae.seeds) == count
    return outputs, used_seeds, len(groups)


@pytest.mark.parametrize(
    "requests",
    [
        [(1.0, 42), (2.0, 123)],
        [(2.0, 123), (1.0, 42)],
        [(1.0, 42), (3.0, 42)],
        [(3.0, 42), (1.0, 42)],
        [(1.0, 42), (2.0, 123), (3.0, 0)],
    ],
)
def test_batch_composition_does_not_change_request_sample(requests):
    actual, seeds, groups = _batch(requests)
    assert groups == 1
    assert seeds == [seed for _, seed in requests]
    for output, (marker, seed) in zip(actual, requests):
        alone, _, _ = _batch([(marker, seed)])
        assert torch.equal(output, alone[0])


@pytest.mark.parametrize("seed", [0, 42, 2**63 - 1])
def test_single_image_matches_independent_cpu_rng(seed):
    outputs, seeds, _ = _batch([(2.0, seed)])
    expected = 2.0 + torch.randn((1, 1, 2, 2), generator=torch.Generator().manual_seed(seed))
    assert seeds == [seed]
    assert torch.equal(outputs[0][:4], expected.flatten().unsqueeze(1))


def test_omitting_another_request_does_not_change_sample():
    full, _, _ = _batch([(1.0, 42), (2.0, 123)])
    miss_subset, _, _ = _batch([(2.0, 123)])
    assert torch.equal(full[1], miss_subset[0])


def test_seeded_requests_do_not_advance_global_rng():
    before = torch.random.get_rng_state()
    _batch([(1.0, 42), (2.0, 123)])
    assert torch.equal(before, torch.random.get_rng_state())


def test_seeded_and_unseeded_fields_form_separate_groups():
    _, seeds, groups = _batch([(1.0, 42), (3.0, None)])
    assert groups == 2
    assert seeds == [42, None]


def test_unseeded_images_keep_global_rng_stream():
    generator = torch.Generator().manual_seed(777)
    expected = [marker + torch.randn((1, 1, 2, 2), generator=generator) for marker in [1.0, 2.0]]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(777)
        outputs, seeds, groups = _batch([(1.0, None), (2.0, None)])
    assert groups == 1
    assert seeds == [None, None]
    for output, latent in zip(outputs, expected):
        assert torch.equal(output[:4], latent.flatten().unsqueeze(1))


@pytest.mark.parametrize("count", [1, 2, 3])
def test_processor_exposes_one_seed_per_image(count):
    groups = list(group_and_batch_mm_kwargs(_item(1.0, 42, count), device="cpu", pin_memory=False))
    assert len(groups) == 1
    assert groups[0][2]["vae_generator_seed"].tolist() == [42] * count


def test_multi_image_fields_remain_aligned_when_only_one_item_is_encoded():
    items = _item(2.0, 42, count=2)
    full_group = list(group_and_batch_mm_kwargs(items, device="cpu", pin_memory=False))
    subset_group = list(group_and_batch_mm_kwargs(items[1:], device="cpu", pin_memory=False))
    full_model, subset_model = _model(), _model()
    full = full_model.embed_multimodal(**full_group[0][2])
    subset = subset_model.embed_multimodal(**subset_group[0][2])
    assert len(full) == 2 and len(subset) == 1
    assert full_model.vae.seeds == [42, 42] and subset_model.vae.seeds == [42]
    assert torch.equal(full[1], subset[0])


def test_empty_multimodal_input_is_unchanged():
    assert _model().embed_multimodal() == []
