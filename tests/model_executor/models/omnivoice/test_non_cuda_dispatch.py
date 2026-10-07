# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression coverage for non-CUDA OmniVoice kernel dispatch."""

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture
def non_cuda_kernels(monkeypatch):
    from vllm_omni.model_executor.models.omnivoice import omnivoice_generator

    def reject_cuda_kernel(*args, **kwargs):
        raise AssertionError("Non-CUDA tensor dispatched to a CUDA Triton kernel")

    monkeypatch.setattr(omnivoice_generator, "_TRITON_AVAILABLE", True)
    for name in ("triton_rms_norm", "triton_swiglu", "triton_fused_add_rms_norm"):
        monkeypatch.setattr(omnivoice_generator, name, reject_cuda_kernel, raising=False)
    return omnivoice_generator


@pytest.mark.core_model
@pytest.mark.cpu
def test_cpu_rmsnorm_with_triton_installed(non_cuda_kernels):
    torch.manual_seed(0)
    norm = non_cuda_kernels.OmniVoiceRMSNorm(16)
    x = torch.randn(2, 3, 16)
    expected = F.rms_norm(x, (16,), norm.weight, norm.eps)
    torch.testing.assert_close(norm(x), expected, atol=1e-6, rtol=1e-6)


@pytest.mark.core_model
@pytest.mark.cpu
def test_cpu_mlp_with_triton_installed(non_cuda_kernels):
    from vllm_omni.transformers_utils.configs.omnivoice import OmniVoiceConfig

    torch.manual_seed(1)
    config = OmniVoiceConfig(llm_config={"hidden_size": 16, "intermediate_size": 32})
    mlp = non_cuda_kernels.OmniVoiceMLP(config)
    x = torch.randn(2, 3, 16)
    expected = F.linear(
        F.silu(F.linear(x, mlp.gate_proj.weight)) * F.linear(x, mlp.up_proj.weight), mlp.down_proj.weight
    )
    torch.testing.assert_close(mlp(x), expected, atol=1e-6, rtol=1e-6)


@pytest.mark.core_model
@pytest.mark.cpu
def test_cpu_block_with_triton_installed(non_cuda_kernels, monkeypatch):
    from vllm_omni.transformers_utils.configs.omnivoice import OmniVoiceConfig

    torch.manual_seed(2)
    monkeypatch.setattr(non_cuda_kernels, "OmniVoiceAttention", lambda *args: torch.nn.Identity())
    config = OmniVoiceConfig(llm_config={"hidden_size": 16, "intermediate_size": 32})
    block = non_cuda_kernels.OmniVoiceTransformerBlock(config)
    monkeypatch.setattr(block.self_attn, "forward", lambda hidden, **kwargs: hidden.sin())
    x = torch.randn(2, 3, 16)
    normed = F.rms_norm(x, (16,), block.input_layernorm.weight, block.input_layernorm.eps)
    residual = x + normed.sin()
    post_norm = F.rms_norm(residual, (16,), block.post_attention_layernorm.weight, block.post_attention_layernorm.eps)
    expected = residual + F.linear(
        F.silu(F.linear(post_norm, block.mlp.gate_proj.weight)) * F.linear(post_norm, block.mlp.up_proj.weight),
        block.mlp.down_proj.weight,
    )
    torch.testing.assert_close(block(x), expected, atol=1e-6, rtol=1e-6)
