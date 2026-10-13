# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the NPU torch profiler wrapper (issue #8877).

``torch_profiler_with_stack`` must be forwarded to
``torch_npu.profiler.profile`` as the native ``with_stack`` kwarg. The option
used to be hard-coded to ``False``, so a user asking for Python stacks got a
trace without them. These tests run on CPU with a fake ``torch_npu`` module
injected into ``sys.modules``; no real NPU or kernel is required.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from types import SimpleNamespace
from unittest import mock

import pytest

import vllm_omni.profiler.omni_torch_profiler as base_mod
from vllm_omni.platforms.npu.profiler import NPUTorchProfilerWrapper

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class DummyProfilerConfig:
    torch_profiler_dir: str
    torch_profiler_use_gzip: bool = False
    torch_profiler_record_shapes: bool = True
    torch_profiler_with_memory: bool = True
    torch_profiler_with_stack: bool = True
    torch_profiler_with_flops: bool = False
    torch_profiler_dump_cuda_time_total: bool = False


@pytest.fixture(autouse=True)
def patch_worker_profiler_init(monkeypatch):
    def fake_init(self, profiler_config):
        self.profiler_config = profiler_config

    monkeypatch.setattr(base_mod.WorkerProfiler, "__init__", fake_init)


def _install_fake_torch_npu(monkeypatch, captured: dict):
    """Inject a minimal ``torch_npu.profiler`` surface into sys.modules."""

    def fake_profile(**kwargs):
        captured.update(kwargs)
        return mock.Mock()

    profiler = SimpleNamespace(
        ProfilerActivity=SimpleNamespace(CPU="CPU", NPU="NPU"),
        ExportType=SimpleNamespace(Text="Text"),
        ProfilerLevel=SimpleNamespace(Level1="Level1"),
        AiCMetrics=SimpleNamespace(PipeUtilization="PipeUtilization"),
        _ExperimentalConfig=lambda **_: mock.Mock(),
        tensorboard_trace_handler=lambda *args, **kwargs: mock.Mock(),
        profile=fake_profile,
    )
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(profiler=profiler))


def _build_wrapper(tmp_path, monkeypatch, captured: dict, with_stack: bool):
    _install_fake_torch_npu(monkeypatch, captured)
    config = DummyProfilerConfig(
        torch_profiler_dir=str(tmp_path),
        torch_profiler_with_stack=with_stack,
    )
    return NPUTorchProfilerWrapper(config, worker_name="stage_0", local_rank=0)


def test_with_stack_true_is_forwarded(tmp_path, monkeypatch):
    """A user asking for stacks must reach ``torch_npu`` with ``with_stack=True``."""
    captured: dict = {}
    _build_wrapper(tmp_path, monkeypatch, captured, with_stack=True)
    assert captured["with_stack"] is True
    assert captured["with_modules"] is True


def test_with_stack_false_is_forwarded(tmp_path, monkeypatch):
    captured: dict = {}
    _build_wrapper(tmp_path, monkeypatch, captured, with_stack=False)
    assert captured["with_stack"] is False
    assert captured["with_modules"] is False


def test_record_shapes_still_forwarded(tmp_path, monkeypatch):
    """Guard the #8770 forwarding against regressions from this change."""
    captured: dict = {}
    _build_wrapper(tmp_path, monkeypatch, captured, with_stack=True)
    assert captured["record_shapes"] is True
