# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the NPU profiler stop-schedule fix (issue #6620).

``torch_npu.profiler.profile`` installs its internal ``_default_schedule_fn``
when no ``schedule`` is supplied, and that function returns
``ProfilerAction.RECORD`` forever. ``stop()`` then always takes the
``transit_action(RECORD, None)`` branch, whose trace export can segfault on
Ascend. The NPU wrapper must install a closable schedule and close it (one
``step()``) before stopping, on the *default* profiler configuration.

The regression is therefore driven through the default config: no schedule is
configured by the user, as before the fix. These tests run on CPU against a
fake ``torch_npu`` module injected into ``sys.modules``.
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
    torch_profiler_with_memory: bool = False
    torch_profiler_with_stack: bool = False
    torch_profiler_with_flops: bool = False
    torch_profiler_dump_cuda_time_total: bool = False


@pytest.fixture(autouse=True)
def patch_worker_profiler_init(monkeypatch):
    def fake_init(self, profiler_config):
        self.profiler_config = profiler_config
        self._running = False

    monkeypatch.setattr(base_mod.WorkerProfiler, "__init__", fake_init)


class _FakeProfiler:
    """Minimal stand-in recording the call order of the profiler API."""

    def __init__(self, kwargs: dict, calls: list):
        self.kwargs = kwargs
        self.calls = calls
        self.stopped = False

    def start(self):
        self.calls.append("start")

    def step(self):
        self.calls.append("step")

    def stop(self):
        self.calls.append("stop")


def _install_fake_torch_npu(monkeypatch, created: list, calls: list):
    class _FakeProfilerAction:
        RECORD = "RECORD"
        RECORD_AND_SAVE = "RECORD_AND_SAVE"

    def fake_profile(**kwargs):
        profiler = _FakeProfiler(kwargs, calls)
        created.append(profiler)
        return profiler

    profiler_ns = SimpleNamespace(
        ProfilerActivity=SimpleNamespace(CPU="CPU", NPU="NPU"),
        ExportType=SimpleNamespace(Text="Text"),
        ProfilerLevel=SimpleNamespace(Level1="Level1"),
        AiCMetrics=SimpleNamespace(PipeUtilization="PipeUtilization"),
        ProfilerAction=_FakeProfilerAction,
        _ExperimentalConfig=lambda **_: SimpleNamespace(),
        tensorboard_trace_handler=lambda *args, **kwargs: mock.Mock(),
        profile=fake_profile,
    )
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(profiler=profiler_ns))
    # ``from torch_npu.profiler import ...`` resolves the submodule through
    # sys.modules, so the fake has to be registered under both names.
    monkeypatch.setitem(sys.modules, "torch_npu.profiler", profiler_ns)


def _build_wrapper(tmp_path, monkeypatch, created: list, calls: list):
    _install_fake_torch_npu(monkeypatch, created, calls)
    return NPUTorchProfilerWrapper(
        DummyProfilerConfig(torch_profiler_dir=str(tmp_path)),
        worker_name="stage_0",
        local_rank=0,
    )


def test_default_config_installs_a_schedule(tmp_path, monkeypatch):
    """No user-configured wait/warmup window: a schedule must still be passed.

    Without it torch_npu falls back to ``_default_schedule_fn`` and the
    profiler can only ever be stopped mid-RECORD.
    """
    created: list = []
    calls: list = []
    _build_wrapper(tmp_path, monkeypatch, created, calls)

    assert created, "the wrapper must create a torch_npu profiler"
    schedule = created[0].kwargs.get("schedule")
    assert schedule is not None, "NPU wrapper must not rely on _default_schedule_fn"
    assert schedule(0) == "RECORD"


def test_stop_closes_the_window_before_stopping(tmp_path, monkeypatch):
    """``_stop`` must leave RECORD via one ``step()`` before ``stop()``."""
    created: list = []
    calls: list = []
    wrapper = _build_wrapper(tmp_path, monkeypatch, created, calls)
    schedule = created[0].kwargs["schedule"]

    wrapper._start()
    assert schedule(1) == "RECORD", "must keep recording until stop_profile"

    wrapper._stop()

    assert calls == ["start", "step", "stop"]
    assert schedule(2) == "RECORD_AND_SAVE"


def test_first_window_reuses_the_init_instance(tmp_path, monkeypatch):
    """The instance built by ``__init__`` must not be discarded unstarted."""
    created: list = []
    calls: list = []
    wrapper = _build_wrapper(tmp_path, monkeypatch, created, calls)

    wrapper._start()
    wrapper._stop()

    assert len(created) == 1
    assert wrapper.profiler is created[0]


def test_second_window_rebuilds_profiler_and_schedule(tmp_path, monkeypatch):
    """Each later window receives a fresh profiler and an open schedule."""
    created: list = []
    calls: list = []
    wrapper = _build_wrapper(tmp_path, monkeypatch, created, calls)

    wrapper._start()
    wrapper._stop()
    first = wrapper.profiler
    first_schedule = created[0].kwargs["schedule"]

    wrapper._start()

    assert len(created) == 2
    assert wrapper.profiler is created[1]
    assert wrapper.profiler is not first
    # The rebuilt schedule starts open again, so the new window records.
    assert created[1].kwargs["schedule"] is not first_schedule
    assert created[1].kwargs["schedule"](0) == "RECORD"

    wrapper._stop()
    assert calls == ["start", "step", "stop", "start", "step", "stop"]
