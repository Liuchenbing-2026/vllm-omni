# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the NPU profiler stop-schedule fix (issue #6620).

``torch_npu.profiler.profile`` installs its internal ``_default_schedule_fn``
when no ``schedule`` is supplied, and that function returns
``ProfilerAction.RECORD`` forever. ``stop()`` then always takes the
``transit_action(RECORD, None)`` branch, whose trace export can segfault on
Ascend. The NPU wrapper must install a closable schedule and finish its recording cycle before stopping, on the *default* profiler configuration.

The regression is therefore driven through the default config: no schedule is
configured by the user, as before the fix. These tests run on CPU against a
fake ``torch_npu`` module injected into ``sys.modules``.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest import mock

import pytest
from torch.profiler import ProfilerAction
from vllm.config import ProfilerConfig

from vllm_omni.platforms.npu import profiler as npu_profiler
from vllm_omni.platforms.npu.profiler import NPUTorchProfilerWrapper

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeProfiler:
    """Minimal stand-in recording the call order of the profiler API."""

    def __init__(self, kwargs: dict, calls: list):
        self.kwargs = kwargs
        self.calls = calls
        self.stopped = False
        self.step_num = 0
        schedule = kwargs.get("schedule", lambda _: ProfilerAction.RECORD)
        self.current_action = schedule(self.step_num)

    def start(self):
        self.calls.append("start")

    def step(self):
        self.calls.append("step")
        self.step_num += 1
        schedule = self.kwargs.get("schedule", lambda _: ProfilerAction.RECORD)
        self.current_action = schedule(self.step_num)

    def stop(self):
        self.calls.append("stop")


def _install_fake_torch_npu(monkeypatch, created: list, calls: list):
    def fake_profile(**kwargs):
        profiler = _FakeProfiler(kwargs, calls)
        created.append(profiler)
        return profiler

    profiler_ns = ModuleType("torch_npu.profiler")
    for name, attrs in {
        "ProfilerActivity": {"CPU": "CPU", "NPU": "NPU"},
        "ExportType": {"Text": "Text"},
        "ProfilerLevel": {"Level1": "Level1"},
        "AiCMetrics": {"PipeUtilization": "PipeUtilization"},
    }.items():
        namespace = ModuleType(name)
        for key, value in attrs.items():
            setattr(namespace, key, value)
        setattr(profiler_ns, name, namespace)
    setattr(profiler_ns, "ProfilerAction", ProfilerAction)
    setattr(profiler_ns, "_ExperimentalConfig", lambda **_: object())
    setattr(profiler_ns, "tensorboard_trace_handler", lambda *args, **kwargs: mock.Mock())
    setattr(profiler_ns, "profile", fake_profile)
    npu_module = ModuleType("torch_npu")
    setattr(npu_module, "profiler", profiler_ns)
    monkeypatch.setitem(sys.modules, "torch_npu", npu_module)
    # ``from torch_npu.profiler import ...`` resolves the submodule through
    # sys.modules, so the fake has to be registered under both names.
    monkeypatch.setitem(sys.modules, "torch_npu.profiler", profiler_ns)


def _build_wrapper(tmp_path, monkeypatch, created: list, calls: list):
    _install_fake_torch_npu(monkeypatch, created, calls)
    return NPUTorchProfilerWrapper(
        ProfilerConfig(profiler="torch", torch_profiler_dir=str(tmp_path)),
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
    assert all(schedule(step) is ProfilerAction.RECORD for step in [0, 1, 100, 10000])


def test_stop_closes_the_window_before_stopping(tmp_path, monkeypatch):
    """``stop`` must finish the recording cycle before backend ``stop``."""
    created: list = []
    calls: list = []
    wrapper = _build_wrapper(tmp_path, monkeypatch, created, calls)
    schedule = created[0].kwargs["schedule"]

    wrapper.start()
    assert schedule(1) is ProfilerAction.RECORD, "must keep recording until stop_profile"

    wrapper.stop()

    assert calls == ["start", "step", "step", "stop"]
    assert wrapper.profiler.current_action is ProfilerAction.NONE
    assert schedule(2) is ProfilerAction.NONE
    assert not wrapper.is_running


def test_first_window_reuses_the_init_instance(tmp_path, monkeypatch):
    """The instance built by ``__init__`` must not be discarded unstarted."""
    created: list = []
    calls: list = []
    wrapper = _build_wrapper(tmp_path, monkeypatch, created, calls)

    wrapper.start()
    wrapper.stop()

    assert len(created) == 1
    assert wrapper.profiler is created[0]


def test_second_window_rebuilds_profiler_and_schedule(tmp_path, monkeypatch):
    """A new capture window needs a fresh, open schedule after stop."""
    created: list = []
    calls: list = []
    wrapper = _build_wrapper(tmp_path, monkeypatch, created, calls)

    wrapper.start()
    wrapper.stop()
    first = wrapper.profiler
    first_schedule = created[0].kwargs["schedule"]

    wrapper.start()

    assert len(created) == 2
    assert wrapper.profiler is created[1]
    assert wrapper.profiler is not first
    # The rebuilt schedule starts open again, so the new window records.
    assert created[1].kwargs["schedule"] is not first_schedule
    assert created[1].kwargs["schedule"](0) is ProfilerAction.RECORD

    wrapper.stop()
    assert calls == ["start", "step", "step", "stop", "start", "step", "step", "stop"]


def test_public_start_stop_are_idempotent(tmp_path, monkeypatch):
    created: list = []
    calls: list = []
    wrapper = _build_wrapper(tmp_path, monkeypatch, created, calls)

    wrapper.stop()
    wrapper.start()
    wrapper.start()
    wrapper.stop()
    wrapper.stop()

    assert calls == ["start", "step", "step", "stop"]
    assert len(created) == 1
    assert not wrapper.is_running


def test_closed_schedule_finishes_after_one_save_step(monkeypatch):
    _install_fake_torch_npu(monkeypatch, [], [])
    schedule = npu_profiler._ClosableRecordSchedule()
    assert schedule(10000) is ProfilerAction.RECORD
    schedule.close()
    assert schedule(10001) is ProfilerAction.RECORD_AND_SAVE
    assert schedule(10001) is ProfilerAction.RECORD_AND_SAVE
    assert schedule(10002) is ProfilerAction.NONE
    assert schedule(10003) is ProfilerAction.NONE
