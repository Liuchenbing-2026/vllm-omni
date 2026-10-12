# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Preserve resolved fault-tolerance settings when rebuilding stage arguments."""

import copy
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from vllm.config.parallel import FaultToleranceConfig

from vllm_omni.engine import stage_init_utils
from vllm_omni.engine.arg_utils import OmniEngineArgs

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def capture_stage_args(monkeypatch):
    captured = []

    def create_engine_config(args, **kwargs):
        captured.append(args)
        return SimpleNamespace(quant_config=None)

    # Keep real OmniEngineArgs/EngineArgs initialization; only skip model load
    # and executor selection, which are outside this argument-boundary test.
    monkeypatch.setattr(OmniEngineArgs, "create_engine_config", create_engine_config)
    monkeypatch.setattr(stage_init_utils.Executor, "get_class", lambda config: object)
    return captured


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("timeout", [120, 30])
def test_stage_reconstruction_preserves_serialized_fault_tolerance(tmp_path, capture_stage_args, enabled, timeout):
    config = FaultToleranceConfig(engine_recovery_timeout_sec=timeout)
    args = OmniEngineArgs(model=str(tmp_path), enable_fault_tolerance=enabled, fault_tolerance_config=config)
    # Isolate the serialized FT fields: full asdict also involves compilation
    # config normalization, which has a separate contract.
    serialized = asdict(args)
    engine_args = {
        "model": str(tmp_path),
        "enable_fault_tolerance": serialized["enable_fault_tolerance"],
        "fault_tolerance_config": serialized["fault_tolerance_config"],
    }
    original = copy.deepcopy(engine_args)

    stage_init_utils.build_vllm_config(SimpleNamespace(stage_id=0), str(tmp_path), engine_args_dict=engine_args)

    rebuilt = capture_stage_args[0]
    assert rebuilt.enable_fault_tolerance is enabled
    assert isinstance(rebuilt.fault_tolerance_config, FaultToleranceConfig)
    assert rebuilt.fault_tolerance_config.engine_recovery_timeout_sec == timeout
    assert engine_args == original


def test_explicit_fault_tolerance_dict_still_enables_without_resolved_flag(tmp_path, capture_stage_args):
    # Raw CLI/library overrides have no resolved flag; retain upstream's
    # automatic opt-in behavior, including an explicitly supplied default.
    engine_args = {"model": str(tmp_path), "fault_tolerance_config": {"engine_recovery_timeout_sec": 120}}

    stage_init_utils.build_vllm_config(SimpleNamespace(stage_id=0), str(tmp_path), engine_args_dict=engine_args)

    assert capture_stage_args[0].enable_fault_tolerance is True


def test_typed_fault_tolerance_config_is_preserved(tmp_path, capture_stage_args):
    config = FaultToleranceConfig(engine_recovery_timeout_sec=30)
    engine_args = {"model": str(tmp_path), "enable_fault_tolerance": False, "fault_tolerance_config": config}

    stage_init_utils.build_vllm_config(SimpleNamespace(stage_id=0), str(tmp_path), engine_args_dict=engine_args)

    assert capture_stage_args[0].enable_fault_tolerance is False
    assert capture_stage_args[0].fault_tolerance_config is config


def test_missing_fault_tolerance_config_keeps_default_disabled(tmp_path, capture_stage_args):
    stage_init_utils.build_vllm_config(
        SimpleNamespace(stage_id=0), str(tmp_path), engine_args_dict={"model": str(tmp_path)}
    )

    assert capture_stage_args[0].enable_fault_tolerance is False
