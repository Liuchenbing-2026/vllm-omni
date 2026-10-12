# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
from vllm.config import FaultToleranceConfig

from vllm_omni.engine.stage_init_utils import _normalize_fault_tolerance_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_serialized_default_dict_is_dropped_when_fault_tolerance_is_off():
    """An untouched serialized default must not read as an explicit opt-in."""
    engine_args = {"enable_fault_tolerance": False, "fault_tolerance_config": {"engine_recovery_timeout_sec": 120}}

    _normalize_fault_tolerance_config(engine_args)

    assert "fault_tolerance_config" not in engine_args
    assert engine_args["enable_fault_tolerance"] is False


def test_missing_enable_flag_still_drops_the_serialized_default():
    engine_args = {"fault_tolerance_config": {"engine_recovery_timeout_sec": 120}}

    _normalize_fault_tolerance_config(engine_args)

    assert "fault_tolerance_config" not in engine_args


def test_explicit_opt_in_reconstructs_the_dataclass():
    engine_args = {"enable_fault_tolerance": True, "fault_tolerance_config": {"engine_recovery_timeout_sec": 5}}

    _normalize_fault_tolerance_config(engine_args)

    assert isinstance(engine_args["fault_tolerance_config"], FaultToleranceConfig)
    assert engine_args["fault_tolerance_config"].engine_recovery_timeout_sec == 5


def test_already_constructed_config_is_left_untouched():
    config = FaultToleranceConfig(engine_recovery_timeout_sec=7)
    engine_args = {"enable_fault_tolerance": True, "fault_tolerance_config": config}

    _normalize_fault_tolerance_config(engine_args)

    assert engine_args["fault_tolerance_config"] is config


def test_absent_field_is_a_no_op():
    engine_args = {"model": "dummy"}

    _normalize_fault_tolerance_config(engine_args)

    assert engine_args == {"model": "dummy"}
