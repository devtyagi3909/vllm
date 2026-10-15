# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Script to test add_lora, remove_lora, pin_lora, list_loras functions."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config.lora import LoRAConfig
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.entrypoints.launchers.api_server.entry import (
    build_async_engine_client_from_engine_args,
)
from vllm.lora.request import LoRARequest
from vllm.v1.engine.llm_engine import LLMEngine

MODEL_PATH = "Qwen/Qwen3-0.6B"
LORA_MODULE_PATH = "charent/self_cognition_Alice"
LORA_RANK = 8


def make_lora_request(lora_id: int):
    return LoRARequest(
        lora_name=f"{lora_id}", lora_int_id=lora_id, lora_path=LORA_MODULE_PATH
    )


def test_lora_functions_sync():
    max_loras = 4
    # Create engine in eager-mode. Due to high max_loras, the CI can
    # OOM during cuda-graph capture.
    engine_args = EngineArgs(
        model=MODEL_PATH,
        enable_lora=True,
        max_loras=max_loras,
        max_lora_rank=LORA_RANK,
        max_model_len=128,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
    )

    llm = LLMEngine.from_engine_args(engine_args)

    def run_check(fn, args, expected: list):
        fn(args)
        assert set(llm.list_loras()) == set(expected)

    run_check(llm.add_lora, make_lora_request(1), [1])
    run_check(llm.add_lora, make_lora_request(2), [1, 2])

    # Pin LoRA 1 and test that it is never removed on subsequent adds.
    run_check(llm.pin_lora, 1, [1, 2])
    run_check(llm.add_lora, make_lora_request(3), [1, 2, 3])
    run_check(llm.add_lora, make_lora_request(4), [1, 2, 3, 4])
    run_check(llm.add_lora, make_lora_request(5), [1, 5, 3, 4])
    run_check(llm.add_lora, make_lora_request(6), [1, 5, 6, 4])
    run_check(llm.add_lora, make_lora_request(7), [1, 5, 6, 7])
    run_check(llm.add_lora, make_lora_request(8), [1, 8, 6, 7])
    run_check(llm.add_lora, make_lora_request(9), [1, 8, 9, 7])
    run_check(llm.add_lora, make_lora_request(10), [1, 8, 9, 10])

    # Remove LoRA 1 and continue adding.
    run_check(llm.remove_lora, 1, [8, 9, 10])
    run_check(llm.add_lora, make_lora_request(11), [8, 9, 10, 11])
    run_check(llm.add_lora, make_lora_request(12), [12, 9, 10, 11])
    run_check(llm.add_lora, make_lora_request(13), [12, 13, 10, 11])

    # Remove all LoRAs.
    run_check(llm.remove_lora, 13, [12, 10, 11])
    run_check(llm.remove_lora, 12, [10, 11])
    run_check(llm.remove_lora, 11, [10])
    run_check(llm.remove_lora, 10, [])


@pytest.mark.asyncio
async def test_lora_functions_async():
    max_loras = 4
    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        enable_lora=True,
        max_loras=max_loras,
        max_lora_rank=LORA_RANK,
        max_model_len=128,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
    )

    async def run_check(fn, args, expected: list):
        await fn(args)
        assert set(await llm.list_loras()) == set(expected)

    async with build_async_engine_client_from_engine_args(engine_args) as llm:
        await run_check(llm.add_lora, make_lora_request(1), [1])
        await run_check(llm.add_lora, make_lora_request(2), [1, 2])

        # Pin LoRA 1 and test that it is never removed on subsequent adds.
        await run_check(llm.pin_lora, 1, [1, 2])
        await run_check(llm.add_lora, make_lora_request(3), [1, 2, 3])
        await run_check(llm.add_lora, make_lora_request(4), [1, 2, 3, 4])
        await run_check(llm.add_lora, make_lora_request(5), [1, 5, 3, 4])
        await run_check(llm.add_lora, make_lora_request(6), [1, 5, 6, 4])
        await run_check(llm.add_lora, make_lora_request(7), [1, 5, 6, 7])
        await run_check(llm.add_lora, make_lora_request(8), [1, 8, 6, 7])
        await run_check(llm.add_lora, make_lora_request(9), [1, 8, 9, 7])
        await run_check(llm.add_lora, make_lora_request(10), [1, 8, 9, 10])

        # Remove LoRA 1 and continue adding.
        await run_check(llm.remove_lora, 1, [8, 9, 10])
        await run_check(llm.add_lora, make_lora_request(11), [8, 9, 10, 11])
        await run_check(llm.add_lora, make_lora_request(12), [12, 9, 10, 11])
        await run_check(llm.add_lora, make_lora_request(13), [12, 13, 10, 11])

        # Remove all LoRAs
        await run_check(llm.remove_lora, 13, [12, 10, 11])
        await run_check(llm.remove_lora, 12, [10, 11])
        await run_check(llm.remove_lora, 11, [10])
        await run_check(llm.remove_lora, 10, [])


def test_lora_config_dtype_validation():
    # Model dtype float32 should fast-fail with ValueError when LoRA is enabled
    lora_config = LoRAConfig(max_loras=1, max_lora_rank=8)
    model_config_fp32 = SimpleNamespace(dtype=torch.float32, architectures=[])
    with pytest.raises(ValueError, match="LoRA only supports float16 and bfloat16"):
        lora_config.verify_with_model_config(model_config_fp32)

    # Explicit lora_dtype float32 should also fail
    lora_config_explicit_fp32 = LoRAConfig(
        max_loras=1, max_lora_rank=8, lora_dtype=torch.float32
    )
    with pytest.raises(ValueError, match="LoRA only supports float16 and bfloat16"):
        lora_config_explicit_fp32.verify_with_model_config(model_config_fp32)

    # Valid dtypes should succeed
    lora_config_fp16 = LoRAConfig(max_loras=1, max_lora_rank=8, lora_dtype="float16")
    model_config_fp16 = SimpleNamespace(dtype=torch.float16, architectures=[])
    lora_config_fp16.verify_with_model_config(model_config_fp16)
    assert lora_config_fp16.lora_dtype == torch.float16

    lora_config_bf16 = LoRAConfig(max_loras=1, max_lora_rank=8)
    model_config_bf16 = SimpleNamespace(dtype=torch.bfloat16, architectures=[])
    lora_config_bf16.verify_with_model_config(model_config_bf16)
    assert lora_config_bf16.lora_dtype == torch.bfloat16


def test_lora_ops_dtype_validation():
    from vllm.lora.ops.triton_ops.lora_expand_op import _lora_expand
    from vllm.lora.ops.triton_ops.lora_shrink_op import _lora_shrink

    # Test lora_expand rejects invalid weight dtype
    inputs = torch.zeros((1, 10, 16), dtype=torch.float16)
    fp32_weight = torch.zeros((16, 16), dtype=torch.float32)
    no_lora_flag = torch.tensor([False])
    num_active = torch.tensor([1])
    with pytest.raises(
        ValueError, match="lora_b_weights must have dtype float16 or bfloat16"
    ):
        _lora_expand(
            inputs,
            [fp32_weight],
            torch.zeros((10, 16), dtype=torch.float16),
            torch.zeros(10, dtype=torch.long),
            torch.zeros(10, dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
            torch.zeros(2, dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
            no_lora_flag,
            num_active,
        )

    # Test lora_shrink rejects mismatched dtype
    with pytest.raises(
        ValueError, match="must match lora_a_weights dtype"
    ):
        _lora_shrink(
            inputs,
            [fp32_weight],
            torch.zeros((1, 10, 8), dtype=torch.float32),
            torch.zeros(10, dtype=torch.long),
            torch.zeros(10, dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
            torch.zeros(2, dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
            no_lora_flag,
            num_active,
            1.0,
        )

    # Test lora_shrink rejects float32 inputs
    fp32_inputs = torch.zeros((1, 10, 16), dtype=torch.float32)
    with pytest.raises(
        ValueError, match="inputs.dtype must be float16 or bfloat16"
    ):
        _lora_shrink(
            fp32_inputs,
            [fp32_weight],
            torch.zeros((1, 10, 8), dtype=torch.float32),
            torch.zeros(10, dtype=torch.long),
            torch.zeros(10, dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
            torch.zeros(2, dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
            no_lora_flag,
            num_active,
            1.0,
        )
