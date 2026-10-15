# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for fused_moe get_default_config."""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.fused_moe import get_default_config
import vllm.utils.mem_utils as mem_utils


def test_fused_moe_float32_default_config_shared_memory(monkeypatch):
    # SM89/SM86 hardware shared memory per thread block limit is 101,376 bytes
    sm89_smem_limit = 101376
    monkeypatch.setattr(
        mem_utils, "get_max_shared_memory_bytes", lambda gpu=0: sm89_smem_limit
    )
    max_smem = mem_utils.get_max_shared_memory_bytes()

    for m in [1, 16, 32, 64, 128, 256, 512, 1024]:
        for dtype in [torch.float32, "float32", "float"]:
            cfg = get_default_config(
                M=m, E=8, N=4096, K=4096, topk=2, dtype=dtype
            )
            assert cfg["BLOCK_SIZE_M"] <= 64
            assert cfg["BLOCK_SIZE_K"] == 64
            assert cfg["BLOCK_SIZE_N"] == 64
            assert cfg["num_stages"] == 2

            # Pipeline stage shared memory footprint for fused_moe kernel:
            # num_stages * (BLOCK_SIZE_M * BLOCK_SIZE_K + BLOCK_SIZE_K * BLOCK_SIZE_N) * sizeof(float32)
            bytes_per_elem = 4
            smem = (
                cfg["num_stages"]
                * (
                    cfg["BLOCK_SIZE_M"] * cfg["BLOCK_SIZE_K"]
                    + cfg["BLOCK_SIZE_K"] * cfg["BLOCK_SIZE_N"]
                )
                * bytes_per_elem
            )
            assert smem <= 65536, f"Shared memory {smem} exceeds 64 KB cap"
            assert smem <= max_smem, (
                f"Shared memory {smem} exceeds hardware limit {max_smem} "
                f"for M={m}, dtype={dtype}"
            )
            if m <= 32:
                # Decode / small batch shared memory stays ~40 KB (40,960 bytes)
                assert smem == 40960
