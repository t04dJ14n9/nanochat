"""CPU-safe tests for the FSDP2 support helpers."""

import pytest
import torch

from nanochat import core_eval
from nanochat.parallelism import parameter_numel, setup_fsdp_adamw


def test_fsdp_adamw_groups_parameters_once():
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 4, bias=True),
        torch.nn.LayerNorm(4),
    )
    optimizer = setup_fsdp_adamw(model, lr=2e-4, weight_decay=0.2)

    optimized_params = [param for group in optimizer.param_groups for param in group["params"]]
    assert len(optimized_params) == len(list(model.parameters()))
    assert len({id(param) for param in optimized_params}) == len(optimized_params)
    assert all(group["kind"] == "adamw" for group in optimizer.param_groups)
    assert all(group["initial_lr"] == 2e-4 for group in optimizer.param_groups)
    assert optimizer.defaults["foreach"] is False

    decay_by_param = {
        id(param): group["weight_decay"]
        for group in optimizer.param_groups
        for param in group["params"]
    }
    for param in model.parameters():
        expected_decay = 0.2 if param.ndim >= 2 else 0.0
        assert decay_by_param[id(param)] == expected_decay


def test_parameter_numel_on_unsharded_model():
    model = torch.nn.Linear(7, 3, bias=True)
    expected = sum(param.numel() for param in model.parameters())
    assert parameter_numel(model) == expected
    assert parameter_numel(model, local=True) == expected


def test_model_parallel_core_eval_runs_every_example(monkeypatch):
    visited = []

    def fake_evaluate_example(idx, model, tokenizer, data, device, task_meta):
        visited.append(idx)
        return idx != 1

    monkeypatch.setattr(core_eval, "evaluate_example", fake_evaluate_example)
    monkeypatch.setattr(core_eval.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(core_eval.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(core_eval.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        core_eval.dist,
        "all_reduce",
        lambda *args, **kwargs: pytest.fail("model-parallel evaluation must not data-reduce replicated scores"),
    )

    accuracy = core_eval.evaluate_task(
        model=object(),
        tokenizer=object(),
        data=[object(), object(), object()],
        device=torch.device("cpu"),
        task_meta={},
        model_parallel=True,
    )

    assert visited == [0, 1, 2]
    assert accuracy == pytest.approx(2 / 3)
