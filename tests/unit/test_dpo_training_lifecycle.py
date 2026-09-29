"""Regression tests for stale success, interrupted checkpoints and saved progress."""
import json
import sys
from types import SimpleNamespace

import pytest

from scenesmith.scene_expert.slow_memory.training import apply_training_profile
from scenesmith.scene_expert.slow_memory.training_lifecycle import (
    checkpoint_complete, check_training_completion, make_progress_callback,
)


def test_pilot_saves_each_step_and_evaluates_once():
    config = apply_training_profile({}, 'furniture_initial_pilot')
    assert config['training']['save_steps'] == 1
    assert config['training']['eval_strategy'] == 'no'
    assert config['training']['gradient_accumulation_steps'] == 4
    assert config['training']['num_train_epochs'] == 2


def test_stale_manifest_and_partial_schedule_are_not_success(tmp_path):
    (tmp_path/'training_manifest.json').write_text(json.dumps({
        'execution_id': 'old',
        'completion': {'completed': True, 'optimizer_steps': 4, 'expected_optimizer_steps': 8},
    }))
    check = check_training_completion(tmp_path, 'new')
    assert not check['completed']
    assert 'training manifest belongs to another invocation' in check['errors']
    assert 'optimizer schedule is incomplete' in check['errors']


def test_callback_marks_only_fully_saved_checkpoints_and_persists_metrics(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(TrainerCallback=object))
    cb = make_progress_callback(tmp_path)
    state = SimpleNamespace(global_step=1, max_steps=8, epoch=0.25)
    cb.on_log(None, state, None, logs={'loss':0.69, 'grad_norm':0.2})
    assert json.loads((tmp_path/'training_progress.json').read_text())['metrics']['loss'] == 0.69
    ckpt = tmp_path/'checkpoint-1'
    ckpt.mkdir()
    with pytest.raises(ValueError, match='incomplete resume'):
        cb.on_save(None, state, None)
    assert not (ckpt/'checkpoint_complete.json').exists()
    for name in ('optimizer.pt','scheduler.pt','adapter_config.json','adapter_model.safetensors','rng_state.pth'):
        (ckpt/name).write_bytes(b'fixture')
    (ckpt/'trainer_state.json').write_text('{"global_step":1}')
    cb.on_save(None,state,None)
    checkpoint_complete(ckpt)
    (ckpt/'trainer_state.json').write_text('{"global_step":2}')
    with pytest.raises(ValueError, match='does not match'):
        checkpoint_complete(ckpt)
