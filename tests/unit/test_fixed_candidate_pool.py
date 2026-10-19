"""A complete, same-request pool is one task group, with no success stopping."""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from scenesmith.scene_expert.slow_memory.candidate_pool import (
    candidate_names, pool_policy, validate_pool_execution,
)
from scenesmith.scene_expert.slow_memory.paired import (
    audit_pairs, digest, read_json, validate_pair_inputs, write_json,
)
from scenesmith.scene_expert.slow_memory.paired_runtime import InitialPair, run_shadow, shadow_environment
from tests.unit.test_initial_pairs import _group


def four_group(root: Path) -> Path:
    group = _group(root)
    shutil.copytree(group / 'B', group / 'C')
    shutil.copytree(group / 'A', group / 'D')
    snapshot = read_json(group / 'snapshot.json')
    snapshot.update(decision_boundary='first_mutation', candidate_pool=pool_policy(4))
    write_json(group / 'snapshot.json', snapshot)
    for name in ('A', 'B', 'C', 'D'):
        directory = group / name
        proof = read_json(group / 'continuation_proof.json')
        if name != 'A':
            write_json(group / f'continuation_proof_{name}.json', proof)
        result = read_json(directory / 'result.json')
        result.update(candidate=name, snapshot_hash=digest(snapshot))
        path = directory / 'slow_memory/trajectories.jsonl'
        row = json.loads(path.read_text(encoding='utf8'))
        row.update(trajectory_id=name, run_id=name, response='Placed bed ' + name,
                   response_hash=name, context_hash=digest(snapshot))
        row['spatial_context']['initial_snapshot_sha256'] = digest(snapshot)
        row['evidence']['evidence_id'] = name
        path.write_text(json.dumps(row) + '\n', encoding='utf8')
        result['trajectory_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        write_json(directory / 'result.json', result)
    write_json(group / 'candidate_pool_execution.json', {
        'policy': pool_policy(4), 'scheduled': ['B', 'C', 'D'],
        'worker_exits': {'B': 0, 'C': 0, 'D': 0},
    })
    return group


def test_four_candidates_export_one_independent_task_group(tmp_path):
    group = four_group(tmp_path)
    result = audit_pairs(tmp_path)
    assert result['execution_integrity_passed'] is True
    assert result['candidate_count'] == 4
    assert result['valid_group_count'] == 1
    assert result['independent_task_group_count'] == 1
    assert result['eligible_pair_count'] == 1
    assert read_json(group / 'candidate_pool_execution.json')['worker_exits'] == {'B': 0, 'C': 0, 'D': 0}


@pytest.mark.parametrize('change', ['request_C', 'missing_D', 'failed_C', 'proof_C', 'identity_D'])
def test_pool_cannot_hide_a_failed_omitted_or_different_candidate(tmp_path, change):
    group = four_group(tmp_path)
    if change == 'request_C':
        request = read_json(group / 'C/first_request.json')
        request['messages'].append({'role': 'user', 'content': 'Now make a bad scene'})
        write_json(group / 'C/first_request.json', request)
    elif change == 'missing_D':
        shutil.rmtree(group / 'D')
    elif change == 'failed_C':
        execution = read_json(group / 'candidate_pool_execution.json')
        execution['worker_exits']['C'] = 2
        write_json(group / 'candidate_pool_execution.json', execution)
    elif change == 'proof_C':
        proof = read_json(group / 'continuation_proof_C.json')
        proof['memory_after'] = 'mutated'
        write_json(group / 'continuation_proof_C.json', proof)
    else:
        result = read_json(group / 'D/result.json')
        result['candidate'] = 'B'
        write_json(group / 'D/result.json', result)
    sources, groups, errors = validate_pair_inputs(tmp_path)
    assert not sources and errors and groups[0]['valid'] is False


@pytest.mark.parametrize('codes', [{'B': 0, 'C': 0, 'D': 0}, {'B': 0, 'C': 2, 'D': 0}])
def test_execution_schedule_does_not_stop_after_a_winner_or_failed_worker(tmp_path, monkeypatch, codes):
    snapshot = {'decision_boundary': 'first_mutation', 'candidate_pool': pool_policy(4)}
    pair = InitialPair(tmp_path, snapshot)
    calls = []
    monkeypatch.setattr(pair, 'capture_returned', lambda *args: calls.append('A'))

    async def worker(agent, name):
        calls.append(name)
        return codes[name]

    monkeypatch.setattr(pair, '_finish_shadow', worker)
    asyncio.run(pair.finish(object(), 'native safety'))
    assert calls == ['A', 'B', 'C', 'D']
    execution = read_json(tmp_path / 'candidate_pool_execution.json')
    assert execution['worker_exits'] == codes
    assert read_json(tmp_path / 'status.json')['status'] == ('pool_failed' if codes['C'] else 'completed')


def test_legacy_remains_two_candidates_and_cannot_run_an_unapproved_extra(tmp_path):
    assert candidate_names({}) == ('A', 'B')
    write_json(tmp_path / 'snapshot.json', {})
    with pytest.raises(ValueError, match='absent from the frozen schedule'):
        asyncio.run(run_shadow(tmp_path, 'C'))
    assert not (tmp_path / 'C').exists()


@pytest.mark.parametrize('mutation', ['count', 'quality', 'early_stop', 'order'])
def test_frozen_policy_rejects_adaptive_or_unapproved_sampling(mutation):
    policy = copy.deepcopy(pool_policy(4))
    key, value = {'count': ('candidate_count', 8), 'quality': ('quality_conditioning', True),
                  'early_stop': ('early_stop_on_preference', True), 'order': ('execution_order', ['D', 'B', 'C'])}[mutation]
    policy[key] = value
    with pytest.raises(ValueError, match='frozen four-candidate'):
        candidate_names({'decision_boundary': 'first_mutation', 'candidate_pool': policy})


def test_private_memory_paths_are_candidate_specific(tmp_path):
    for name in ('B', 'C', 'D'):
        assert shadow_environment(tmp_path, name)['SCENEEXPERT_ACTIVE_MEMORY_BANK_DIR'] == str(tmp_path / name / 'private_memory')
    with pytest.raises(ValueError, match='invalid isolated shadow'):
        shadow_environment(tmp_path, 'A')


def test_real_worker_dispatch_names_and_proofs_are_candidate_specific(tmp_path, monkeypatch):
    from scenesmith.scene_expert.slow_memory import paired_runtime

    calls = []
    snapshot = {'decision_boundary': 'first_mutation', 'candidate_pool': pool_policy(4),
                'source_scene_root': str(tmp_path / 'canonical')}
    pair = InitialPair(tmp_path, snapshot)
    agent = SimpleNamespace(scene=SimpleNamespace(to_state_dict=lambda: {'stable': True}))
    monkeypatch.setattr(pair, 'capture_returned', lambda *args: None)
    monkeypatch.setattr(paired_runtime, 'canonical_asset_hashes', lambda path: {'mesh': 'same'})

    async def spawn(*args, **kwargs):
        name = args[args.index('--worker-candidate') + 1]
        assert kwargs['env']['SCENEEXPERT_ACTIVE_MEMORY_BANK_DIR'] == str(tmp_path / name / 'private_memory')
        calls.append(name)

        async def wait():
            return 0

        return SimpleNamespace(wait=wait, returncode=0, pid=1)

    monkeypatch.setattr(paired_runtime.asyncio, 'create_subprocess_exec', spawn)
    asyncio.run(pair.finish(agent, 'safety'))
    assert calls == ['B', 'C', 'D']
    for name in calls:
        proof = read_json(tmp_path / f'continuation_proof_{name}.json')
        assert proof['canonical_before'] == proof['canonical_after']
        assert proof['assets_before'] == proof['assets_after']
        assert proof['memory_before'] == proof['memory_after']
    validate_pool_execution(snapshot, read_json(tmp_path / 'candidate_pool_execution.json'))


def test_native_decision_freezes_the_approved_pool_and_exact_request(tmp_path, monkeypatch):
    from tests.unit.test_direct_decision_pairs import direct_pair, tool_item

    monkeypatch.setenv('SCENEEXPERT_POOL_CANDIDATES', '4')
    monkeypatch.setenv('SCENEEXPERT_SHADOW_SCOPE_SAMPLES', '3')
    agent, pair = direct_pair(tmp_path, monkeypatch)
    inputs = [{'role': 'user', 'content': 'immutable original prompt'}]
    request = {'model': 'unsloth/Qwen3.8-27B-GGUF', 'messages': inputs}
    pair.observe_request(request)
    asyncio.run(pair.observe_response(SimpleNamespace(output=[tool_item('generate_assets')]),
                                     (), {'system_instructions': 'system', 'input': inputs}))
    assert pair.frozen and not pair.disabled
    assert candidate_names(pair.snapshot) == ('A', 'B', 'C', 'D')
    assert pair.snapshot['shadow_sampling']['max_samples'] == 3
    assert read_json(pair.group / 'A/first_request.json') == request
    assert pair.snapshot['state'] == agent.scene.to_state_dict()


def test_legacy_rescoring_cannot_silently_drop_the_other_pool_candidates(tmp_path):
    from scenesmith.scene_expert.slow_memory.paired_rescore import rescore_pairs

    source = tmp_path / 'native'
    four_group(source)
    output = tmp_path / 'rescore'
    with pytest.raises(ValueError, match='legacy A/B rescoring is unsupported'):
        rescore_pairs(source, output)
    assert not output.exists()
