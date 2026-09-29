"""Exercise actual pinned HF/PEFT checkpoint persistence and resume on CPU."""
import json

import pytest


def test_real_peft_checkpoint_and_resume_on_cpu(tmp_path):
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    peft = pytest.importorskip('peft')
    from scenesmith.scene_expert.slow_memory.training_lifecycle import (
        checkpoint_complete, make_progress_callback,
    )
    torch.set_num_threads(1)

    def model():
        torch.manual_seed(42)
        base = transformers.LlamaForCausalLM(transformers.LlamaConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=32,
        ))
        return peft.get_peft_model(base, peft.LoraConfig(
            r=2, lora_alpha=4, target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM',
        ))

    def trainer(output):
        args = transformers.TrainingArguments(
            output_dir=str(output), use_cpu=True, max_steps=2, per_device_train_batch_size=1,
            save_strategy='steps', save_steps=1, logging_steps=1, report_to='none',
            disable_tqdm=True, optim='adamw_torch',
        )
        return transformers.Trainer(
            model=model(), args=args,
            train_dataset=[{'input_ids': [1,2,3,4], 'labels': [1,2,3,4]}]*4,
            callbacks=[make_progress_callback(output)],
        )

    first = tmp_path/'first'
    first.mkdir()
    run = trainer(first)
    run.train()
    checkpoint_complete(first/'checkpoint-1')
    second = tmp_path/'second'
    second.mkdir()
    resumed = trainer(second)
    resumed.train(resume_from_checkpoint=str(first/'checkpoint-1'))
    assert resumed.state.global_step == 2
    checkpoint_complete(second/'checkpoint-2')
    events = [json.loads(x) for x in (second/'training_events.jsonl').read_text().splitlines()]
    assert any(e['event']=='checkpoint_saved' and e['global_step']==2 for e in events)
