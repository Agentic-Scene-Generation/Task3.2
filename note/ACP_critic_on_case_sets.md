# ACP Critic-On Case Sets

Run these commands from the Task3.2 checkout inside the ACP job. The wrapper
starts the embedding service and Qwen llama.cpp server, waits for the model,
then runs `scripts/run_parallel_critic_on.sh` with the selected registry.

```bash
cd /data/task3_2/L202500266_hrk/code/Task3.2
```

## New Three Scenes

Create a new floor-plan shared base and run critic-on:

```bash
bash scripts/run_acp_critic_on.sh new3 create
```

Reuse a compatible three-scene shared base:

```bash
bash scripts/run_acp_critic_on.sh new3 reuse \
  /data/task3_2/L202500266_hrk/code/Task3.2/outputs/critic_probe/shared_base_3scene_20260812_215933/shared_base
```

## Legacy Eight Scenes

Create a new floor-plan shared base and run critic-on:

```bash
bash scripts/run_acp_critic_on.sh legacy8 create
```

Reuse a compatible eight-scene shared base:

```bash
bash scripts/run_acp_critic_on.sh legacy8 reuse \
  /absolute/path/to/legacy8_run/shared_base
```

The legacy registry is restored from commit `1c45466` and keeps its original
eight IDs, prompts, order, batch labels, and scene indices. The `new3`
registry keeps `bedroom`, `office`, and `long_living_room` in their current
order. A shared base is valid only for the registry that generated it.

Pass scene selection through to the critic-on runner when needed:

```bash
bash scripts/run_acp_critic_on.sh new3 reuse /path/to/new3/shared_base \
  --scenes office

bash scripts/run_acp_critic_on.sh legacy8 reuse /path/to/legacy8/shared_base \
  --scenes default_classroom
```

Use `DRY_RUN=true` to validate the resolved case set, batch mapping, and
runner arguments without starting ACP services or generating scenes:

```bash
DRY_RUN=true bash scripts/run_acp_critic_on.sh legacy8 create
```
