# Critic Probe Web

Read-only workspace for reviewing SceneSmith critic-probe logs in
`../outputs/critic_probe`.

## Start

Use two terminals from `Task3.2`.

Terminal 1 starts the read-only Flask API:

```bash
.venv/bin/python tools/critic_probe_web.py
```

When port `5055` is already held by another instance of this same script, the
new command stops that instance and starts the current version. It never stops
an unrelated process that happens to use the port.

Terminal 2 starts the Vite frontend:

```bash
cd web
pnpm install --frozen-lockfile
pnpm dev --host 0.0.0.0
```

Open `http://127.0.0.1:5175/` in a browser. The frontend proxies `/api` requests
to `http://127.0.0.1:5055`, so the API does not need to be publicly exposed.

For a CCI machine, Vite also prints the machine's network URL after startup.

## Commands

```bash
pnpm run build
pnpm run lint
```

Stop each process with `Ctrl+C` in its terminal.

After changing `tools/critic_probe_web.py`, restart the Flask process before
refreshing the browser. Vite reloads frontend source changes automatically.

## Data Boundary

The Flask service only serves files under `outputs/critic_probe`; it provides no
write or deletion endpoints. It reads the existing render snapshots, scene states,
timing JSONL, action logs, score files, and critic SQLite session messages.
The audit drawer combines timing records with full SQLite agent traces for old
runs. New runs also persist complete LLM request/response payloads under the
scene audit directory.

LLM audit events expose normalized input/cache and output/reasoning token
breakdowns when the provider reports them, the resolved API response time, and
the selected scene's largest observed request input context. Historical runs
without usage details remain readable and show these values as unavailable.
