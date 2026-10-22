# Future exact-byte transport entry point

`scripts/package_sceneexpert_exact_transport.py` is an opt-in entry point for a
future, separately frozen exact-byte packer. It does not change the default review
packager or any already closed run's scripts, source, artifacts, verdicts or exits.

Image-like strings in JSON/JSONL can match the packer's regex while containing
invalid padding or noncanonical base64. The entry point factors only canonical,
strictly decodable matches. Other matches remain literal original bytes, so the
existing recipe reconstructs the entire original file, including malformed text.
No image is resized and no JSON is reserialized. The external packer's size limits,
SHA reconstruction, missing-output diagnostics and actual return code are retained.

Before a future allocation, copy its reviewed packer into its new immutable control
directory. Bind that file's SHA256 **and** this entry point and
`scenesmith/scene_expert/transport_media.py` in the new plan's source/control hashes.
Use that new plan and control directory, never a closed run's paths. For example:

```bash
python scripts/package_sceneexpert_exact_transport.py \
  --packer tmp/acp_controls/new_run/native_pack.py \
  --packer-sha256 <SHA256-from-the-new-frozen-plan> \
  --plan tmp/acp_controls/new_run/plan.json \
  --prefix tmp/results/new_run/review --include-postpack
```

The wrapper imports only a matching pinned packer inside the project. Its existing
`--verify`, `--server-originals` and `--cache` modes remain available. This is a
transport utility, not an allocation command or evidence of native completion or
qualification. Planned-but-uncreated candidate slots still need an independent
full-review ledger; a packer's created-directory scan does not establish that all
planned native outputs exist. Late exits, review certificates and actual download
proofs must still fit inside the final whole-set budget.
