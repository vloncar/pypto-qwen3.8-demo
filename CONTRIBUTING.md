# Working on this preview

Keep model implementation in pypto-lib and serving integration in pypto-serving.
This repository owns reproducibility, comparison scripts and documentation.
Preserve the common timing client and report default vLLM separately from eager.
Use Python 3.10-compatible syntax and concise imperative commit subjects.

After changing benchmark code or pinned inputs, review the changes and refresh
`bundle_manifest.json` with `python3 freeze_bundle.py`. Do not refresh hashes to
hide an unexpected mismatch during a run. Check with:

```bash
python3 -m unittest -v test_protocol.py test_packaging.py
python3 -m compileall -q .
ruff check --config ruff.toml .
bash -n run_pypto.sh preflight.sh run_offline.sh environment.example.sh
git diff --check
```

Changes to engine launch arguments need an actual device/host test before a new
performance claim. Keep partial results and state which execution paths have
been validated. Never commit private local configuration or generated binaries.
