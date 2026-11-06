# Published preview and future updates

The preview is public under `vloncar` on GitHub:

- [pypto-lib preview branch](https://github.com/vloncar/pypto-lib/tree/preview/qwen3.8-27b)
- [pypto-serving preview branch](https://github.com/vloncar/pypto-serving/tree/preview/qwen3.8-27b)
- [Companion repository](https://github.com/vloncar/pypto-qwen3.8-demo)

Exact code commits and source URLs are in [versions.json](versions.json).
Publishing these previews does not create an upstream MR/PR.

For a later preview update:

1. Validate and push the new kernel/serving revisions, then update the exact
   commits and source hashes in this companion repository.
2. Refresh the bundle with `python3 freeze_bundle.py` and run the checks in
   CONTRIBUTING. Keep old benchmark bundles and results intact so their
   original integrity checks remain usable.
3. Test `checkout.py` in a fresh directory using the public URLs.
4. Run a complete matched host comparison before claiming a new performance
   result against vLLM, and update the results accordingly.

The existing review branches remain separate. Upstreaming still needs review
against the actual base revisions, logical commit splitting and dependency
ordering; the preview snapshots are not a substitute for that work.

Keep `config.local.json`, `environment.local.sh`, model weights, build output,
local validation logs and fresh benchmark results out of publication. The
portable examples and historical reference export are the intended public files.
