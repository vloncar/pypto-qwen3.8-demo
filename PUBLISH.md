# Publish when ready

Everything is currently local. Nothing in this preparation creates an upstream
MR/PR. The two code branches are named `preview/qwen3.8-27b`; exact commits and
planned GitHub URLs are in `versions.json`.

When publication is authorized:

1. Push the pypto-lib preview branch to `vloncar/pypto-lib`.
2. Create/use a `vloncar/pypto-serving` fork and push its preview branch.
3. Create `vloncar/pypto-qwen3.8-demo` and publish this repository's `main` branch.
4. Verify both remote commits match `versions.json`. Update the local-preview
   notice in README and `versions.json` only after the URLs resolve. Refresh
   the bundle hashes with `python3 freeze_bundle.py`, run the checks in
   CONTRIBUTING, then commit and push that documentation/status update.
5. Test a fresh clone using the default URLs in `checkout.py`, run a new complete
   host comparison if making a claim about these exact preview commits, and
   replace `<repo-link>` in `SHARE.md` before sharing the message.

The existing review branches remain separate. Upstreaming still needs review
against the actual base revisions, logical commit splitting and dependency
ordering; the preview snapshots are not a substitute for that work.

Keep `config.local.json`, `environment.local.sh`, model weights, build output,
local validation logs and fresh benchmark results out of publication. The
portable examples and historical reference export are the intended public files.
