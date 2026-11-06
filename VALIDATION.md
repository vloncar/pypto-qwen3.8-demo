# Validation of this package

Prepared on 6 November 2026.

| Check | Result |
|---|---|
| pypto-lib / serving source commits | Published preview branches; commits match `versions.json` |
| Fresh source checkouts | Public GitHub URLs resolve; all three source revisions and all 150 model, harness and serving Python source hashes match; required submodules initialize |
| Protocol and packaging tests | 12 passed, including wrong-output/source rejection, portable payload, allocation loss and cleanup |
| Python compatibility | All scripts parse as Python 3.10; byte compilation passes |
| Lint and shell syntax | Ruff and all documented shell checks pass |
| Documentation links and publication scan | Local links resolve; tracked files contain no private machine paths/configuration |
| Historical result export | All three throughput medians and all 4,800 output IDs independently reproduced |
| Packaged PyPTO runner on NPU | Fresh compilation of seven programs; five warmup plus twenty measured rounds; all 800 IDs across 100 requests match; worker shutdown verified |
| Existing implementation validation | 623 serving tests, 16 contract tests and the NPU cleanup regression described in `RESULTS.md` passed |

The clean-checkout check uses the public GitHub URLs at the pinned commits,
including the compiler/runtime and serving kernel submodules. A completely new
virtual-environment installation was not performed; the setup guide uses the
selected projects' installation procedures and the verified environment's
dependency versions.

The host coordinator's configuration, portable payload and failure cleanup are
covered by host-only tests. A fresh six-block Docker comparison of this portable
package has not run. Perform that on the Docker host before making a new
performance claim about the exact preview revisions.

The NPU run validates the packaged PyPTO entry and the exact preview model/
serving sources. Its raw result also passes the independent per-block validator.
It is a single-engine packaging check, not a new matched vLLM comparison.
