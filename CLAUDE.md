# element-calcium-imaging (BrainCOGS fork)

BrainCOGS's fork of DataJoint's calcium imaging element (Suite2p, CaImAn), used by
U19-pipeline-python. Python >= 3.11, managed with uv.

## Setup and tests

```bash
uv sync --locked
uv run --locked pytest      # CI: Python 3.11 and 3.14
```

Lint: `uvx prek run --all-files` (the same as CI's Lint workflow).

## Pins and gotchas

- `datajoint==0.14.10` exactly; `suite2p[io]>=1.1.0`. This fork holds the suite2p
  1.x settings converter.
- The DataJoint CaImAn fork can't run on numpy 2, and PyPI's `caiman` is an
  unrelated package.
- Changes stay on the BrainCOGS fork; nothing goes upstream to datajoint/.

## Agent workflow

- Errors reported in Slack: use the `bugfix-from-slack` skill (agent-tooling). The braincogs-claude-agent bot opens issues and PRs; commits and pushes are the user's.
- Commits end with one `Assisted-by: <Harness>:<model>` trailer per agent, e.g. `Assisted-by: ClaudeCode:claude-opus-5-5`. No AI co-authors, and no `Generated with` lines. `prek install` sets up the `ai-attribution` commit-msg hook (`.github/scripts/check_ai_attribution.py`, vendored from agent-tooling), and CI checks every commit of a PR.
- PRs target `main` on `BrainCOGS/element-calcium-imaging`. Merge PRs the bot opened with **Rebase and merge**, not Squash: GitHub attributes a squash commit to the PR's opener.
- This is a fork. Always pass `-R BrainCOGS/element-calcium-imaging` to `gh`, since plain `gh pr create` targets the upstream.
- Issues are turned off here; put the bug report sections in the PR body instead.
