# Using Claude Code with This Repo

This guide covers the Claude Code tooling available in `proto-language` and `proto-tools`. Both repos ship custom skills and CI integrations that make common workflows faster.

## Skills (auto-loaded)

Skills load on demand when Claude detects a relevant task. Don't invoke them by name.

**proto-language** (`.claude/skills/`):

| Skill | When It Activates |
|-------|------------------|
| `implement-constraint` | Implementing / modifying / debugging constraints |
| `implement-generator` | Implementing / modifying / debugging generators |
| `implement-optimizer` | Implementing / modifying / debugging optimizers |
| `write-program` | Composing optimization programs (Segments, Constructs, Programs) |

General coding conventions live in `CLAUDE.md`. Long-form testing reference (templates, fixtures, mocks) lives in `notes/testing.md`.

**proto-tools** (`.claude/skills/`):

| Skill | When It Activates |
|-------|------------------|
| `implement-tool` | New bioinformatics tool wrapper (6-phase parallelized pipeline) |
| `fix-env` | Tool environment setup failures (compute detection, env isolation, deps) |

## CI Integration

| Trigger | What Happens |
|---------|-------------|
| `@claude` in a PR comment | Claude reviews the PR or answers a specific question |
| `@claude` in an issue body/title | Claude responds when the issue is opened or assigned |

## How the Pieces Fit Together

- **CLAUDE.md** — high-leverage conventions Claude reads at session start. Both repos have their own.
- **`.claude/skills/`** — domain knowledge loaded on demand; templates, patterns, decision trees.
- **`notes/`** — long-form team docs (this file, `dev.md`, `batching.md`, `error-handling.md`). Where CLAUDE.md links for detail.
- **Auto-memory** — per-developer, not committed; sits under `~/.claude/projects/.../memory/`. Where Claude remembers debugging patterns and project quirks across sessions.

## Tips

- **Sub-agent parallelism**: Claude fans out investigation to multiple sub-agents on complex tasks.
- **Worktrees**: Claude creates worktrees under `.claude/worktrees/` for isolated work. Multiple issues in flight at once, no branch switching. Clean up with `git worktree remove .claude/worktrees/<name>`.
- **Submodule in worktrees**: worktrees don't auto-initialize submodules. If a fix touches `proto-tools` code, run `git submodule update --init --recursive` inside the worktree before running tests.
- **Squash merge**: always use squash merge for PRs in `evo-design/*` repos.
