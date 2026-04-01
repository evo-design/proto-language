# Using Claude Code with This Repo

This guide covers the Claude Code tooling available in `proto-language` and `proto-tools`. Both repos have custom commands, skills, and CI integrations that make common workflows faster.

## What's Available

### Commands

Commands are workflows you invoke with `/command-name`. They orchestrate multi-step processes.

| Command | Repo | What It Does |
|---------|------|-------------|
| `/fix-issue <number>` | Both repos | End-to-end issue resolution: reads issue, investigates codebase, presents findings for your approval, writes failing test, implements fix, verifies, creates PR. Uses git worktrees for isolation. |

### Skills

Skills are domain knowledge that Claude loads on demand. You don't invoke them directly; Claude activates them when working on relevant tasks.

**proto-language:**

| Skill | When It Activates |
|-------|------------------|
| `general-dev` | Architecture overview, coding conventions, Pydantic config patterns |
| `implement-constraint` | Implementing, modifying, or debugging constraints in the DSL |
| `implement-generator` | Implementing, modifying, or debugging generators |
| `implement-optimizer` | Implementing, modifying, or debugging optimizers |
| `testing` | Writing tests: pytest markers, fixtures, CPU/GPU patterns |
| `write-program` | Composing optimization programs (Segments, Constructs, Programs) |

**proto-tools:**

| Skill | When It Activates |
|-------|------------------|
| `implement-tool` | Implementing a new bioinformatics tool wrapper (6-phase parallelized pipeline) |
| `fix-env` | Debugging tool environment setup failures (compute detection, env isolation, deps) |

### CI Integration

| Trigger | What Happens |
|---------|-------------|
| `@claude` in a PR comment | Claude reviews the PR or answers a specific question |
| `@claude` in an issue body/title | Claude responds when the issue is opened or assigned |

## Common Workflows

### Fix a GitHub issue

```
/fix-issue 42
```

This runs through 9 steps automatically:
1. Creates an isolated git worktree (your current work is untouched)
2. Reads the issue and comments from GitHub
3. Investigates the codebase with parallel sub-agents
4. **Stops and presents findings**: you confirm the interpretation before any code is written
5. Writes a failing test
6. Implements the fix
7. Runs the full verification suite
8. Pushes and creates a PR that closes the issue
9. Offers to clean up the worktree

The branch is named `<your-name>/fix-issue-<number>` (derived from `git config user.name`).

### Implement a new component

For constraints, generators, or optimizers, just describe what you want. Claude will activate the relevant skill automatically and follow the established patterns (registry decorator, Pydantic models, test templates).

Example prompts:
- "Implement a new constraint that checks codon usage bias"
- "Add a generator that wraps the ProGen3 model"

### Implement a new tool (proto-tools)

The `implement-tool` skill runs a 6-phase parallelized pipeline: research → contract → scaffolding → implementation → testing → integration. Describe the tool and point Claude at the paper or model docs.

### Get a PR reviewed

In any PR, comment:
```
@claude
```
Claude will do a full code review. Or ask a specific question:
```
@claude why does this function use caching instead of recomputing?
```

## How the Pieces Fit Together

```
CLAUDE.md          → Conventions Claude always follows (coding style, architecture, commands)
.claude/skills/    → Domain knowledge loaded on demand (how to implement a constraint, etc.)
.claude/commands/  → Multi-step workflows you invoke explicitly (/fix-issue)
notes/             → Human documentation (this file, dev.md, batching.md)
```

- **CLAUDE.md** is the first thing Claude reads. It contains architecture, coding conventions, and key commands. Both repos have their own.
- **Skills** are reference material Claude consults when doing specific tasks. They contain templates, patterns, and decision trees.
- **Commands** are step-by-step workflows. They're the "do this task end-to-end" instructions.
- **notes/** is human documentation. `dev.md` covers git workflows, CI, pre-commit hooks, and the export chain validator. This file covers the Claude Code layer on top of that.

## Tips

- **Sub-agent parallelism**: Claude can fan out investigation to multiple sub-agents. The `/fix-issue` command does this in Step 2 (codebase exploration). When giving Claude complex tasks, it will parallelize where possible.
- **Worktrees**: `/fix-issue` creates worktrees under `.claude/worktrees/`. You can have multiple issues in-flight simultaneously without branch switching. Clean up with `git worktree remove .claude/worktrees/issue-<N>`.
- **Submodule in worktrees**: Worktrees don't auto-initialize submodules. If your fix touches `proto-tools` code, run `git submodule update --init --recursive` inside the worktree before running tests.
- **Auto-memory**: Claude maintains personal notes across sessions in `~/.claude/projects/.../memory/`. This is per-developer and not committed. It's where Claude remembers debugging patterns and project quirks.
- **Squash merge**: Always use squash merge for PRs in `evo-design/*` repos.
