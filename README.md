# Autoneuro

## Projects

Each project lives under `projects/` and provides a `template/` directory. The
launch scripts copy that template into a timestamped workspace before starting a
research session.

Create a project like this:

```text
projects/<name>/
  template/
    RESEARCH_QUESTION.md
    RESEARCH_LOG.md
    ...
```

Launch Claude Code:

```bash
./launch_claude.sh projects/<name>
```

Launch Codex:

```bash
./launch_codex.sh projects/<name>
```

Each launch creates:

```text
projects/<name>/results/<timestamp>/workspace/
```

Attach to the session:

```bash
tmux attach -t <name>-claude
```

Stop the session:

```bash
./launch_claude.sh projects/<name> stop
```

Example:

```bash
./launch_claude.sh projects/yang-tasks
```
