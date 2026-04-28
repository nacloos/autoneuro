# Autoneuro

## Yang Tasks

Run one research session from a template workspace:

```bash
./launch_claude.sh projects/yang-tasks
```

Or launch Codex:

```bash
./launch_codex.sh projects/yang-tasks
```

By default, the launcher creates:

```text
projects/yang-tasks/results/<timestamp>/workspace/
```

The template research question uses `dlygointr` as the held-out Yang task.

Attach to the session:

```bash
tmux attach -t yang-tasks-claude
```

Stop the session:

```bash
./launch_claude.sh projects/yang-tasks stop
```
