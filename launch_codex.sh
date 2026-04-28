#!/usr/bin/env bash
set -euo pipefail

COMMAND="start"
PROJECT_ARG="${PROJECT_DIR:-}"

if [[ $# -gt 0 ]]; then
  case "$1" in
    start|attach|stop|status|-h|--help|help)
      COMMAND="$1"
      if [[ $# -gt 1 ]]; then
        PROJECT_ARG="$2"
      fi
      ;;
    *)
      PROJECT_ARG="$1"
      if [[ $# -gt 1 ]]; then
        COMMAND="$2"
      fi
      ;;
  esac
fi

if [[ -z "$PROJECT_ARG" ]]; then
  PROJECT_ARG="$PWD"
fi

PROJECT_DIR="$(cd "$PROJECT_ARG" && pwd)"
PROJECT_NAME="$(basename "$PROJECT_DIR" | tr -c 'A-Za-z0-9._-' '-')"
TMUX_SESSION="${TMUX_SESSION:-$PROJECT_NAME-codex}"
TEMPLATE_DIR="${TEMPLATE_DIR:-$PROJECT_DIR/template}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_DIR/results}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${RUN_DIR:-$RESULTS_ROOT/$RUN_ID}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$RUN_DIR/workspace}"
RUNTIME_DIR="${RUNTIME_DIR:-$RUN_DIR/runtime}"
CODEX_EXTRA_ARGS="${CODEX_EXTRA_ARGS:-}"
CODEX_INITIAL_PROMPT="${CODEX_INITIAL_PROMPT:-Read RESEARCH_QUESTION.md, then begin. Work only in this workspace.}"
CODEX_BIN="${CODEX_BIN:-codex}"
LAUNCHER_OWNS_SESSION=0

usage() {
  cat <<EOF
Usage: ./launch_codex.sh [project-dir] [start|attach|stop|status]
       ./launch_codex.sh [start|attach|stop|status] [project-dir]

Environment:
  PROJECT_DIR=$PROJECT_DIR
  TMUX_SESSION=$TMUX_SESSION
  TEMPLATE_DIR=$TEMPLATE_DIR
  RESULTS_ROOT=$RESULTS_ROOT
  RUN_ID=$RUN_ID
  WORKSPACE_DIR=$WORKSPACE_DIR
  CODEX_EXTRA_ARGS=$CODEX_EXTRA_ARGS
EOF
}

prepare_workspace() {
  if [[ ! -d "$TEMPLATE_DIR" ]]; then
    echo "Template directory not found: $TEMPLATE_DIR" >&2
    exit 1
  fi

  mkdir -p "$WORKSPACE_DIR"
  cp -R -n "$TEMPLATE_DIR"/. "$WORKSPACE_DIR"/
}

cleanup_session() {
  if [[ "$LAUNCHER_OWNS_SESSION" == "1" ]]; then
    tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
  fi
}

wait_for_session() {
  trap cleanup_session INT TERM
  while tmux has-session -t "$TMUX_SESSION" 2>/dev/null; do
    sleep 1
  done
  trap - INT TERM
}

start_session() {
  if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "tmux session already exists: $TMUX_SESSION"
    echo "Attach with: tmux attach -t $TMUX_SESSION"
    return 0
  fi

  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is required." >&2
    exit 1
  fi
  if ! command -v "$CODEX_BIN" >/dev/null 2>&1; then
    echo "Codex CLI not found: $CODEX_BIN" >&2
    exit 1
  fi

  prepare_workspace

  mkdir -p "$RUNTIME_DIR"
  local command_file
  command_file="$RUNTIME_DIR/run-codex.sh"
  cat >"$command_file" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd $(printf '%q' "$WORKSPACE_DIR")
CODEX_EXTRA_ARGS=$(printf '%q' "$CODEX_EXTRA_ARGS")

extra_args=()
if [[ -n "\$CODEX_EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  extra_args=(\$CODEX_EXTRA_ARGS)
fi

exec $(printf '%q' "$CODEX_BIN") \\
  "\${extra_args[@]}" \\
  $(printf '%q' "$CODEX_INITIAL_PROMPT")
EOF
  chmod +x "$command_file"

  tmux new-session -d -s "$TMUX_SESSION" -n research "bash $(printf '%q' "$command_file")"
  LAUNCHER_OWNS_SESSION=1
  echo "Started Codex in tmux session: $TMUX_SESSION"
  echo "Result run: $RUN_DIR"
  echo "Workspace: $WORKSPACE_DIR"
  echo "Attach with: tmux attach -t $TMUX_SESSION"
  echo "Waiting for session to finish. Press Ctrl-C to stop it."
  wait_for_session
}

case "$COMMAND" in
  start)
    start_session
    ;;
  attach)
    exec tmux attach -t "$TMUX_SESSION"
    ;;
  stop)
    tmux kill-session -t "$TMUX_SESSION"
    ;;
  status)
    tmux has-session -t "$TMUX_SESSION" 2>/dev/null \
      && echo "running: $TMUX_SESSION" \
      || echo "not running: $TMUX_SESSION"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
