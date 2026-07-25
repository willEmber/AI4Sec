#!/usr/bin/env bash
# One-command local dev launcher: brings up the FastAPI backend and the Next.js
# frontend together, wires the frontend proxy to the right backend port, and
# tears both down on a single Ctrl-C.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/.dev-logs"

BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
TARGET="all"      # all | backend | frontend
DO_INSTALL=1      # auto-sync deps when lockfiles are newer than the install dir
KILL_PORT=0
OPEN_BROWSER=0

C_RESET=$'\033[0m'; C_BE=$'\033[36m'; C_FE=$'\033[35m'
C_OK=$'\033[32m'; C_WARN=$'\033[33m'; C_ERR=$'\033[31m'; C_DIM=$'\033[2m'

usage() {
  cat <<'EOF'
用法: scripts/dev.sh [选项]

一键启动本地开发环境（后端 FastAPI + 前端 Next.js），Ctrl-C 同时退出。

选项:
  --backend, -b        只启动后端
  --frontend, -f       只启动前端
  --backend-port N     后端端口（默认 8000，也可用 BACKEND_PORT 环境变量）
  --frontend-port N    前端端口（默认 3000，也可用 FRONTEND_PORT 环境变量）
  --no-install         跳过依赖自动同步（uv sync / npm install）
  --kill-port          端口被占用时自动结束占用进程，而不是直接报错
  --open               就绪后自动打开浏览器
  --help, -h           显示本帮助

说明:
  * 前端通过 next.config.ts 的 rewrites 把 /api/* 代理到后端，脚本会自动把
    BACKEND_URL 指向本次启动的后端端口（默认 8001 是 Docker 用的端口）。
  * 后端读取仓库根目录的 .env；缺失时会从 .env.example 复制一份。
  * 完整日志写入 .dev-logs/backend.log 与 .dev-logs/frontend.log。
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -b|--backend)   TARGET="backend" ;;
    -f|--frontend)  TARGET="frontend" ;;
    --backend-port)  BACKEND_PORT="$2"; shift ;;
    --frontend-port) FRONTEND_PORT="$2"; shift ;;
    --no-install)   DO_INSTALL=0 ;;
    --kill-port)    KILL_PORT=1 ;;
    --open)         OPEN_BROWSER=1 ;;
    -h|--help)      usage; exit 0 ;;
    *) printf '%s未知参数: %s%s\n' "$C_ERR" "$1" "$C_RESET"; usage; exit 2 ;;
  esac
  shift
done

log()  { printf '%s>>%s %s\n' "$C_OK" "$C_RESET" "$*"; }
warn() { printf '%s!!%s %s\n' "$C_WARN" "$C_RESET" "$*"; }
die()  { printf '%sXX%s %s\n' "$C_ERR" "$C_RESET" "$*"; exit 1; }

# --- process management -----------------------------------------------------
PIDS=()

kill_tree() {
  local pid="$1" child
  for child in $(pgrep -P "$pid" 2>/dev/null); do kill_tree "$child"; done
  kill -TERM "$pid" 2>/dev/null || true
}

cleanup() {
  trap - INT TERM EXIT
  printf '\n%s>>%s 正在停止服务...\n' "$C_OK" "$C_RESET"
  local pid
  for pid in "${PIDS[@]:-}"; do
    [[ -n "$pid" ]] && kill_tree "$pid"
  done
  # give children a moment to flush, then hard-kill leftovers
  sleep 1
  for pid in "${PIDS[@]:-}"; do
    [[ -n "$pid" ]] && kill -KILL "$pid" 2>/dev/null || true
  done
  log "已退出"
}
trap cleanup INT TERM EXIT

# Reads the listening-socket table instead of dialing the port: on WSL2 a
# connect() to an unbound port can hang instead of being refused.
port_in_use() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltnH 2>/dev/null | awk -v p=":$port\$" '$4 ~ p {found=1} END {exit !found}'
  elif command -v lsof >/dev/null 2>&1; then
    lsof -iTCP:"$port" -sTCP:LISTEN -t >/dev/null 2>&1
  else
    timeout 1 bash -c "exec 3<>/dev/tcp/127.0.0.1/$port" 2>/dev/null
  fi
}

free_port() {
  local port="$1" name="$2"
  port_in_use "$port" || return 0
  if [[ "$KILL_PORT" -eq 1 ]]; then
    warn "$name 端口 $port 被占用，正在结束占用进程"
    if command -v fuser >/dev/null 2>&1; then
      fuser -k "${port}/tcp" >/dev/null 2>&1 || true
    elif command -v lsof >/dev/null 2>&1; then
      lsof -ti "tcp:${port}" | xargs -r kill -9 2>/dev/null || true
    else
      die "找不到 fuser / lsof，无法自动释放端口 $port"
    fi
    sleep 1
    port_in_use "$port" && die "端口 $port 仍被占用"
  else
    die "$name 端口 $port 已被占用（Docker 也在跑？）。换端口：--${name}-port N，或加 --kill-port 自动结束占用进程"
  fi
}

wait_for_port() {
  local port="$1" tries="${2:-120}"
  while ((tries-- > 0)); do
    port_in_use "$port" && return 0
    # a service that died mid-startup should not keep us waiting
    local pid alive=0
    for pid in "${PIDS[@]:-}"; do
      [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && alive=1
    done
    [[ "$alive" -eq 0 ]] && return 1
    sleep 1
  done
  return 1
}

# starts a service detached from the terminal, logging to its own file so the
# recorded PID is the service itself (not a pipeline subshell)
start_service() {
  local name="$1" dir="$2"; shift 2
  ( cd "$dir" && exec "$@" ) >"$LOG_DIR/$name.log" 2>&1 &
  PIDS+=("$!")
}

stream_log() {
  local name="$1" color="$2"
  tail -n 0 -qF "$LOG_DIR/$name.log" 2>/dev/null \
    | sed -u "s/^/${color}[${name}]${C_RESET} /" &
  PIDS+=("$!")
}

# --- preflight --------------------------------------------------------------
mkdir -p "$LOG_DIR"

if [[ ! -f "$ROOT/.env" ]]; then
  if [[ -f "$ROOT/.env.example" ]]; then
    cp "$ROOT/.env.example" "$ROOT/.env"
    warn "已从 .env.example 生成 .env，请填入 LLM_APIKEY / MINERU_TOKEN 后再跑分析任务"
  else
    warn "缺少 .env，后端将以空配置启动"
  fi
elif grep -qE '^LLM_APIKEY=("?)(your-api-key)?\1$' "$ROOT/.env" 2>/dev/null; then
  warn ".env 中的 LLM_APIKEY 尚未填写，分析任务会失败"
fi

if [[ "$TARGET" != "frontend" ]]; then
  command -v uv >/dev/null 2>&1 || die "未找到 uv，请先安装: https://docs.astral.sh/uv/"
  if [[ "$DO_INSTALL" -eq 1 ]]; then
    if [[ ! -d "$ROOT/backend/.venv" ]] \
       || [[ "$ROOT/backend/pyproject.toml" -nt "$ROOT/backend/.venv" ]] \
       || [[ -f "$ROOT/backend/uv.lock" && "$ROOT/backend/uv.lock" -nt "$ROOT/backend/.venv" ]]; then
      log "同步后端依赖 (uv sync)..."
      (cd "$ROOT/backend" && uv sync) || die "uv sync 失败"
      touch "$ROOT/backend/.venv"
    fi
  fi
  free_port "$BACKEND_PORT" backend
fi

if [[ "$TARGET" != "backend" ]]; then
  command -v npm >/dev/null 2>&1 || die "未找到 npm，请先安装 Node.js 20+"
  if [[ "$DO_INSTALL" -eq 1 ]]; then
    if [[ ! -d "$ROOT/frontend/node_modules" ]] \
       || [[ "$ROOT/frontend/package-lock.json" -nt "$ROOT/frontend/node_modules" ]]; then
      log "安装前端依赖 (npm install)..."
      (cd "$ROOT/frontend" && npm install) || die "npm install 失败"
      touch "$ROOT/frontend/node_modules"
    fi
  fi
  free_port "$FRONTEND_PORT" frontend
fi

# --- launch -----------------------------------------------------------------
if [[ "$TARGET" != "frontend" ]]; then
  : >"$LOG_DIR/backend.log"
  # The SSE stream is the one cross-origin call (EventSource dials the backend
  # directly, bypassing the Next proxy), so the backend must allow the dev
  # frontend's origin. Respect an explicit CORS_ORIGINS in .env if present.
  if ! grep -qE '^[[:space:]]*CORS_ORIGINS=' "$ROOT/.env" 2>/dev/null; then
    export CORS_ORIGINS="[\"http://localhost:$FRONTEND_PORT\",\"http://127.0.0.1:$FRONTEND_PORT\"]"
  fi
  log "启动后端 http://localhost:$BACKEND_PORT  ${C_DIM}(--reload)${C_RESET}"
  start_service backend "$ROOT/backend" \
    uv run uvicorn app.main:app --reload --host 0.0.0.0 --port "$BACKEND_PORT"
  stream_log backend "$C_BE"

  if ! wait_for_port "$BACKEND_PORT" 90; then
    printf '%s' "$C_ERR"; tail -n 30 "$LOG_DIR/backend.log"; printf '%s' "$C_RESET"
    die "后端启动失败，完整日志见 .dev-logs/backend.log"
  fi
fi

if [[ "$TARGET" != "backend" ]]; then
  : >"$LOG_DIR/frontend.log"
  # Two separate knobs, both defaulting to the Docker port :8001 — point each at
  # the backend this script actually started:
  #   BACKEND_URL             server-side, backs the /api/* rewrite (next.config.ts)
  #   NEXT_PUBLIC_BACKEND_URL browser-side, where EventSource dials the SSE stream
  #                           directly (lib/api.ts) — the proxy buffers streams
  export BACKEND_URL="http://localhost:$BACKEND_PORT"
  export NEXT_PUBLIC_BACKEND_URL="http://localhost:$BACKEND_PORT"
  log "启动前端 http://localhost:$FRONTEND_PORT  ${C_DIM}(代理 /api + SSE → $BACKEND_URL)${C_RESET}"
  start_service frontend "$ROOT/frontend" npm run dev -- --port "$FRONTEND_PORT"
  stream_log frontend "$C_FE"

  if ! wait_for_port "$FRONTEND_PORT" 120; then
    printf '%s' "$C_ERR"; tail -n 30 "$LOG_DIR/frontend.log"; printf '%s' "$C_RESET"
    die "前端启动失败，完整日志见 .dev-logs/frontend.log"
  fi
fi

printf '\n%s✔ 就绪%s\n' "$C_OK" "$C_RESET"
[[ "$TARGET" != "backend"  ]] && printf '  前端  %shttp://localhost:%s%s\n' "$C_FE" "$FRONTEND_PORT" "$C_RESET"
[[ "$TARGET" != "frontend" ]] && printf '  后端  %shttp://localhost:%s/docs%s\n' "$C_BE" "$BACKEND_PORT" "$C_RESET"
printf '  日志  %s%s%s\n' "$C_DIM" "$LOG_DIR" "$C_RESET"
printf '%s  Ctrl-C 停止全部服务%s\n\n' "$C_DIM" "$C_RESET"

if [[ "$OPEN_BROWSER" -eq 1 && "$TARGET" != "backend" ]]; then
  url="http://localhost:$FRONTEND_PORT"
  if command -v wslview >/dev/null 2>&1; then wslview "$url" >/dev/null 2>&1 &
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$url" >/dev/null 2>&1 &
  elif command -v open >/dev/null 2>&1; then open "$url" >/dev/null 2>&1 &
  fi
fi

wait
