#!/usr/bin/env bash
#
# setup-mcp.sh - Corrected MCP setup for Codex CLI + ChatGPT desktop.
#
# WHY THIS SCRIPT EXISTS
# ----------------------
# The original runbook this replaces could not work as written. Verified failures:
#
#   1. `@vercel/mcp` does not exist on npm (HTTP 404). There is no stdio Vercel MCP
#      package and no VERCEL_TOKEN is involved. Vercel's official MCP server is a
#      *remote OAuth* endpoint at https://mcp.vercel.com.
#   2. Codex CLI, the ChatGPT desktop app and the Codex IDE extension all share ONE
#      file: ~/.codex/config.toml (TOML). There is no ~/.config/chatgpt/mcp.json,
#      and symlinking a JSON file to ~/.codex/mcp.json is inert -- Codex never reads
#      it. JSON `mcpServers` blocks are a Claude/Cursor convention, not Codex's.
#   3. `omc setup --agent-target=openai` is not a real flag
#      (error: unknown option '--agent-target=openai').
#
# This script writes the real config, idempotently, and never clobbers unrelated
# settings.
#
# USAGE
#   ./setup-mcp.sh            # configure servers
#   ./setup-mcp.sh --verify   # configure, then run a live MCP handshake per server
#   CODEX_HOME=/path ./setup-mcp.sh
#
set -euo pipefail

CODEX_HOME_DIR="${CODEX_HOME:-$HOME/.codex}"
CONFIG="$CODEX_HOME_DIR/config.toml"
VERIFY=0
[[ "${1:-}" == "--verify" ]] && VERIFY=1

c_ok()   { printf '\033[32m  ok\033[0m   %s\n' "$*"; }
c_warn() { printf '\033[33m  warn\033[0m %s\n' "$*"; }
c_err()  { printf '\033[31m  fail\033[0m %s\n' "$*"; }
hdr()    { printf '\n\033[1m%s\033[0m\n' "$*"; }

# ---------------------------------------------------------------------------
# Phase 1 - Prerequisites
# ---------------------------------------------------------------------------
hdr "Phase 1 - Prerequisites"

if ! command -v node >/dev/null 2>&1; then
  c_err "node not found. Install Node.js >= 20.19 (https://nodejs.org)."
  exit 1
fi

NODE_V="$(node -v)"; NODE_MAJOR="${NODE_V#v}"; NODE_MAJOR="${NODE_MAJOR%%.*}"
NODE_MINOR="$(node -p 'process.versions.node.split(".")[1]' 2>/dev/null || echo 0)"
if (( NODE_MAJOR > 23 )) || (( NODE_MAJOR == 22 && NODE_MINOR >= 12 )) || (( NODE_MAJOR == 20 && NODE_MINOR >= 19 )); then
  c_ok "node $NODE_V (satisfies chrome-devtools-mcp: ^20.19 || ^22.12 || >=23)"
else
  c_warn "node $NODE_V is below chrome-devtools-mcp's range (^20.19 || ^22.12 || >=23)."
  c_warn "Most servers will run, but chrome-devtools-mcp may not."
fi
command -v npm >/dev/null 2>&1 && c_ok "npm $(npm -v)" || { c_err "npm not found"; exit 1; }

# Chrome is a HARD requirement for chrome-devtools-mcp -- not a soft one.
# Verified: the server advertises 30 tools and then every single call fails with
#   "Could not find Google Chrome executable for channel 'stable' at: /opt/google/chrome/chrome."
CHROME_BIN=""
for cand in google-chrome google-chrome-stable chromium chromium-browser \
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"; do
  if command -v "$cand" >/dev/null 2>&1 || [[ -x "$cand" ]]; then CHROME_BIN="$cand"; break; fi
done
if [[ -n "$CHROME_BIN" ]]; then
  c_ok "Chrome found: $CHROME_BIN"
else
  c_warn "No Chrome/Chromium binary found. The chrome-devtools server will register"
  c_warn "and list tools, but EVERY tool call will fail. Install Chrome, or pass"
  c_warn "  --executablePath /path/to/chrome   (see README troubleshooting)."
fi

if command -v codex >/dev/null 2>&1; then
  c_ok "codex CLI: $(codex --version 2>/dev/null || echo present)"
  HAVE_CODEX=1
else
  c_warn "codex CLI not found. Will write config.toml directly instead of using"
  c_warn "'codex mcp add'. Install with: npm install -g @openai/codex"
  HAVE_CODEX=0
fi

echo
if command -v tmux >/dev/null 2>&1; then c_ok "tmux present ($(tmux -V))"; else
  c_warn "tmux not found -- 'omc team' (tmux worker panes) cannot run without it."
fi

# ---------------------------------------------------------------------------
# Phase 2 - Write the real config
# ---------------------------------------------------------------------------
hdr "Phase 2 - Configure MCP servers in $CONFIG"

mkdir -p "$CODEX_HOME_DIR"

# Back up an existing config so a hand-edit is always recoverable.
if [[ -f "$CONFIG" ]]; then
  BAK="$CONFIG.bak.$(date +%Y%m%d-%H%M%S)"
  cp "$CONFIG" "$BAK"
  c_ok "Backed up existing config -> $BAK"
fi

# Canonical server definitions. Remote servers use OAuth (no API token);
# local servers are stdio subprocesses spawned via npx.
read -r -d '' SPEC <<'JSON' || true
[
  {"name": "vercel",         "url": "https://mcp.vercel.com"},
  {"name": "supabase",       "url": "https://mcp.supabase.com/mcp"},
  {"name": "next-devtools",  "command": "npx", "args": ["-y", "next-devtools-mcp@latest"]},
  {"name": "chrome-devtools","command": "npx", "args": ["-y", "chrome-devtools-mcp@latest", "--isolated"]}
]
JSON

if [[ "$HAVE_CODEX" == "1" ]]; then
  # Preferred path: let Codex write its own schema. `codex mcp add` is idempotent
  # (re-adding a name updates it in place, verified: no duplicate sections).
  # NB: delimiter must be a NON-whitespace char. A tab is whitespace, so `read`
  # would collapse the empty url field of stdio servers and shift every field.
  while IFS=$'\x1f' read -r name url command args; do
    if [[ -n "$url" ]]; then
      # shellcheck disable=SC2086
      if err="$(codex mcp add "$name" --url "$url" 2>&1 >/dev/null)"; then
        c_ok "registered (http)  $name -> $url"
      else
        c_err "failed to register $name: $(printf '%s' "$err" | head -1)"
      fi
    else
      # shellcheck disable=SC2086
      if err="$(codex mcp add "$name" -- $command $args 2>&1 >/dev/null)"; then
        c_ok "registered (stdio) $name -> $command $args"
      else
        c_err "failed to register $name: $(printf '%s' "$err" | head -1)"
      fi
    fi
  done < <(printf '%s' "$SPEC" | python3 -c '
import json,sys
for s in json.load(sys.stdin):
    print("\x1f".join([s["name"], s.get("url",""), s.get("command",""), " ".join(s.get("args",[]))]))
')
else
  # Fallback: merge TOML ourselves, preserving unrelated keys. Validated with
  # tomllib so we cannot emit a file that breaks the user's whole Codex config.
  # The spec travels in an env var: stdin is already taken by this heredoc.
  MCP_SPEC="$SPEC" python3 - "$CONFIG" <<'PY'
import json, os, sys, tomllib
from pathlib import Path

config = Path(sys.argv[1])
spec = json.loads(os.environ["MCP_SPEC"])
text = config.read_text() if config.exists() else ""

lines = text.splitlines()
for s in spec:
    name = s["name"]
    if s.get("url"):
        block = [f"[mcp_servers.{name}]", f'url = "{s["url"]}"']
    else:
        args = ", ".join(f'"{a}"' for a in s["args"])
        block = [f"[mcp_servers.{name}]", f'command = "{s["command"]}"', f"args = [{args}]"]
    header = f"[mcp_servers.{name}]"
    if header in lines:                       # replace existing block in place
        i = lines.index(header)
        j = i + 1
        while j < len(lines) and not lines[j].lstrip().startswith("["):
            j += 1
        lines[i:j] = block
        # keep a blank line between TOML tables for readability
        k = i + len(block)
        if k < len(lines) and lines[k].strip():
            lines.insert(k, "")
    else:                                     # append new block
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend(block)
    print(f"  ok   merged (toml) {name}")

out = "\n".join(lines).rstrip() + "\n"
tomllib.loads(out)                            # refuse to write invalid TOML
config.write_text(out)
PY
fi

# ---------------------------------------------------------------------------
# Phase 3 - Verify
# ---------------------------------------------------------------------------
hdr "Phase 3 - Verify"

if [[ "$HAVE_CODEX" == "1" ]]; then
  codex mcp list 2>&1 | sed 's/^/  /'
else
  echo "  (codex CLI absent) config.toml written at $CONFIG:"
  sed 's/^/  /' "$CONFIG"
fi

cat <<'NEXT'

  Remote servers authenticate via OAuth, NOT an API token. Complete the browser
  consent flow once per server:

      codex mcp login vercel
      codex mcp login supabase

NEXT

if [[ "$VERIFY" == "1" ]]; then
  hdr "Phase 4 - Live MCP handshake"
  # Speaks real JSON-RPC (initialize -> tools/list) to each stdio server, so
  # "configured" is proven rather than assumed.
  python3 - <<'PY'
import json, subprocess, sys

SPEC = [
    ("next-devtools",   ["npx", "-y", "next-devtools-mcp@latest"]),
    ("chrome-devtools", ["npx", "-y", "chrome-devtools-mcp@latest", "--isolated"]),
]

def handshake(name, cmd):
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    def send(o):
        p.stdin.write(json.dumps(o) + "\n"); p.stdin.flush()
    try:
        send({"jsonrpc":"2.0","id":1,"method":"initialize","params":{
            "protocolVersion":"2025-06-18","capabilities":{},
            "clientInfo":{"name":"setup-mcp","version":"1.0"}}})
        server = tools = None
        for line in p.stdout:
            if not line.lstrip().startswith("{"): continue
            try: m = json.loads(line)
            except ValueError: continue
            if m.get("id") == 1:
                server = (m.get("result") or {}).get("serverInfo", {}).get("name")
                send({"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}})
            elif m.get("id") == 2:
                tools = len((m.get("result") or {}).get("tools", []))
                break
        if server:
            print(f"  ok   {name}: handshake ok (server={server}, tools={tools})")
        else:
            print(f"  warn {name}: no initialize response")
    except Exception as e:
        print(f"  fail {name}: {e}")
    finally:
        p.kill()

for name, cmd in SPEC:
    handshake(name, cmd)
PY
  echo
  echo "  Note: remote OAuth servers are not handshaken here (a browser is required);"
  echo "  their reachability is confirmed by 'codex mcp login <name>'."
fi

hdr "Done"
echo "  Config: $CONFIG"
echo "  Note: this one file is shared by the Codex CLI, the Codex IDE extension and"
echo "        the ChatGPT desktop app -- configure once, and all three see it."
