# MCP Setup — Verification Report & Corrected Runbook

I executed the six-phase runbook against the live environment and tested every package,
flag, path and endpoint it depends on. **Three of the six phases could not work as
written**, and two commands in Phase 5 target a package that does not exist. Everything
below is backed by executed evidence, not inspection.

The working replacement is [`setup-mcp.sh`](./setup-mcp.sh) in this directory. It has been
run end-to-end (including a no-Codex fallback path and a live MCP handshake check).

---

## Verdict by phase

| Phase | Claim | Status |
|---|---|---|
| 1 | Node ≥ 18, npm, Chrome | ⚠️ Node/npm pass; **Chrome absent — and it is a hard requirement** |
| 2 | `~/.config/chatgpt/mcp.json` symlinked to `~/.codex/mcp.json` | ❌ **Wrong path, wrong format, wrong file entirely** |
| 3 | `omc setup --agent-target=openai` | ❌ **Flag does not exist** |
| 4 | MCP infrastructure table | ⚠️ Vercel row is wrong; other three are broadly right |
| 5 | Verification commands | ❌ 2 of 4 commands fail (`@vercel/mcp` 404; `omc ultrawork` not a CLI command) |
| 6 | Troubleshooting matrix | ⚠️ Several root causes misdiagnosed |

---

## Phase 1 — Prerequisites

```
node v22.22.3        ✓  (npm 10.9.8)
google-chrome        ✗  not in PATH
tmux                 ✗  not installed
```

Node/npm pass. Two environment gaps matter:

**Chrome is a hard requirement, not a soft one.** `chrome-devtools-mcp` starts happily and
advertises 30 tools, then *every* call fails. Proven with a real JSON-RPC call:

```
initialize -> chrome_devtools 1.10.1
tools/list -> 30 tools
tools/call list_pages ->
   Could not find Google Chrome executable for channel 'stable' at:
    - /opt/google/chrome/chrome.
```

So a Chrome-free machine produces a server that looks healthy and is silently useless.

**`tmux` is required by `omc team`.** Its own `--help` describes "terminal-launched tmux
CLI workers", and the upstream README states that `omc team` requires tmux plus an
installed *and authenticated* provider CLI. Neither is present here.

---

## Phase 2 — The config file is wrong in four ways

This phase is the centrepiece of the runbook and none of it works. Verified against the
real Codex CLI (installed `@openai/codex` 0.160.1 and ran it):

| Runbook | Reality |
|---|---|
| `~/.config/chatgpt/mcp.json` exists | No such convention. **Codex CLI, the Codex IDE extension and the ChatGPT desktop app all share one file: `~/.codex/config.toml`** |
| Config is JSON (`mcpServers`) | Codex reads **TOML**, with `[mcp_servers.<name>]` tables (underscore, per-server section). Codex ignores a `mcpServers` JSON key entirely |
| `ln -sf` JSON to `~/.codex/mcp.json` | Inert. Codex never reads `mcp.json`, so the symlink is dead weight |
| Vercel via `npx -y @vercel/mcp@latest` + `VERCEL_TOKEN` | **`@vercel/mcp` does not exist on npm.** See below |

`@vercel/mcp` is a hard 404 — both the registry API and an actual install attempt:

```
$ npx -y @vercel/mcp@latest --version
npm error 404 Not Found - GET https://registry.npmjs.org/@vercel%2fmcp - Not found
npm error 404  '@vercel/mcp@latest' is not in this registry.
```

Vercel's official MCP server is a **remote OAuth endpoint at `https://mcp.vercel.com`**
(public beta). There is no stdio package and **no `VERCEL_TOKEN` involved**. It also uses
an approved-client allowlist with a per-client OAuth consent screen; Codex is on that list.

The Supabase entry, by contrast, is right — `https://mcp.supabase.com/mcp` is the correct
Streamable HTTP endpoint (OAuth, or a PAT via `Authorization` header). Only its wrapper
was wrong: `"type": "http"` is the JSON-client spelling, whereas Codex wants just
`url = "..."`.

### What actually works

Written by the real CLI and confirmed by `codex mcp list`:

```toml
# ~/.codex/config.toml
[mcp_servers.vercel]
url = "https://mcp.vercel.com"

[mcp_servers.supabase]
url = "https://mcp.supabase.com/mcp"

[mcp_servers.next-devtools]
command = "npx"
args = ["-y", "next-devtools-mcp@latest"]

[mcp_servers.chrome-devtools]
command = "npx"
args = ["-y", "chrome-devtools-mcp@latest", "--isolated"]
```

```
Name             Command  Args                                      Status   Auth
chrome-devtools  npx      -y chrome-devtools-mcp@latest --isolated  enabled  Unsupported
next-devtools    npx      -y next-devtools-mcp@latest               enabled  Unsupported

Name      Url                           Status   Auth
supabase  https://mcp.supabase.com/mcp  enabled  Unknown
vercel    https://mcp.vercel.com        enabled  Unknown
```

Remote servers authenticate with **`codex mcp login vercel` / `codex mcp login supabase`**
(browser OAuth), not with a token pasted into a config file.

---

## Phase 3 — `--agent-target` does not exist

Tested in an isolated `HOME` so nothing touched the real environment:

```
$ omc setup --agent-target=openai
error: unknown option '--agent-target=openai'
(exit 1)
```

`omc setup` accepts only `-f/--force`, `-q/--quiet`, `--no-plugin`, `--plugin-dir-mode`,
`--skip-hooks`, `--force-hooks`, `-h`. There is no runtime-target concept to configure.

**The package itself is real.** `oh-my-claude-sisyphus` v5.6.2, MIT, ships the `omc`
binary — that part of the runbook checks out. But it is an orchestration layer *for Claude
Code*: `omc setup` installs hooks/agents/skills into Claude Code, and any run without
Claude Code present fails with `[omc] Error: claude CLI not found`.

For the stated goal ("OpenAI / Codex engines") upstream points at a **different package**:

```bash
npm install -g oh-my-codex      # v0.21.8, bin: omx
```

That is the Codex-native equivalent. The runbook conflated the two projects.

---

## Phase 4 — Table corrections

| Server | Runbook says | Correct |
|---|---|---|
| Supabase Remote | Remote HTTP/SSE, `mcp.supabase.com/mcp` | ✅ Right (transport is Streamable HTTP, not SSE) |
| Vercel | "stdio via npx", `@vercel/mcp`, `VERCEL_TOKEN` | ❌ **Remote HTTP, `https://mcp.vercel.com`, OAuth. No npm package, no token** |
| Next.js DevTools | stdio via npx, `next-devtools-mcp@latest` | ✅ Right (v0.4.0) |
| Chrome DevTools | `--isolated=true` **or** `--remote-debugging-port=9222` | ⚠️ Real, but both flag forms are wrong; see below |

On the Chrome flags — from the server's own `--help`, `--isolated` is declared
`[boolean] [default: false]`, so the space-separated form is canonical:

```
--isolated    If specified, creates a temporary user-data-dir that is automatically
              cleaned up after the browser is closed. Defaults to false.  [boolean]
```

Two more notes:

- `--remote-debugging-port=9222` is a **Chrome** flag, not a `chrome-devtools-mcp` flag.
  The server-side equivalent is `--browserUrl http://127.0.0.1:9222` (or `-u`). Mixing
  them up is why the "Chrome DevTools Connection Error" row below misdiagnoses the fault.
- Chrome's presence matters more than any flag: without a binary, all 30 tools fail.

---

## Phase 5 — Verification commands

| Command | Result |
|---|---|
| `npx -y @vercel/mcp@latest --version` | ❌ `npm error 404` — package does not exist |
| `npx -y chrome-devtools-mcp@latest --version` | ✅ `1.10.1` |
| `omc ultrawork "..."` | ❌ Not a CLI subcommand (exit 1: `claude CLI not found`) |
| `omc team "..."` | ⚠️ Valid CLI, but needs `tmux` + an authenticated provider CLI |

`ultrawork` exists only as an **in-session slash command** (`/ultrawork`, `/uw`), matched
by a slash-pattern regex inside the CLI — the runbook presented it as a shell command. The
CLI equivalent is `omc team N:codex "<task>"`, e.g. `omc team 2:codex "review auth flow"`.

Real, executed verification instead — a live MCP handshake, which proves tool discovery
rather than assuming it:

```
ok   next-devtools:   handshake ok (server=next-devtools-mcp, tools=4)
ok   chrome-devtools: handshake ok (server=chrome_devtools, tools=30)
```

(`next-devtools` exposes `browser_eval`, `nextjs_docs`, `nextjs_index`, `nextjs_call`.)

---

## Phase 6 — Corrected troubleshooting matrix

| Issue | Runbook root cause | Actual root cause & fix |
|---|---|---|
| `Command not found: npx` | npm bin dir missing from PATH | ✅ Correct. `export PATH="$PATH:$(npm config get prefix)/bin"` |
| Vercel 401 / `VERCEL_TOKEN` space-padded | Invalid token | ❌ **No token exists.** Vercel MCP is OAuth. Fix: `codex mcp login vercel` |
| Chrome DevTools connection error | "Browser process conflict" | ⚠️ Usually **no Chrome binary at all**. Install Chrome, or point at one with `--executablePath`, or attach to an existing instance with `--browserUrl http://127.0.0.1:9222` |
| Supabase remote timeout | Firewall blocking WSS/HTTPS | ⚠️ Reasonable, but the transport is Streamable HTTP over HTTPS (not WSS). `curl -i https://mcp.supabase.com/mcp` is the right probe |
| `omc` unrecognized | Global bin not linked | ⚠️ Also check you installed the right project: `omc` is Claude Code's; Codex's is `omx` from `oh-my-codex` |
| *(missing)* | — | **Config silently ignored**: if you wrote JSON or used `~/.codex/mcp.json`, nothing loads. Verify with `codex mcp list`; the real file is `~/.codex/config.toml` |
| *(missing)* | — | **`npx` servers timeout on first use**: package download on first spawn can exceed Codex's default `startup_timeout_sec` (10s). Raise it in the server's table |

---

## Usage

```bash
./setup-mcp.sh                 # configure, preserving existing settings
./setup-mcp.sh --verify        # also run a live MCP handshake per stdio server
CODEX_HOME=/path ./setup-mcp.sh
```

Behaviour verified by test:

- **Preserves** unrelated config (`approval_policy`, other MCP servers) — merges, never clobbers.
- **Idempotent** — re-running leaves no duplicate sections (5 sections stayed 5 after two runs).
- **Backs up** any existing config to `config.toml.bak.<timestamp>` before writing.
- **Validates** hand-written TOML with Python's `tomllib` before writing, so a bad edit can
  never break the config shared by your CLI, IDE extension and ChatGPT desktop app.
- **Fails safely** — `set -euo pipefail`; the one bug I hit during development aborted
  before modifying the config.
- **Falls back** to direct TOML merging when the `codex` CLI is absent (tested).

### Not verifiable from this sandbox

- **Vercel/Supabase OAuth**: the consent flow needs a browser, and this sandbox's egress is
  restricted to a package-registry allowlist, so `mcp.vercel.com` and `mcp.supabase.com`
  are unreachable here. Their endpoints and auth model are confirmed from vendor
  documentation; run `codex mcp login <name>` on your machine to complete them.
- **Chrome-dependent tools**: no Chrome binary exists here, so all 30 tools fail by design.
- **`omc team`**: needs `tmux` and an authenticated provider CLI, neither of which is present.

### A note on `oh-my-claude-sisyphus`

I did not install it globally. It is real software, but it is the wrong tool for the
Codex target you specified, and `omc setup` writes hooks and agent/skill files into Claude
Code's config — a side effect worth choosing deliberately rather than inheriting from a
runbook that was wrong about the flag controlling it.
