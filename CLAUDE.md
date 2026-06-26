# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Open-source foundation for building autonomous web-research agents on top of Firecrawl. The repo is a layered stack: a publishable core library (`agent-core/`), deployment templates that embed it (`agent-templates/`), and internal tooling (`.internal/`, gitignored from the published surface conceptually but checked in here). Users scaffold projects via the `firecrawl` CLI (`firecrawl create agent -t next|express|library`).

## Layout

| Path | What it is |
|------|-----------|
| `agent-core/` | **The canonical source.** Publishable library `@firecrawl/agent-core`. All real work happens here. |
| `agent-templates/{next,express,library}/` | Deployable apps. Each contains a **vendored copy** of `agent-core/` at `./agent-core` (not a symlink, not a dependency). |
| `.internal/cli/` | The `firecrawl-agent` scaffolding CLI (TypeScript, commander). |
| `.internal/scripts/sync-agent-core.mjs` | Copies canonical `agent-core/` into each template. |
| `.internal/agent-core-py/` | Experimental Python port. |
| `.internal/experimental/agent-sdks/` | Generated OpenAPI client SDKs (many languages) — generated artifacts, don't hand-edit. |

### Critical: agent-core is vendored into templates

`agent-templates/*/agent-core/` are **copies**, kept in sync by `.internal/scripts/sync-agent-core.mjs`. When you change `agent-core/src/...`, the template copies are now stale.

```bash
node .internal/scripts/sync-agent-core.mjs            # sync all templates
node .internal/scripts/sync-agent-core.mjs --check    # CI drift check (fails if out of sync)
node .internal/scripts/sync-agent-core.mjs --dry-run
node .internal/scripts/sync-agent-core.mjs --target agent-templates/next
```

Don't edit `agent-templates/*/agent-core/` directly — edit `agent-core/` and re-sync.

## Commands

All from `agent-core/` unless noted. Package manager: pnpm (`pnpm-lock.yaml` present) but npm works.

```bash
# agent-core
npm test                         # vitest run (all tests)
npm run test:watch
npx vitest run src/toolkit.test.ts          # single test file
npx vitest run -t "name of test"            # single test by name
npm run typecheck                # tsc --noEmit
npm run build                    # tsup → dist/

# *.live.test.ts files hit real provider APIs — they need API keys in env and are slow.

# Examples (need FIRECRAWL_API_KEY + a provider key in env)
npm run example:basic | example:structured | example:parallel | example:skills | example:stream

# Templates (next / express / library) — run from the template dir
npm run dev                      # express: tsx watch; next: next dev; library: tsx watch index.ts
npm run doctor                   # express/library: env preflight check
npm run typecheck

# CLI (.internal/cli)
npm run build                    # tsc
```

## Architecture

The agent is a plan-act-observe loop combining Firecrawl web tools with any LLM provider.

- **Harness**: LangChain's **Deep Agents** (`deepagents`, `createDeepAgent`) provides the agent loop, sub-agent (`task`) dispatch, on-demand SKILL.md loading, a virtual filesystem, and built-in summarization middleware.
- **Models**: provider-agnostic. `ModelConfig = { provider, model }` where provider is `google | anthropic | openai | gateway | custom-openai`. Two resolvers exist — `resolveModel` (→ AI SDK `LanguageModel`, used by the orchestrator path) and `resolveLcModel` in `agent.ts` (→ LangChain chat model via `initChatModel`, used by Deep Agents). Keys come from `apiKeys` map or env.
- **Tools** are authored once in Vercel **AI SDK** `ToolSet` shape (so the same toolkit drops into either runtime) and wrapped to LangChain tools by the `aiToLc` / `aiToolToLc` adapter for Deep Agents.

### Two agent runtimes (important)

There are **two** orchestration implementations and they are not the same path:

1. **`src/agent.ts` — `FirecrawlAgent` / `createAgent`** — the **primary, public API**. Built on Deep Agents (`createDeepAgent`). This is what templates and the README use. `run()`, `stream()`, `plan()`, `toResponse()`, `sse()`.
2. **`src/orchestrator/index.ts` — `createOrchestrator`** — a lower-level path built on the AI SDK's `ToolLoopAgent`, with its own skill tools, worker tools, and explicit context compaction (`compaction.ts`). Exported for advanced users who want to bypass `createAgent`. Don't assume a change to one path affects the other.

### Toolkit abstraction

`agent-core` never imports a tool provider directly in the agent loop — it consumes a `Toolkit` (`{ tools, systemPrompt?, createFiltered? }`). `buildFirecrawlToolkit()` (in `toolkit.ts`) is the default, wrapping `firecrawl-aisdk` (search / scrape / interact / map / crawl). `createToolkit()` (`toolkit-builder.ts`) builds a `Toolkit` from any custom search/scrape backend. `createFiltered(enabledTools)` produces a reduced toolset for sub-agents.

- `bash: true` swaps `scrape` for `scrapeBash` — pages load into a WASM sandbox queried with rg/grep/sed so full markdown never enters LLM context.
- `interact` calls are wrapped with a hard timeout (`wrapInteractWithTimeout`, default 60s) and have null fields stripped so the model doesn't echo `null`.

### Skills

SKILL.md playbooks under `src/skills/definitions/<name>/`. Auto-discovered by `discoverSkills()` (reads frontmatter via `gray-matter`). A skill may include `sites/*.md` site-playbooks keyed by `domains:` frontmatter — `buildDomainIndex()` maps a hostname → the right playbook so site-specific guidance loads when the agent visits a known URL. Deep Agents loads skills on demand. `exportSkill` turns a successful run into a reusable SKILL.md (+ workflow script + schema).

### Structured output & enforcement (a defining design choice)

Output is produced by the model calling the **`formatOutput`** tool — the run "isn't done until formatOutput is called". Two hard gates are enforced in code (not just prompt advice), in the `aiToolToLc` wrapper in `agent.ts`:

1. **Data-collection gate**: `formatOutput` is rejected with an error until at least one data tool (search/scrape/interact/…) has returned non-empty output (`resultHasData`). Stops premature/stub output.
2. **Schema-adherence gate**: when a `schema` + `format:"json"` is set, every `formatOutput` call is validated (`validateAgainstSchema`) and bounced back with the missing/extra fields until it passes, bounded by `MAX_SCHEMA_REPAIRS` (3). A final post-run check sets `RunResult.schemaMismatch` if it still slipped through.

Sub-agents get a **restricted** toolset: data tools + `bashExec`, but **not** `formatOutput`/`exportSkill` (those are orchestrator-only, or a sub-agent would emit final output mid-run with partial data). The default `general-purpose` sub-agent is explicitly overridden in `agent.ts` so it inherits this restricted set instead of cloning the parent's tools.

### HTTP API

`agent-core/openapi.yaml` is the contract; all templates implement it (`POST /v1/run`, etc.). `createAgentFromEnv()` reads `FIRECRAWL_API_KEY`, provider keys, and `MODEL`/`MODEL_PROVIDER`/`MODEL_ID` env vars, and throws clear errors when the selected provider's key is missing.

## Conventions

- ESM throughout (`"type": "module"`), TypeScript, Node ≥ 20.
- `.gitattributes` forces **LF** line endings repo-wide — keep it that way on Windows.
- LangChain/AI-SDK provider packages (`@langchain/google`, `@ai-sdk/openai`, etc.) are **optional peer deps**, imported lazily by provider. Install only the one(s) for the provider you use.
- Tests live next to source as `*.test.ts`; `*.live.test.ts` are real-API integration tests (slow, need keys).
