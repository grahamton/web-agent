/**
 * mcp.ts — bridge any running MCP server into agent-core's Toolkit, and a
 * first-class `createMcpAgentFromEnv()` factory for local-model setups.
 *
 * Instead of Firecrawl, this wraps a Model Context Protocol server's tools as
 * AI SDK tools (the exact `ToolSet` shape a `Toolkit` expects), so all of
 * agent-core's orchestration, sub-agents, skills, schema enforcement and
 * streaming keep working unchanged.
 *
 * Pairs naturally with a local OpenAI-compatible model (Ollama, llama.cpp,
 * LM Studio, vLLM): point `OPENAI_BASE_URL` at the endpoint and run a
 * tool-calling-capable model. The tool wrapper adds timeouts, retries and
 * output truncation — small local models are far more reliable when a single
 * tool call can't hang the loop or flood a short context window.
 *
 * The `@modelcontextprotocol/sdk` package is an OPTIONAL peer dependency and is
 * imported lazily — only installs/loads when you actually use this path.
 *
 * ```ts
 * import { createMcpAgentFromEnv } from "@firecrawl/agent-core";
 *
 * const { agent, close } = await createMcpAgentFromEnv();
 * try {
 *   for await (const ev of agent.stream({ prompt: "..." })) { ... }
 * } finally {
 *   await close();
 * }
 * ```
 */

import { tool, jsonSchema, type ToolSet } from "ai";
import { createAgent, FirecrawlAgent } from "./agent";
import type { CreateAgentOptions, Toolkit } from "./types";

// Type-only imports are erased at runtime, so they don't pull the optional
// peer dependency into the runtime require graph.
import type { Client } from "@modelcontextprotocol/sdk/client/index.js";

/** Transport selection for the MCP connection. */
export type McpTransport = "auto" | "http" | "sse";

/** Knobs that harden tool calls against small/local model failure modes. */
export interface McpHardeningOptions {
  /**
   * Hard cap for a single MCP tool call. On timeout the call is retried (up to
   * `toolRetries`), then resolves with an error envelope instead of hanging the
   * agent loop. Default: 60_000 (60s). Set `0` to disable.
   */
  toolTimeoutMs?: number;
  /**
   * Extra attempts after the first failure/timeout. Default: 1 (so 2 attempts
   * total). Set `0` for no retries.
   */
  toolRetries?: number;
  /**
   * Max characters of a single tool result handed back to the model. Long
   * scrapes are truncated with a marker so they can't blow a small context
   * window. Default: 12_000. Set `0` to disable truncation.
   */
  maxToolChars?: number;
}

export interface BuildMcpToolkitOptions extends McpHardeningOptions {
  /** MCP endpoint, e.g. http://host:3100/mcp (Streamable HTTP) or .../sse. */
  url: string;
  /** Force a transport. Default "auto": try Streamable HTTP, fall back to SSE. */
  transport?: McpTransport;
  /**
   * Allowlist of tool names to expose. When omitted, every tool the server
   * publishes is kept. Use this to drop side-effectful or workflow-specific
   * tools and keep a clean read-only research surface.
   */
  keep?: string[];
  /** System prompt snippet describing the tools. Defaults to a generic one. */
  systemPrompt?: string;
  /** MCP client identifier reported in the handshake. Default "agent-core". */
  clientName?: string;
}

/** A connected MCP-backed toolkit plus the lifecycle handle to close it. */
export interface McpToolkitHandle {
  toolkit: Toolkit;
  /** The underlying MCP client — already connected. */
  client: Client;
  /** Which transport actually connected ("http" | "sse"). */
  transport: string;
  /** Names of the tools that were kept and wrapped. */
  toolNames: string[];
  /** Close the MCP connection so the process can exit cleanly. */
  close: () => Promise<void>;
}

const DEFAULT_SYSTEM_PROMPT = `You are a general web research agent connected to search/scrape/crawl tools over MCP.
Use the available tools to gather information before answering. Prefer a quick
search or single-page fetch over heavier multi-page tools unless one page can't
answer the question. Always cite the source URLs you used.`;

const DEFAULTS = {
  toolTimeoutMs: 60_000,
  toolRetries: 1,
  maxToolChars: 12_000,
};

/**
 * Flatten an MCP callTool result into a string the model (and the formatOutput
 * data-gate) can consume.
 */
function flattenToolResult(result: {
  content?: Array<Record<string, unknown>>;
  isError?: boolean;
  structuredContent?: unknown;
}): string {
  if (result.structuredContent !== undefined) {
    return typeof result.structuredContent === "string"
      ? result.structuredContent
      : JSON.stringify(result.structuredContent);
  }
  const parts = (result.content ?? [])
    .map((c) => (c.type === "text" && typeof c.text === "string" ? c.text : JSON.stringify(c)))
    .filter(Boolean);
  const text = parts.join("\n");
  if (result.isError) return JSON.stringify({ error: text || "MCP tool returned an error" });
  return text;
}

function truncate(text: string, maxChars: number): string {
  if (!maxChars || text.length <= maxChars) return text;
  const dropped = text.length - maxChars;
  return (
    text.slice(0, maxChars) +
    `\n\n[...truncated ${dropped} chars. Narrow your query, or use a focused tool ` +
    `like ask_page to extract just the fact you need.]`
  );
}

function errMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/** Race a promise against a timeout; rejects with a labeled error on timeout. */
function withTimeout<T>(p: Promise<T>, ms: number, label: string): Promise<T> {
  if (!ms || ms <= 0) return p;
  let timer: ReturnType<typeof setTimeout>;
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new Error(`timed out after ${ms}ms`)), ms);
  });
  return Promise.race([p, timeout]).finally(() => clearTimeout(timer)) as Promise<T>;
}

/**
 * Call an MCP tool with a timeout and bounded retries. On final failure it
 * resolves with an error-shaped result (rather than throwing) so the agent loop
 * survives — the model sees the error and can adapt or try a different tool.
 */
async function callToolWithRetry(
  client: Client,
  name: string,
  args: Record<string, unknown>,
  timeoutMs: number,
  retries: number,
): Promise<{ content?: Array<Record<string, unknown>>; isError?: boolean; structuredContent?: unknown }> {
  let lastErr: unknown;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      return (await withTimeout(
        client.callTool({ name, arguments: args }),
        timeoutMs,
        name,
      )) as never;
    } catch (err) {
      lastErr = err;
    }
  }
  return {
    isError: true,
    content: [
      {
        type: "text",
        text: `Tool "${name}" failed after ${retries + 1} attempt(s): ${errMessage(lastErr)}`,
      },
    ],
  };
}

type JsonSchemaShape = { properties?: Record<string, unknown>; required?: string[] };

// Common wrong keys small models reach for, grouped by the canonical param they
// usually mean. Used as a first pass before the positional fallback below.
const ARG_ALIASES: Record<string, string[]> = {
  url: ["file_path", "filepath", "path", "file", "link", "href", "uri", "page"],
  query: ["q", "search", "term", "text", "question", "keywords"],
};

/**
 * Repair argument keys a small model got wrong before sending them to the MCP
 * server. Many local models confuse one tool's schema with another's — e.g.
 * calling `scrape_url` with `file_path` (bleeding in from Deep Agents' built-in
 * filesystem tools) instead of `url`. A wrong key fails server-side validation,
 * stalls the data gate, and burns retries.
 *
 * Two passes, both schema-driven (no per-tool hardcoding):
 *   1. Alias map: rename a known wrong key to a missing required param.
 *   2. Positional fallback: if exactly one provided key is unrecognized and
 *      exactly one required param is still missing, assume the model just
 *      mislabeled it and remap.
 *
 * Keys already valid per the schema are passed through untouched.
 */
function normalizeArgs(args: Record<string, unknown>, schema: JsonSchemaShape): Record<string, unknown> {
  const props = schema.properties;
  if (!props || typeof props !== "object") return args;
  const valid = new Set(Object.keys(props));
  if (Object.keys(args).every((k) => valid.has(k))) return args;

  const out: Record<string, unknown> = {};
  const unmatched: string[] = [];
  for (const [k, v] of Object.entries(args)) {
    if (valid.has(k)) out[k] = v;
    else unmatched.push(k);
  }
  const required = (Array.isArray(schema.required) ? schema.required : []).filter((r) => valid.has(r));
  const missing = () => required.filter((r) => !(r in out));

  // 1) alias-based remap
  for (const key of [...unmatched]) {
    const canon = missing().find((r) => ARG_ALIASES[r]?.includes(key.toLowerCase()));
    if (canon) {
      out[canon] = args[key];
      unmatched.splice(unmatched.indexOf(key), 1);
    }
  }

  // 2) positional fallback: one stray key + one missing required → remap
  const stillMissing = missing();
  if (unmatched.length === 1 && stillMissing.length === 1) {
    out[stillMissing[0]] = args[unmatched[0]];
    unmatched.length = 0;
  }

  // Preserve anything we couldn't place (server will report it) so behavior
  // only ever improves, never silently drops data the model intended to send.
  for (const key of unmatched) out[key] = args[key];
  return out;
}

/** Build an AI SDK ToolSet from the MCP server's tool list, with hardening. */
function mcpToToolSet(
  client: Client,
  mcpTools: Array<{ name: string; description?: string; inputSchema?: unknown }>,
  opts: { keep?: string[]; timeoutMs: number; retries: number; maxChars: number },
): ToolSet {
  const toolSet: ToolSet = {};
  for (const t of mcpTools) {
    if (opts.keep && !opts.keep.includes(t.name)) continue;
    const schema = (t.inputSchema ?? { type: "object", properties: {} }) as JsonSchemaShape &
      Record<string, unknown>;
    const required = Array.isArray(schema.required) ? schema.required : [];
    // Name the required params in the description — a cheap up-front nudge that
    // cuts down on mis-keyed calls from small models.
    const hint = required.length ? ` (required arguments: ${required.join(", ")})` : "";
    toolSet[t.name] = tool({
      description: (t.description ?? "") + hint,
      // MCP inputSchema is already JSON Schema; jsonSchema() adapts it for the AI SDK.
      inputSchema: jsonSchema(schema as never),
      execute: async (args) => {
        const fixed = normalizeArgs((args ?? {}) as Record<string, unknown>, schema);
        const res = await callToolWithRetry(client, t.name, fixed, opts.timeoutMs, opts.retries);
        return truncate(flattenToolResult(res), opts.maxChars);
      },
    });
  }
  return toolSet;
}

const isHttp = (t: string) => t === "http" || t === "streamable" || t === "streamablehttp";

/**
 * Connect to the MCP server, picking a transport. "auto" tries Streamable HTTP
 * (the current MCP standard) first and falls back to legacy SSE.
 */
async function connectMcp(
  url: string,
  transport: McpTransport,
  clientName: string,
): Promise<{ client: Client; transport: string }> {
  const { Client } = await import("@modelcontextprotocol/sdk/client/index.js");
  const { StreamableHTTPClientTransport } = await import(
    "@modelcontextprotocol/sdk/client/streamableHttp.js"
  );
  const { SSEClientTransport } = await import("@modelcontextprotocol/sdk/client/sse.js");

  const target = new URL(url);
  const pref = transport.toLowerCase();
  const order = pref === "sse" ? ["sse"] : isHttp(pref) ? ["http"] : ["http", "sse"];

  let lastErr: unknown;
  for (const label of order) {
    const client = new Client({ name: clientName, version: "0.1.0" });
    try {
      await client.connect(
        label === "http"
          ? new StreamableHTTPClientTransport(target)
          : new SSEClientTransport(target),
      );
      return { client, transport: label };
    } catch (err) {
      lastErr = err;
      await client.close().catch(() => {});
    }
  }
  throw lastErr;
}

/**
 * Connect to an MCP server and expose its tools as an agent-core `Toolkit`.
 * The caller owns the connection lifecycle — call `close()` when done.
 */
export async function buildMcpToolkit(options: BuildMcpToolkitOptions): Promise<McpToolkitHandle> {
  const { client, transport } = await connectMcp(
    options.url,
    options.transport ?? "auto",
    options.clientName ?? "agent-core",
  );

  try {
    const { tools: mcpTools } = await client.listTools();
    const tools = mcpToToolSet(client, mcpTools as never, {
      keep: options.keep,
      timeoutMs: options.toolTimeoutMs ?? DEFAULTS.toolTimeoutMs,
      retries: options.toolRetries ?? DEFAULTS.toolRetries,
      maxChars: options.maxToolChars ?? DEFAULTS.maxToolChars,
    });
    const toolNames = Object.keys(tools);
    if (toolNames.length === 0) {
      const exposed = (mcpTools as Array<{ name: string }>).map((t) => t.name).join(", ") || "(none)";
      throw new Error(
        `MCP server at ${options.url} exposed no matching tools. ` +
          `Server published: ${exposed}. ` +
          (options.keep ? `Adjust 'keep' to match those names.` : ``),
      );
    }

    const toolkit: Toolkit = {
      tools,
      systemPrompt: options.systemPrompt ?? DEFAULT_SYSTEM_PROMPT,
      createFiltered: (enabled) =>
        enabled
          ? Object.fromEntries(Object.entries(tools).filter(([name]) => enabled.includes(name)))
          : tools,
    };

    return { toolkit, client, transport, toolNames, close: () => client.close() };
  } catch (err) {
    await client.close().catch(() => {});
    throw err;
  }
}

export interface CreateMcpAgentOptions extends McpHardeningOptions {
  /** MCP endpoint. Defaults to env MCP_URL. */
  url?: string;
  /** Transport override. Defaults to env MCP_TRANSPORT, else "auto". */
  transport?: McpTransport;
  /** Local OpenAI-compatible endpoint. Defaults to env OPENAI_BASE_URL. */
  baseURL?: string;
  /** Model id served at that endpoint. Defaults to env MODEL_ID. */
  model?: string;
  /** API key for the endpoint. Defaults to env OPENAI_API_KEY, else "sk-local". */
  apiKey?: string;
  /** Tool allowlist passed through to the toolkit. */
  keep?: string[];
  /** System prompt describing the tools. */
  systemPrompt?: string;
  /**
   * Tool names that count as "data collected" and so unblock formatOutput.
   * Defaults to every kept MCP tool name — for a research backend, any tool
   * returning data should satisfy the gate.
   */
  dataToolNames?: string[];
  /** Extra agent options merged into the created agent (skills, appSections, …). */
  agentOptions?: Partial<CreateAgentOptions>;
}

/** A ready-to-run MCP-backed agent plus its connection lifecycle handle. */
export interface McpAgentHandle {
  agent: FirecrawlAgent;
  client: Client;
  transport: string;
  toolNames: string[];
  close: () => Promise<void>;
}

function requireEnv(value: string | undefined, name: string): string {
  if (!value) {
    throw new Error(
      `Missing required value for ${name}. Set the ${name} env var or pass it explicitly.`,
    );
  }
  return value;
}

/**
 * First-class factory: connect to an MCP server and wire it to a (typically
 * local) OpenAI-compatible model, returning a ready agent.
 *
 * The model uses provider "openai" + `baseURL` — the `createAgent` path resolves
 * models through LangChain's `initChatModel`, which has no "custom-openai"
 * provider, so "openai" + baseURL is how you target a local endpoint here.
 *
 * The returned `close()` must be called to release the MCP connection.
 */
export async function createMcpAgentFromEnv(
  overrides: CreateMcpAgentOptions = {},
): Promise<McpAgentHandle> {
  const url = overrides.url ?? requireEnv(process.env.MCP_URL, "MCP_URL");
  const baseURL = overrides.baseURL ?? requireEnv(process.env.OPENAI_BASE_URL, "OPENAI_BASE_URL");
  const modelId = overrides.model ?? requireEnv(process.env.MODEL_ID, "MODEL_ID");
  const apiKey = overrides.apiKey ?? process.env.OPENAI_API_KEY ?? "sk-local";
  const transport = overrides.transport ?? (process.env.MCP_TRANSPORT as McpTransport) ?? "auto";

  const handle = await buildMcpToolkit({
    url,
    transport,
    keep: overrides.keep,
    systemPrompt: overrides.systemPrompt,
    toolTimeoutMs: overrides.toolTimeoutMs,
    toolRetries: overrides.toolRetries,
    maxToolChars: overrides.maxToolChars,
  });

  const agent = createAgent({
    toolkit: handle.toolkit,
    model: { provider: "openai", model: modelId, baseURL, apiKey },
    dataToolNames: overrides.dataToolNames ?? handle.toolNames,
    ...overrides.agentOptions,
  });

  return {
    agent,
    client: handle.client,
    transport: handle.transport,
    toolNames: handle.toolNames,
    close: handle.close,
  };
}
