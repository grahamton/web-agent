/**
 * general-agent.ts — A general-purpose web agent (search / scrape / crawl)
 * powered by an existing MCP server and a local OpenAI-compatible model.
 *
 * Instead of Firecrawl, this bridges a running MCP server (here: the
 * `dtc-research` server, which wraps SearXNG + crawl4AI + ask_page + research)
 * into agent-core's Toolkit abstraction. We connect with the official MCP SDK,
 * list the server's tools, and wrap each one as an AI SDK tool — the exact
 * `ToolSet` shape a Toolkit expects — so all of agent-core's orchestration,
 * sub-agents, skills, schema enforcement and streaming keep working unchanged.
 *
 * (AI SDK v6 removed its built-in `experimental_createMCPClient`, so we use
 * `@modelcontextprotocol/sdk` directly.)
 *
 * Requirements:
 *   - The MCP server must be reachable over HTTP (Streamable HTTP or legacy SSE).
 *   - The local model MUST support OpenAI-style tool calling, or the agent
 *     loop won't function.
 *
 * Env:
 *   MCP_URL          MCP endpoint, e.g. http://homelab:3100/mcp (Streamable HTTP) or .../sse  (required)
 *   MCP_TRANSPORT    Force a transport: "http" (Streamable HTTP) or "sse". Default: auto —
 *                    try Streamable HTTP first, fall back to SSE.
 *   OPENAI_BASE_URL  Local OpenAI-compatible endpoint, e.g. http://homelab:1234/v1  (required)
 *   MODEL_ID         Model id served there, e.g. "gemma4:12b-tools"  (required)
 *   OPENAI_API_KEY   API key — a dummy like "sk-local" is fine for most local servers
 *
 * Run (after copying .env.example to .env and filling it in):
 *   npm run agent -- "your question or task"
 *   npm run agent -- "find the latest Firecrawl pricing and summarize it"
 */

import { tool, jsonSchema, type ToolSet } from "ai";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { SSEClientTransport } from "@modelcontextprotocol/sdk/client/sse.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { createAgent, type Toolkit } from "../src/index.js";

// ---------------------------------------------------------------------------
// 0.  Config from env
// ---------------------------------------------------------------------------

const MCP_URL = requireEnv("MCP_URL");
const MCP_TRANSPORT = (process.env.MCP_TRANSPORT ?? "auto").toLowerCase();
const OPENAI_BASE_URL = requireEnv("OPENAI_BASE_URL");
const MODEL_ID = requireEnv("MODEL_ID");
const OPENAI_API_KEY = process.env.OPENAI_API_KEY ?? "sk-local";

function requireEnv(name: string): string {
  const v = process.env[name];
  if (!v) {
    console.error(`Missing required env var: ${name}. See the header of this file.`);
    process.exit(1);
  }
  return v;
}

// Curated, general-purpose surface. We omit the dtc-research tools that have
// side effects or are workflow-specific (notify_telegram, save_report, track,
// site_memory, audit_storefront, …) and keep the read-only research tools.
const KEEP = ["search", "scrape_url", "browse", "ask_page", "research"];

const TOOLKIT_SYSTEM_PROMPT = `You are a general web research agent connected to a search/scrape/crawl backend over MCP.

Tool guidance:
- search: discover relevant pages for a query. Start here when you don't have a URL.
- scrape_url: fetch and read the full content of a single known URL.
- ask_page: ask a specific question of a single page and get a focused answer (cheaper than scraping when you only need one fact).
- browse / sweep: explore or crawl across multiple pages of a site when one page isn't enough.
- research: run a deeper multi-step investigation for a broad question.
- compare: evaluate two URLs head-to-head.

Prefer search → scrape_url/ask_page for most tasks. Reach for research/sweep only when a single page can't answer the question. Always cite the source URLs you used.`;

// ---------------------------------------------------------------------------
// MCP → AI SDK ToolSet
// ---------------------------------------------------------------------------

/** Flatten an MCP callTool result into a string the model (and the
 *  formatOutput data-gate) can consume. */
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

/** Build an AI SDK ToolSet from the MCP server's tool list. */
async function mcpToToolSet(client: Client, keep: string[]): Promise<ToolSet> {
  const { tools: mcpTools } = await client.listTools();
  const toolSet: ToolSet = {};
  for (const t of mcpTools) {
    if (!keep.includes(t.name)) continue;
    toolSet[t.name] = tool({
      description: t.description ?? "",
      // MCP inputSchema is already JSON Schema; jsonSchema() adapts it for the AI SDK.
      inputSchema: jsonSchema((t.inputSchema ?? { type: "object", properties: {} }) as never),
      execute: async (args) => {
        const res = await client.callTool({
          name: t.name,
          arguments: (args ?? {}) as Record<string, unknown>,
        });
        return flattenToolResult(res as never);
      },
    });
  }
  return toolSet;
}

const isHttp = (t: string) => t === "http" || t === "streamable" || t === "streamablehttp";

/**
 * Connect, picking a transport. Default ("auto") tries Streamable HTTP first
 * (the current MCP standard) and falls back to legacy SSE. Set MCP_TRANSPORT
 * to "http" or "sse" to force one. Returns the connected client and the
 * transport label that worked.
 */
async function connectMcp(): Promise<{ client: Client; transport: string }> {
  const url = new URL(MCP_URL);
  const order =
    MCP_TRANSPORT === "sse" ? ["sse"] : isHttp(MCP_TRANSPORT) ? ["http"] : ["http", "sse"];

  let lastErr: unknown;
  for (const label of order) {
    const client = new Client({ name: "general-agent", version: "0.1.0" });
    try {
      await client.connect(
        label === "http" ? new StreamableHTTPClientTransport(url) : new SSEClientTransport(url),
      );
      return { client, transport: label };
    } catch (err) {
      lastErr = err;
      await client.close().catch(() => {});
    }
  }
  throw lastErr;
}

// ---------------------------------------------------------------------------

async function main() {
  // 1. Connect to the MCP server (auto-detect transport by default)
  const { client, transport } = await connectMcp();

  try {
    const tools = await mcpToToolSet(client, KEEP);
    const available = Object.keys(tools);
    if (available.length === 0) {
      const { tools: all } = await client.listTools();
      console.error(
        `Connected to MCP at ${MCP_URL} but none of the expected tools were found.\n` +
          `Server exposed: ${all.map((t) => t.name).join(", ") || "(none)"}\n` +
          `Adjust KEEP in this file to match the tool names your server publishes.`,
      );
      return;
    }
    console.log(`MCP connected (${transport}). Tools: ${available.join(", ")}\n`);

    // 2. Wrap into a Toolkit
    const toolkit: Toolkit = {
      tools,
      systemPrompt: TOOLKIT_SYSTEM_PROMPT,
      createFiltered: (enabled) =>
        enabled
          ? Object.fromEntries(Object.entries(tools).filter(([name]) => enabled.includes(name)))
          : tools,
    };

    // 3. Create the agent — local model via provider "openai" + baseURL.
    //    (The createAgent path uses LangChain's initChatModel, which doesn't
    //    know a "custom-openai" provider — "openai" + baseURL is the way to
    //    target a local OpenAI-compatible endpoint.)
    const agent = createAgent({
      toolkit,
      model: {
        provider: "openai",
        model: MODEL_ID,
        baseURL: OPENAI_BASE_URL,
        apiKey: OPENAI_API_KEY,
      },
      // Make the formatOutput data-gate recognize the MCP tool names, so
      // structured-output runs unblock after these tools return data.
      dataToolNames: KEEP,
    });

    // 4. Run the query from the command line, streaming output live.
    const query = process.argv.slice(2).join(" ").trim();
    if (!query) {
      console.log('Usage: npm run agent -- "your question or task"');
      console.log('   e.g. npm run agent -- "find the latest Firecrawl pricing and summarize it"');
      return;
    }

    console.log(`> ${query}\n`);
    let usage: { totalTokens?: number } | undefined;
    let steps = 0;
    for await (const event of agent.stream({ prompt: query })) {
      switch (event.type) {
        case "tool-call":
          steps++;
          process.stdout.write(`\n  · ${event.toolName}(${JSON.stringify(event.input).slice(0, 100)})\n`);
          break;
        case "text":
          if (event.content) process.stdout.write(event.content);
          break;
        case "done":
          usage = event.usage;
          break;
        case "error":
          console.error(`\n[error] ${event.error}`);
          break;
      }
    }
    console.log(`\n\n— ${steps} tool calls · ${usage?.totalTokens ?? "?"} tokens`);
  } finally {
    // Close the connection so the process exits cleanly.
    await client.close();
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
