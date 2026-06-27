/**
 * general-agent.ts — A general-purpose web agent (search / scrape / crawl)
 * powered by an existing MCP server and a local OpenAI-compatible model.
 *
 * The MCP→Toolkit bridge, transport auto-detection, and small-model hardening
 * (per-tool timeout/retry, output truncation) now live in agent-core itself:
 * `createMcpAgentFromEnv()` in src/mcp.ts. This file is just a thin CLI around
 * it — curate the tool surface, tailor the prompt, stream the run.
 *
 * Requirements:
 *   - The MCP server must be reachable over HTTP (Streamable HTTP or legacy SSE).
 *   - The local model MUST support OpenAI-style tool calling, or the agent
 *     loop won't function.
 *
 * Env:
 *   MCP_URL          MCP endpoint, e.g. http://homelab:3100/mcp or .../sse  (required)
 *   MCP_TRANSPORT    Force a transport: "http" or "sse". Default: auto.
 *   OPENAI_BASE_URL  Local OpenAI-compatible endpoint, e.g. http://homelab:1234/v1  (required)
 *   MODEL_ID         Model id served there, e.g. "gemma4:12b-web"  (required)
 *   OPENAI_API_KEY   API key — a dummy like "sk-local" is fine for most local servers
 *
 * Run (after copying .env.example to .env and filling it in):
 *   npm run agent -- "your question or task"
 *   npm run agent -- "find the latest Firecrawl pricing and summarize it"
 */

import { createMcpAgentFromEnv } from "../src/index.js";

// Curated, general-purpose surface. We omit the dtc-research tools that have
// side effects or are workflow-specific (notify_telegram, save_report, track,
// site_memory, audit_storefront, …) and keep the read-only research tools.
const KEEP = ["search", "scrape_url", "browse", "ask_page", "research"];

const SYSTEM_PROMPT = `You are a general web research agent connected to a search/scrape/crawl backend over MCP.

Tool guidance:
- search: discover relevant pages for a query. Start here when you don't have a URL.
- scrape_url: fetch and read the full content of a single known URL.
- ask_page: ask a specific question of a single page and get a focused answer (cheaper than scraping when you only need one fact).
- browse: explore or crawl across multiple pages of a site when one page isn't enough.
- research: run a deeper multi-step investigation for a broad question.

Prefer search → scrape_url/ask_page for most tasks. Reach for browse/research only when a single page can't answer the question. Always cite the source URLs you used.`;

async function main() {
  const query = process.argv.slice(2).join(" ").trim();
  if (!query) {
    console.log('Usage: npm run agent -- "your question or task"');
    console.log('   e.g. npm run agent -- "find the latest Firecrawl pricing and summarize it"');
    return;
  }

  const { agent, transport, toolNames, close } = await createMcpAgentFromEnv({
    keep: KEEP,
    systemPrompt: SYSTEM_PROMPT,
  });

  console.log(`MCP connected (${transport}). Tools: ${toolNames.join(", ")}\n`);
  console.log(`> ${query}\n`);

  try {
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
    await close();
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
