/**
 * custom-toolkit.ts — General web research agent with a custom search/scrape backend.
 *
 * Drop in your own search API and scraper; the orchestration, skills,
 * schema enforcement, parallel workers and streaming all work unchanged.
 *
 * Run:
 *   ANTHROPIC_API_KEY=sk-ant-... npx tsx examples/custom-toolkit.ts
 */

import { createAgent, createToolkit } from "../src/index.js";

// ---------------------------------------------------------------------------
// 1.  Implement your own search / scrape functions
//     Replace the stubs below with real calls to your backend.
// ---------------------------------------------------------------------------

async function mySearch(query: string, opts?: { limit?: number }) {
  // Example: call Brave Search API, Tavily, SerpAPI, DuckDuckGo, etc.
  console.log(`[search] ${query} (limit=${opts?.limit})`);
  // Return at least { title, url, description } per result.
  return [
    { title: "Example result", url: "https://example.com", description: "A placeholder result." },
  ];
}

async function myScrape(url: string, opts?: { selector?: string }) {
  // Example: fetch with Playwright, Puppeteer, your HTTP client, etc.
  console.log(`[scrape] ${url} (selector=${opts?.selector})`);
  // Return { markdown, html, metadata } — any subset is fine.
  return {
    markdown: `# ${url}\n\nPlaceholder content scraped from ${url}.`,
    metadata: { url },
  };
}

// ---------------------------------------------------------------------------
// 2.  Build the toolkit from your functions
// ---------------------------------------------------------------------------

const toolkit = createToolkit({
  search: mySearch,
  scrape: myScrape,
  // interact, crawl, map — add as needed
  systemPrompt: "You are connected to a custom search and scrape backend.",
});

// ---------------------------------------------------------------------------
// 3.  Create the agent — no firecrawlApiKey required
// ---------------------------------------------------------------------------

const agent = createAgent({
  toolkit,
  model: { provider: "anthropic", model: "claude-sonnet-4-6" },
  // subAgentModel, skillsDir, maxSteps, etc. all work as usual
});

// ---------------------------------------------------------------------------
// 4.  Run a research task
// ---------------------------------------------------------------------------

const result = await agent.run({
  prompt: "What are the top 5 open-source LLM frameworks? List name, GitHub URL, and main use case.",
  format: "json",
  schema: {
    frameworks: [{ name: null, github_url: null, use_case: null }],
  },
});

console.log("\n--- Result ---");
console.log(result.text);
if (result.data) {
  console.log("\n--- Structured data ---");
  console.log(JSON.stringify(result.data, null, 2));
}
console.log(`\nSteps: ${result.steps?.length} | Tokens: ${result.usage?.totalTokens}`);
