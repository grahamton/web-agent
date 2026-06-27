import { tool } from "ai";
import { z } from "zod";
import type { ToolSet } from "ai";
import type { Toolkit } from "./types";

// --- Result types for custom tool functions ---

export interface SearchResult {
  title: string;
  url: string;
  description?: string;
}

export interface ScrapeResult {
  markdown?: string;
  html?: string;
  metadata?: Record<string, unknown>;
}

export interface InteractResult {
  output?: string;
  screenshot?: string;
}

export interface CrawlPage {
  url: string;
  markdown?: string;
}

// --- Config accepted by createToolkit ---

export interface ToolkitConfig {
  /**
   * Web search function. Return a ranked list of results for a query.
   * Example backends: Brave Search API, SerpAPI, DuckDuckGo, Tavily.
   */
  search?: (
    query: string,
    opts?: { limit?: number; lang?: string },
  ) => Promise<SearchResult[]>;

  /**
   * Page scraping function. Fetch a URL and return its content as markdown
   * and/or HTML. Example backends: Playwright, Puppeteer, your own HTTP client.
   */
  scrape?: (
    url: string,
    opts?: { selector?: string; waitMs?: number },
  ) => Promise<ScrapeResult>;

  /**
   * Browser interaction function. Given a URL and a natural-language prompt,
   * perform actions (click, fill, scroll) and return the result.
   */
  interact?: (url: string, prompt: string) => Promise<InteractResult>;

  /**
   * Site crawler. Recursively fetch pages starting from a URL and return
   * their content. Useful for indexing documentation or product catalogs.
   */
  crawl?: (
    url: string,
    opts?: { limit?: number; depth?: number },
  ) => Promise<CrawlPage[]>;

  /**
   * Sitemap discovery. Return all URLs discoverable from a root URL.
   */
  map?: (url: string) => Promise<string[]>;

  /**
   * Optional system prompt snippet injected into the agent's system prompt,
   * useful for documenting backend-specific quirks or rate limits.
   */
  systemPrompt?: string;
}

/**
 * Wrap your own search/scrape/crawl functions into a Toolkit that any
 * agent created with `createAgent({ toolkit, model })` can consume.
 *
 * Only the functions you provide become tools — omit any you don't need.
 * The returned toolkit mirrors the Firecrawl toolkit's interface so all
 * sub-agent and worker features work unchanged.
 *
 * Example:
 * ```ts
 * import { createToolkit, createAgent } from "@firecrawl/agent-core";
 *
 * const toolkit = createToolkit({
 *   search: async (query) => mySearchApi(query),
 *   scrape: async (url)   => myScraper(url),
 * });
 *
 * const agent = createAgent({ toolkit, model: { provider: "anthropic", model: "claude-sonnet-4-6" } });
 * ```
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function createToolkit(config: ToolkitConfig): Toolkit {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const tools: Record<string, any> = {};

  if (config.search) {
    const fn = config.search;
    tools.search = tool({
      description:
        "Search the web for information. Returns a ranked list of results with titles, URLs, and descriptions.",
      inputSchema: z.object({
        query: z.string().describe("The search query"),
        limit: z.number().int().min(1).max(20).optional().describe("Max number of results to return"),
        lang: z.string().optional().describe("Language/locale for results (e.g. 'en', 'fr')"),
      }),
      execute: async ({ query, limit, lang }) => fn(query, { limit, lang }),
    });
  }

  if (config.scrape) {
    const fn = config.scrape;
    tools.scrape = tool({
      description:
        "Fetch and extract content from a URL. Returns markdown and/or HTML. Use for reading articles, product pages, documentation.",
      inputSchema: z.object({
        url: z.string().url().describe("The URL to scrape"),
        selector: z.string().optional().describe("CSS selector to target a specific section of the page"),
        waitMs: z.number().int().min(0).optional().describe("Milliseconds to wait for JS rendering before extracting"),
      }),
      execute: async ({ url, selector, waitMs }) => fn(url, { selector, waitMs }),
    });
  }

  if (config.interact) {
    const fn = config.interact;
    tools.interact = tool({
      description:
        "Interact with a web page using a natural-language prompt — click buttons, fill forms, scroll, handle JavaScript-heavy pages.",
      inputSchema: z.object({
        url: z.string().url().describe("The page URL to interact with"),
        prompt: z.string().describe("Natural-language instruction describing what to do on the page"),
      }),
      execute: async ({ url, prompt }) => fn(url, prompt),
    });
  }

  if (config.crawl) {
    const fn = config.crawl;
    tools.crawl = tool({
      description:
        "Recursively crawl a website starting from a URL. Returns content from multiple pages. Use when you need to index a site or find pages beyond the first.",
      inputSchema: z.object({
        url: z.string().url().describe("Root URL to start crawling from"),
        limit: z.number().int().min(1).optional().describe("Max number of pages to crawl"),
        depth: z.number().int().min(1).optional().describe("Max link depth to follow"),
      }),
      execute: async ({ url, limit, depth }) => fn(url, { limit, depth }),
    });
  }

  if (config.map) {
    const fn = config.map;
    tools.map = tool({
      description:
        "Discover all URLs on a website by following its sitemap and links. Use before crawl when you need to understand site structure.",
      inputSchema: z.object({
        url: z.string().url().describe("Root URL to map"),
      }),
      execute: async ({ url }) => fn(url),
    });
  }

  const enabledNames = new Set(Object.keys(tools));
  const toolSet = tools as ToolSet;

  return {
    tools: toolSet,
    systemPrompt: config.systemPrompt,
    createFiltered: (enabled) => {
      if (!enabled) return toolSet;
      const normalized = enabled.map((name) => name.trim().toLowerCase());
      const unknown = normalized.filter((n) => !enabledNames.has(n));
      if (unknown.length > 0) {
        console.warn(
          `createFiltered: unknown tool name(s): ${unknown.join(", ")}. Available: ${[...enabledNames].join(", ")}`,
        );
      }
      return Object.fromEntries(
        Object.entries(tools).filter(([name]) => normalized.includes(name)),
      ) as ToolSet;
    },
  };
}
