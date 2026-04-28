/**
 * 通用 Brave Search 工具
 *
 * 用途：给 Agent 提供互联网搜索能力。
 * API：https://api-dashboard.search.brave.com/api-reference/web/search/get
 * 环境变量：BRAVE_API_KEY
 *
 * 使用：
 *   import { searchTool } from "./shared/search-tool.js";
 *   const llm = baseLlm.bindTools([searchTool]);
 *
 * 设计要点：
 * - 工具内部捕获所有错误并返回错误字符串（而非 throw），
 *   让 LLM 能读到失败原因并自主重试或换关键词（参见 02-tool-use 的 Strategy A）。
 * - 只取前 count 条结果并压缩成精简文本，避免 ToolMessage 让上下文爆炸。
 */

import { tool } from "@langchain/core/tools";
import { z } from "zod";

interface BraveWebResult {
  title: string;
  url: string;
  description: string;
}

interface BraveSearchResponse {
  web?: { results?: BraveWebResult[] };
}

export interface SearchToolOptions {
  /** 返回前 N 条结果，默认 5。范围 1-20。 */
  count?: number;
  /** 是否打印调用日志，默认 true。 */
  verbose?: boolean;
}

export function createSearchTool(options: SearchToolOptions = {}) {
  const { count = 5, verbose = true } = options;

  return tool(
    async ({ query }: { query: string }) => {
      if (verbose) console.log(`  [tool:search] 搜索: ${query}`);

      const apiKey = process.env.BRAVE_API_KEY;
      if (!apiKey) {
        return "搜索失败：缺少 BRAVE_API_KEY 环境变量。";
      }

      const url = new URL("https://api.search.brave.com/res/v1/web/search");
      url.searchParams.set("q", query);
      url.searchParams.set("count", String(count));

      try {
        const res = await fetch(url, {
          headers: {
            Accept: "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": apiKey,
          },
        });
        if (!res.ok) {
          return `搜索失败：HTTP ${res.status} ${res.statusText}`;
        }
        const data = (await res.json()) as BraveSearchResponse;
        const results = data.web?.results ?? [];
        if (results.length === 0) {
          return `未找到关于 "${query}" 的资料。`;
        }
        return results
          .map(
            (r, i) =>
              `[${i + 1}] ${r.title}\n${r.description}\n来源：${r.url}`
          )
          .join("\n\n");
      } catch (err) {
        return `搜索异常：${(err as Error).message}`;
      }
    },
    {
      name: "search",
      description:
        "搜索互联网技术资料（Brave Search）。输入搜索关键词，返回 top N 结果（标题/摘要/URL）。",
      schema: z.object({
        query: z.string().describe("搜索关键词，建议使用英文专业术语"),
      }),
    }
  );
}

/** 默认实例：count=5, verbose=true，开箱即用。 */
export const searchTool = createSearchTool();
