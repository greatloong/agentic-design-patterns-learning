/**
 * AG-UI 官方方案 · 图模块（被 langgraph dev 加载，仅定义图，不碰 SSE）
 *
 * 用 @langchain/anthropic 指向 DashScope（百炼）的 Anthropic 兼容端点 + 开启 thinking，
 * 思维链便以 Anthropic thinking content block 流出，被官方 LangGraphAgent 的
 * resolveReasoningContent 识别为 REASONING_* 事件（它不读 OpenAI 兼容接口的
 * additional_kwargs.reasoning_content）。详见 07-streaming-ag-ui.ts 顶部说明。
 */

import dotenv from "dotenv";
import { StateGraph, START, END, MessagesAnnotation } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { ChatAnthropic } from "@langchain/anthropic";
import { AIMessage, SystemMessage } from "@langchain/core/messages";
import { searchTool } from "../shared/search-tool.js";

// langgraph dev 跑图时 cwd 不一定是本包目录，按模块位置显式定位包根 .env。
dotenv.config({ path: new URL("../../.env", import.meta.url) });

// 清掉可能从外部进程（如配过第三方中转的 shell/IDE）继承来的 Anthropic 凭据，
// 否则 @anthropic-ai/sdk 的 credential chain 会兜底用它们，盖过下面显式传的配置导致 401。
delete process.env.ANTHROPIC_AUTH_TOKEN;
delete process.env.ANTHROPIC_BASE_URL;

const tools = [searchTool];

// 走 DashScope（阿里云百炼）的 Anthropic 兼容端点：国内直连、稳定。
// DeepSeek 自家的 api.deepseek.com 走 AWS CloudFront，国内访问会间歇性 TLS 重置，
// 流式长连接尤其容易超时（Connection error）；DashScope 同样以 thinking block 返回思维链。
const modelWithTools = new ChatAnthropic({
  model: "deepseek-v4-pro",
  apiKey: process.env.DASHSCOPE_API_KEY,
  anthropicApiUrl: "https://dashscope.aliyuncs.com/apps/anthropic",
  thinking: { type: "enabled", budget_tokens: 2000 },
  maxTokens: 4096,
}).bindTools(tools);

// 约束搜索次数，避免模型搜不到确切答案时无限换关键词重搜、撞 25 步递归上限。
const systemPrompt = new SystemMessage(
  "你是一个有联网搜索能力的助手。同一问题最多搜索 2-3 次；" +
    "若仍拿不到确切信息，就基于已有结果如实作答并说明不确定之处，不要反复换关键词重搜。"
);

async function agentNode(state: typeof MessagesAnnotation.State) {
  const response = await modelWithTools.invoke([systemPrompt, ...state.messages]);
  return { messages: [response] };
}

const workflow = new StateGraph(MessagesAnnotation)
  .addNode("agent", agentNode)
  .addNode("tools", new ToolNode(tools))
  .addEdge(START, "agent")
  .addConditionalEdges("agent", (state) => {
    const last = state.messages.at(-1) as AIMessage;
    return last.tool_calls?.length ? "tools" : END;
  })
  .addEdge("tools", "agent");

export const graph = workflow.compile();
