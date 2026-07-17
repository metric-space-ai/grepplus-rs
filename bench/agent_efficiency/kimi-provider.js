// pi extension: register Moonshot Kimi as an Anthropic-compatible provider.
// API key is read from $KIMI_API_KEY at runtime — NEVER stored here.
// Endpoint: the flatrate subscription (Kimi For Coding) is served from
// api.kimi.com, NOT the metered platform.moonshot.ai (a key for one is a
// 401 on the other — verified live 2026-07-17). Anthropic-messages
// compatibility surface; kimi-k3 ships built-in thinking.
export default function (pi) {
  pi.registerProvider("kimi", {
    name: "Kimi", baseUrl: "https://api.kimi.com/coding",
    apiKey: "$KIMI_API_KEY", api: "anthropic-messages",
    models: [{ id: "kimi-k3", name: "Kimi K3", reasoning: true,
      input: ["text"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 262144, maxTokens: 8192 }],
  });
}
