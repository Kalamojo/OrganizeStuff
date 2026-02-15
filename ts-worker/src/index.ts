import { Workspace } from "@vowpalwabbit/vowpalwabbit";

const corsHeaders = {
  "Content-Type": "application/json",
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

export default {
  async fetch(request: Request, env: any, ctx: ExecutionContext): Promise<Response> {
    // CORS preflight
    if (request.method === "OPTIONS") return new Response(null, { headers: corsHeaders });

    // Health check (prevents wrangler/browser GET / from becoming a 500)
    if (request.method === "GET") {
      return new Response(JSON.stringify({ ok: true }), { status: 200, headers: corsHeaders });
    }

    // Only accept JSON POST
    if (request.method !== "POST") {
      return new Response(JSON.stringify({ error: "Method not allowed" }), { status: 405, headers: corsHeaders });
    }

    const ct = request.headers.get("content-type") || "";
    if (!ct.includes("application/json")) {
      return new Response(JSON.stringify({ error: "Expected application/json" }), { status: 415, headers: corsHeaders });
    }

    try {
      const body: any = await request.json();
      const { action, state = {} } = body;

      if (!action) {
        return new Response(JSON.stringify({ error: "Missing 'action'" }), { status: 400, headers: corsHeaders });
      }

      const vw = new Workspace([
        "--cb_explore_adf",
        "--epsilon", "0.2",
        "--learning_rate", "0.5",
        "--power_t", "0",
      ]);

      let items = state.items || {};
      let cm = state.cm || { clusters: {} };

      if (action === "CLUSTER_ITEM") {
        const item = body.item;
        const embedding: number[] = item.full_embedding;

        const adf_examples = [
          "shared | s_features " + embedding.join(" "),
          "| a_action_1",
          "| a_action_2",
        ];

        const prediction = vw.predict(adf_examples);
        vw.learn(adf_examples);

        items[item.id] = {
          ...item,
          cluster: prediction[0].action,
        };
      }

      vw.delete();

      return new Response(JSON.stringify({ items, cm }), { status: 200, headers: corsHeaders });
    } catch (e: any) {
      return new Response(JSON.stringify({ error: e?.message || String(e) }), { status: 500, headers: corsHeaders });
    }
  },
};
