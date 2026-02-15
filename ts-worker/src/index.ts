import VW from "@vowpalwabbit/vowpalwabbit";
console.log("VW:", VW);

const corsHeaders: Record<string, string> = {
  "Content-Type": "application/json",
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS, GET",
  "Access-Control-Allow-Headers": "Content-Type",
};

let loggedOnce = false;

function resolveWorkspaceCtorOrFactory(): any {
  // VW may be:
  // - the constructor/factory itself
  // - a CJS default wrapper (VW.default)
  // - an object with Workspace
  const candidate =
    (VW as any)?.Workspace ??
    (VW as any)?.default?.Workspace ??
    (VW as any)?.default ??
    VW;

  return candidate;
}

function createVW(args: string[]) {
  const WorkspaceOrFactory = resolveWorkspaceCtorOrFactory();

  if (!WorkspaceOrFactory) {
    throw new Error("VW export is empty/undefined");
  }

  // If it looks like a class (has prototype.predict), use `new`.
  if (WorkspaceOrFactory.prototype && typeof WorkspaceOrFactory.prototype.predict === "function") {
    return new WorkspaceOrFactory(args);
  }

  // Otherwise assume it's a factory function.
  if (typeof WorkspaceOrFactory === "function") {
    return WorkspaceOrFactory(args);
  }

  throw new Error(
    `VW export is not callable/constructible (type=${typeof WorkspaceOrFactory})`
  );
}

export default {
  async fetch(request: Request): Promise<Response> {
    // CORS preflight
    if (request.method === "OPTIONS") {
      return new Response(null, { headers: corsHeaders });
    }

    // Health check (also prevents Wrangler GET / from producing JSON parse errors)
    if (request.method === "GET") {
      if (!loggedOnce) {
        loggedOnce = true;
        console.log("VW typeof:", typeof VW);
        console.log("VW keys:", VW && typeof VW === "object" ? Object.keys(VW as any) : []);
        console.log("VW.default typeof:", typeof (VW as any)?.default);
      }
      return new Response(JSON.stringify({ ok: true }), { status: 200, headers: corsHeaders });
    }

    // Only POST is supported for actions
    if (request.method !== "POST") {
      return new Response(JSON.stringify({ error: "Method not allowed" }), {
        status: 405,
        headers: corsHeaders,
      });
    }

    const ct = request.headers.get("content-type") || "";
    if (!ct.includes("application/json")) {
      return new Response(JSON.stringify({ error: "Expected application/json" }), {
        status: 415,
        headers: corsHeaders,
      });
    }

    try {
      const body: any = await request.json();
      const { action, state = {} } = body;

      if (!action) {
        return new Response(JSON.stringify({ error: "Missing 'action'" }), {
          status: 400,
          headers: corsHeaders,
        });
      }

      // Initialize VW instance
      const vw = createVW([
        "--cb_explore_adf",
        "--epsilon",
        "0.2",
        "--learning_rate",
        "0.5",
        "--power_t",
        "0",
      ]);

      let items = state.items || {};
      let cm = state.cm || { clusters: {} };

      if (action === "CLUSTER_ITEM") {
        const item = body.item;
        if (!item?.full_embedding || !Array.isArray(item.full_embedding)) {
          throw new Error("Missing item.full_embedding (expected number[])");
        }

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
          cluster: prediction?.[0]?.action ?? 0,
        };
      }

      if (typeof vw.delete === "function") vw.delete();

      return new Response(JSON.stringify({ items, cm }), { status: 200, headers: corsHeaders });
    } catch (err: any) {
      console.error("Worker error:", err);
      return new Response(JSON.stringify({ error: err?.message || String(err) }), {
        status: 500,
        headers: corsHeaders,
      });
    }
  },
};
