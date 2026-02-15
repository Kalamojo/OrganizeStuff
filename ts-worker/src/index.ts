import { Workspace } from '@vowpalwabbit/vowpalwabbit';

const corsHeaders = {
  "Content-Type": "application/json",
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

export default {
  async fetch(request: Request, env: any, ctx: ExecutionContext): Promise<Response> {
    if (request.method === 'OPTIONS') return new Response(null, { headers: corsHeaders });

    try {
      const body: any = await request.json();
      const { action, state = {} } = body;
      
      // 1. Initialize Workspace with your specific Python args
      // In JS, we pass these as an array of strings
      const vw = new Workspace([
        "--cb_explore_adf", 
        "--epsilon", "0.2", 
        "--learning_rate", "0.5", 
        "--power_t", "0"
      ]);

      let items = state.items || {};
      let cm = state.cm || { clusters: {} };

      if (action === "CLUSTER_ITEM") {
        const item = body.item;
        const embedding: number[] = item.full_embedding;

        // 2. Format ADF examples for JS
        // Unlike Python's single multi-line string, the WASM version 
        // often expects an array of strings for multi-example ADF
        const adf_examples = [
          "shared | s_features " + embedding.join(" "),
          "| a_action_1",
          "| a_action_2"
          // Add your dynamic cluster/action logic here
        ];

        // 3. Predict & Learn
        // The JS .predict() returns an array of scores/actions
        const prediction = vw.predict(adf_examples);
        vw.learn(adf_examples); 

        // Update your item state
        items[item.id] = {
          ...item,
          cluster: prediction[0].action // or however your logic picks the top action
        };
      } 
      else if (action === "APPLY_CORRECTION") {
        // Implement your manual correction logic here
        // vw.learn(correction_example)
      }

      // 4. CRITICAL: Manual Memory Cleanup
      // Because this is WebAssembly, you must free the memory!
      vw.delete();

      return new Response(JSON.stringify({ items, cm }), { 
        status: 200, 
        headers: corsHeaders 
      });

    } catch (e: any) {
      return new Response(JSON.stringify({ error: e.message }), { 
        status: 500, 
        headers: corsHeaders 
      });
    }
  }
};
