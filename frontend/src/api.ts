import axios from 'axios';
import type { Item } from './types';

const VERCEL_API_BASE = '/api';
const CLOUDFLARE_WORKER_URL = import.meta.env.VITE_CLOUDFLARE_WORKER_URL || "https://ts-worker.bandit-cluster.workers.dev";
if (!CLOUDFLARE_WORKER_URL) {
  throw new Error("Missing VITE_CLOUDFLARE_WORKER_URL (not injected at build time).");
}

export const api = {
  // Get embedding for an image from the Vercel backend
  embedImage: async (imageUrl: string, metadata?: string): Promise<Item> => {
    const response = await axios.post(`${VERCEL_API_BASE}/embed_image`, {
      image_url: imageUrl,
      metadata,
    });
    return response.data;
  },

  // Get embedding for a URL from the Vercel backend
  embedUrl: async (url: string, metadata?: string): Promise<Item> => {
    const response = await axios.post(`${VERCEL_API_BASE}/embed_url`, {
      url: url,
      metadata,
    });
    return response.data;
  },
  
  // --- Clustering operations (to be handled by Cloudflare Worker) ---

  // This function will now be responsible for all communication with the worker
  // It will send the current state and the desired action
  postToWorker: async (action: string, payload: any): Promise<any> => {
    const response = await axios.post(CLOUDFLARE_WORKER_URL, {
        action,
        ...payload
    });
    return response.data;
  },
};
