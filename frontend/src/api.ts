import axios from 'axios';
import type { ItemsResponse, CorrectionRequest } from './types';

const API_BASE = 'http://localhost:8000/api';

export const api = {
  // Get all items
  getItems: async (): Promise<ItemsResponse> => {
    const response = await axios.get(`${API_BASE}/items`);
    return response.data;
  },

  // Add a new random item
  addItem: async () => {
    const response = await axios.post(`${API_BASE}/items`);
    return response.data;
  },

  // Apply human correction
  applyCorrection: async (itemId: number, targetCluster: string): Promise<ItemsResponse> => {
    const request: CorrectionRequest = {
      item_id: itemId,
      target_cluster: targetCluster,
    };
    const response = await axios.post(`${API_BASE}/correct`, request);
    return response.data;
  },

  // Recluster all items
  reclusterAll: async (): Promise<ItemsResponse> => {
    const response = await axios.post(`${API_BASE}/recluster`);
    return response.data;
  },
};
