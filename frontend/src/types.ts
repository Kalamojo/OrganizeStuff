export interface Item {
  id: number;
  features: number[];
  cluster: string;
  metadata: string;
}

export interface ItemsResponse {
  items: Item[];
  clusters: Record<string, number>;
}

export interface CorrectionRequest {
  item_id: number;
  target_cluster: string;
}
