const API_BASE = "http://127.0.0.1:5000";

export interface AnomalyObject {
  oid: string;
  ra: number;
  dec: number;
  score: number;
  rec_error?: number;
  knn_dist?: number;
  n_detections?: number;
  mag_mean?: number;
  mag_err_mean?: number;
  triage: string;
  triage_reason?: string;
  auto_class?: string | null;
  class_distance?: number | null;
  simbad_match?: string | null;
  simbad_otype?: string | null;
  human_feedback?: string | null;
  scored_at?: string;
  model_version?: number;
  flag_count?: number;
  last_flagged_at?: string | null;
}

export interface DismissedObject {
  oid: string;
  ra: number;
  dec: number;
  score: number;
  rec_error: number;
  knn_dist: number;
  n_detections: number;
  mag_err_mean: number;
  model_version: number;
}

export interface Discovery {
  oid: string;
  score: number;
  ra: number;
  dec: number;
  simbad_match: string | null;
  flagged_at: string;
  confirmed_by: string;
  notes: string | null;
  flag_count?: number;
  last_flagged_at?: string | null;
}

export interface Stats {
  total: number;
  flagged: number;
  classified: number;
  dismissed: number;
  discoveries: number;
  feedback: number;
}

export interface Health {
  status: string;
  model_loaded: boolean;
  feedback_clf: boolean;
  isolation_forest: boolean;
  db: boolean;
}

export interface FeedbackRequest {
  oid: string;
  action: "interesting" | "noise" | "classify";
  label?: string;
}

export interface RetrainResult {
  status: string;
  feedback_clf_trained: boolean;
  iso_forest_trained: boolean;
  stats: {
    n_samples: number;
    n_positive: number;
    n_negative: number;
    timestamp: string;
  };
}

export interface LLMReview {
  oid: string;
  verdict: "interesting" | "known_type" | "noise";
  confidence: number;
  reasoning: string;
  suggested_class: string | null;
  is_candidate: number;
  timestamp: string;
}

export interface ScoreBin {
  range: string;
  count: number;
}

export interface LLMStats {
  verdicts: Record<string, number>;
  total: number;
  candidates: number;
}

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export const api = {
  getStats: () => apiFetch<Stats>("/api/stats"),
  getHealth: () => apiFetch<Health>("/api/health"),
  getFlagged: (limit = 50) => apiFetch<AnomalyObject[]>(`/api/flagged?limit=${limit}`),
  getClassified: (limit = 200) => apiFetch<AnomalyObject[]>(`/api/objects?triage=classified&limit=${limit}`),
  getDismissed: (limit = 200) => apiFetch<DismissedObject[]>(`/api/dismissed?limit=${limit}`),
  getDiscoveries: () => apiFetch<Discovery[]>("/api/discoveries"),
  getLLMReviews: () => apiFetch<LLMReview[]>("/api/llm-reviews"),
  getLLMReview: (oid: string) => apiFetch<LLMReview[]>(`/api/llm-review/${oid}`),
  getLLMStats: () => apiFetch<LLMStats>("/api/llm-stats"),
  getScoreDistribution: () => apiFetch<ScoreBin[]>("/api/score-distribution"),
  sendFeedback: (data: FeedbackRequest) =>
    apiFetch<{ status: string; message: string }>("/api/feedback", {
      method: "POST",
      body: JSON.stringify(data),
    }),
  rescan: () =>
    apiFetch<{ status: string; promoted: number }>("/api/rescan", { method: "POST" }),
  retrain: () =>
    apiFetch<RetrainResult>("/api/retrain", { method: "POST" }),
};
