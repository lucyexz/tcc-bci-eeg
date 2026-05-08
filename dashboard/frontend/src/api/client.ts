const BASE = "http://localhost:8000";

const _cache = new Map<string, unknown>();

async function cachedFetch<T>(url: string): Promise<T> {
  if (_cache.has(url)) return _cache.get(url) as T;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${url}`);
  const data: T = await res.json();
  _cache.set(url, data);
  return data;
}

export interface ModelSummary {
  name: string;
  display_name: string;
  test_accuracy: number;
  test_loss: number | null;
  trained_at: string | null;
}

export interface ClassMetrics {
  precision: number;
  recall: number;
  "f1-score": number;
  support: number;
}

export interface ClassificationReport {
  [key: string]: ClassMetrics | number;
}

export interface ModelConfig {
  batch_size?: number;
  learning_rate?: number;
  dropout?: number;
  [key: string]: unknown;
}

export interface ModelDetail {
  model_name: string;
  display_name: string;
  test_accuracy: number;
  test_loss: number;
  trained_at: string;
  config: ModelConfig;
  history: Record<string, number[]>;
  confusion_matrix: number[][];
  classification_report: ClassificationReport;
  per_user_accuracy: Record<string, number>;
}

export interface ComparisonModel {
  name: string;
  display_name: string;
  test_accuracy: number;
  test_loss: number | null;
  per_class_f1: Record<string, number>;
  macro_f1: number | null;
  per_user_accuracy: Record<string, number>;
}

export interface ComparisonResponse {
  models: ComparisonModel[];
}

export function fetchModels(): Promise<ModelSummary[]> {
  return cachedFetch<ModelSummary[]>(`${BASE}/api/models`);
}

export function fetchModel(name: string): Promise<ModelDetail> {
  return cachedFetch<ModelDetail>(`${BASE}/api/models/${name}`);
}

export function fetchComparison(): Promise<ComparisonResponse> {
  return cachedFetch<ComparisonResponse>(`${BASE}/api/models/comparison`);
}

export function figureUrl(filename: string): string {
  return `${BASE}/figures/${filename}`;
}
