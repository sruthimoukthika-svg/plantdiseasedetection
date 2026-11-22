export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export type ConditionInfo = {
  summary: string;
  symptoms: string[];
  spreads: string;
  confirmation_note: string;
};

export type ScanSuccess = {
  success: true;
  plant: { name: string; id: string; confidence: number };
  prediction: { name: string; id: string; kind: string; confidence: number };
  alternatives: { name: string; id: string; kind: string; confidence: number }[];
  info: ConditionInfo;
  confidence_note: string;
  diagnostics?: Record<string, unknown>;
  inference_ms: number;
  disclaimer: string;
};

export type ScanRejected = {
  success: false;
  rejected: true;
  reason: string;
  message: string;
  diagnostics?: Record<string, unknown>;
  inference_ms?: number;
};

export type ScanResult = ScanSuccess | ScanRejected;

export type PlantEntry = {
  crop: string;
  display: string;
  conditions: {
    condition: string;
    display: string;
    kind: string;
    pathogen: string;
    info?: ConditionInfo;
  }[];
};

async function getJSON<T>(path: string): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!r.ok) throw new Error(`${path} returned ${r.status}`);
  return r.json() as Promise<T>;
}

export const getSupportedPlants = () =>
  getJSON<{ plants: PlantEntry[] }>("/api/supported-plants");

export const getDiseases = () =>
  getJSON<{ crops: PlantEntry[]; confirmation_note: string }>("/api/diseases");

export const getModelInfo = () => getJSON<Record<string, any>>("/api/model-info");

export async function scanImage(file: File): Promise<ScanResult> {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch(`${API_BASE}/api/scan`, { method: "POST", body: fd });
  const body = await r.json().catch(() => null);
  if (!body) throw new Error(`scan failed with status ${r.status}`);
  return body as ScanResult;
}

export const KIND_LABEL: Record<string, string> = {
  disease: "Disease",
  pest: "Pest damage",
  deficiency: "Nutrient deficiency",
  healthy: "Healthy",
};
