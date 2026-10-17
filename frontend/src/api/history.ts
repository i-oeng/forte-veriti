// Batch upload and check history (backend: /ui/analyze/batch, /ui/batches/{id}, /ui/history).
// Key-less /ui/ endpoints: the jury API key never goes into the browser bundle.

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "";

export type HistoryItem = {
  job_id: string;
  filename: string;
  status: "processing" | "done" | "failed";
  stage?: string | null;
  progress?: number | null;
  verdict?: "violation" | "no_violation" | null;
  violations_count?: number | null;
  error?: string | null;
  batch_id?: string | null;
  created_at: number;
  finished_at?: number | null;
};

export type BatchJob = {
  job_id: string | null;
  filename: string;
  status: "processing" | "rejected";
  error?: string;
};

export type BatchResponse = {
  batch_id: string;
  jobs: BatchJob[];
};

async function errorText(
  response: Response
): Promise<string> {
  try {
    const body = await response.json();

    if (typeof body?.detail === "string") {
      return body.detail;
    }
  } catch {
    // not JSON
  }

  return `HTTP ${response.status}`;
}

// Several videos in one request; one job per file, analysed in parallel on the server.
// A rejected file (wrong format) comes back with status "rejected" and does not fail the others.
export async function analyzeBatch(
  files: File[]
): Promise<BatchResponse> {
  const formData = new FormData();

  files.forEach((file) =>
    formData.append("files", file)
  );

  const response = await fetch(
    `${API_BASE_URL}/ui/analyze/batch`,
    {
      method: "POST",
      body: formData,
    }
  );

  if (!response.ok) {
    throw new Error(
      `Не удалось загрузить видео: ${await errorText(response)}`
    );
  }

  return response.json();
}

// Past checks made through the web interface (newest first).
export async function fetchHistory(
  limit = 50
): Promise<HistoryItem[]> {
  const response = await fetch(
    `${API_BASE_URL}/ui/history?limit=${limit}`
  );

  if (!response.ok) {
    throw new Error(
      `Не удалось получить историю: ${await errorText(response)}`
    );
  }

  const body = await response.json();

  return body.jobs ?? [];
}

// Browser-friendly copy of an analysed video (for opening a past check without the original file).
export function previewUrl(jobId: string): string {
  return `${API_BASE_URL}/media/${jobId}/preview.mp4`;
}
