import type {
  AnalysisJob,
  AnalyzeResponse,
} from "../types/analysis";

import {
  mockAnalyzeVideo,
  mockGetJob,
} from "./mock";

const USE_MOCK =
  import.meta.env.DEV && import.meta.env.VITE_USE_MOCK === "true";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "";

// Browser requests always use the UI contract; jury credentials stay on the server.
const ANALYZE_PATH = "/ui/analyze";
const jobPath = (jobId: string) => `/ui/jobs/${encodeURIComponent(jobId)}`;

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

export async function analyzeVideo(
  file: File
): Promise<AnalyzeResponse> {
  if (USE_MOCK) {
    return mockAnalyzeVideo(file);
  }

  const formData = new FormData();

  formData.append("file", file);

  const response = await fetch(
    `${API_BASE_URL}${ANALYZE_PATH}`,
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

export async function getAnalysisJob(
  jobId: string
): Promise<AnalysisJob> {
  if (USE_MOCK) {
    return mockGetJob(jobId);
  }

  const response = await fetch(
    `${API_BASE_URL}${jobPath(jobId)}`
  );

  if (!response.ok) {
    throw new Error(
      `Не удалось получить результат: ${await errorText(response)}`
    );
  }

  return response.json();
}

export function mediaUrl(path: string): string {
  return `${API_BASE_URL}${path}`;
}
