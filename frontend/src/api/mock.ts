import type {
  AnalysisJob,
  AnalyzeResponse,
} from "../types/analysis";

type MockJob = {
  createdAt: number;
};

const jobs = new Map<string, MockJob>();

function generateJobId() {
  return Math.random().toString(36).substring(2, 10);
}

export async function mockAnalyzeVideo(
  file: File
): Promise<AnalyzeResponse> {
  console.log("Mock upload:", file.name);

  await new Promise((resolve) => setTimeout(resolve, 700));

  const jobId = generateJobId();

  jobs.set(jobId, {
    createdAt: Date.now(),
  });

  return {
    job_id: jobId,
  };
}

export async function mockGetJob(
  jobId: string
): Promise<AnalysisJob> {
  await new Promise((resolve) => setTimeout(resolve, 200));

  const job = jobs.get(jobId);

  if (!job) {
    return {
      job_id: jobId,
      status: "failed",
      error: "Job not found",
    };
  }

  const elapsed = Date.now() - job.createdAt;

  if (elapsed < 3000) {
    return {
      job_id: jobId,
      status: "processing",
    };
  }

  return {
    job_id: jobId,
    status: "done",
    verdict: "violation",

    violations: [
      {
        type: "cash_counted_without_customer_present",
        title: "Пересчёт денег без присутствия клиента",
        timecode_sec: 12.4,
        description:
          "Кассир продолжает пересчёт наличных после ухода клиента.",
        confidence: 0.94,
        severity: "high",
        law: "Постановление НБРК №231",
        rule: "Пункт 20",
        evidence_frames: [],
      },

      {
        type: "cashier_left_during_active_customer_service",
        title: "Кассир покинул рабочее место",
        timecode_sec: 24.8,
        description:
          "Кассир покинул рабочее место до завершения обслуживания клиента.",
        confidence: 0.89,
        severity: "high",
        law: "Постановление НБРК №231",
        rule: "Пункт 22",
        evidence_frames: [],
      },

      {
        type: "PHONE_WITH_CAMERA_IN_CASH_NODE",
        title: "Смартфон в кассовом узле",
        timecode_sec: 35.1,
        description:
          "На рабочем месте кассира обнаружено устройство, похожее на смартфон.",
        confidence: 0.82,
        severity: "medium",
        law: "Постановление НБРК №231",
        rule: "Пункт 12",
        evidence_frames: [],
      },
    ],
  };
}
