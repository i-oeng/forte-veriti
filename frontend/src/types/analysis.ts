export type JobStatus = "processing" | "done" | "failed";

export type Verdict = "violation" | "no_violation";

export type EvidenceFrame = {
  timestamp_sec: number;
  image_base64: string;
};

export type Violation = {
  // fields from the case specification (section 10)
  type: string;
  timecode_sec: number;
  description: string;
  evidence_frames?: EvidenceFrame[];

  // display fields returned by /ui/jobs (and /api/v1/jobs?extended=true)
  type_id?: string;
  title?: string;
  title_en?: string;
  tier?: "A" | "B" | "C";
  confidence?: number;
  law?: string | null;
  law_short?: string | null;
  law_url?: string | null;
  law_reg?: string | null;
  law_about?: string | null;
  rule_section?: string | null;
  rule?: string | null;
  observation?: string;
  why?: string;
  frame_urls?: string[];
  clip_url?: string | null;
  severity?: "low" | "medium" | "high";
};

export type AnalyzeResponse = {
  job_id: string;
};

export type JobProgress = {
  filename?: string;
  state?: string;
  stage?: string;
  progress?: number;
  queue_position?: number;
};

export type AnalysisJob = {
  job_id: string;
  status: JobStatus;
  verdict?: Verdict;
  violations?: Violation[];
  error?: string;
  job?: JobProgress;
};
