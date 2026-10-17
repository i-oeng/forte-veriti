import { useEffect, useRef, useState } from "react";
import { analyzeVideo, getAnalysisJob, mediaUrl } from "../api/client";
import { analyzeBatch, previewUrl, type HistoryItem } from "../api/history";
import HistoryPanel from "../components/HistoryPanel";
import UploadPanel from "../components/UploadPanel";
import ViolationTimeline from "../components/ViolationTimeline";
import type { AnalysisJob } from "../types/analysis";
import { base64ToImageSrc } from "../utils/image";
import { formatTime } from "../utils/time";

type VideoTask = {
  localId: string;
  name: string;
  file?: File;
  videoUrl: string;
  jobId?: string;
  status: "uploading" | "processing" | "done" | "failed";
  result?: AnalysisJob;
  error?: string;
};

function createLocalId() {
  return crypto.randomUUID?.() ?? `${Date.now()}-${Math.random()}`;
}

export default function AnalyzePage() {
  const [tasks, setTasks] = useState<VideoTask[]>([]);
  const [activeTaskId, setActiveTaskId] = useState<string | null>(null);
  const [videoDuration, setVideoDuration] = useState(0);
  const [historyVersion, setHistoryVersion] = useState(0);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const mounted = useRef(true);
  const cancelled = useRef(new Set<string>());
  const timers = useRef(new Map<string, number>());
  const objectUrls = useRef(new Map<string, string>());
  const activeTask = tasks.find((task) => task.localId === activeTaskId);

  useEffect(() => {
    mounted.current = true;
    const pendingTimers = timers.current;
    const urls = objectUrls.current;
    return () => {
      mounted.current = false;
      pendingTimers.forEach((timer) => window.clearTimeout(timer));
      pendingTimers.clear();
      urls.forEach((url) => URL.revokeObjectURL(url));
      urls.clear();
    };
  }, []);

  function isActive(localId: string) {
    return mounted.current && !cancelled.current.has(localId);
  }

  function updateTask(localId: string, patch: Partial<VideoTask>) {
    if (!isActive(localId)) return;
    setTasks((current) => current.map((task) =>
      task.localId === localId ? { ...task, ...patch } : task));
  }

  async function pollJob(localId: string, jobId: string) {
    if (!isActive(localId)) return;
    timers.current.delete(localId);
    try {
      const job = await getAnalysisJob(jobId);
      if (!isActive(localId)) return;
      updateTask(localId, {
        status: job.status,
        result: job,
        ...(job.status === "done" ? { videoUrl: previewUrl(jobId) } : {}),
        error: job.status === "failed" ? job.error || "Ошибка обработки видео" : undefined,
      });
      if (job.status === "processing") {
        timers.current.set(localId, window.setTimeout(() => void pollJob(localId, jobId), 1500));
      } else {
        setHistoryVersion((version) => version + 1);
      }
    } catch (error) {
      updateTask(localId, {
        status: "failed",
        error: error instanceof Error ? error.message : "Ошибка получения результата",
      });
    }
  }

  async function processUploads(newTasks: VideoTask[], files: File[]) {
    try {
      // Batch results preserve file order, including individually rejected videos.
      const jobs = files.length > 1
        ? (await analyzeBatch(files)).jobs
        : [{ ...(await analyzeVideo(files[0])), status: "processing" as const, error: undefined }];
      if (!mounted.current) return;
      setHistoryVersion((version) => version + 1);
      newTasks.forEach((task, index) => {
        if (!isActive(task.localId)) return;
        const job = jobs[index];
        if (!job?.job_id || job.status === "rejected") {
          updateTask(task.localId, { status: "failed", error: job?.error || "Видео не принято сервером" });
          return;
        }
        updateTask(task.localId, { jobId: job.job_id, status: "processing" });
        void pollJob(task.localId, job.job_id);
      });
    } catch (error) {
      newTasks.forEach((task) => updateTask(task.localId, {
        status: "failed",
        error: error instanceof Error ? error.message : "Ошибка загрузки видео",
      }));
    }
  }

  function handleUpload(files: File[]) {
    if (!files.length) return;
    const newTasks: VideoTask[] = files.map((file) => {
      const localId = createLocalId();
      const videoUrl = URL.createObjectURL(file);
      objectUrls.current.set(localId, videoUrl);
      return { localId, name: file.name, file, videoUrl, status: "uploading" };
    });
    setTasks((current) => [...current, ...newTasks]);
    setActiveTaskId((current) => current ?? newTasks[0].localId);
    void processUploads(newTasks, files);
  }

  function openFromHistory(item: HistoryItem) {
    const existing = tasks.find((task) => task.jobId === item.job_id);
    if (existing) {
      setActiveTaskId(existing.localId);
      if (existing.status === "failed" && item.status !== "failed") {
        updateTask(existing.localId, { status: "processing", error: undefined });
        void pollJob(existing.localId, item.job_id);
      }
      return;
    }
    const localId = createLocalId();
    setTasks((current) => [...current, {
      localId, name: item.filename, videoUrl: previewUrl(item.job_id),
      jobId: item.job_id, status: "processing",
    }]);
    setActiveTaskId(localId);
    void pollJob(localId, item.job_id);
  }

  function releaseTask(task: VideoTask) {
    cancelled.current.add(task.localId);
    window.clearTimeout(timers.current.get(task.localId));
    timers.current.delete(task.localId);
    const url = objectUrls.current.get(task.localId);
    if (url) URL.revokeObjectURL(url);
    objectUrls.current.delete(task.localId);
  }

  function removeTask(localId: string) {
    const task = tasks.find((item) => item.localId === localId);
    if (task) releaseTask(task);
    const remaining = tasks.filter((item) => item.localId !== localId);
    setTasks((current) => current.filter((item) => item.localId !== localId));
    setActiveTaskId((current) => current === localId ? remaining[0]?.localId ?? null : current);
  }

  function clearAll() {
    tasks.forEach(releaseTask);
    setTasks([]);
    setActiveTaskId(null);
  }

  function seekTo(seconds: number) {
    if (!videoRef.current) return;
    videoRef.current.currentTime = seconds;
    void videoRef.current.play().catch(() => {});
  }

  const completedCount = tasks.filter((task) => task.status === "done").length;
  const processingCount = tasks.filter((task) => ["processing", "uploading"].includes(task.status)).length;
  const progress = Math.round(Math.min(1, Math.max(0, activeTask?.result?.job?.progress ?? 0)) * 100);

  return (
    <main className="container">
      <header className="header">
        <div className="logo">F</div>
        <div><h1>Forte Compliance AI</h1><p>Контроль кассовой дисциплины</p></div>
      </header>
      <section className="hero">
        <h2>Анализ кассовых операций</h2>
        <p>Загружайте несколько видеозаписей. Каждая запись анализируется независимо.</p>
      </section>
      <UploadPanel onUpload={handleUpload} />
      <HistoryPanel onOpen={openFromHistory} refreshKey={historyVersion} />

      {tasks.length > 0 && <>
        <section className="batch-summary" aria-label="Статус проверок">
          <div><span>Всего видео</span><strong>{tasks.length}</strong></div>
          <div><span>Обрабатывается</span><strong>{processingCount}</strong></div>
          <div><span>Завершено</span><strong>{completedCount}</strong></div>
          <button onClick={clearAll} title="Результаты останутся в истории; обработка продолжится">Очистить список</button>
        </section>
        <div className="multi-layout">
          <aside className="task-list" aria-label="Видео">
            <div className="task-list-title"><span>Видео</span><span>{tasks.length}</span></div>
            {tasks.map((task) => <button
              key={task.localId}
              className={`task-card ${activeTaskId === task.localId ? "task-card-active" : ""}`}
              aria-pressed={activeTaskId === task.localId}
              onClick={() => setActiveTaskId(task.localId)}
            >
              <div className="task-card-main">
                <strong title={task.name}>{task.name}</strong>
                {task.file && <small>{(task.file.size / 1024 / 1024).toFixed(1)} МБ</small>}
              </div>
              <TaskStatusBadge task={task} />
              {task.status === "done" && <div className="task-result-small">
                {task.result?.verdict === "violation"
                  ? `⚠ ${task.result.violations?.length ?? 0} событий`
                  : "Нарушение не обнаружено"}
              </div>}
            </button>)}
          </aside>
          <section className="selected-task" aria-label="Результат проверки">
            {activeTask && <>
              <div className="selected-task-header">
                <div><small>Текущее видео</small><h2>{activeTask.name}</h2></div>
                <button className="remove-button" onClick={() => removeTask(activeTask.localId)}
                  title="Результат останется в истории; обработка продолжится">Убрать из списка</button>
              </div>
              <div className="analysis-grid">
                <div>
                  <div className="video-panel">
                    <video key={activeTask.localId} ref={videoRef} src={activeTask.videoUrl}
                      controls playsInline preload="metadata" onLoadStart={() => setVideoDuration(0)}
                      onLoadedMetadata={(event) => setVideoDuration(event.currentTarget.duration)} />
                  </div>
                  {activeTask.status === "done" && activeTask.result && <ViolationTimeline
                    duration={videoDuration} violations={activeTask.result.violations ?? []} onSeek={seekTo} />}
                </div>
                <div className="results">
                  {(activeTask.status === "uploading" || activeTask.status === "processing") &&
                    <div className="processing" role="status">
                      <div className="spinner" aria-hidden="true" />
                      <h3>{activeTask.status === "uploading" ? "Загружаем видео…" : "Анализируем видео…"}</h3>
                      {activeTask.status === "processing" && <>
                        <p>{activeTask.result?.job?.stage || "Ожидает обработки"} · {progress}%</p>
                        <progress value={progress} max={100} aria-label="Прогресс анализа" />
                        {!!activeTask.result?.job?.queue_position &&
                          <p>Место в очереди: {activeTask.result.job.queue_position}</p>}
                      </>}
                      <p>Другие загруженные видео продолжают обрабатываться параллельно.</p>
                    </div>}
                  {activeTask.status === "failed" && <div className="error" role="alert">
                    <strong>Ошибка</strong><p>{activeTask.error}</p>
                  </div>}
                  {activeTask.status === "done" && activeTask.result &&
                    <ResultView result={activeTask.result} onSeek={seekTo} />}
                </div>
              </div>
            </>}
          </section>
        </div>
      </>}
    </main>
  );
}

function TaskStatusBadge({ task }: { task: VideoTask }) {
  const label = task.status === "uploading" ? "Загрузка"
    : task.status === "processing" ? "Анализ"
    : task.status === "failed" ? "Ошибка"
    : task.result?.verdict === "violation" ? "Требует проверки" : "Готово";
  const style = task.status === "uploading" || task.status === "processing" ? "processing"
    : task.status === "failed" ? "failed"
    : task.result?.verdict === "violation" ? "warning" : "done";
  return <span className={`task-status status-${style}`}>{label}</span>;
}

function ResultView({ result, onSeek }: { result: AnalysisJob; onSeek: (seconds: number) => void }) {
  const violations = result.violations ?? [];
  const hasViolations = result.verdict === "violation";
  return <>
    <div className={`verdict ${hasViolations ? "violation" : "success"}`} role="status">
      <h2>{hasViolations ? "Обнаружены признаки возможных нарушений" : "Нарушение не обнаружено"}</h2>
      {hasViolations && <p>Найдено событий: {violations.length}</p>}
    </div>
    {violations.map((violation, index) => <div className="violation-card" key={`${violation.type}-${index}`}>
      <div className="violation-header">
        <button onClick={() => onSeek(violation.timecode_sec)}>▶ {formatTime(violation.timecode_sec)}</button>
        {violation.confidence != null && <span>{Math.round(violation.confidence * 100)}%</span>}
      </div>
      <h3>{violation.title || violation.type}</h3>
      <p>{violation.description}</p>
      {(violation.law || violation.law_short || violation.rule) && <div className="law">
        {(violation.law || violation.law_short) && <strong>
          {violation.law_url
            ? <a href={violation.law_url} target="_blank" rel="noopener noreferrer">
              {violation.law_short || violation.law}</a>
            : violation.law_short || violation.law}
        </strong>}
        {violation.rule_section && <strong>{violation.rule_section}</strong>}
        {violation.rule && <span>{violation.rule}</span>}
        {violation.law_about && <p>{violation.law_about}</p>}
      </div>}
      {!!violation.evidence_frames?.length && <div className="evidence-section">
        <div className="evidence-heading">Кадры-доказательства</div>
        <div className="evidence-grid">
          {violation.evidence_frames.map((frame, frameIndex) => <button key={frameIndex}
            className="evidence-frame" onClick={() => onSeek(frame.timestamp_sec)}>
            <img src={base64ToImageSrc(frame.image_base64)} loading="lazy"
              alt={`Кадр в ${formatTime(frame.timestamp_sec)}: ${violation.title || violation.type}`} />
            <span>▶ {formatTime(frame.timestamp_sec)}</span>
          </button>)}
        </div>
      </div>}
      {violation.clip_url && <a className="clip-link" href={mediaUrl(violation.clip_url)}
        target="_blank" rel="noopener noreferrer">Открыть фрагмент события ↗</a>}
    </div>)}
  </>;
}
