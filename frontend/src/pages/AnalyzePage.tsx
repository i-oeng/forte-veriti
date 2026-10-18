import {
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import {
  AlertCircle,
  CheckCircle2,
  Clock3,
  LoaderCircle,
  MoreHorizontal,
  Trash2,
  Video as VideoIcon,
} from "lucide-react";

import {
  analyzeVideo,
  getAnalysisJob,
} from "../api/client";

import UploadPanel from "../components/UploadPanel";
import ComplianceVideoPlayer from "../components/ComplianceVideoPlayer";

import type {
  AnalysisJob,
} from "../types/analysis";

import {
  formatTime,
} from "../utils/time";

import {
  addEvidenceFrames,
} from "../utils/evidence";

import {
  base64ToImageSrc,
} from "../utils/image";

type TaskStatus =
  | "uploading"
  | "processing"
  | "done"
  | "failed";

type VideoTask = {
  localId: string;
  file: File;
  videoUrl: string;
  jobId?: string;
  status: TaskStatus;
  result?: AnalysisJob;
  error?: string;
};

function createLocalId() {
  if (
    typeof crypto !== "undefined" &&
    crypto.randomUUID
  ) {
    return crypto.randomUUID();
  }

  return `${Date.now()}-${Math.random()}`;
}

export default function AnalyzePage() {
  const [tasks, setTasks] =
    useState<VideoTask[]>([]);

  const [
    activeTaskId,
    setActiveTaskId,
  ] =
    useState<string | null>(null);


  const videoRef =
    useRef<HTMLVideoElement | null>(
      null
    );

  const tasksRef =
    useRef<VideoTask[]>([]);

  const cancelledTasksRef =
    useRef<Set<string>>(
      new Set()
    );

  useEffect(() => {
    tasksRef.current = tasks;
  }, [tasks]);

  useEffect(() => {
    return () => {
      tasksRef.current.forEach(
        (task) => {
          URL.revokeObjectURL(
            task.videoUrl
          );
        }
      );
    };
  }, []);

  const activeTask =
    useMemo(() => {
      return (
        tasks.find(
          (task) =>
            task.localId ===
            activeTaskId
        ) ?? null
      );
    }, [
      tasks,
      activeTaskId,
    ]);


  function updateTask(
    localId: string,
    patch: Partial<VideoTask>
  ) {
    setTasks((current) =>
      current.map((task) =>
        task.localId === localId
          ? {
              ...task,
              ...patch,
            }
          : task
      )
    );
  }

  async function pollJob(
    localId: string,
    jobId: string,
    file: File
  ) {
    if (
      cancelledTasksRef.current.has(
        localId
      )
    ) {
      return;
    }

    try {
      const job =
        await getAnalysisJob(jobId);

      if (
        cancelledTasksRef.current.has(
          localId
        )
      ) {
        return;
      }

      if (
        job.status ===
        "processing"
      ) {
        updateTask(localId, {
          status: "processing",
          result: job,
        });

        window.setTimeout(
          () =>
            void pollJob(
              localId,
              jobId,
              file
            ),
          1500
        );

        return;
      }

      if (
        job.status === "done"
      ) {
        let finalJob = job;

        try {
          finalJob =
            await addEvidenceFrames(
              file,
              job
            );
        } catch (error) {
          console.warn(
            "Evidence generation failed",
            error
          );
        }

        updateTask(localId, {
          status: "done",
          result: finalJob,
        });

        return;
      }

      updateTask(localId, {
        status: "failed",
        result: job,
        error:
          job.error ??
          "Ошибка обработки",
      });
    } catch (error) {
      console.error(error);

      updateTask(localId, {
        status: "failed",
        error:
          error instanceof Error
            ? error.message
            : "Ошибка обработки",
      });
    }
  }

  async function processTask(
    task: VideoTask
  ) {
    try {
      const response =
        await analyzeVideo(
          task.file
        );

      if (
        cancelledTasksRef.current.has(
          task.localId
        )
      ) {
        return;
      }

      updateTask(task.localId, {
        jobId:
          response.job_id,
        status:
          "processing",
      });

      await pollJob(
        task.localId,
        response.job_id,
        task.file
      );
    } catch (error) {
      updateTask(task.localId, {
        status: "failed",

        error:
          error instanceof Error
            ? error.message
            : "Ошибка загрузки",
      });
    }
  }

  function handleUpload(
    files: File[]
  ) {
    const newTasks:
      VideoTask[] =
      files.map((file) => ({
        localId:
          createLocalId(),

        file,

        videoUrl:
          URL.createObjectURL(
            file
          ),

        status:
          "uploading",
      }));

    setTasks((current) => [
      ...current,
      ...newTasks,
    ]);

    if (
      !activeTaskId &&
      newTasks.length > 0
    ) {
      setActiveTaskId(
        newTasks[0].localId
      );
    }

    newTasks.forEach(
      (task) => {
        void processTask(task);
      }
    );
  }

  function removeTask(
    localId: string
  ) {
    cancelledTasksRef.current.add(
      localId
    );

    const task =
      tasks.find(
        (item) =>
          item.localId ===
          localId
      );

    if (task) {
      URL.revokeObjectURL(
        task.videoUrl
      );
    }

    const remaining =
      tasks.filter(
        (item) =>
          item.localId !==
          localId
      );

    setTasks(remaining);

    if (
      activeTaskId ===
      localId
    ) {
      setActiveTaskId(
        remaining[0]
          ?.localId ?? null
      );
    }
  }

  function clearAll() {
    tasks.forEach((task) => {
      cancelledTasksRef.current.add(
        task.localId
      );

      URL.revokeObjectURL(
        task.videoUrl
      );
    });

    setTasks([]);
    setActiveTaskId(null);
  }

  function seekTo(
    seconds: number
  ) {
    if (!videoRef.current) {
      return;
    }

    videoRef.current.currentTime =
      seconds;

    void videoRef.current
      .play()
      .catch(() => {});
  }

  return (
    <div className="app">
      <header className="topbar">
  <div className="brand">
    <span className="brand-primary">
      FORTE
    </span>

    <span className="brand-secondary">
      VER
      <span className="brand-accent">
        IT
      </span>
      I
    </span>
  </div>
      </header>

      <main className="main">

        <section
          className={`top-layout ${
            !activeTask
              ? "top-layout-single"
              : ""
          }`}
        >
          <UploadPanel
            onUpload={
              handleUpload
            }
          />

          {activeTask && (
    <ComplianceVideoPlayer
      videoUrl={
        activeTask.result?.preview_url ??
        activeTask.videoUrl
      }
      videoRef={
        videoRef
      }
      violations={
        activeTask.status ===
          "done"
          ? activeTask.result
              ?.violations ?? []
          : []
      }
    />
          )}
        </section>

        {tasks.length > 0 && (
          <section className="bottom-layout">
            <div className="panel uploads-panel">
              <div className="panel-title">
                <div className="title-with-count">
                  <h2>
                    Загруженные видео
                  </h2>

                  <span>
                    {
                      tasks.length
                    }
                  </span>
                </div>

                <button
                  className="icon-button"
                  onClick={clearAll}
                  title="Очистить"
                >
                  <Trash2
                    size={18}
                  />
                </button>
              </div>

              <div className="task-list">
                {tasks.map(
                  (task) => (
                    <TaskRow
                      key={
                        task.localId
                      }
                      task={
                        task
                      }
                      active={
                        task.localId ===
                        activeTaskId
                      }
                      onClick={() =>
                        setActiveTaskId(
                          task.localId
                        )
                      }
                      onRemove={() =>
                        removeTask(
                          task.localId
                        )
                      }
                    />
                  )
                )}
              </div>
            </div>

            <div className="panel result-panel">
              {activeTask && (
                <ResultPanel
                  task={
                    activeTask
                  }
                  onSeek={
                    seekTo
                  }
                />
              )}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

function TaskRow({
  task,
  active,
  onClick,
  onRemove,
}: {
  task: VideoTask;
  active: boolean;
  onClick: () => void;
  onRemove: () => void;
}) {
  return (
    <div
      className={`task-row ${
        active
          ? "task-row-active"
          : ""
      }`}
      onClick={onClick}
    >
      <TaskPreview
        task={task}
      />

      <div className="task-name">
        {task.file.name}
      </div>

      <StatusBadge
        task={task}
      />

      <button
        className="task-menu"
        onClick={(event) => {
          event.stopPropagation();
          onRemove();
        }}
        title="Удалить"
      >
        <MoreHorizontal
          size={20}
        />
      </button>
    </div>
  );
}

function TaskPreview({
  task,
}: {
  task: VideoTask;
}) {
  const evidence =
    task.result?.violations?.[0]
      ?.evidence_frames?.[0];

  if (evidence) {
    return (
      <img
        className="task-thumbnail"
        src={base64ToImageSrc(
          evidence.image_base64
        )}
        alt=""
      />
    );
  }

  return (
    <div className="task-thumbnail task-thumbnail-empty">
      <VideoIcon
        size={22}
      />
    </div>
  );
}

function StatusBadge({
  task,
}: {
  task: VideoTask;
}) {
  if (
    task.status ===
      "uploading" ||
    task.status ===
      "processing"
  ) {
    return (
      <div className="status status-analysis">
        <LoaderCircle
          className="rotate"
          size={16}
        />

        Анализ
      </div>
    );
  }

  if (
    task.status ===
    "failed"
  ) {
    return (
      <div className="status status-error">
        <AlertCircle
          size={16}
        />

        Ошибка
      </div>
    );
  }

  const violations =
    task.result
      ?.violations?.length ??
    0;

  if (violations > 0) {
    return (
      <div className="status status-analysis">
        <AlertCircle
          size={16}
        />

        {violations}
      </div>
    );
  }

  return (
    <div className="status status-done">
      <CheckCircle2
        size={16}
      />

      Готово
    </div>
  );
}

function ResultPanel({
  task,
  onSeek,
}: {
  task: VideoTask;
  onSeek: (
    seconds: number
  ) => void;
}) {
  if (
    task.status ===
      "uploading" ||
    task.status ===
      "processing"
  ) {
    return (
      <div className="center-state">
        <LoaderCircle
          className="rotate"
          size={34}
        />

        <strong>
          Анализ
        </strong>
      </div>
    );
  }

  if (
    task.status === "failed"
  ) {
    return (
      <div className="center-state error-state">
        <AlertCircle
          size={34}
        />

        <strong>
          Ошибка
        </strong>

        {task.error && (
          <span>
            {task.error}
          </span>
        )}
      </div>
    );
  }

  const result =
    task.result;

  if (!result) {
    return null;
  }

  const violations =
    result.violations ?? [];

  const hasViolations =
    violations.length > 0;

  return (
    <>
      <div className="result-header">
        <h2>
          {task.file.name}
        </h2>

        <StatusBadge
          task={task}
        />
      </div>

      <div
        className={`verdict-box ${
          hasViolations
            ? "verdict-warning"
            : "verdict-success"
        }`}
      >
        {hasViolations ? (
          <AlertCircle
            size={20}
          />
        ) : (
          <CheckCircle2
            size={20}
          />
        )}

        <strong>
          {hasViolations
            ? "Найдены признаки нарушений"
            : "Нарушений не обнаружено"}
        </strong>
      </div>

      {violations.length > 0 && (
        <div className="violations">
          {violations.map(
            (
              violation,
              index
            ) => (
              <article
                className="violation-item"
                key={`${violation.type}-${index}`}
              >
                <div className="violation-heading">
                  <div className="violation-index">
                    {index + 1}
                  </div>

                  <h3>
                    {violation.title ??
                      violation.type}
                  </h3>

                  <button
                    className="time-button"
                    onClick={() =>
                      onSeek(
                        violation.timecode_sec
                      )
                    }
                  >
                    <Clock3
                      size={15}
                    />

                    {formatTime(
                      violation.timecode_sec
                    )}
                  </button>
                </div>

                <p>
                  {
                    violation.description
                  }
                </p>

                {(violation.rule ||
                  violation.law) && (
                  <div className="rule">
                    {violation.rule}

                    {violation.rule &&
                      violation.law &&
                      " · "}

                    {violation.law}
                  </div>
                )}

                {violation.evidence_frames &&
                  violation.evidence_frames
                    .length >
                    0 && (
                    <div className="evidence-grid">
                      {violation.evidence_frames.map(
                        (
                          frame,
                          frameIndex
                        ) => (
                          <button
                            className="evidence"
                            key={
                              frameIndex
                            }
                            onClick={() =>
                              onSeek(
                                frame.timestamp_sec
                              )
                            }
                          >
                            <img
                              src={base64ToImageSrc(
                                frame.image_base64
                              )}
                              alt=""
                            />

                            <span>
                              {formatTime(
                                frame.timestamp_sec
                              )}
                            </span>
                          </button>
                        )
                      )}
                    </div>
                  )}
              </article>
            )
          )}
        </div>
      )}
    </>
  );
}
