import type {
  AnalysisJob,
  EvidenceFrame,
} from "../types/analysis";

function waitForMetadata(
  video: HTMLVideoElement
): Promise<void> {
  return new Promise((resolve, reject) => {
    if (video.readyState >= 1) {
      resolve();
      return;
    }

    const onLoaded = () => {
      cleanup();
      resolve();
    };

    const onError = () => {
      cleanup();
      reject(
        new Error(
          "Не удалось прочитать видео для evidence frames"
        )
      );
    };

    const cleanup = () => {
      video.removeEventListener(
        "loadedmetadata",
        onLoaded
      );

      video.removeEventListener(
        "error",
        onError
      );
    };

    video.addEventListener(
      "loadedmetadata",
      onLoaded
    );

    video.addEventListener(
      "error",
      onError
    );
  });
}

function seekVideo(
  video: HTMLVideoElement,
  time: number
): Promise<void> {
  return new Promise((resolve, reject) => {
    if (
      Math.abs(
        video.currentTime - time
      ) < 0.01 &&
      video.readyState >= 2
    ) {
      resolve();
      return;
    }

    const onSeeked = () => {
      cleanup();
      resolve();
    };

    const onError = () => {
      cleanup();

      reject(
        new Error(
          "Ошибка перехода к кадру"
        )
      );
    };

    const cleanup = () => {
      video.removeEventListener(
        "seeked",
        onSeeked
      );

      video.removeEventListener(
        "error",
        onError
      );
    };

    video.addEventListener(
      "seeked",
      onSeeked
    );

    video.addEventListener(
      "error",
      onError
    );

    video.currentTime = time;
  });
}

function getEvidenceTimestamps(
  eventTime: number,
  duration: number
): number[] {
  const rawTimes = [
    eventTime - 0.4,
    eventTime,
    eventTime + 0.4,
  ];

  const safeDuration =
    Math.max(duration - 0.05, 0);

  const normalized = rawTimes.map(
    (time) =>
      Math.min(
        Math.max(time, 0),
        safeDuration
      )
  );

  const unique =
    normalized.filter(
      (time, index, array) =>
        array.findIndex(
          (other) =>
            Math.abs(other - time) <
            0.05
        ) === index
    );

  return unique.slice(0, 3);
}

export async function addEvidenceFrames(
  file: File,
  job: AnalysisJob
): Promise<AnalysisJob> {
  if (
    job.status !== "done" ||
    !job.violations?.length
  ) {
    return job;
  }

  const video =
    document.createElement("video");

  video.muted = true;
  video.preload = "auto";
  video.playsInline = true;

  const videoUrl =
    URL.createObjectURL(file);

  video.src = videoUrl;

  try {
    await waitForMetadata(video);

    const duration =
      video.duration;

    if (
      !Number.isFinite(duration) ||
      duration <= 0
    ) {
      return job;
    }

    const canvas =
      document.createElement("canvas");

    const context =
      canvas.getContext("2d");

    if (!context) {
      return job;
    }

    const sourceWidth =
      video.videoWidth || 1280;

    const sourceHeight =
      video.videoHeight || 720;

    const maxWidth = 960;

    const scale = Math.min(
      1,
      maxWidth / sourceWidth
    );

    canvas.width =
      Math.round(
        sourceWidth * scale
      );

    canvas.height =
      Math.round(
        sourceHeight * scale
      );

    const updatedViolations = [];

    for (
      const violation of
      job.violations
    ) {
      // Если backend уже вернул
      // реальные evidence frames,
      // ничего не перезаписываем.
      if (
        violation.evidence_frames &&
        violation.evidence_frames
          .length > 0
      ) {
        updatedViolations.push(
          violation
        );

        continue;
      }

      const timestamps =
        getEvidenceTimestamps(
          violation.timecode_sec,
          duration
        );

      const frames: EvidenceFrame[] =
        [];

      for (
        const timestamp of timestamps
      ) {
        await seekVideo(
          video,
          timestamp
        );

        context.drawImage(
          video,
          0,
          0,
          canvas.width,
          canvas.height
        );

        const dataUrl =
          canvas.toDataURL(
            "image/jpeg",
            0.85
          );

        const base64 =
          dataUrl.split(",")[1];

        if (!base64) {
          continue;
        }

        frames.push({
          timestamp_sec:
            Number(
              timestamp.toFixed(2)
            ),

          image_base64:
            base64,
        });
      }

      updatedViolations.push({
        ...violation,

        evidence_frames:
          frames,
      });
    }

    return {
      ...job,

      violations:
        updatedViolations,
    };
  } finally {
    video.pause();

    video.removeAttribute("src");

    video.load();

    URL.revokeObjectURL(
      videoUrl
    );
  }
}
