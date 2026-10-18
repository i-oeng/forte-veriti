import {
  useEffect,
  useRef,
  useState,
} from "react";

import type {
  RefObject,
} from "react";

import {
  Maximize,
  Minimize,
  Pause,
  Play,
} from "lucide-react";

import type {
  Violation,
} from "../types/analysis";

import {
  formatTime,
} from "../utils/time";

type Props = {
  videoUrl: string;

  violations: Violation[];

  videoRef: RefObject<HTMLVideoElement | null>;
};

export default function ComplianceVideoPlayer({
  videoUrl,
  violations,
  videoRef,
}: Props) {
  const containerRef =
    useRef<HTMLDivElement | null>(null);

  const [duration, setDuration] =
    useState(0);

  const [currentTime, setCurrentTime] =
    useState(0);

  const [playing, setPlaying] =
    useState(false);

  const [fullscreen, setFullscreen] =
    useState(false);

  // keep the button in sync however fullscreen ends (button, Esc, browser UI)
  useEffect(() => {
    const sync = () => {
      const doc = document as Document & { webkitFullscreenElement?: Element | null };
      setFullscreen(
        Boolean(document.fullscreenElement ?? doc.webkitFullscreenElement)
      );
    };

    document.addEventListener("fullscreenchange", sync);
    document.addEventListener("webkitfullscreenchange", sync);

    return () => {
      document.removeEventListener("fullscreenchange", sync);
      document.removeEventListener("webkitfullscreenchange", sync);
    };
  }, []);

  function togglePlay() {
    const video =
      videoRef.current;

    if (!video) {
      return;
    }

    if (video.paused) {
      void video.play();
    } else {
      video.pause();
    }
  }

  function seekTo(
    seconds: number
  ) {
    const video =
      videoRef.current;

    if (!video) {
      return;
    }

    video.currentTime =
      seconds;

    setCurrentTime(
      seconds
    );
  }

  function handleProgressClick(
    event: React.MouseEvent<HTMLDivElement>
  ) {
    if (
      duration <= 0
    ) {
      return;
    }

    const rect =
      event.currentTarget.getBoundingClientRect();

    const position =
      (event.clientX -
        rect.left) /
      rect.width;

    seekTo(
      position *
        duration
    );
  }

  async function toggleFullscreen() {
    const doc = document as Document & {
      webkitFullscreenElement?: Element | null;
      webkitExitFullscreen?: () => void;
    };
    const container = containerRef.current as (HTMLDivElement & {
      webkitRequestFullscreen?: () => void;
    }) | null;
    const video = videoRef.current as (HTMLVideoElement & {
      webkitEnterFullscreen?: () => void;
    }) | null;

    try {
      if (document.fullscreenElement) {
        await document.exitFullscreen();
      } else if (doc.webkitFullscreenElement) {
        doc.webkitExitFullscreen?.();
      } else if (container?.requestFullscreen) {
        await container.requestFullscreen();
      } else if (container?.webkitRequestFullscreen) {
        container.webkitRequestFullscreen();
      } else {
        video?.webkitEnterFullscreen?.();   // iPhone: only the video itself can go fullscreen
      }
    } catch {
      // the browser refused (e.g. not allowed in this frame); the player keeps working inline
    }
  }

  const progress =
    duration > 0
      ? (currentTime /
          duration) *
        100
      : 0;

  return (
    <div
      ref={containerRef}
      className="custom-player"
    >
      <video
        ref={videoRef}
        src={videoUrl}
        className="custom-player-video"
        muted
        playsInline
        onClick={togglePlay}
        onLoadedMetadata={(
          event
        ) => {
          setDuration(
            event.currentTarget
              .duration
          );
        }}
        onTimeUpdate={(
          event
        ) => {
          setCurrentTime(
            event.currentTarget
              .currentTime
          );
        }}
        onPlay={() =>
          setPlaying(true)
        }
        onPause={() =>
          setPlaying(false)
        }
      />

      <div className="player-controls">
        <button
          className="player-control-button"
          onClick={togglePlay}
        >
          {playing ? (
            <Pause
              size={20}
            />
          ) : (
            <Play
              size={20}
            />
          )}
        </button>

        <span className="player-time">
          {formatTime(
            currentTime
          )}
        </span>

        <div
          className="player-progress"
          onClick={
            handleProgressClick
          }
        >
          <div
            className="player-progress-filled"
            style={{
              width: `${progress}%`,
            }}
          />

          {violations.map(
            (
              violation,
              index
            ) => {
              const position =
                duration > 0
                  ? Math.min(
                      100,
                      Math.max(
                        0,
                        (violation.timecode_sec /
                          duration) *
                          100
                      )
                    )
                  : 0;

              return (
                <button
                  key={`${violation.type}-${index}`}
                  className="player-violation-marker"
                  style={{
                    left: `${position}%`,
                  }}
                  title={`${
                    violation.title ??
                    violation.type
                  } — ${formatTime(
                    violation.timecode_sec
                  )}`}
                  onClick={(
                    event
                  ) => {
                    event.stopPropagation();

                    seekTo(
                      violation.timecode_sec
                    );
                  }}
                >
                  <span />
                </button>
              );
            }
          )}
        </div>

        <span className="player-time">
          {formatTime(
            duration
          )}
        </span>

        <button
          className="player-control-button"
          onClick={
            toggleFullscreen
          }
          aria-label={
            fullscreen
              ? "Выйти из полноэкранного режима"
              : "На весь экран"
          }
          title={
            fullscreen
              ? "Выйти из полноэкранного режима"
              : "На весь экран"
          }
        >
          {fullscreen ? (
            <Minimize
              size={19}
            />
          ) : (
            <Maximize
              size={19}
            />
          )}
        </button>
      </div>
    </div>
  );
}
