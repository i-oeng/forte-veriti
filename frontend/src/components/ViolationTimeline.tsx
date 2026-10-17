import type {
  Violation,
} from "../types/analysis";

import {
  formatTime,
} from "../utils/time";

type Props = {
  duration: number;
  violations: Violation[];
  onSeek: (seconds: number) => void;
};

export default function ViolationTimeline({
  duration,
  violations,
  onSeek,
}: Props) {
  if (
    !Number.isFinite(duration) ||
    duration <= 0
  ) {
    return null;
  }

  return (
    <div className="timeline">
      <div className="timeline-header">
        <span>Временная шкала событий</span>

        <span>
          {violations.length} событий
        </span>
      </div>

      <div className="timeline-track">
        {violations.map(
          (violation, index) => {
            const position =
              Math.min(
                100,
                Math.max(
                  0,
                  (violation.timecode_sec /
                    duration) *
                    100
                )
              );

            return (
              <button
                key={`${violation.type}-${index}`}
                className="timeline-marker"
                style={{
                  left: `${position}%`,
                }}
                onClick={() =>
                  onSeek(
                    violation.timecode_sec
                  )
                }
                title={`${
                  violation.title ||
                  violation.type
                } — ${formatTime(
                  violation.timecode_sec
                )}`}
              >
                <span />
              </button>
            );
          }
        )}
      </div>

      <div className="timeline-times">
        <span>00:00</span>

        <span>
          {formatTime(
            duration / 2
          )}
        </span>

        <span>
          {formatTime(duration)}
        </span>
      </div>

      {violations.length > 0 && (
        <div className="timeline-events">
          {violations.map(
            (violation, index) => (
              <button
                key={`${violation.type}-event-${index}`}
                onClick={() =>
                  onSeek(
                    violation.timecode_sec
                  )
                }
              >
                <strong>
                  {formatTime(
                    violation.timecode_sec
                  )}
                </strong>

                <span>
                  {violation.title ||
                    violation.type}
                </span>
              </button>
            )
          )}
        </div>
      )}
    </div>
  );
}
