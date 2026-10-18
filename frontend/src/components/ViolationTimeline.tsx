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
    duration <= 0 ||
    violations.length === 0
  ) {
    return null;
  }

  return (
    <div className="timeline-card">
      <div className="timeline-top">
        <strong>
          События
        </strong>

        <span>
          {violations.length}
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
                title={`${
                  violation.title ??
                  violation.type
                } · ${formatTime(
                  violation.timecode_sec
                )}`}
                onClick={() =>
                  onSeek(
                    violation.timecode_sec
                  )
                }
              >
                <span />
              </button>
            );
          }
        )}
      </div>

      <div className="timeline-scale">
        <span>00:00</span>

        <span>
          {formatTime(duration)}
        </span>
      </div>
    </div>
  );
}
