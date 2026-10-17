import { useCallback, useEffect, useState } from "react";

import { fetchHistory, type HistoryItem } from "../api/history";

import "./HistoryPanel.css";

// List of past checks from the server. `onOpen` receives the chosen item; the page loads its result
// with getAnalysisJob(item.job_id) and plays previewUrl(item.job_id).
export default function HistoryPanel({
  onOpen,
  refreshKey = 0,
}: {
  onOpen: (item: HistoryItem) => void;
  refreshKey?: number;
}) {
  const [items, setItems] = useState<HistoryItem[]>([]);
  const [open, setOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);

    try {
      setItems(await fetchHistory(100));
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Ошибка загрузки истории");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (open) {
      void load();
    }
  }, [open, load, refreshKey]);

  // refresh while something is still being analysed
  useEffect(() => {
    if (!open || !items.some((item) => item.status === "processing")) {
      return;
    }

    const timer = window.setTimeout(() => void load(), 5000);

    return () => window.clearTimeout(timer);
  }, [open, items, load]);

  return (
    <section className="history">
      <div className="history-header">
        <button className="history-toggle" onClick={() => setOpen((v) => !v)}>
          {open ? "▾" : "▸"} История проверок
        </button>

        {open && (
          <button className="history-refresh" onClick={() => void load()} disabled={loading}>
            {loading ? "Обновляем…" : "Обновить"}
          </button>
        )}
      </div>

      {open && error && <p className="history-error">{error}</p>}

      {open && !error && items.length === 0 && !loading && (
        <p className="history-empty">Проверок пока нет.</p>
      )}

      {open && items.length > 0 && (
        <ul className="history-list">
          {items.map((item) => (
            <li key={item.job_id}>
              <button
                className="history-item"
                onClick={() => onOpen(item)}
                disabled={item.status === "failed"}
                title={item.error || "Открыть результат"}
              >
                <span className="history-name">{item.filename}</span>

                <span className="history-date">
                  {new Date(item.created_at * 1000).toLocaleString("ru-RU", {
                    day: "2-digit",
                    month: "2-digit",
                    hour: "2-digit",
                    minute: "2-digit",
                  })}
                </span>

                <HistoryBadge item={item} />
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function HistoryBadge({ item }: { item: HistoryItem }) {
  if (item.status === "processing") {
    const percent = Math.round((item.progress || 0) * 100);

    return <span className="history-badge history-processing">Анализ {percent}%</span>;
  }

  if (item.status === "failed") {
    return <span className="history-badge history-failed">Ошибка</span>;
  }

  if (item.verdict === "violation") {
    return (
      <span className="history-badge history-violation">
        ⚠ {item.violations_count ?? 0} событ.
      </span>
    );
  }

  return <span className="history-badge history-ok">✓ Нарушение не обнаружено</span>;
}
