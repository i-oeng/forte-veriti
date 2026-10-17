# Frontend ↔ backend (ML) integration

The backend runs on the server at `127.0.0.1:8000`. In production nginx (`:8080`) serves `dist/` from `/var/www/forte`
and proxies `/api`, `/ui`, `/media` and `/healthz` to it. Tailscale Funnel publishes nginx on
`https://xixi-ms-7e02.tail8331b7.ts.net`. **For the jury, 443 must point to nginx (8080), not to the Vite dev server:**
the dev server does not proxy the API, so uploads fail there.

## What the case requires from the UI (section 4)

1. Upload a video file; 2. show the processing status; 3. one of exactly two results:
   **«Нарушение не обнаружено»** or **«Обнаружены признаки возможных нарушений»** (use these exact words);
4. for each event: type, timecode, description; 5. one or more evidence frames; 6. what is seen and why it may be
   a violation (`observation`, `why`); 7. the rule and normative source: **document name + paragraph**
   (`law_short` / `law` + `rule_section`, link `law_url`).

## Endpoints for the browser (no key: the jury key must never be in the JS bundle)

| Method | Path | What |
|---|---|---|
| POST | `/ui/analyze` (field `file`) | one video → `{job_id, status}` |
| POST | `/ui/analyze/batch` (field `files`, repeated) | several videos in one request → `{batch_id, jobs: [{job_id, filename, status, error?}]}`; a rejected file does not fail the others |
| GET | `/ui/batches/{batch_id}` | progress of a batch: `{status, finished, total, jobs: [...]}` |
| GET | `/ui/jobs/{job_id}` | status + result (see below) + `job: {stage, progress, queue_position, filename}` |
| GET | `/ui/history?limit=50` | past checks made through the web UI (jury jobs are never listed) |
| GET | `/media/{job_id}/preview.mp4`, `/media/{job_id}/frames/v0_0.jpg`, `/media/{job_id}/clips/v0.mp4` | browser copy of the video, annotated evidence frames, clip around each event |

The jury API is `/api/v1/analyze` and `/api/v1/jobs/{job_id}` with header `X-API-Key` (exact case-spec JSON).
`POST /api/v1/analyze/batch` and `GET /api/v1/batches/{id}` exist there too.

## Fields of each violation (`/ui/jobs/{id}` → `violations[]`)

Case-spec fields: `type` (e.g. «Сейф оставлен открытым (SAFE_LEFT_OPEN)»), `timecode_sec`, `description` (Russian:
what is seen, why it may be a violation, the act and paragraph, «Требуется проверка сотрудником»),
`evidence_frames: [{timestamp_sec, image_base64}]` (first one = best frame, at `timecode_sec`).

Display fields:

| Field | Example / meaning |
|---|---|
| `title` | «Сейф оставлен открытым» |
| `type_id`, `tier` | `SAFE_LEFT_OPEN`, `A` (visible on video) / `B` (needs a reviewer to check context) |
| `confidence`, `confidence_basis` | 0–1, measured per type on the labelled clips (share of real violations where this type was reported); text for a tooltip. Missing when there is no data: hide the percentage then |
| `law` | full title of the act |
| `law_short` | «Правила кассовых операций и инкассации (пост. НБРК № 231)» |
| `law_url` | adilet.zan.kz link |
| `law_reg` | justice registration, edition |
| `law_about` | what the act covers and why it applies |
| `rule_section` | «абз. 2 п. 20, п. 80» |
| `rule` | the requirement, close to the act's wording |
| `observation`, `why` | the two halves of `description` |
| `frame_urls[]` | annotated versions of `evidence_frames` (same order) |
| `clip_url` | ~8 s clip around the event |

Suggested card: `title` + tier chip; evidence frames (click → seek to `timestamp_sec`); `observation`; `why`;
`law_short` as a link to `law_url` + `rule_section` + `rule`, with `law_about` in a tooltip or «Подробнее».

## Dev server

`.env`: `VITE_USE_MOCK=false` (keep `VITE_API_KEY` empty). `vite.config.ts`:

```ts
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    allowedHosts: ["xixi-ms-7e02.tail8331b7.ts.net"],
    proxy: Object.fromEntries(["/api", "/ui", "/media", "/healthz"].map((p) => [p, "http://127.0.0.1:8000"])),
  },
});
```

## History and batch upload (ready-made files)

* `src/api/history.ts`: `analyzeBatch(files)`, `fetchHistory(limit)`, `previewUrl(jobId)`.
* `src/components/HistoryPanel.tsx` (+ `.css`): collapsible «История проверок» list; calls `onOpen(item)`.

To mount them in `AnalyzePage.tsx`:

1. `VideoTask`: make `file` optional and add `name: string` (use `task.name` where `task.file.name` is shown; show
   the size only when `task.file` exists).
2. Batch: in `handleUpload`, when `files.length > 1`, call `analyzeBatch(files)` once, then for each task set
   `jobId` from `jobs[i].job_id` and start `pollJob(localId, jobId)`; a `rejected` entry → task `failed` with
   `jobs[i].error`. One file → keep `processTask`.
3. History: render `<HistoryPanel onOpen={openFromHistory} />` under `<UploadPanel/>`, with

```ts
function openFromHistory(item: HistoryItem) {
  const existing = tasks.find((t) => t.jobId === item.job_id);
  if (existing) { setActiveTaskId(existing.localId); return; }
  const localId = createLocalId();
  setTasks((cur) => [...cur, { localId, name: item.filename, videoUrl: previewUrl(item.job_id),
                               jobId: item.job_id, status: "processing", startedAt: Date.now() }]);
  setActiveTaskId(localId);
  void pollJob(localId, item.job_id);
}
```

## Production deploy

```bash
npm run build && sudo rsync -a --delete dist/ /var/www/forte/
sudo tailscale funnel --bg --https=443 http://127.0.0.1:8080
```
