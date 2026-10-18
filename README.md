# ForteBank Datathon 26 — cash-desk violation detection

A self-hosted AI service that reviews cashier CCTV video and reports cash-handling and security violations: a verdict
(`violation` / `no_violation`), and for each finding its type, timecode, a description with the rule it may break,
and 1–3 evidence frames. Everything runs on the team's own GPU servers with open-source models; no cloud AI services,
no audio, no face recognition.

**Live service:** https://forteveriti.tail8331b7.ts.net/ (web UI)
**Jury API:** `https://forteveriti.tail8331b7.ts.net/api/v1/...` with header `X-API-Key` (the key is sent to the jury
separately and is not in this repository).

```bash
curl -X POST https://forteveriti.tail8331b7.ts.net/api/v1/analyze -H "X-API-Key: <key>" -F "file=@video_017.mov"
# {"job_id": "3f2a9c1e0b7d", "status": "processing"}
curl https://forteveriti.tail8331b7.ts.net/api/v1/jobs/3f2a9c1e0b7d -H "X-API-Key: <key>"
# {"job_id": "...", "status": "done", "verdict": "violation", "violations": [{"type": ..., "timecode_sec": ...,
#   "description": ..., "evidence_frames": [{"timestamp_sec": ..., "image_base64": "/9j/..."}]}]}
```

## Documents

| Document | Contents |
|---|---|
| [`service/SUBMISSION_RU.md`](service/SUBMISSION_RU.md) | **Описание решения для жюри** (на русском): адреса, инструкция, архитектура и модели, нормативные источники, результаты, ограничения, план развития |
| [`service/README.md`](service/README.md) | Architecture, API, operation and deployment (English) |
| [`service/MODEL_RESULTS.md`](service/MODEL_RESULTS.md) | Model comparison: why Qwen3.5-4B inside the pipeline, the neutral VLM-only test of 8 models |
| [`frontend/BACKEND_INTEGRATION.md`](frontend/BACKEND_INTEGRATION.md) | Web UI ↔ backend contract |

## How it works (short)

1. Any common video format is normalised to MP4 with ffmpeg.
2. RF-DETR (Apache-2.0) finds people as body boxes and objects (phones, bags, cups); ByteTrack tracks people.
3. A small VLM (Qwen3.5-4B AWQ, Apache-2.0, served by vLLM) locates the safe, door and desk in each video and answers
   narrow factual questions about sampled frames; close-up checks at key moments (departures, entries, handovers).
4. A pixel-based door check and fixed rules turn these facts into findings; an open-ended review looks for cash
   handled out of view and other violations; findings are verified, mapped to the bank's catalogue (`service/app/catalog.yaml`,
   32 types with paragraphs of Kazakhstan regulations) and reported in Russian with evidence frames.
5. A second, larger VLM (Qwen3.5-27B AWQ) reviews the whole clip with the bank's catalogue: it vetoes noisy findings
   when it sees no violation and adds the 15 catalogue types the small pipeline has no detector for, plus violations
   named outside the catalogue, at low confidence. If it is unreachable, the result of steps 1–4 stands.

Measured on the 42 labelled clips (40 unique videos; 22 with a violation, 20 clean): **21 of 22 violations found,
10 false alarms, 73.8 % verdict accuracy**, timecode median error 1.2 s, every clip under 181 s (limit 300 s). The
pipeline deliberately prefers false alarms to misses. All 32 catalogue types cite a Kazakhstan regulation and paragraph.

## Deployment modes (both fully local)

| Mode | Hardware | Models | Status |
|---|---|---|---|
| Lite | one GPU with 8 GB | Qwen3.5-4B + RF-DETR + rules (17 catalogue types) | v11: 76.2 %, 1 miss, 9 false alarms |
| **Full** | one GPU with 48 GB, or 8 + 24 GB | Lite + Qwen3.5-27B as a second checker (all 32 types + unseen) | **live** (v12): 73.8 %, 1 miss, 10 false alarms |

The full mode is switched on with `FORTE_SECOND_VLM_URL` / `FORTE_SECOND_VLM_KEY`; without them the service runs the
lite mode. The next step is per-finding "propose and verify" (see `service/SUBMISSION_RU.md`, §8).

## Repository layout

```
service/            backend + ML pipeline (FastAPI API, SQLite job queue, worker)
  app/              API, worker, pipeline (perception, facts, rules, door, review, evidence), violation catalogue
  deploy/           systemd units, nginx config, setup script, requirements, env template
  scripts/          evaluation, calibration, end-to-end checks, model comparison tools
  model-results/    raw results of the model comparison
  web/              built-in review page
frontend/           React + TypeScript + Vite web UI (upload, batch, history, evidence, timeline, rule links)
```

Datasets and test videos are not included (the organisers' data may not be published).

## Run it

Requirements: Linux, NVIDIA GPU with 8 GB+ VRAM, ffmpeg, Python 3.12, Node 22. `service/deploy/setup_server.sh`
creates the virtualenvs, downloads the open-source weights and generates an API key in `forte.env`; the systemd units
and nginx config are in `service/deploy/` (see `service/README.md`). Frontend: `cd frontend && npm ci && npm run build`, then serve
`dist/` with `service/deploy/nginx-forte.conf`.
