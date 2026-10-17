# Cash Desk Review — local AI service for cash-discipline violations in CCTV video

Upload a cashier video (`.mov`, `.mp4`, `.avi`, `.mkv`, `.ts`, `.mts`, `.m2ts`, `.mpg`, `.mpeg`, `.m4v`, `.webm`)
and get a verdict (`violation` / `no_violation`). Each possible violation comes with:

* its type (readable name plus catalogue ID)
* a timecode
* a description of what is visible and why it may break a rule
* 1–3 evidence frames, the best snapshot first
* a short clip around the event
* the normative act (Republic of Kazakhstan) and the paragraph it relates to

Types and descriptions are in Russian (`FORTE_OUTPUT_LANG=ru`; `en` is available). Several videos can be
uploaded at once (batch upload), and past checks stay available in the history.

Everything runs on our own machine. No cloud AI service is called, and once the weights are downloaded the
service works offline. Every result is a hint for a human reviewer, not a final conclusion.

## Architecture

```
upload (.mov) ─► API (FastAPI) ─► SQLite job queue ─► worker (4 videos in parallel, 270 s budget each)
                                                       │
  0. normalize  any accepted container → MP4 (H.264/HEVC in MP4/MOV/MKV is remuxed losslessly; anything else,
                and all transport streams, re-encoded to H.264: timestamps from 0, deinterlaced, square pixels)
  1. decode     ffmpeg → 5 fps; handles rotation, HEVC, VFR, any resolution; crops black letterbox borders
  2. perceive   RF-DETR Small (people as silhouettes only; phones, cups, bags) + ByteTrack tracks
  3. events     entries, exits, walking away, two people close, objects appearing (precise times)
  4. layout     the VLM locates the safe, the glass door and the desk once per video, so a new camera
                angle is handled without calibration
  5. facts      the VLM reads FACTS, not judgements, at ~0.5–2 fps: people in the room, safe open,
                someone at the safe, what the cashier holds, counting by hand, machine in use, food,
                bags, cash bag … (frame + zoomed safe crop; JSON constrained by a schema with xgrammar)
  6. moments    close-ups at the moments that decide a violation:
                · the cashier has just left → desk, laptop screen and safe close-ups (what was left behind?)
                · someone walks in → is cash out on the desk or in the cashier's hands?
                · a client's visit → was an ID shown before cash changed hands?
                · a collection bag arrives → was it inspected / were documents checked?
                · door region (pixels, not the VLM) → did the door stay open after someone passed?
  7. rules      facts + moments + perception → violations typed with catalogue IDs (app/pipeline/rules.py);
                timecode = the moment the violating action starts
  8. review     open-ended VLM review only for what facts cannot capture (cash handled out of view,
                replacement banknotes, and `other`, where the model names an unlisted violation itself),
                followed by a focused yes/no verification
  9. evidence   sharpest native frame at the event + frames ±2 s, an 8-second H.264 clip, an annotated copy
```

**Why facts and rules.** A small VLM asked "is this a violation?" says yes to almost everything (about
55% accuracy in our tests). Asked for facts ("is the safe door open?") it was 80% correct. Rules then
decide, give exact timecodes, and explain every verdict in terms a reviewer can check.

**Privacy.** People are handled only as body boxes (silhouettes). No face detection or recognition is used.
The uploaded original is deleted as soon as the analysis ends (`FORTE_KEEP_SOURCE=0`) and is never served.
Jury (API) jobs get no browser preview, their frames and clips are not served under `/media`, they are not
readable through the key-less `/ui/` endpoints and never appear in the history: test videos are not kept,
published or reused (case section 10).

**No invented citations.** Acts and paragraphs come only from `app/catalog.yaml`, never from the model. Every
paragraph was checked against the current text of the act on adilet.zan.kz (2026-10-17). The model's own
words (the observation of a review finding, the name of a new type) are translated into Russian by the same
local model; the reason and the norm come from the catalogue.

**Offline.** After the weights are on disk nothing needs the internet: models load from local paths,
`HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` are set, vLLM usage statistics are off. Watching the service
processes during analysis showed no outbound connections.

## Models

| Role | Model | License |
|---|---|---|
| Vision-language model (layout, facts, moment checks, review) | **Qwen3.5-4B** (AWQ 4-bit) on 8 GB; candidates on 24 GB: Qwen3.5-9B FP8, Qwen3-VL-8B FP8, InternVideo2.5-8B, InternVL3.5-14B, Cosmos-Reason2-8B | Apache-2.0 (Cosmos: NVIDIA Open Model License, commercial use allowed) |
| Person / object detector | RF-DETR Small (Roboflow, DINOv2 backbone, COCO) | Apache-2.0 |
| Tracker | ByteTrack (`trackers`) | Apache-2.0 |
| Serving | vLLM 0.31 + xgrammar structured output (OpenAI-compatible API, so the backend can be swapped) | Apache-2.0 |
| Video | FFmpeg | LGPL/GPL (CLI) |
| Backend / frontend | FastAPI, SQLite / React, Vite | MIT / public domain / MIT |

All the VLM candidates above (Qwen3.5, Qwen3-VL, InternVL3.5, InternVideo2.5) are Apache-2.0; Cosmos-Reason2 is
under the NVIDIA Open Model License, which allows commercial use. Everything is usable commercially (case FAQ
24 and 34).

## Violation types and normative sources

All types are defined in `app/catalog.yaml`, based on the team catalogue. Tier A means the breach itself is
visible on video; tier B means what is visible needs a human to check the context or documents. Every type
cites a public act of the Republic of Kazakhstan (the case says no dataset violation rests on internal
ForteBank rules only, FAQ 32). All tiers are reported as possible violations.

| ID | Name (in reports) | Tier | Act, paragraph |
|---|---|---|---|
| `previous_customer_cash_left_exposed` | Деньги предыдущего клиента оставлены на столе | A | №231 абз. 2 п. 20 |
| `cash_counted_without_customer_present` | Пересчёт денег клиента после его ухода | A | №231 абз. 1 п. 20 |
| `cashier_left_during_active_customer_service` | Кассир покинул рабочее место во время обслуживания | A | №231 п. 22 |
| `customer_identification_not_observed` | Идентификация клиента не наблюдалась | B | №231 пп. 1) п. 19, пп. 1) п. 31 |
| `possible_unauthorized_vault_access` | Возможный доступ постороннего в зону кассы | B | №14 п. 8, пп. 1) п. 16 |
| `collection_exposed_to_third_parties` | Кассовая операция доступна посторонним | B | №14 п. 21 |
| `cash_desk_access_door_left_unsecured` | Дверь кассы оставлена открытой | B | №14 пп. 1) п. 16, п. 8 |
| `KEY_LEFT_IN_SAFE` | Ключ оставлен в сейфе | B | №231 п. 80, п. 76 |
| `BANK_DOCUMENTS_EXPOSED` | Банковские документы или незаблокированный компьютер оставлены без присмотра | B | №53 пп. 9) п. 71; №231 пп. 17) п. 2, п. 80 |
| `SAFE_LEFT_OPEN` | Сейф оставлен открытым | B | №231 абз. 2 п. 20, п. 80 |
| `COLLECTION_BAG_ACCEPTED_WITHOUT_INTEGRITY_CHECK` | Инкассаторская сумка принята без проверки | A | №231 п. 24 (пп. 1, 3, 4) |
| `PERSONAL_ITEM_IN_CASH_NODE` | Личные вещи кассира в кассовом узле | B | №231 абз. 1 п. 12 |
| `PHONE_WITH_CAMERA_IN_CASH_NODE` | Смартфон с камерой в кассовом узле | B | №231 абз. 2 п. 12 |
| `CASH_PROCESSING_WITHOUT_CONTROLLER` | Обработка наличных без контролирующего работника | A | №231 п. 42 |
| `CASH_OUT_OF_VIEW` | Наличные вне поля зрения камеры (review) | B | №231 абз. 2 п. 20, п. 12 |
| `REPLACEMENT_CASH_NOT_RECOUNTED` | Заменённые банкноты не пересчитаны (review) | A | №231 пп. 2), 4) п. 19 |
| `other` | Иное видимое нарушение порядка, named by the model (review) | B | №231 (act as a whole; the reviewer sets the paragraph, FAQ 31) |

Another 15 types (sorting, counterfeit procedure, receipts, end-of-day reconciliation, …) are in the
catalogue with checked paragraphs (№231 пп. 21, 25, 32, 34, 41, 43, 56, 57, 59–61; №230) but are not detected
automatically yet. They usually need audio, documents or a stronger model. The `other` type still reports
such cases under a name the model chooses.

**The acts.** Each result shows the act's short name, a link, the paragraph, the requirement close to the
act's wording, and what the act is about:

| Ref | Act | What it covers and why we use it |
|---|---|---|
| [№231](https://adilet.zan.kz/rus/docs/V1900019680) | Постановление Правления НБРК от 29.11.2019 № 231 «Об утверждении Правил осуществления кассовых операций и операций по инкассации банкнот, монет и ценностей в банках второго уровня, …» (рег. МЮ РК № 19680) | The main cash-discipline act: how a cashier accepts, issues, counts, sorts, packs and stores cash, handles suspicious banknotes, audits and collection. Source of the rules on the cashier's workplace, counting in the client's presence, identification, storage of cash and keys, collection bags, personal items in the cash unit. |
| [№14](https://adilet.zan.kz/rus/docs/V2000020075) | Постановление Правления НБРК от 24.02.2020 № 14 «Об утверждении Правил организации охраны и устройства помещений банков второго уровня, …» (рег. МЮ РК № 20075) | Security and layout of rooms where cash is handled: cash unit, cash desk room, vault, safe room, collection box, alarms, video surveillance. Source of the isolation of the cash unit, the lockable door of each cash room, and protecting collection from third parties. |
| [№53](https://adilet.zan.kz/rus/docs/V2600038336) | Постановление Правления АРРФР от 03.04.2026 № 53 «Об утверждении Требований к обеспечению информационной безопасности банков, …» (рег. МЮ РК № 38336) | The regulator's information-security requirements, including passwords, accounts and locking of workstations. Used for a work computer left unattended with a banking application open. |
| [№230](https://adilet.zan.kz/rus/docs/V1700016120) | Постановление Правления НБРК от 29.11.2017 № 230 «Об утверждении Правил определения платежности банкнот и монет национальной валюты РК» (рег. МЮ РК № 16120) | Which tenge banknotes and coins are legal tender, worn or unfit; referenced by №231 п. 57. Used by the banknote checking and sorting types. |

**How rules map to video.** Each type has a visual definition (what must be seen) and a detector built
from checkable facts. For example, previous customer's cash left exposed (№231 абз. 2 п. 20) fires when a
close-up of the desk at the moment a new person walks in, or right after the cashier leaves, shows
banknotes lying on the desk. The description states what is seen, why it may break the rule, and the act
and paragraph, then «Требуется проверка сотрудником».

**Timecodes** (FAQ 10 leaves the convention to the team): for an action, the moment it starts (the door
opens, the phone comes out); for something left behind (cash, an open safe, a key, an unlocked computer),
the moment the cashier walks away from it; for a missing step (no ID, no bag inspection, no client present
during counting), the moment the step should have happened (cash changes hands, the bag is accepted, the
client leaves). The first evidence frame is always at the timecode.

## Using the web interface

1. Open the service URL and drop one or several `.mov` files on the upload area (several files are analysed
   in parallel; a wrong file is rejected without stopping the others).
2. Each video shows its stage and progress («поиск силуэтов…», «анализ сцены…», …) until it is done.
3. The result is «Нарушение не обнаружено» or «Обнаружены признаки возможных нарушений». For each event:
   the type, the timecode (click it to jump the player there), the evidence frames (click one to jump), what
   is seen and why it may be a violation, and the act with its paragraph, a link to adilet.zan.kz and a note
   on what the act covers; a clip around the event.
4. «История проверок» lists earlier checks made through the interface; open one to see its result again.

## API

All API calls need the header `X-API-Key`.

```bash
curl -X POST https://<host>/api/v1/analyze -H "X-API-Key: $KEY" -F "file=@video_017.mov"
# {"job_id": "3f2a9c1e0b7d", "status": "processing"}
curl https://<host>/api/v1/jobs/3f2a9c1e0b7d -H "X-API-Key: $KEY"
# {"job_id": "...", "status": "done", "verdict": "violation", "violations": [{"type":
#   "Previous customer's cash left exposed (previous_customer_cash_left_exposed)", "timecode_sec": 33.0,
#   "description": "...", "evidence_frames": [{"timestamp_sec": 33.0, "image_base64": "/9j/..."}]}]}
```

`status` is `processing`, `done` or `failed`. The first evidence frame is the best snapshot and its
`timestamp_sec` equals `timecode_sec`. Requests without a valid key get 401. Add `?extended=true` to also get,
per violation, the display fields: `title` (Russian name), `type_id`, `tier`, `confidence`, `law` (full act
title), `law_short`, `law_url`, `law_reg`, `law_about`, `rule_section`, `rule` (the requirement), `observation`,
`why`. Interactive docs are at `/api/docs`.

**Confidence** (UI and `?extended=true` only) is measured, not guessed: for each type, the share of clips that
really contain a violation among the labelled clips where the service reported that type, smoothed as
(hits + 1) / (n + 2). `confidence_basis` says it in words («нарушение подтвердилось на 5 из 5 размеченных
роликов…»). Types without evaluation data get no percentage. Recompute after an evaluation with
`python -m scripts.calibrate --run NAME` (writes `app/calibration.json`). The rule's internal score is
`rule_score`.

**Batch upload** (an extension; the case's two endpoints stay unchanged):

```bash
curl -X POST https://<host>/api/v1/analyze/batch -H "X-API-Key: $KEY" -F "files=@a.mov" -F "files=@b.mov"
# {"batch_id": "...", "jobs": [{"job_id": "...", "filename": "a.mov", "status": "processing"}, ...]}
curl https://<host>/api/v1/batches/<batch_id> -H "X-API-Key: $KEY"   # {status, finished, total, jobs: [...]}
```

Each file becomes its own job, so results are read with `/api/v1/jobs/{job_id}`. Videos are not analysed one
by one: the worker runs 4 at a time and the VLM server batches their requests together (tested with 12
concurrent videos, all finished within 181 s; 3 clips in one batch finished in 50 s). API jobs are served
before web-UI jobs.

The web interface uses the key-less `/ui/analyze`, `/ui/analyze/batch`, `/ui/batches/{id}`, `/ui/jobs/{id}`,
`/ui/history` and `/media/{job_id}/...` (preview, annotated frames, clips). These see web-UI jobs only.

## Running

**Host with systemd** (`~/forte/{service,models,venv-app,venv-vllm,var}`):

```bash
cp service/deploy/forte.env.example ~/forte/forte.env      # set FORTE_API_KEY (and the GPU settings)
cp service/deploy/*.service ~/.config/systemd/user/ && systemctl --user daemon-reload
systemctl --user enable --now forte-vllm forte-api forte-worker
sudo cp service/deploy/nginx-forte.conf /etc/nginx/sites-available/forte   # frontend + /api /ui /media proxy
sudo tailscale funnel --bg --https=443 http://127.0.0.1:8080      # public URL: frontend + jury API
sudo tailscale funnel --bg --https=8443 http://127.0.0.1:8000     # optional: backend directly
```

**Fresh GPU box / container without systemd** (Runpod, Vast.ai):

```bash
FORTE_HOME=/workspace/forte VLM_REPO=RedHatAI/Qwen3.5-9B-FP8-dynamic bash service/deploy/setup_server.sh
FORTE_HOME=/workspace/forte bash service/deploy/start_all.sh        # vLLM + API + worker with auto-restart
```

**Evaluate** on the labelled clips. This reports verdict accuracy, misses and false alarms, timecode error,
and type accuracy against the team labels:

```bash
cd ~/forte/service && FORTE_HOME=~/forte ~/forte/venv-app/bin/python -m scripts.eval --run NAME
```

## Results on the 42 labelled clips (22 violations, 20 clean)

| Version | Verdict accuracy | Missed | False alarms | Type accuracy | Timecode ≤ 3 s |
|---|---|---|---|---|---|
| v7, Qwen3.5-4B | 67% | 5 | 9 | 82% | 65% |
| **v8 (live)**, Qwen3.5-4B, Russian output | **69%** | 4 | 9 | 83% | 67% (median 2.1 s) |

All 42 clips finished within 198 s (limit 300 s).

The model comparison on the 24 GB GPU is in [`MODEL_RESULTS.md`](MODEL_RESULTS.md).

## Limitations and known failure cases

* **Small model on 8 GB.** Fine hand actions are the weakest area: replacement banknotes, hand counting
  versus machine counting, cash out of view. The tested 9B and 8B alternatives have not beaten the current
  4B pipeline. See `MODEL_RESULTS.md` for measured results and the validation needed before an upgrade.
* **Glass walls.** People behind the glass and reflections confuse people counts and the door signal. The
  door detector still fires on some clean clips.
* **Small details.** A key left in the safe lock is a few pixels at this resolution and is usually missed.
* **Client identification** is tier B. An ID can be shown out of view or be digital, so it can fire on
  correct visits.
* **Departure timecodes** can be 2–4 s late when the cashier stays visible behind the glass after leaving.
* **Nothing that needs data invisible on video** is judged (postings, reconciliation, documents), and
  people are never identified.
* **Door violations look like normal walk-throughs.** In the labelled clips the door is open for about
  3.2–3.4 s both when the label says «door left open» (`Х-22`, `Х-24`, `Х-26`, `Х-28`, `Х-13`) and when
  someone simply walks through on a clean clip (`Х-16`, `Х-19`, `Х-25`, `Х-27`, `Х-29`, `Х-35`, `Х-37`). The
  rule keeps only longer openings or openings tied to people passing, so some door labels are missed and
  some clean clips are flagged.
* **Known misses** (v8): `Х-13` cash counted alone after a collection bag (the bag is seen in a single
  frame), `Х-1` replacement banknotes, `Х-44` hands below the desk edge (subtle hand actions), `Х-24` (found in
  some runs, missed in others). `Х-22` is now flagged, but as an unlocked computer, not the key in the lock.
* **Known false alarms** (v8): phone use on `Х-5`/`Х-48` (the same clip twice), client identification not
  seen on `Х-19`, `Х-37` (an ID can be shown out of view), door events on `Х-12`, `Х-31`, `Х-39`, personal item
  on `Х-16`, cash processed without a controller on `Х-14`.
* **Determinism.** Two byte-identical clips (`Х-4` = `Х-47`) can get different results, because the VLM
  server batches requests from parallel jobs and that changes the numerics slightly.
