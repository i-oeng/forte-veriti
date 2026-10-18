# Models and accuracy — 17 October 2026

The deployed **complete live-service result** is Qwen3.5-4B AWQ, running locally on the 8 GB GPU.
This is a result for this pipeline and dataset, not a general ranking of the models.

The new isolated, compiled-inference baseline on the Vast RTX 3090 Ti finished at **31/42 = 73.8%**:
TP18 / FN4 / FP7 / TN13, zero job errors, zero incomplete fact sequences, and zero skipped review windows.
Median latency was 53.2 seconds, p95 99.94 seconds, maximum 116.85 seconds; type hits were 14/18.
This is a direct-pipeline run with previews disabled, not public API latency. It establishes the
same-machine comparison baseline for Cosmos Reason2-8B, Qwen3.5-27B AWQ and InternVL3.5-14B AWQ,
whose results remain pending. It does not establish that a production deployment would score 73.8%.
Artifacts: `model-results/candidates-20261017-baseline.json`; methodology: `model-results/CANDIDATE_PROTOCOL.md`.

| VLM | Recorded verdict accuracy | Missed violations | False alarms | Status |
|---|---:|---:|---:|---|
| Qwen3.5-4B AWQ, production v8 | 29/42 = 69.0% | 4/22 | 9/20 | Deployed |
| Qwen3.5-9B FP8, v7 bake-off | 27/42 = 64.3% | 6/22 | 9/20 | Completed |
| Qwen3-VL-8B-Instruct FP8, v7 bake-off | 25/42 = 59.5% | 9/22 | 8/20 | Completed |
| InternVL3.5-14B, FP8 at load | No score in retrieved summary | — | — | Summary ends at start marker |
| Qwen3.5-4B AWQ, 24 GB control | No score in retrieved summary | — | — | Completion unconfirmed |
| Cosmos-Reason2-8B | No score in retrieved summary | — | — | Completion unconfirmed |
| InternVideo2.5-Chat-8B | No score | — | — | Failed to start in vLLM |

Source: the team's run notes and the retrieved `model-results/vast-summary-20261017.txt`. The larger-model bake-off used v7 rules;
production's newest score uses v8. A clean comparison needs the same pipeline revision, prompts,
sampling, concurrency and repeated runs on the same GPU. The rules were calibrated on the 4B model.

Other small-model checkpoints (Qwen3-VL-2B/4B, Qwen3.5-2B, Ovis2.5-2B, InternVL3.5-2B/8B) were downloaded; that alone does not establish
a completed evaluation or justify calling them better or worse.

## What actually makes the decision

RF-DETR detects people and objects; ByteTrack follows their tracks. These are not VLMs.
Qwen reads scene layout and observable facts from frames and close-ups. Temporal rules combine those
facts with object tracks and the pixel door signal. An open-ended VLM review covers some cash-handling
events and unknown types, with a separate verification step. Thus verdict accuracy measures the entire
pipeline, including sampling and rules, rather than the VLM alone.

## What 69% means

The current confusion matrix is TP18 / FN4 / FP9 / TN11. Recall is **81.8%**, precision **66.7%**,
and the false-alarm rate on clean clips is **45%**. Reported type accuracy is 83%, a separate metric.
The slowest clip in the recorded v8 run took 198 seconds.

One clip changes overall accuracy by 2.38 percentage points. The set includes byte-identical duplicates
(`Х-4`/`Х-47`, `Х-5`/`Х-48`), and some answers change between runs. These 42 clips have already guided
rule tuning: their accuracy is not an unbiased estimate for the jury's unseen camera view.

## Next experiments, in order

**Required scope:** the supplied label PDF, `forte_ejiggas.pdf`, and `Copy of sheets.xlsx` are
non-exhaustive examples. Evaluate novel violation types as well as known types. Preserve a free-form
name, timestamp and concrete visible evidence for novel findings; do not invent a matching law.
Reserve whole violation categories from prompt examples and tuning, then measure their recall
alongside false alarms on unfamiliar clean scenes. Categories already used in development are not
a genuinely untouched holdout; independently collected examples are needed for a strong claim.

**Unresolved implementation limitation:** `analyze.select` requires four supporting windows for
`other`, even when the verification answer is yes. Brief novel events can be discarded. Clustering
and deduplication use the shared `other` ID and may merge different novel event names. Review these
mechanisms with short novel events, distinct nearby events and normal lookalikes before deploying changes.

1. **Establish a repeatable comparison.** Freeze the deployed pipeline. Record its revision, model,
   prompts, concurrency and seed/settings; repeat the baseline to measure variation. Group duplicate
   footage and staged scene pairs together, and reserve camera views for validation. Do not select
   changes solely because they fix named clips used to tune them.
2. **Improve evidence for misses.** Test tracked desk/hand/cash-bag crops and denser sampling only around
   exchanges, a brief bag appearance, and hands crossing the desk edge. Target the mechanisms behind
   `Х-1`, `Х-13` and `Х-44`; inspect `Х-24` across repeated runs. Measure recall, false alarms and latency
   together. Extra frames are useful only if their gain fits the five-minute budget.
3. **Reduce specific false alarms.** Test temporal/object corroboration for phone detections and door
   reflections. Treat an ID check outside the camera view as missing evidence, rather than proof of an
   omitted action. Validate on both clean and violating examples before changing thresholds.
4. **Compare stronger verification, not just model size.** Finish the same-machine 4B control and pending
   models. Test a larger VLM on ambiguous candidate events with identical crops and prompts. This needs
   a suitable deployment GPU and measured end-to-end latency; the 24 GB rental result does not make a
   model deployable on the current 8 GB server.
5. **Collect more representative labels before fine-tuning.** Add independently captured clean and
   violating examples from new views, annotate event intervals and visible objects/actions, and keep
   held-out scenes separate. Forty-two repeatedly inspected clips are insufficient to demonstrate
   reliable generalization from fine-tuning.

These are proposed experiments, not measured improvements. Preserve the validated production model
until a candidate improves recall without an unacceptable increase in false alarms or processing time.

## Completed candidate comparison and InternVideo3 — 17 October 2026, 23:50 (+05)

Same pipeline snapshot on the Vast RTX 3090 Ti (protocol: `model-results/CANDIDATE_PROTOCOL.md`):

| Model | Accuracy (42 clips) | Misses | False alarms | Median / max s per clip | Notes |
|---|---|---|---|---|---|
| **Qwen3.5-4B AWQ (production)** | **73.8%** | 4 | 7 | 53 / 117 | full coverage |
| Qwen3.5-27B AWQ | 66.7% | 6 | 8 | 171 / 265 | 77 review windows skipped by the deadline |
| Cosmos Reason2-8B | 61.9% | 8 | 8 | 93 / 211 | |
| InternVL3.5-14B AWQ | — | — | — | 274 (5 clips) | stopped: too slow for the 270 s budget |
| InternVideo3-8B (BF16) | not run in the pipeline | | | | vLLM 0.31 has no support for its architecture (custom MLA attention, Transformers-only remote code) |

**Label-blind open probe** (8 frames around the labelled moment / clean midpoint, generic review prompt, no type
names, JSON schema, temperature 0). It is a diagnostic on known footage, not unseen-type accuracy:

| Model | Flags a violation window | Flags a clean window |
|---|---|---|
| Qwen3.5-4B | 21/22 | 16/20 |
| Cosmos Reason2-8B | 11/22 | 4/20 |
| Qwen3.5-27B | 2/22 | 0/20 |
| InternVideo3-8B, frames as images (protocol) | **0/22** | 0/20 |
| InternVideo3-8B, same frames as a video (native input) | **0/22** | 0/20 |

InternVideo3 ran under Transformers 4.57.3 + SDPA + BF16 with xgrammar-constrained JSON, using the same
JPEG bytes as the vLLM candidates. It returned an empty list on all 84 probes (about 1.1 s each) and also
without the grammar, so the format constraint was not the cause. Asked to *describe* the same Х-11 frames, it
correctly reports the cashier leaving through the glass door and leaving it open (`smoke-result.json`). So it
sees the scene but does not raise concerns under this prompt. It also needs 17.5 GB of GPU memory, with no
quantized checkpoint or vLLM path, and cannot run on the 8 GB production GPU. **Decision: keep Qwen3.5-4B.**
Artifacts: `model-results/internvideo3-open*-20261017.json`; Vast
`/workspace/forte/var/bakeoff/internvideo3-20261017/`.

## Final model comparison — 18 October 2026 (night before submission)

**Production stays Qwen3.5-4B inside the facts + rules pipeline (v11: 76.2 %, 1 miss, 9 false alarms).**

Neutral VLM-only test: each model sees the whole clip as 32 uniform timestamped frames (identical JPEGs for every
model), one fixed prompt written before any result, JSON schema, temperature 0, no pipeline, rules or detector.
Accuracy on the 42 labelled clips (caught / 22 violations, false alarms / 20 clean). Always answering "violation"
scores 52.4 %; always "no violation" 47.6 %.

| Model | plain prompt | + bank catalogue | v2 (authorised-review framing, timeline first) |
|---|---|---|---|
| Qwen3.5-27B AWQ | 40.5 % (9, 12) | 59.5 % (20, 15) | **66.7 % (19, 11)** |
| Qwen3.5-4B AWQ | 42.9 % (8, 10) | 52.4 % (7, 5) | 57.1 % (11, 7) |
| Qwen3.6-27B AWQ | — | 52.4 % (15, 13) | 52.4 % (15, 13) |
| InternVL3.5-8B AWQ | 50.0 % (9, 8) | 47.6 % (0, 0) | 47.6 % (0, 0) |
| InternVL3.5-14B AWQ | 47.6 % (0, 0) | 47.6 % (0, 0) | 50.0 % (1, 0) |
| Cosmos-Reason2-8B | 47.6 % (0, 0) | 47.6 % (0, 0) | 47.6 % (0, 0) |
| InternVideo3-8B (Transformers) | 47.6 % (0, 0) | 47.6 % (0, 0) | not finished |
| MiniCPM-V 4.5 AWQ | most answers cut off (run-on text) | | |

Full-pipeline runs (same snapshot, Vast RTX 3090 Ti): Qwen3.5-4B 73.8 %, Qwen3.5-27B 66.7 %, InternVL3.5-8B 64.3 %,
Cosmos-Reason2-8B 61.9 %, MiniCPM-V 4.5 52.4 %; InternVL3.5-14B too slow for the 270 s budget.

What this shows:
* No VLM on its own judges these clips reliably; the decisions come from the detector, tracker, door check and rules,
  and the VLM answers narrow factual questions. The pipeline was tuned on the 4B's answers, so it favours the 4B.
* Giving a model the bank's rules (catalogue) helps; the 27B then catches 19–20 of 22 but also flags 11–15 of 20
  clean clips. Its mistakes differ from the pipeline's (it misses door cases the pixel door check finds).
* No refusals: free-text answers from the 4B, Cosmos, 27B and InternVL-14B contain no refusal or privacy language
  (0 of 6 each); the models describe the scene and judge it routine.
* Offline estimate (same 42 clips, optimistic): letting the 27B confirm only the noisy finding types would keep 1
  miss and cut false alarms 10 → 6 (83 %). Not deployed: it needs a second 24 GB GPU, and the brief forbids passing
  test videos to third parties, which a rented remote GPU would do.
* Simulated new camera angle (mirrored, rotated 5°, cropped to 85 %): 5 of 5 violation clips still caught; the door
  check found the moved door on 3 of 4 door clips.
* Roadmap: a fast mixture-of-experts model (e.g. Qwen3.6-35B-A3B, 3B active parameters, Apache-2.0) on one local
  48 GB GPU as the single model for both fact reading and whole-clip review.

### Combining two VLMs: what the experimental module does, and the better design

The experimental `app/pipeline/second_opinion.py` (off unless `FORTE_SECOND_VLM_URL` is set) asks the larger model
for one verdict on 32 uniform frames of the whole clip and drops noisy finding types when it says `no_violation`.
That is crude: the veto is clip-level, not about the flagged moment; sparse frames can miss a 3-second event; it can
only remove findings, never add one.

The design to build next is **propose and verify per finding**: the 4B pipeline proposes candidates generously
(recall), and the larger model checks each one on a dense clip around its time (about ±4 s at 2–4 fps), with
detector crops and a type-specific question (precision). For unseen types the larger model reviews key moments from
the tracker with the catalogue and re-checks anything named outside it with a differently worded question (the 4B's
own open-ended signal fires on 13 of 20 clean clips, so it should not vote). Both models fit one local 48 GB GPU.

### v12 — both models live (measured, 18 October, 05:30)

Lite pipeline (Qwen3.5-4B on the 8 GB GPU) plus Qwen3.5-27B AWQ as the second model on a separate 24 GB GPU
(team server, HTTPS with an API key; if it is unreachable the job falls back to the lite result). Full 42-clip
evaluation through the live API: **73.8 % (TP21 FP10 TN10 FN1)**, missed Х-1; timecode median 1.2 s; slowest 181 s;
the 27B answered in 15–33 s per video. Its veto removed false alarms on Х-16 and Х-35; its added type "cash recounted
without verification equipment" found Х-44 (missed by v11) but also fired on clean Х-27, Х-43 and Х-45. Х-1 depends on
a recall-mode finding that does not appear in every run. Catalogue coverage: 17 types by the lite pipeline, the other
15 and unseen types only via the 27B (2 of the 15 observed on these clips; unseen types not measurable here).
