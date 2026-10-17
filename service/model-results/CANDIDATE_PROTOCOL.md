# Candidate comparison, 17 October 2026

The user authorized this comparison on the existing Vast RTX 3090 Ti (24 GB).
Production is unchanged. Code is copied to an isolated experiment directory, with SHA-256
source hashes; candidate weights downloaded for this run are pinned to repository revisions.

## Candidates

1. Qwen3.5-4B AWQ, same-machine baseline.
2. Cosmos Reason2-8B, existing BF16 checkpoint.
3. cyankiwi/Qwen3.5-27B-AWQ-4bit, revision 5cba9b5f1bca69fab9b75c88ae7817ae22d7d898.
4. cyankiwi/InternVL3_5-14B-AWQ-4bit, revision f5ccf9c80ccaa9ca579603708864bb55a4417779.

The proposed InternVL3.5-38B AWQ checkpoint contains 31,923,014,696 bytes of weights,
exceeding this GPU's 24,564 MiB before runtime overhead. It is excluded from the fully
GPU-resident comparison; clearing disk does not remove that limit.

## Shared setup

- Same local pipeline snapshot, prompts, frame sampling and rules for every model.
- 42 supplied labelled clips, four jobs in parallel, 270-second per-clip deadline.
- Eight concurrent VLM requests, 16,384-token context, 24-image input cap, temperature zero.
- vLLM 0.31.0, compiled inference/CUDA graphs, seed zero; 90% GPU memory budget leaves room for the detector.
- Video previews disabled consistently; this is a direct pipeline evaluation, not public API latency.
- Results are separate per model. Errors are recorded and never counted as clean true negatives.
- Original 69% production result used a different GPU/runtime arrangement: compare candidates
  primarily against this run's baseline, not by attributing changes from 69% solely to model quality.

## Unlisted violation types

The supplied PDFs and spreadsheets are not an exhaustive taxonomy. Alongside the pipeline test,
each VLM receives the same short sequences with a generic bank-security review prompt, no label
names or catalogue entries, and free-form event names. Positive clips use the first annotated event
time; clean clips use their midpoint. This oracle window selection makes the probe a diagnostic
of recognition and description, NOT an end-to-end search test or unseen-type accuracy benchmark.
The footage has already been used in development. Claims about new violation categories require
independent labelled footage of categories excluded from examples and tuning.

An additional technical check demonstrates the current filter's rejection of a positively verified
single-window `other` event. It is an implementation limitation shared by candidates, not a VLM score.
Different novel event names can also be merged by the shared `other` ID. Neither limitation is
silently changed during this comparison.

## Setup incident

The first baseline attempt mixed the detector's and vLLM's CUDA library paths, causing cuDNN
loading errors. That attempt was archived as `candidates-setup-cudnn-failure` and excluded.
The restarted runner gives each runtime its own NVIDIA library directories.

The initial working run used eager execution to simplify startup, but per-clip measurements showed
scene-fact reading dominated latency (138.7 of 183 seconds for one clip). The user asked about speed.
The partial eager run is retained separately as `candidates-eager-partial`; the comparison was
restarted with normal compiled inference and CUDA graphs enabled. Do not combine partial eager
results with the optimized candidates as though their runtime settings were identical.

## Artifacts

Vast: `/workspace/forte/var/bakeoff/candidates-20261017/`.
Each model has server/evaluation logs, per-clip results, metrics, label-blind diagnostic responses,
and a saved inference command. Root `status.json` records progress and resource limits.
No new accuracy result is established until the corresponding evaluation finishes and is reviewed.
