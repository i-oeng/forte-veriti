# InternVideo3 bank-video adaptation

Authorized by the user after reviewing the limitations of the earlier model comparison.

Checkpoint: `yanziang/InternVideo3-8B-Instruct`, pinned revision
`c4602918b65225650d152db2850fe34e01d21fcd`. Official source:
https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo3.
The existing VA `video-analyst` implementation uses this model through a custom SGLang
integration, with a small fixed activity taxonomy and Hopper-specific runtime settings.
Those prompts and performance claims do not establish bank-violation accuracy.

## Scope

Adapt for all supplied bank categories plus explicitly named, evidence-backed unlisted
findings. Retain timestamps, Russian presentation, and catalogue legal references where
applicable. Never invent a legal citation for a novel finding. No face identification or audio.
The 42 supplied clips are development data, not an untouched generalization test.

## Gates

1. Compatibility: isolated official Transformers 4.57.3, SDPA, BF16 on the existing 24 GB
   RTX 3090 Ti. Pin weights, log input token count, represented frames, output truncation,
   GPU memory and generation time. A failure is an error, never a clean verdict.
2. Quality: complete model responses on a frozen, label-independent frame/window manifest.
   Compare identical footage and prompts against the 4B control; preserve raw answers.
   Score verdicts, correctly matched types, event timing, false alarms and evidence validity
   separately. Do not use the 270-second pipeline cutoff for this quality measurement.
3. Deployment: separately measure full pipeline coverage and latency under the service
   deadline. Partial frame/review coverage must be explicit and cannot pass silently.
4. Novel categories: include free-form findings and distinct event identities; verified
   short events must not require four windows just because their type is unknown.
   Existing footage can test the implementation, but unseen-type accuracy requires independently
   labelled categories/footage excluded from development. Do not manufacture that claim.

## Isolation

Vast experiment: `/workspace/forte/experiments/internvideo3-20261017`.
Artifacts: `/workspace/forte/var/bakeoff/internvideo3-20261017`.
Pinned runtime packages are installed in the experiment's `runtime/` directory; the existing
application and vLLM environments are not upgraded. No new instance is rented.
The live server is being edited/evaluated by another session (handoff update at 22:15 +05).
Do not overwrite its pipeline or restart it during these experiments.

The first smoke test uses eight frames from clip 11 at 224x384 solely to establish
runtime compatibility. It is not a model accuracy evaluation or a full coverage claim.
