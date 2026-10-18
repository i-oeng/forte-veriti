"""End-to-end analysis of one video: perception -> event proposals -> VLM review -> fusion -> evidence."""
from __future__ import annotations

import base64
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .. import config
from ..catalog import Catalog
from . import door, events, facts, perception, rules, second_opinion, video, vlm

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, float], None]

SWEEP_LEN = config.fenv("FORTE_SWEEP_LEN", 12.0)         # seconds per coverage window
SWEEP_STRIDE = config.fenv("FORTE_SWEEP_STRIDE", 12.0)
SWEEP_FRAMES = int(config.fenv("FORTE_SWEEP_FRAMES", 8))
PROPOSAL_BEFORE = 2.5                                      # proposal window around the event
PROPOSAL_AFTER = 3.5
PROPOSAL_FRAMES = int(config.fenv("FORTE_PROPOSAL_FRAMES", 12))
MAX_PROPOSAL_WINDOWS = int(config.fenv("FORTE_MAX_PROPOSAL_WINDOWS", 4))
PROPOSAL_MERGE = config.fenv("FORTE_PROPOSAL_MERGE", 1.0)  # proposals this deep inside a window share it
MAX_WINDOWS = 60
CLUSTER_GAP = 4.0        # same-type findings closer than this are one event
SNAP_RADIUS = config.fenv("FORTE_SNAP_RADIUS", 2.5)       # max shift when snapping to a perception event (0 = off)
EVIDENCE_RESERVE = 25.0  # seconds kept for evidence extraction before the deadline
SAME_TYPE_GAP = 15.0     # seconds within which a second finding of the same type is treated as the same event
SECOND_SCORE = 0.55      # findings added by the larger model: types the lite pipeline cannot check, so low confidence
VERIFY = config.fenv("FORTE_VERIFY", 1) > 0
CANDIDATE_THRESHOLD = config.fenv("FORTE_CANDIDATE_THRESHOLD", 0.35)
MAX_VERIFY = int(config.fenv("FORTE_MAX_VERIFY", 6))
# types the open-ended VLM review may report; everything else comes from the fact timeline + rules
REVIEW_TYPES = [t for t in config._env("FORTE_REVIEW_TYPES", "CASH_OUT_OF_VIEW,other").split(",") if t]


@dataclass
class Window:
    start: float
    end: float
    source: str                       # proposal | sweep
    priority: float
    hints: list[str] = field(default_factory=list)
    focus: list[tuple] = field(default_factory=list)   # boxes to crop for close-ups
    focus_t: float | None = None


@dataclass
class Event:
    type: str
    t: float
    score: float
    evidence: list[str]
    sources: list[str]
    windows: list[tuple[float, float]]
    snapped_to: str | None = None
    t_vlm: float = 0.0
    other_name: str = ""
    support: int = 1
    verified: str = ""   # yes | no | unclear | "" (not checked)
    ru: str = ""         # Russian observation (rules write it; VLM text is translated before the report)


def _nearest(frames, t: float) -> int:
    return int(np.argmin([abs(f.t - t) for f in frames]))


def _crop(img: np.ndarray, boxes: list[tuple], pad: float = 0.35) -> np.ndarray | None:
    if not boxes:
        return None
    b = np.array(boxes)
    x1, y1, x2, y2 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    w, h = x2 - x1, y2 - y1
    x1, y1 = max(0, int(x1 - pad * w)), max(0, int(y1 - pad * h))
    x2, y2 = min(img.shape[1], int(x2 + pad * w)), min(img.shape[0], int(y2 + pad * h))
    if x2 - x1 < 24 or y2 - y1 < 24:
        return None
    # skip crops that are basically the whole frame
    if (x2 - x1) * (y2 - y1) > 0.6 * img.shape[0] * img.shape[1]:
        return None
    return img[y1:y2, x1:x2]


def build_windows(duration: float, props: list[events.Proposal]) -> list[Window]:
    windows: list[Window] = []
    for p in sorted(props, key=lambda p: -p.strength):
        hint = f"t={p.t:.1f}s: {p.kind.replace('_', ' ')} ({p.note})"
        host = next((w for w in windows if w.start + PROPOSAL_MERGE <= p.t <= w.end - PROPOSAL_MERGE), None)
        if host or len(windows) >= MAX_PROPOSAL_WINDOWS:
            if host:
                host.hints.append(hint)
            continue
        windows.append(Window(max(0.0, p.t - PROPOSAL_BEFORE), min(duration, p.t + PROPOSAL_AFTER), "proposal",
                              1.0 + p.strength, [hint],
                              p.boxes if p.kind.startswith("object_") or p.kind == "interaction" else [], p.t))
    t = 0.0
    while t < duration - 0.5:
        end = min(duration, t + SWEEP_LEN)
        hints = [f"t={p.t:.1f}s: {p.kind.replace('_', ' ')} ({p.note})" for p in props if t <= p.t <= end]
        windows.append(Window(max(0.0, end - SWEEP_LEN), end, "sweep", 0.5, hints[:6]))
        if end >= duration:
            break
        t += SWEEP_STRIDE
    windows.sort(key=lambda w: -w.priority)
    return windows[:MAX_WINDOWS]


def _window_frames(frames, w: Window, n: int | None = None) -> list[vlm.WindowFrame]:
    n = n or (PROPOSAL_FRAMES if w.source == "proposal" else SWEEP_FRAMES)
    times = np.linspace(w.start, w.end, n)
    idx = sorted({_nearest(frames, t) for t in times})
    out = [vlm.WindowFrame(frames[i].t, frames[i].image) for i in idx]
    if w.focus and w.focus_t is not None:
        for k in {_nearest(out, w.focus_t), _nearest(out, w.focus_t + 1.0)}:
            out[k].crop = _crop(out[k].image, w.focus)
    return out


def _snap(ev: Event, props: list[events.Proposal], catalog: Catalog) -> None:
    kinds = catalog.by_id[ev.type].proposal_kinds if ev.type in catalog.by_id else []
    near = [p for p in props if p.kind in kinds and abs(p.t - ev.t_vlm) <= SNAP_RADIUS]
    if near:
        p = min(near, key=lambda p: abs(p.t - ev.t_vlm))
        ev.t, ev.snapped_to = p.t, f"{p.kind} @ {p.t:.2f}s"
    else:
        ev.t, ev.snapped_to = ev.t_vlm, None


def cluster(findings: list[vlm.Finding], props: list[events.Proposal], catalog: Catalog) -> list[Event]:
    clusters: list[list[vlm.Finding]] = []
    for f in sorted(findings, key=lambda f: f.t):
        for c in clusters:
            if c[0].type == f.type and abs(np.mean([x.t for x in c]) - f.t) <= CLUSTER_GAP:
                c.append(f)
                break
        else:
            clusters.append([f])

    evts: list[Event] = []
    for c in clusters:
        best = max(c, key=lambda f: (f.confidence, f.source == "proposal"))
        n_windows = len({x.window for x in c})
        score = min(0.98, best.confidence + 0.08 * (n_windows - 1))
        # proposal windows sample densely, so prefer their timestamps
        dense = [x for x in c if x.source == "proposal"] or c
        weights = np.array([x.confidence for x in dense])
        t = float(np.average([x.t for x in dense], weights=weights))
        ev = Event(best.type, t, score, [best.evidence] + [x.evidence for x in c if x is not best][:2],
                   sorted({x.source for x in c}), sorted({x.window for x in c}), t_vlm=round(t, 2),
                   other_name=next((x.other_name for x in sorted(c, key=lambda x: -x.confidence) if x.other_name), ""),
                   support=n_windows)
        _snap(ev, props, catalog)
        evts.append(ev)
    return sorted(evts, key=lambda e: -e.score)


def _count_series(dets: list[perception.FrameDets], fps: float) -> np.ndarray:
    k = max(3, int(round(fps * 0.8)) | 1)
    return events._median_filter(np.array([len(d.persons) for d in dets], dtype=float), k)


def passes_gate(ev: Event, gate: dict, dets: list[perception.FrameDets], counts: np.ndarray) -> tuple[bool, str]:
    """Perception as a second witness: the detector must agree with what the VLM claims."""
    if not gate:
        return True, ""
    times = np.array([d.t for d in dets])
    t = ev.t_vlm
    near = (times >= t - 3.0) & (times <= t + 3.0)
    if "objects" in gate:
        wanted = set(gate["objects"])
        hits = sum(1 for d, ok in zip(dets, near) if ok and any(o.label in wanted for o in d.objects))
        if hits < 2:
            return False, f"detector saw none of {sorted(wanted)} near {t:.1f}s"
    if "min_people" in gate:
        if not near.any() or counts[near].max() < gate["min_people"]:
            return False, f"fewer than {gate['min_people']} people in view near {t:.1f}s"
    if "room_empty_within" in gate:
        after = (times >= t - 1.0) & (times <= t + float(gate["room_empty_within"]))
        if not after.any() or counts[after].min() > 0:
            return False, f"room never empty within {gate['room_empty_within']}s after {t:.1f}s"
    return True, ""


def _unconfirmed(e: Event) -> bool:
    """Recall mode keeps a review finding that verification rejected or could not settle (never 'other')."""
    return config.RECALL_MODE and e.type != "other" and e.verified in ("no", "unclear")


def select(evts: list[Event], verified_only: bool) -> list[Event]:
    evts = sorted(evts, key=lambda e: -e.score)
    accepted: list[Event] = []
    for e in evts:
        unconfirmed = _unconfirmed(e)
        # screening confidences are not calibrated, so with verification on only confirmed events count;
        # an event that could not be checked in time survives only with strong multi-window support
        if (verified_only and e.verified not in ("yes", "rule") and not (e.verified == "" and e.support >= 3)
                and not unconfirmed):
            continue
        if e.type == "other" and e.support < 4:   # the catch-all is where hallucinations live
            continue
        # the catch-all type is where false positives live, so it needs stronger support
        need = max(config.VIOLATION_THRESHOLD, 0.8) if e.type == "other" else config.VIOLATION_THRESHOLD
        if e.score < (config.RECALL_THRESHOLD if unconfirmed else need):
            continue
        # a weaker review finding next to an accepted one usually describes the same moment; rule findings of
        # different types at one moment are separate violations (door left open + laptop unlocked as she leaves)
        if e.verified != "rule" and any(abs(a.t - e.t) < 3.0 for a in accepted) and e.score < 0.85:
            continue
        if any(a.type == e.type and abs(a.t - e.t) < SAME_TYPE_GAP for a in accepted):
            continue   # one finding per situation: the same type again a few seconds later is the same event
        accepted.append(e)
        if len(accepted) >= config.MAX_VIOLATIONS:
            break
    return sorted(accepted, key=lambda e: e.t)


def _annotate(img: np.ndarray, dets: perception.FrameDets | None, label: str) -> np.ndarray:
    out = img.copy()
    if dets is not None:
        for p in dets.persons:
            x1, y1, x2, y2 = map(int, p.box)
            cv2.rectangle(out, (x1, y1), (x2, y2), (60, 200, 255), 2)
        for o in dets.objects:
            if perception.group_of(o.label):
                x1, y1, x2, y2 = map(int, o.box)
                cv2.rectangle(out, (x1, y1), (x2, y2), (80, 80, 255), 2)
                cv2.putText(out, o.label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 255), 1, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _evidence(src: Path, out_dir: Path, idx: int, ev: Event, frames, dets, duration: float) -> dict:
    """Best snapshot = sharpest native frame near the event, plus one frame before and after."""
    dense = video.dense_frames(src, ev.t - 0.6, ev.t + 0.6, config.MAX_LONG_SIDE)
    if dense:
        sharp = np.array([video.sharpness(img) for _, img in dense])
        sharp = sharp / (sharp.max() or 1.0)
        score = sharp - 0.35 * np.abs(np.array([t for t, _ in dense]) - ev.t)
        best_t, best_img = dense[int(np.argmax(score))]
    else:
        grabbed = video.grab_frames(src, [ev.t], config.MAX_LONG_SIDE)
        best_t, best_img = grabbed[0] if grabbed else (frames[_nearest(frames, ev.t)].t, frames[_nearest(frames, ev.t)].image)
    shots = [(round(best_t, 2), best_img)]
    for t in (best_t - 2.0, best_t + 2.0):
        if 0 <= t <= duration:
            shots += [(round(tt, 2), img) for tt, img in video.grab_frames(src, [t], config.MAX_LONG_SIDE)]

    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    api_frames, ui_frames = [], []
    for k, (t, img) in enumerate(shots):
        raw = video.jpeg(img, 88)
        api_frames.append({"timestamp_sec": t, "image_base64": base64.b64encode(raw).decode()})
        name = f"v{idx}_{k}.jpg"
        near = dets[_nearest(dets, t)] if dets else None
        (frames_dir / name).write_bytes(video.jpeg(_annotate(img, near, f"{t:.2f}s" + ("  best frame" if k == 0 else "")), 88))
        ui_frames.append({"timestamp_sec": t, "url": f"frames/{name}", "best": k == 0})

    clip_name = f"clips/v{idx}.mp4"
    start = max(0.0, best_t - config.CLIP_PAD_SEC)
    end = min(duration, best_t + config.CLIP_PAD_SEC)
    ok = video.cut_clip(src, out_dir / clip_name, start, end)
    return {"timecode": round(best_t, 2), "api_frames": api_frames, "ui_frames": ui_frames,
            "clip": {"url": clip_name, "start_sec": round(start, 2), "end_sec": round(end, 2)} if ok else None}


def _laptop_box(dets) -> tuple[int, int, int, int] | None:
    """Where the laptop is: median of the detector's laptop boxes over the clip."""
    boxes = [o.box for d in dets for o in d.objects if o.label == "laptop" and o.conf >= 0.4]
    if len(boxes) < 3:
        return None
    return tuple(int(v) for v in np.median(np.array(boxes), axis=0))


def _moment_checks(frames, layout, laptop_box, departures, entry_times, cit_times, duration, deadline):
    """Close-ups at decisive moments: right after the cashier left, when someone walks in (is cash out? is the client
    identified before cash changes hands?), when a cash bag appears."""
    entries = [{"t": t} for t in entry_times[:6]]
    id_checks = [{"t": t} for t in entry_times[:4]]
    cits = [{"t": round(t, 2)} for t in cit_times[:3]]
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs_id = []
        for c in id_checks:
            idx = sorted({_nearest(frames, t) for t in np.linspace(c["t"], min(duration, c["t"] + 16.0), 8)})
            futs_id.append((c, pool.submit(facts.id_check, [frames[i] for i in idx], deadline)))
        futs = []
        for d in departures[:4]:
            f = frames[_nearest(frames, min(duration - 0.1, d["t_empty"] + 0.6))]
            futs.append((d, pool.submit(facts.departure_check, f, layout, laptop_box, deadline)))
        for e in entries:
            f = frames[_nearest(frames, min(duration - 0.1, e["t"] + 0.3))]
            futs.append((e, pool.submit(facts.entry_check, f, layout, deadline)))
        for c in cits:
            idx = sorted({_nearest(frames, t) for t in np.linspace(max(0.0, c["t"] - 7.0), min(duration, c["t"] + 2.0), 8)})
            futs.append((c, pool.submit(facts.cit_check, [frames[i] for i in idx], deadline)))
        for item, fut in futs + futs_id:
            item["check"] = fut.result()
    return departures, entries, cits, id_checks


def _sentence(s: str) -> str:
    s = s.strip()
    return s if not s or s[-1] in ".!?" else s + "."


def analyze(src: Path, out_dir: Path, deadline: float, progress: ProgressFn = lambda s, v: None,
            preview: bool = True) -> dict:
    t_start = time.time()
    catalog = Catalog.load()   # re-read per job so rule edits apply without a restart
    out_dir.mkdir(parents=True, exist_ok=True)

    progress("reading video", 0.03)
    info = video.probe(src)
    second = second_opinion.Pending(src, info.duration, catalog)   # larger VLM on the whole clip, in the background
    preview_job = threading.Thread(target=video.make_preview, args=(src, out_dir / "preview.mp4"), daemon=True)
    if preview:
        preview_job.start()

    fps = config.PERCEPTION_FPS
    frames = list(video.iter_frames(src, info, fps, config.MAX_LONG_SIDE))
    if not frames:
        raise ValueError("could not decode any frames")
    duration = max(info.duration, frames[-1].t)

    progress("detecting people and objects", 0.12)
    dets = perception.run(frames, fps)
    props = events.propose(dets, fps)

    counts = _count_series(dets, fps)
    timing = {"perception": round(time.time() - t_start, 1)}

    # 1) fact timeline + rules: state-based violation types with transition-precise timecodes
    progress("locating safe, door and desk", 0.2)
    layout = facts.locate_layout(frames, counts, deadline - EVIDENCE_RESERVE)
    progress("reading scene facts", 0.25)
    sample_idx = facts.sample_times(duration, props, frames)
    timeline = facts.read_facts(frames, sample_idx, layout, deadline - EVIDENCE_RESERVE,
                                lambda v: progress("reading scene facts", 0.25 + 0.35 * v))
    departures, entry_times, arrivals = rules.plan_moments(timeline, dets, counts, props)
    cit_times = rules.bag_arrivals(timeline, dets, counts, entry_times)
    door_series, door_thr = door.open_series(frames, dets, layout.room_door)
    door_events = door.left_open_events(door_series, door_thr)
    progress("checking key moments", 0.55)
    departures, entries, cits, id_checks = _moment_checks(frames, layout, _laptop_box(dets), departures, entry_times,
                                                          cit_times,
                                               duration, deadline - EVIDENCE_RESERVE)
    rule_events = rules.apply(timeline, dets, counts, layout, props, door_events, departures, entries, cits, arrivals,
                              id_checks)
    timing["facts"] = round(time.time() - t_start - sum(timing.values()), 1)
    evts_rules = [Event(r.type, r.t, r.score, [r.observation], ["rules"], [], t_vlm=round(r.t, 2), verified="rule",
                        ru=r.ru) for r in rule_events]

    # 2) open-ended review only for what facts cannot capture (cash handling, concealment, unknown types)
    progress("reviewing hand actions and unknown types", 0.6)
    review_cat = catalog.subset(REVIEW_TYPES)
    # hand actions happen when two people are close: only those proposals get a dense review window
    windows = build_windows(duration, [p for p in props if p.kind == "interaction"]) if review_cat.types else []
    findings: list[vlm.Finding] = []
    scenes: list[dict] = []
    skipped = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {}
        for w in windows:
            futures[pool.submit(vlm.check_window, review_cat, _window_frames(frames, w), duration, w.hints,
                                w.source, deadline - EVIDENCE_RESERVE)] = w
        done = 0
        for fut in as_completed(futures):
            w = futures[fut]
            done += 1
            progress("reviewing hand actions and unknown types", 0.6 + 0.2 * done / len(futures))
            try:
                scene, found = fut.result()
            except (TimeoutError, vlm.BadAnswer):
                skipped += 1
                continue
            scenes.append({"window": [round(w.start, 2), round(w.end, 2)], "source": w.source, "scene": scene,
                           "findings": [asdict(f) for f in found]})
            findings += found

    progress("verifying candidates", 0.8)
    evts = cluster(findings, props, review_cat)
    gated = []
    for e in evts:
        ok, why = passes_gate(e, catalog.by_id[e.type].gate if e.type in catalog.by_id else {}, dets, counts)
        if not ok:
            gated.append({"type": e.type, "t": e.t_vlm, "reason": why})
    rejected = {(g["type"], g["t"]) for g in gated}
    evts = [e for e in evts if (e.type, e.t_vlm) not in rejected]
    verdicts = []
    if VERIFY:
        cands = sorted([e for e in evts if e.score >= CANDIDATE_THRESHOLD],
                       key=lambda e: (-e.support, e.snapped_to is None, -e.score))[:MAX_VERIFY]
        with ThreadPoolExecutor(max_workers=4) as pool:
            futs = {}
            for e in cands:
                w = Window(max(0.0, e.t_vlm - 3.0), min(duration, e.t_vlm + 3.0), "verify", 0)
                vf = _window_frames(frames, w, PROPOSAL_FRAMES)
                near = [p for p in props if abs(p.t - e.t_vlm) <= 3.0 and p.boxes and p.kind != "person_exit"]
                if near:   # close-ups around whatever perception saw near the event
                    for k in {_nearest(vf, e.t_vlm), _nearest(vf, e.t_vlm + 1.0)}:
                        vf[k].crop = _crop(vf[k].image, [b for p in near for b in p.boxes])
                futs[pool.submit(vlm.verify, review_cat, e.type, e.other_name, vf, deadline - EVIDENCE_RESERVE)] = e
            for fut in as_completed(futs):
                e = futs[fut]
                try:
                    v = fut.result()
                except (TimeoutError, vlm.BadAnswer):
                    continue   # could not verify: keep the screening result
                verdicts.append({"type": e.type, "t": e.t_vlm, "score_before": round(e.score, 3), **v})
                e.verified = v["visible"]
                if v["visible"] == "yes":
                    e.score = min(0.98, max(e.score, 0.75) + 0.1)
                    e.t_vlm = round(v["t"], 2)
                    if v["evidence"]:
                        e.evidence.insert(0, v["evidence"])
                    _snap(e, props, catalog)
                elif v["visible"] == "unclear":
                    e.score *= 0.85
                else:
                    e.score *= 0.4
    accepted = select(evts_rules + evts, verified_only=VERIFY)
    timing["review"] = round(time.time() - t_start - sum(timing.values()), 1)

    opinion = second.result(deadline - EVIDENCE_RESERVE - time.time())
    accepted, dropped_by_second = second_opinion.apply(accepted, opinion)
    added_by_second = []
    for x in second_opinion.additions(opinion, catalog, duration):   # types the lite pipeline cannot detect
        if any(a.type == x["type"] and abs(a.t - x["t"]) < SAME_TYPE_GAP for a in accepted + added_by_second):
            continue
        added_by_second.append(Event(x["type"], x["t"], SECOND_SCORE, [x["observation"]], ["second_vlm"], [],
                                     t_vlm=round(x["t"], 2), other_name=x["other_name"], verified="second"))
    accepted = sorted(accepted + added_by_second, key=lambda e: e.t)

    progress("extracting evidence", 0.86)
    russian = config.OUTPUT_LANG == "ru"
    if russian:   # model-written text (review findings, names of new types) -> Russian, one call per video
        todo = [(ev, "ru") for ev in accepted if not ev.ru and ev.evidence and ev.evidence[0]]
        todo += [(ev, "other_name") for ev in accepted if ev.type == "other" and ev.other_name]
        texts = [ev.evidence[0] if field_ == "ru" else ev.other_name for ev, field_ in todo]
        for (ev, field_), text in zip(todo, vlm.to_russian(texts, deadline - 10.0)):
            setattr(ev, field_, text)
    violations_api, violations_ui = [], []
    for i, ev in enumerate(accepted):
        vt = catalog.by_id.get(ev.type) or catalog.by_id["other"]
        if vt.id == "other" and ev.other_name:
            type_name = f"{ev.other_name[:1].upper()}{ev.other_name[1:]} (other)"
        else:
            type_name = vt.label_ru if russian else vt.label
        ev_files = _evidence(src, out_dir, i, ev, frames, dets, duration)
        observation_en = _sentence(ev.evidence[0]) if ev.evidence and ev.evidence[0] else ""
        observation = _sentence(ev.ru) if russian and ev.ru else observation_en
        why = _sentence(vt.why_ru if russian else vt.why)
        rule_text = vt.rule.as_text()
        caveat = ""
        if "second_vlm" in ev.sources:
            caveat = ("Найдено второй, более крупной моделью при просмотре всего ролика; время указано приблизительно."
                      if russian else "Found by the second, larger model reviewing the whole clip; the time is approximate.")
        elif _unconfirmed(ev):
            caveat = ("Повторная автоматическая проверка это не подтвердила; показано, потому что пропуск нарушения "
                      "хуже ложной тревоги." if russian else
                      "A second automatic check did not confirm this; shown because a missed violation is worse "
                      "than a false alarm.")
        if russian:
            parts = (observation, caveat, f"Почему это может быть нарушением: {why}",
                     f"Норма: {rule_text}." if vt.rule.filled else "", "Требуется проверка сотрудником.")
        else:
            parts = (observation, caveat, f"Why it may be a violation: {why}",
                     f"Rule: {rule_text}." if vt.rule.filled else "", "Needs confirmation by a reviewer.")
        description = " ".join(p for p in parts if p)
        violations_api.append({"type": type_name, "timecode_sec": ev_files["timecode"], "description": description,
                               "evidence_frames": ev_files["api_frames"]})
        violations_ui.append({
            "type": type_name, "type_id": vt.id, "tier": vt.tier, "timecode_sec": ev_files["timecode"],
            "confidence": round(ev.score, 2),
            "observation": observation, "observation_en": observation_en, "why": why, "description": description,
            "rule": {**asdict(vt.rule), "status": "filled" if vt.rule.filled else "pending"},
            "frames": ev_files["ui_frames"], "clip": ev_files["clip"], "sources": ev.sources,
            "snapped_to": ev.snapped_to, "t_vlm": ev.t_vlm, "other_evidence": ev.evidence[1:],
        })

    if preview:
        preview_job.join(timeout=max(1.0, deadline - time.time() - 2))
    debug = {
        "video": asdict(info), "perception_fps": fps,
        "person_count": events.person_count_series(dets),
        "proposals": [asdict(p) for p in props],
        "windows": scenes, "skipped_windows": skipped, "verification": verdicts, "gated": gated,
        "candidates": [{"type": e.type, "t": e.t, "t_vlm": e.t_vlm, "score": round(e.score, 3), "support": e.support,
                        "verified": e.verified} for e in evts],
        "layout": asdict(layout), "timing": timing, "fact_samples": len(sample_idx),
        "facts": [{"t": round(f.t, 2), **f.facts} for f in timeline],
        "rule_events": [asdict(r) for r in rule_events],
        "door_series": [(round(t, 2), None if v is None else round(v, 3)) for t, v in door_series],
        "door_events": door_events, "door_threshold": door_thr,
        "departures": departures, "entries": entries, "cits": cits, "id_checks": id_checks,
        "model": config.VLM_MODEL, "detector": config.DETECTOR,
        "second_opinion": opinion, "dropped_by_second_opinion": dropped_by_second,
        "added_by_second_opinion": [{"type": e.type, "t": e.t, "other_name": e.other_name} for e in added_by_second],
    }
    (out_dir / "debug.json").write_text(json.dumps(debug, ensure_ascii=False, default=float), encoding="utf-8")
    return {
        "verdict": "violation" if violations_api else "no_violation",
        "violations": violations_api,
        "extended": {
            "violations": violations_ui,
            "video": {"duration_sec": round(duration, 2), "width": info.width, "height": info.height,
                      "fps": round(info.fps, 2), "codec": info.codec},
            "preview_url": "preview.mp4" if (out_dir / "preview.mp4").exists() else None,
            "processing_sec": round(time.time() - t_start, 1),
            "windows_reviewed": len(scenes), "windows_skipped": skipped, "timing": timing,
        },
    }
