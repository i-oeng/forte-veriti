"""Rules over the fact timeline, moment checks and perception -> violation events.

Each rule states its condition in plain terms, so every event can be explained and traced to the frames.
Timecodes are the moment the violating action starts (the convention seen in the labelled clips): the start of the
walk away from the desk, the moment a person walks in, the moment counting by hand begins.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import config
from .facts import FactFrame, Layout
from .perception import FrameDets

# rules that need facts the 4B model reads unreliably; re-enable with a stronger VLM (FORTE_RULES_EXTRA=hand_counting)
EXTRA = set(filter(None, config._env("FORTE_RULES_EXTRA", "").split(",")))

FOOD = {"cup", "bottle", "wine glass", "bowl", "banana", "apple", "sandwich", "orange", "hot dog", "pizza", "donut", "cake"}


@dataclass
class RuleEvent:
    type: str
    t: float
    score: float
    observation: str
    trace: dict = field(default_factory=dict)
    ru: str = ""         # the same observation in Russian


def _fmt(t: float) -> str:
    m = int(t // 60)
    return f"{m:02d}:{t - 60 * m:04.1f}"


class Timeline:
    """Fact samples with majority smoothing, plus perception lookups."""

    def __init__(self, facts: list[FactFrame], dets: list[FrameDets], counts: np.ndarray, layout: Layout):
        self.f = sorted(facts, key=lambda x: x.t)
        self.t = np.array([x.t for x in self.f])
        self.dets = dets
        self.dt = np.array([d.t for d in dets])
        self.counts = counts
        self.layout = layout

    def raw(self, key: str) -> list:
        return [x.facts.get(key) for x in self.f]

    def smooth(self, key: str, half: int = 1) -> list:
        vals = self.raw(key)
        out = []
        for i in range(len(vals)):
            win = vals[max(0, i - half):i + half + 1]
            out.append(max(set(win), key=win.count) if win else None)
        return out

    def runs(self, mask: list[bool], min_len: int = 2, min_sec: float = 0.0) -> list[tuple[int, int]]:
        """(start, end) sample index ranges where mask holds for >= min_len samples and >= min_sec seconds."""
        out, i, n = [], 0, len(mask)
        while i < n:
            if mask[i]:
                j = i
                while j + 1 < n and mask[j + 1]:
                    j += 1
                if j - i + 1 >= min_len and self.t[j] - self.t[i] >= min_sec:
                    out.append((i, j))
                i = j + 1
            else:
                i += 1
        return out

    def _frames_between(self, a: float, b: float):
        return [d for d, t in zip(self.dets, self.dt) if a <= t <= b]

    def count_at(self, a: float, b: float, fn=np.max) -> float:
        m = (self.dt >= a) & (self.dt <= b)
        return float(fn(self.counts[m])) if m.any() else 0.0

    def objects_near(self, labels: set[str], a: float, b: float, min_conf: float = 0.35) -> int:
        return sum(1 for d in self._frames_between(a, b)
                   if any(o.label in labels and o.conf >= min_conf for o in d.objects))


def _point_box_dist(box, x: float, y: float) -> float:
    dx = max(box[0] - x, 0.0, x - box[2])
    dy = max(box[1] - y, 0.0, y - box[3])
    return float(np.hypot(dx, dy))


def phone_with_cashier(tl: Timeline, a: float, b: float, min_conf: float = 0.6) -> bool:
    """Is the detected phone in the hands of the person nearest the safe (the cashier's seat) and not a client's?

    A client's own phone is normal. With one person in view, or without a located safe, the phone counts as the
    cashier's (the old behaviour).
    """
    safe = tl.layout.safe if tl.layout else None
    votes = []
    for d in tl._frames_between(a, b):
        phones = [o for o in d.objects if o.label == "cell phone" and o.conf >= min_conf]
        if not phones or not d.persons:
            continue
        if safe is None or len(d.persons) == 1:
            votes.append(True)
            continue
        sx, sy = (safe[0] + safe[2]) / 2, (safe[1] + safe[3]) / 2
        cashier = min(d.persons, key=lambda p: np.hypot((p.box[0] + p.box[2]) / 2 - sx, (p.box[1] + p.box[3]) / 2 - sy))
        for ph in phones:
            px, py = (ph.box[0] + ph.box[2]) / 2, (ph.box[1] + ph.box[3]) / 2
            owner = min(d.persons, key=lambda p: (_point_box_dist(p.box, px, py),
                                                  np.hypot((p.box[0] + p.box[2]) / 2 - px, (p.box[1] + p.box[3]) / 2 - py)))
            votes.append(owner is cashier)
    return not votes or sum(votes) >= len(votes) / 2


def empty_intervals(dets: list[FrameDets], counts: np.ndarray, min_sec: float = 1.5) -> list[tuple[float, float]]:
    """Intervals where nobody is in view (perception count == 0)."""
    out, start = [], None
    dt = [d.t for d in dets]
    for t, c in zip(dt, counts):
        if c == 0 and start is None:
            start = t
        elif c > 0 and start is not None:
            if t - start >= min_sec:
                out.append((float(start), float(t)))
            start = None
    if start is not None and dt and dt[-1] - start >= min_sec:
        out.append((float(start), float(dt[-1])))
    return out


def plan_moments(facts: list[FactFrame], dets: list[FrameDets], counts: np.ndarray, props):
    """Departures (the room empties after someone was inside) and entries (someone joins people already inside).

    Occupancy comes from the VLM people count, which ignores people seen through the glass walls; the exact moment is
    taken from the detector's count change nearby.
    """
    tl = Timeline(facts, dets, counts, Layout())
    if len(tl.t) < 3:
        return [], [], []
    T = tl.t
    P = [int(v or 0) for v in tl.smooth("people", half=1)]
    dt = tl.dt
    drops = [float(dt[i]) for i in range(1, len(counts)) if counts[i] < counts[i - 1]]
    rises = [float(dt[i]) for i in range(1, len(counts)) if counts[i] > counts[i - 1]]
    moves = sorted(p.t for p in props if p.kind in ("person_moving", "person_exit"))

    departures = []
    empty = [v == 0 for v in P]
    runs = tl.runs(empty, min_len=2, min_sec=1.0)
    # a clip that ends right after the cashier walks out leaves less than a second of empty room: still a departure
    tail = tl.runs(empty, min_len=2)
    if tail and tail[-1][1] == len(P) - 1 and tail[-1] not in runs:
        runs.append(tail[-1])
    for i, j in runs:
        if i == 0 or P[i - 1] < 1:
            continue  # the room was already empty (e.g. the clip starts empty): not a departure
        near = [d for d in drops if T[i - 1] - 1.0 <= d <= T[i] + 0.5]
        t_empty = near[-1] if near else float(T[i])
        # the cashier may stay visible through the glass after leaving, so t_empty can be late: walk back along the
        # chain of movement events (each < 3.5 s apart) that ends at the empty room and take its start
        chain = [m for m in moves if m <= t_empty - 0.5]
        t_dep = None
        while chain and (t_dep is None and t_empty - chain[-1] <= 6.0 or t_dep is not None and t_dep - chain[-1] <= 3.5):
            if t_empty - chain[-1] > 10.0:
                break   # never more than 10 s before the room empties (Х-24 was walked back 11.6 s)
            t_dep = chain.pop()
        t_dep = t_dep if t_dep is not None else max(0.0, t_empty - 2.0)
        departures.append({"t_depart": round(t_dep, 2), "t_empty": round(t_empty, 2)})

    arrivals = []
    for i in range(1, len(P)):
        if P[i - 1] == 0 and P[i] >= 1:
            near = [r for r in rises if T[i - 1] - 1.0 <= r <= T[i] + 0.5]
            arrivals.append(round(near[0] if near else float(T[i]), 2))
    entries: list[float] = []
    for i in range(1, len(P)):
        if P[i] > P[i - 1] >= 1 and (i + 1 >= len(P) or P[i + 1] >= P[i]):
            near = [r for r in rises if T[i - 1] - 1.0 <= r <= T[i] + 0.5]
            t = near[0] if near else float(T[i])
            if not entries or t - entries[-1] >= 4.0:
                entries.append(round(t, 2))
    return departures, entries, arrivals


def bag_arrivals(facts: list[FactFrame], dets: list[FrameDets], counts: np.ndarray, entries: list[float]) -> list[float]:
    """A bag/sack appears on or at the desk within 12 s after a second person walked in: a cash-in-transit handover."""
    tl = Timeline(facts, dets, counts, Layout())
    if not len(tl.t):
        return []
    vals = [bool(a) or bool(b) for a, b in zip(tl.smooth("cash_bag"), tl.smooth("bag_on_desk"))]
    out = []
    for i, j in tl.runs(vals, min_len=2):
        t = float(tl.t[i])
        if any(t - 12.0 <= e <= t + 1.0 for e in entries) and tl.count_at(t - 2, float(tl.t[j]) + 2) >= 2:
            out.append(t)
    return out


_CASH_WORDS = ("banknote", "money", "bills", "cash")   # paper money; bundles, stacks, envelopes may be papers
_DOC_WORDS = ("folder", "document", "paper", "binder", "notebook", "case", "box", "file", "envelope", "stack")
# what always stands on a cash desk is not "left behind" (Х-24: "cash counting machine" was read as banknotes)
_EQUIPMENT_WORDS = ("machine", "counter", "register", "drawer", "printer", "phone", "laptop", "computer", "monitor",
                    "screen", "keyboard", "mouse", "cable", "lamp", "scanner", "terminal", "calculator", "tablet")
_PERSONAL_WORDS = ("handbag", "backpack", "purse", "jacket", "coat", "clothes", "scarf", "wallet")


_WHAT_RU = {"cash": "наличные деньги", "personal": "личные вещи (сумка или одежда)", "docs": "документы или папка",
            "bag": "папка, кейс или сумка", "screen": "включённый компьютер с открытым рабочим окном (экран не заблокирован)"}


def _left_behind(c: dict) -> list[tuple[str, str, str]]:
    """Departure close-up -> [(type id, what, what in Russian)] for the things left on the desk."""
    objects = [str(o).lower() for o in c.get("objects_on_desk", [])]
    found: dict[str, list[tuple[str, str]]] = {}
    if c.get("cash_on_desk"):
        found.setdefault("previous_customer_cash_left_exposed", []).append(("banknotes", "cash"))
    if c.get("bag_bundle_case_or_folder"):
        for o in objects:
            if any(w in o for w in _EQUIPMENT_WORDS):
                continue
            if any(w in o for w in _CASH_WORDS):
                found.setdefault("previous_customer_cash_left_exposed", []).append((o, "cash"))
            elif any(w in o for w in _PERSONAL_WORDS):
                found.setdefault("PERSONAL_ITEM_IN_CASH_NODE", []).append((o, "personal"))
            elif any(w in o for w in _DOC_WORDS):
                found.setdefault("BANK_DOCUMENTS_EXPOSED", []).append((o, "docs"))
        if not found:
            found["BANK_DOCUMENTS_EXPOSED"] = [("a folder, case or bag", "bag")]
    if c.get("screen") == "work_window":
        found.setdefault("BANK_DOCUMENTS_EXPOSED", []).append(("the computer showing an open banking application", "screen"))
    return [(k, ", ".join(dict.fromkeys(w for w, _ in v)), ", ".join(dict.fromkeys(_WHAT_RU[c] for _, c in v)))
            for k, v in found.items()]


def apply(facts: list[FactFrame], dets: list[FrameDets], counts: np.ndarray, layout: Layout, props,
          door_events: list[tuple[float, float]], departures: list[dict], entries: list[dict],
          cits: list[dict], arrivals: list[float] = (), id_checks: list[dict] = ()) -> list[RuleEvent]:
    """Rules -> events typed with the catalogue IDs (catalog.yaml)."""
    tl = Timeline(facts, dets, counts, layout)
    T = tl.t
    out: list[RuleEvent] = []
    enter_times = [e["t"] for e in entries] + [p.t for p in props if p.kind == "person_enter"]

    # --- what the cashier leaves behind (close-ups right after the room empties)
    for d in departures:
        c = d.get("check") or {}
        t0 = d["t_depart"]
        for typ, what, what_ru in _left_behind(c):
            out.append(RuleEvent(typ, t0, 0.8, f"At {_fmt(t0)} the cashier leaves the workplace while {what} "
                                 f"remain(s) on the desk.", d,
                                 f"В {_fmt(t0)} кассир покидает рабочее место; на столе остаются: {what_ru}."))
        if c.get("safe_door") == "open":
            out.append(RuleEvent("SAFE_LEFT_OPEN", t0, 0.8, f"At {_fmt(t0)} the cashier leaves the workplace while the "
                                 f"safe door is open.", d,
                                 f"В {_fmt(t0)} кассир покидает рабочее место, дверь сейфа остаётся открытой."))
        elif c.get("key_in_safe_lock"):
            out.append(RuleEvent("KEY_LEFT_IN_SAFE", t0, 0.7, f"At {_fmt(t0)} the cashier leaves the workplace and a key "
                                 f"is left in the safe lock.", d,
                                 f"В {_fmt(t0)} кассир покидает рабочее место, в замке сейфа остаётся ключ."))

    if len(T):
        P = [int(v or 0) for v in tl.smooth("people")]

        # --- safe open while nobody is at it (cashier back at the desk)
        safe_open = tl.smooth("safe_open", half=2)
        at_safe = tl.smooth("someone_at_safe", half=1)
        for i, j in tl.runs([bool(o) and not a for o, a in zip(safe_open, at_safe)], min_len=2, min_sec=1.0):
            before = [k for k in range(i) if at_safe[k] and T[i] - T[k] <= 6.0]
            if not before:
                continue   # nobody was just working at the safe: a stray "open" reading, not someone walking away
            t0 = float(T[before[-1]] + 0.25)
            if tl.count_at(T[i], T[j], np.max) >= 1:   # someone is still in the room (else: a departure, see above)
                out.append(RuleEvent("SAFE_LEFT_OPEN", t0, min(0.95, 0.65 + 0.04 * (j - i)),
                                     f"At {_fmt(t0)} the cashier moves away from the safe and its door stays open "
                                     f"(still open at {_fmt(T[j])}).", {"open": [float(T[i]), float(T[j])]},
                                     f"В {_fmt(t0)} кассир отходит от сейфа, дверь сейфа остаётся открытой "
                                     f"(открыта и в {_fmt(T[j])})."))

        # --- a third person at the desk / someone else on the cashier's side
        for i, j in tl.runs([p >= 3 and tl.count_at(t - 1.5, t + 1.5) >= 3 for p, t in zip(P, T)], min_len=2):
            ent = [e for e in enter_times if T[i] - 6.0 <= e <= T[i]]
            t0 = ent[-1] if ent else float(T[i])
            out.append(RuleEvent("collection_exposed_to_third_parties", t0, min(0.9, 0.6 + 0.05 * (j - i)),
                                 f"From {_fmt(t0)} a third person is at the cash desk besides the cashier and the "
                                 f"client and can watch the cash operation.", {},
                                 f"С {_fmt(t0)} у кассового стола кроме кассира и клиента находится третий человек, "
                                 f"который может наблюдать за кассовой операцией."))
        side = tl.smooth("extra_on_staff_side")
        for i, j in tl.runs([bool(s) for s in side], min_len=2):
            out.append(RuleEvent("possible_unauthorized_vault_access", float(T[i]), min(0.9, 0.6 + 0.05 * (j - i)),
                                 f"At {_fmt(T[i])} a person other than the cashier is on the cashier's side of the desk "
                                 f"or at the safe.", {},
                                 f"В {_fmt(T[i])} другой человек (не кассир) находится на стороне кассира или у сейфа."))

        # --- cashier gone while a client is still there
        holding = tl.smooth("holding")
        for i, j in tl.runs([h == "no_cashier" and p >= 1 and tl.count_at(t - 1, t + 1, np.median) >= 1
                             for h, p, t in zip(holding, P, T)], min_len=2, min_sec=1.5):
            out.append(RuleEvent("cashier_left_during_active_customer_service", float(T[i]), 0.7,
                                 f"From {_fmt(T[i])} the cashier is not at the workplace while a client is still in "
                                 f"the room.", {},
                                 f"С {_fmt(T[i])} кассира нет на рабочем месте, а клиент остаётся в помещении."))

        # --- the client leaves while the cashier is still counting / loading the client's cash
        hand = tl.raw("counting_by_hand")
        machine = tl.raw("machine_in_use")
        hold = tl.raw("holding")
        dt = tl.dt
        drops = [float(dt[k]) for k in range(1, len(counts)) if counts[k] < counts[k - 1]]
        for i in range(1, len(P)):
            if P[i - 1] >= 2 and P[i] == P[i - 1] - 1 and (i + 1 >= len(P) or P[i + 1] <= P[i]):
                near = [d for d in drops if T[i - 1] - 1.0 <= d <= T[i] + 0.5]
                t_exit = near[-1] if near else float(T[i])
                busy = [k for k in range(len(T)) if t_exit - 1.0 <= T[k] <= t_exit + 4.0
                        and (hand[k] or machine[k] or hold[k] == "cash")]
                if len(busy) >= 2:
                    out.append(RuleEvent("cash_counted_without_customer_present", t_exit, 0.7,
                                         f"At {_fmt(t_exit)} the client leaves while the cashier is still counting or "
                                         f"loading the client's cash into the machine.", {},
                                         f"В {_fmt(t_exit)} клиент уходит, а кассир ещё пересчитывает его деньги "
                                         f"или закладывает их в счётную машину."))

        # --- cash processed alone after a collection bag was seen (no controlling employee)
        bag_seen = [T[k] for k, v in enumerate(tl.smooth("cash_bag")) if v]
        if bag_seen:
            mask = [T[k] > bag_seen[0] and P[k] == 1 and tl.count_at(T[k] - 1, T[k] + 1, np.max) <= 1
                    and bool(hand[k] or machine[k]) for k in range(len(T))]
            for i, j in tl.runs(mask, min_len=2, min_sec=1.0)[:1]:
                out.append(RuleEvent("CASH_PROCESSING_WITHOUT_CONTROLLER", float(T[i]), 0.7,
                                     f"From {_fmt(T[i])} the cashier alone counts cash after a collection bag was "
                                     f"handled; no controlling employee is present.", {},
                                     f"С {_fmt(T[i])} кассир один пересчитывает наличные после работы с инкассаторской "
                                     f"сумкой; контролирующего работника рядом нет."))

        # --- client cash counted by hand instead of on the machine (off by default: needs a stronger VLM)
        mask = [bool(h) and not m and tl.count_at(t - 1.0, t + 1.0, np.median) >= 2 for h, m, t in zip(hand, machine, T)]
        last = -1e9
        for i, j in tl.runs(mask if "hand_counting" in EXTRA else [], min_len=2, min_sec=1.0):
            near_machine = any(machine[k] for k in range(len(T)) if T[i] - 4.0 <= T[k] <= T[j] + 2.0)
            if near_machine or T[i] - last < 10.0:
                continue
            last = float(T[i])
            out.append(RuleEvent("CASH_RECOUNT_WITHOUT_VERIFICATION_EQUIPMENT", float(T[i]),
                                 min(0.85, 0.6 + 0.05 * (j - i)),
                                 f"From {_fmt(T[i])} to {_fmt(T[j])} the cashier counts banknotes by hand; the counting "
                                 f"machine on the desk is not used.", {},
                                 f"С {_fmt(T[i])} до {_fmt(T[j])} кассир пересчитывает банкноты вручную, счётная "
                                 f"машина на столе не используется."))

        # --- phone (the detector must see the phone too, in the cashier's hands rather than a client's)
        phone = tl.raw("phone")
        for i, j in tl.runs([bool(v) for v in phone], min_len=3):
            if (tl.objects_near({"cell phone"}, T[i] - 3.0, T[j] + 3.0, min_conf=0.6) >= 6
                    and phone_with_cashier(tl, T[i] - 3.0, T[j] + 3.0)):
                out.append(RuleEvent("PHONE_WITH_CAMERA_IN_CASH_NODE", float(T[i]), min(0.9, 0.65 + 0.05 * (j - i)),
                                     f"At {_fmt(T[i])} the cashier is using a smartphone at the cash desk.", {},
                                     f"В {_fmt(T[i])} кассир пользуется смартфоном за кассовым столом."))

        # --- personal items: food eaten at the desk, a bag / outerwear put on the desk (drinking alone is fine)
        eating = tl.raw("eating")
        for i, j in tl.runs([bool(v) for v in eating], min_len=1):
            if tl.objects_near(FOOD, T[i] - 3.0, T[j] + 3.0) >= 2:
                out.append(RuleEvent("PERSONAL_ITEM_IN_CASH_NODE", float(T[i]), min(0.9, 0.65 + 0.05 * (j - i)),
                                     f"At {_fmt(T[i])} the cashier is eating at the cash desk (food at the cash "
                                     f"workplace).", {},
                                     f"В {_fmt(T[i])} кассир ест за кассовым столом (еда на рабочем месте)."))
        vals = [bool(v) for v in tl.smooth("bag_on_desk")]
        cash_bag = [bool(v) for v in tl.smooth("cash_bag")]
        for i, j in tl.runs(vals, min_len=2):
            visitor_bag = any(cash_bag[i:j + 1]) or any(abs(c["t"] - T[i]) <= 3.0 for c in cits)
            if i >= 2 and not any(vals[max(0, i - 3):i]) and not visitor_bag:
                out.append(RuleEvent("PERSONAL_ITEM_IN_CASH_NODE", float(T[i]), min(0.9, 0.6 + 0.05 * (j - i)),
                                     f"At {_fmt(T[i])} a bag or outerwear is put on the cash desk.", {},
                                     f"В {_fmt(T[i])} на кассовый стол кладут сумку или верхнюю одежду."))

    # --- the next client walks in while the previous client's cash is still out
    for e in entries:
        c = e.get("check") or {}
        if c.get("cash_on_desk") or c.get("cashier_handling_cash"):
            what = "banknotes lie on the desk" if c.get("cash_on_desk") else "the cashier is still handling cash"
            what_ru = "на столе лежат банкноты" if c.get("cash_on_desk") else "кассир ещё держит наличные"
            out.append(RuleEvent("previous_customer_cash_left_exposed", e["t"], 0.75,
                                 f"At {_fmt(e['t'])} a person walks into the cash room while {what}.", e,
                                 f"В {_fmt(e['t'])} в кассу входит человек, а {what_ru}."))

    # --- no identity document seen before cash changed hands
    for c in id_checks:
        chk = c.get("check") or {}
        if chk.get("cash_exchanged") == "yes" and chk.get("id_shown") == "no":
            t0 = chk.get("t_handover", c["t"])
            out.append(RuleEvent("customer_identification_not_observed", t0, 0.6,
                                 f"At {_fmt(t0)} cash changes hands between the client and the cashier; no identity "
                                 f"document is seen being shown before that.", c,
                                 f"В {_fmt(t0)} клиент и кассир передают друг другу наличные; до этого не видно, "
                                 f"чтобы клиент показывал документ, удостоверяющий личность."))

    # --- door left open while people work inside (door state from pixels)
    passes = sorted(set(enter_times) | {d["t_empty"] for d in departures} | set(arrivals))
    for a, b in door_events:
        if b - a < 4.0 or tl.count_at(a, b, np.median) < 1:
            continue
        near = [x for x in passes if a - 3.0 <= x <= a + 3.0]
        if not near:
            continue   # the door region changed without anyone walking through: lighting / people behind the glass
        # timecode: the cashier walking out through the open door; otherwise the door has stood open for 2 s
        # (on the labelled door clips: median error 1.4 s, vs 3.8 s for the first nearby pass)
        walk_out = [d["t_depart"] for d in departures if a - 1.0 <= d["t_depart"] <= b]
        t0 = walk_out[0] if walk_out else min(b, a + 2.0)
        if walk_out:
            en = f"At {_fmt(t0)} the cashier walks out and the cash room door stays open (open until {_fmt(b)})."
            ru = f"В {_fmt(t0)} кассир выходит, дверь кассы остаётся открытой (открыта до {_fmt(b)})."
        else:
            en = f"From {_fmt(a)} to {_fmt(b)} the cash room door stands open while people are inside."
            ru = f"С {_fmt(a)} до {_fmt(b)} дверь кассы открыта, а внутри находятся люди."
        out.append(RuleEvent("cash_desk_access_door_left_unsecured", t0, min(0.9, 0.65 + 0.03 * (b - a)), en,
                             {"open": [a, b]}, ru))

    # --- collection bag taken without inspection
    for c in cits:
        chk = c.get("check") or {}
        if chk.get("documents_checked") == "no":
            out.append(RuleEvent("COLLECTION_BAG_ACCEPTED_WITHOUT_INTEGRITY_CHECK", c["t"], 0.8,
                                 f"At {_fmt(c['t'])} a collection bag is handed over at the desk and accepted without a "
                                 f"visible check of the bag's integrity or the collector's documents.", c,
                                 f"В {_fmt(c['t'])} у стола передают инкассаторскую сумку, и её принимают без видимой "
                                 f"проверки целостности сумки, пломбы и документов инкассатора."))

    return out
