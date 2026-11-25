"""Deterministic, scored timetable optimisation.

* Hard constraints (never violated by the optimiser): fixed activities and locked periods stay,
  teacher unavailability (own class + other classes' approved timetables), max periods per
  subject per day.
* Soft objective (score, higher is better):
    + load_weight(subject) x attention(slot)   -> demanding subjects in attentive periods
    + feedback affinity (understanding by time of day)
    + small bonus where Groq advice preferences are met
    - consecutive high-load periods with no break between
    - same subject twice in a day / back-to-back
    - more than two high-load periods in one day
* Search: randomised greedy construction + simulated-annealing swaps, several restarts with
  fixed seeds (reproducible). Candidates are ranked by score; the validator has the last word.
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import config
from adaptive.learning import Insights
from adaptive.models import FREE_LABEL, Cell, ClassProfile, SubjectConfig
from adaptive.timing import PeriodTime

LOAD_WEIGHT = {"high": 3.0, "medium": 2.0, "low": 1.0}
FREE_WEIGHT = 1.0
W_CONSEC_HIGH = 4.0
W_SAME_DAY = 2.0
W_BACK_TO_BACK = 1.0
W_HEAVY_DAY = 1.5
W_AFFINITY = 1.5
W_ADVICE = 0.3  # tie-breaker only: teacher data and feedback must outweigh AI advice

Key = tuple[str, int]  # (day, period)


@dataclass
class Slot:
    day: str
    period: int
    attention: float
    bucket: str
    block: int  # periods in the same block have no break between them


@dataclass
class Problem:
    profile: ClassProfile
    slots: dict[Key, Slot]
    subjects: dict[str, SubjectConfig]
    fixed: dict[Key, str]
    locked: dict[Key, str]
    blocked: set[tuple[str, str, int]]  # (subject, day, period) not allowed
    demand: list[str]
    movable: list[Key]
    insights: Insights
    advice: dict[str, str]
    conflicts: list[str] = field(default_factory=list)


@dataclass
class GenerationResult:
    cells: list[Cell]
    score: float
    alternatives: list[float]
    conflicts: list[str]
    unplaced: list[str]
    breakdown: dict[str, float]


def _blocks(profile: ClassProfile) -> dict[int, int]:
    after = {b.after_period for b in profile.breaks}
    block, out = 0, {}
    for p in range(1, profile.num_periods + 1):
        out[p] = block
        if p in after:
            block += 1
    return out


def build_problem(profile: ClassProfile, periods: list[PeriodTime], insights: Insights,
                  advice: dict[str, str] | None = None, locked: dict[Key, str] | None = None,
                  external_busy: set[tuple[str, str, int]] | None = None) -> Problem:
    blocks = _blocks(profile)
    slots: dict[Key, Slot] = {}
    for day in profile.working_days:
        for p in periods:
            slots[(day, p.index)] = Slot(day, p.index, insights.period_effective.get(p.index, p.base_attention),
                                         p.bucket, blocks[p.index])
    subjects = {s.name: s for s in profile.subjects if s.periods_per_week > 0}
    fixed = {(f.day, f.period): f.label for f in profile.fixed if (f.day, f.period) in slots}
    locked = {k: v for k, v in (locked or {}).items() if k in slots and k not in fixed}
    conflicts: list[str] = []

    for k, subj in list(locked.items()):
        if subj not in subjects and subj != FREE_LABEL:
            conflicts.append(f"Locked period {k[0]} P{k[1]} holds '{subj}', which is no longer a subject; "
                             "it was unlocked.")
            del locked[k]

    blocked: set[tuple[str, str, int]] = set()
    for u in profile.unavailability:
        for s in subjects.values():
            if s.teacher and s.teacher.strip().lower() == u.teacher.strip().lower():
                for p in u.periods:
                    blocked.add((s.name, u.day, p))
    for teacher, day, p in external_busy or set():
        for s in subjects.values():
            if s.teacher and s.teacher.strip().lower() == teacher.strip().lower():
                blocked.add((s.name, day, p))

    locked_count = Counter(locked.values())
    demand: list[str] = []
    for s in subjects.values():
        remaining = s.periods_per_week - locked_count.get(s.name, 0)
        if remaining < 0:
            conflicts.append(f"{s.name} has {locked_count[s.name]} locked periods but only "
                             f"{s.periods_per_week} per week are required. Unlock some periods or "
                             "raise its weekly count.")
        demand += [s.name] * max(0, remaining)

    movable = [k for k in slots if k not in fixed and k not in locked]
    if len(demand) > len(movable):
        conflicts.append(
            f"The subjects need {len(demand) + sum(locked_count.values())} periods a week but only "
            f"{len(movable) + len(locked)} teaching periods are available "
            f"({len(slots)} total - {len(fixed)} fixed activities). Options: reduce weekly periods "
            "of some subjects, add a period per day, add a working day, or remove a fixed activity.")

    days = profile.working_days
    for s in subjects.values():
        capacity = 0
        for d in days:
            open_slots = [k for k in movable if k[0] == d and (s.name, d, k[1]) not in blocked]
            already = sum(1 for k, v in locked.items() if k[0] == d and v == s.name)
            capacity += min(max(0, s.max_per_day - already), len(open_slots))
        needed = s.periods_per_week - locked_count.get(s.name, 0)
        if needed > capacity:
            why = []
            if s.max_per_day * len(days) < s.periods_per_week:
                why.append(f"at most {s.max_per_day} per day x {len(days)} days = "
                           f"{s.max_per_day * len(days)}")
            if any(b[0] == s.name for b in blocked):
                why.append(f"teacher {s.teacher or '(unnamed)'} is unavailable in some periods")
            conflicts.append(
                f"{s.name} needs {s.periods_per_week} periods/week but only {capacity + locked_count.get(s.name, 0)} "
                f"can be placed ({'; '.join(why) or 'not enough free periods'}). Options: allow "
                f"{s.max_per_day + 1} per day, reduce to {capacity + locked_count.get(s.name, 0)} per week, "
                "or change the teacher's availability.")

    return Problem(profile, slots, subjects, fixed, locked, blocked, demand, movable, insights,
                   dict(advice or {}), conflicts)


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------


def _load_of(label: str, problem: Problem) -> str | None:
    s = problem.subjects.get(label)
    return s.load if s else None


def score_assignment(assign: dict[Key, str], problem: Problem) -> tuple[float, dict[str, float]]:
    parts = defaultdict(float)
    per_day: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for key, label in assign.items():
        slot = problem.slots[key]
        per_day[key[0]].append((key[1], label))
        if key in problem.fixed:
            continue
        load = _load_of(label, problem)
        weight = LOAD_WEIGHT[load] if load else FREE_WEIGHT
        parts["attention_match"] += weight * slot.attention
        if load:
            parts["feedback_affinity"] += W_AFFINITY * problem.insights.affinity_score(label, slot.bucket)
            pref = problem.advice.get(label)
            if pref and pref == slot.bucket:
                parts["ai_advice"] += W_ADVICE

    for day, items in per_day.items():
        items.sort()
        counts = Counter(lbl for _, lbl in items if lbl in problem.subjects)
        parts["same_day_repeat"] -= W_SAME_DAY * sum(c - 1 for c in counts.values() if c > 1)
        highs = sum(1 for _, lbl in items if _load_of(lbl, problem) == "high")
        parts["heavy_day"] -= W_HEAVY_DAY * max(0, highs - 2)
        for (p1, a), (p2, b) in zip(items, items[1:]):
            if problem.slots[(day, p1)].block != problem.slots[(day, p2)].block:
                continue  # a break separates them
            if _load_of(a, problem) == "high" and _load_of(b, problem) == "high":
                parts["consecutive_high"] -= W_CONSEC_HIGH
            if a == b and a in problem.subjects:
                parts["back_to_back"] -= W_BACK_TO_BACK
    return round(sum(parts.values()), 3), dict(parts)


# --------------------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------------------


def _allowed(subject: str, key: Key, day_count: Counter, problem: Problem) -> bool:
    if subject == FREE_LABEL:
        return True
    if (subject, key[0], key[1]) in problem.blocked:
        return False
    return day_count[(subject, key[0])] < problem.subjects[subject].max_per_day


def _greedy(problem: Problem, rng: random.Random) -> tuple[dict[Key, str], list[str]]:
    assign: dict[Key, str] = {**problem.fixed, **problem.locked}
    day_count: Counter = Counter((v, k[0]) for k, v in problem.locked.items())
    free = set(problem.movable)

    def options(subj):
        return [k for k in free if _allowed(subj, k, day_count, problem)]

    items = list(problem.demand)
    rng.shuffle(items)
    items.sort(key=lambda s: (-LOAD_WEIGHT[problem.subjects[s].load], len(options(s))))
    unplaced: list[str] = []
    for subj in items:
        cands = options(subj)
        if not cands:
            unplaced.append(subj)
            continue
        load = problem.subjects[subj].load

        def gain(k):
            slot = problem.slots[k]
            g = LOAD_WEIGHT[load] * slot.attention
            g += W_AFFINITY * problem.insights.affinity_score(subj, slot.bucket)
            if problem.advice.get(subj) == slot.bucket:
                g += W_ADVICE
            for nb in ((k[0], k[1] - 1), (k[0], k[1] + 1)):
                other = assign.get(nb)
                if other and nb in problem.slots and problem.slots[nb].block == slot.block:
                    if load == "high" and _load_of(other, problem) == "high":
                        g -= W_CONSEC_HIGH
                    if other == subj:
                        g -= W_BACK_TO_BACK
            return g + rng.uniform(0, 0.6)

        best = max(cands, key=gain)
        assign[best] = subj
        day_count[(subj, best[0])] += 1
        free.discard(best)
    for k in free:
        assign[k] = FREE_LABEL
    return assign, unplaced


def _valid_swap(assign: dict[Key, str], a: Key, b: Key, problem: Problem) -> bool:
    sa, sb = assign[a], assign[b]
    if sa == sb:
        return False
    for subj, target, source in ((sa, b, a), (sb, a, b)):
        if subj == FREE_LABEL:
            continue
        if (subj, target[0], target[1]) in problem.blocked:
            return False
        if target[0] != source[0]:
            n = sum(1 for k, v in assign.items() if v == subj and k[0] == target[0])
            if n + 1 > problem.subjects[subj].max_per_day:
                return False
    return True


def _anneal(assign: dict[Key, str], problem: Problem, rng: random.Random, iters: int) -> tuple[dict[Key, str], float]:
    current = dict(assign)
    cur_score, _ = score_assignment(current, problem)
    best, best_score = dict(current), cur_score
    movable = problem.movable
    if len(movable) < 2:
        return best, best_score
    for i in range(iters):
        temp = max(0.02, 2.0 * (1 - i / iters))
        a, b = rng.sample(movable, 2)
        if not _valid_swap(current, a, b, problem):
            continue
        current[a], current[b] = current[b], current[a]
        new_score, _ = score_assignment(current, problem)
        delta = new_score - cur_score
        if delta >= 0 or rng.random() < math.exp(delta / temp):
            cur_score = new_score
            if cur_score > best_score:
                best, best_score = dict(current), cur_score
        else:
            current[a], current[b] = current[b], current[a]
    return best, best_score


def to_cells(assign: dict[Key, str], problem: Problem) -> list[Cell]:
    cells = []
    for (day, period), slot in problem.slots.items():
        label = assign.get((day, period), FREE_LABEL)
        if (day, period) in problem.fixed:
            kind, load = "fixed", None
        elif label in problem.subjects:
            kind, load = "subject", problem.subjects[label].load
        else:
            kind, load = "free", None
        cells.append(Cell(day=day, period=period, subject=label, kind=kind, load=load,
                          attention=slot.attention, locked=(day, period) in problem.locked))
    order = {d: i for i, d in enumerate(problem.profile.working_days)}
    return sorted(cells, key=lambda c: (order[c.day], c.period))


def generate(problem: Problem, seed: int = 7, restarts: int | None = None,
             iters: int | None = None) -> GenerationResult:
    restarts = restarts or config.ADAPTIVE_RESTARTS
    iters = iters or max(600, 60 * len(problem.movable))
    candidates: list[tuple[float, dict[Key, str], list[str]]] = []
    for r in range(restarts):
        rng = random.Random(seed * 1000 + r)
        assign, unplaced = _greedy(problem, rng)
        assign, score = _anneal(assign, problem, rng, iters)
        candidates.append((score, assign, unplaced))
    # Prefer complete timetables, then higher score.
    candidates.sort(key=lambda c: (len(c[2]), -c[0]))
    best_score, best, unplaced = candidates[0]
    unique_scores = sorted({round(c[0], 2) for c in candidates}, reverse=True)
    _, breakdown = score_assignment(best, problem)
    conflicts = list(problem.conflicts)
    if unplaced:
        missing = Counter(unplaced)
        conflicts.append("Could not place: " + ", ".join(f"{s} x{n}" for s, n in missing.items())
                         + ". The timetable is incomplete until the conflicts above are resolved.")
    return GenerationResult(to_cells(best, problem), best_score, unique_scores[1:4], conflicts,
                            unplaced, breakdown)


def assignment_from_cells(cells: list[Cell]) -> dict[Key, str]:
    return {(c.day, c.period): c.subject for c in cells}
