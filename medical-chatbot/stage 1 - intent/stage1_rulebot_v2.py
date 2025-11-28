#!/usr/bin/env python3
"""
Stage 1.5: Rule‑Based Chatbot with Knowledge Packs (CLI)

What changed (vs. Stage 1):
- Pluggable knowledge packs (JSON) for symptoms/synonyms, follow‑ups, red flags, guidance.
- Negation + uncertainty handling.
- Lightweight extraction for duration, severity, frequency, onset.
- Micro‑tree follow‑ups per cluster (drives smarter questions, no repetition).
- Same public interfaces for Stage 2 (BioBERT extractor) & Stage 3 (DiseaseClassifier), so you can
  drop them in later without rewriting the loop.

Run
---
python stage1_rulebot.py

Optional: place JSON files in the same folder (all optional; built‑in defaults are used if missing):
- symptoms.json        (ontology + synonyms + systems)
- negation.json        (negation/uncertainty cues + window)
- followups.json       (per‑cluster micro trees)
- triage_rules.json    (red flags)
- guidance.json        (pattern → generic guidance text)

Commands in chat: 'done', 'help', 'list', 'remove <symptom>', 'restart'.

"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
import re
import sys
import json
from collections import defaultdict, deque
from pathlib import Path

# ----------------------------
# Utilities
# ----------------------------

ROOT = Path(__file__).parent

def load_json_maybe(name: str, fallback: Any) -> Any:
    path = ROOT / name
    if path.exists():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return fallback


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ----------------------------
# Default Knowledge Packs (inline fallbacks)
# ----------------------------

FALLBACK_SYMPTOMS = {
    "fever": {"systems": ["general"], "synonyms": ["feverish", "high temperature", "feeling hot"]},
    "cough": {"systems": ["respiratory"], "synonyms": ["coughing"]},
    "sore throat": {"systems": ["respiratory"], "synonyms": ["throat pain", "scratchy throat"]},
    "runny nose": {"systems": ["respiratory"], "synonyms": ["stuffy nose", "nasal congestion"]},
    "headache": {"systems": ["neuro"], "synonyms": ["head pain", "head ache"]},
    "fatigue": {"systems": ["general"], "synonyms": ["tired", "exhausted", "wiped"]},
    "chills": {"systems": ["general"], "synonyms": []},
    "nausea": {"systems": ["gi"], "synonyms": ["nauseous"]},
    "vomiting": {"systems": ["gi"], "synonyms": ["vomit", "throwing up"]},
    "diarrhea": {"systems": ["gi"], "synonyms": ["diarrhoea", "loose stools"]},
    "abdominal pain": {"systems": ["gi"], "synonyms": ["stomach pain", "tummy ache", "cramps"]},
    "shortness of breath": {"systems": ["respiratory"], "synonyms": ["breathless", "wheezing"]},
    "chest pain": {"systems": ["cardio"], "synonyms": ["chest tightness"]},
    "dizziness": {"systems": ["neuro"], "synonyms": ["lightheaded"]},
    "rash": {"systems": ["derm"], "synonyms": []},
    "itching": {"systems": ["derm"], "synonyms": ["itchy"]},
    "body aches": {"systems": ["general"], "synonyms": ["aches", "soreness", "body ache"]},
    "sneezing": {"systems": ["respiratory"], "synonyms": ["sneeze"]},
    "loss of taste": {"systems": ["neuro"], "synonyms": ["ageusia"]},
    "loss of smell": {"systems": ["neuro"], "synonyms": ["anosmia"]},
}

FALLBACK_NEGATION = {
    "negation_cues": ["no", "not", "never", "without", "denies"],
    "uncertainty_cues": ["maybe", "might", "unsure", "not sure", "kind of", "kinda", "probably"],
    "window": 3  # tokens before/after
}

FALLBACK_FOLLOWUPS = {
    "respiratory": [
        ["fever"], ["cough"], ["shortness of breath"], ["chest pain"], ["sore throat"], ["runny nose"], ["loss of smell"], ["loss of taste"]
    ],
    "gi": [["nausea"], ["vomiting"], ["diarrhea"], ["abdominal pain"]],
    "general": [["fatigue"], ["body aches"], ["chills"]],
    "neuro": [["headache"], ["dizziness"]],
    "derm": [["rash"], ["itching"]],
}

FALLBACK_TRIAGE = {
    "red_flags": [
        "severe chest pain", "trouble breathing", "shortness of breath", "fainted", "passed out",
        "blood in vomit", "blood in stool", "stiff neck with fever", "pregnant bleeding", "confusion"
    ],
    "message": "If you're experiencing severe or concerning symptoms (e.g., severe chest pain, trouble breathing, fainting), seek immediate medical care or call local emergency services."
}

FALLBACK_GUIDANCE = {
    "patterns": {
        "fever + cough + sore throat": [
            "General care: rest and hydrate; consider fever reducers per label.",
            "Watch for persistent high fever (>3 days) or breathing trouble and seek care if they occur."
        ],
        "nausea + vomiting + diarrhea": [
            "Oral rehydration and small frequent sips; seek care if signs of dehydration.",
            "Ask about abdominal pain severity, blood in stool, or recent risky food exposure."
        ]
    }
}

# ----------------------------
# Abstractions for Stage 2 & 3
# ----------------------------

class SymptomExtractor:
    def extract(self, text: str) -> Set[str]:
        raise NotImplementedError

@dataclass
class ExtractedSymptom:
    name: str
    present: bool = True
    certainty: float = 1.0
    severity: Optional[int] = None   # 0..3
    severity_raw: Optional[str] = None
    duration_hours: Optional[int] = None
    onset: Optional[str] = None       # sudden/gradual
    frequency: Optional[str] = None   # intermittent/constant/night


class RuleBasedExtractor(SymptomExtractor):
    """Loads ontology + synonyms; handles negation/uncertainty; extracts basic attributes.
       Pure stdlib, regex‑driven; replace with BioBERT in Stage 2.
    """
    def __init__(self,
                 symptoms_pack: Optional[Dict[str, Any]] = None,
                 negation_pack: Optional[Dict[str, Any]] = None):
        self.symptoms_pack = symptoms_pack or load_json_maybe('symptoms.json', FALLBACK_SYMPTOMS)
        self.neg_pack = negation_pack or load_json_maybe('negation.json', FALLBACK_NEGATION)
        self.vocab = set(self.symptoms_pack.keys())
        # Build synonym → canonical map
        self.syn2canon: Dict[str, str] = {}
        for canon, meta in self.symptoms_pack.items():
            self.syn2canon[canon.lower()] = canon
            for s in meta.get('synonyms', []):
                self.syn2canon[s.lower()] = canon

        # Precompile symptom regexes (canonical + synonyms)
        self.term_regexes: List[Tuple[re.Pattern, str]] = []
        for syn, canon in self.syn2canon.items():
            pat = re.compile(rf"(?<![\w-]){re.escape(syn)}(?![\w-])", re.I)
            self.term_regexes.append((pat, canon))

        # Negation/uncertainty
        self.neg_cues = set(self.neg_pack.get('negation_cues', []))
        self.uncertain_cues = set(self.neg_pack.get('uncertainty_cues', []))
        self.window = int(self.neg_pack.get('window', 3))

        # Attribute regexes
        self.re_duration = re.compile(r"(for|since)\s+(\d+)\s*(hour|hours|hr|hrs|day|days|week|weeks)", re.I)
        self.re_severity_word = re.compile(r"\b(mild|moderate|severe)\b", re.I)
        self.re_severity_num = re.compile(r"\b(\d|10)\s*/\s*10|\b(\d{1,2})\b(?=\/10)", re.I)
        self.re_onset = re.compile(r"\b(sudden|gradual)\b", re.I)
        self.re_freq = re.compile(r"\b(intermittent|constant|at\s+night)\b", re.I)

    # --- token helpers ---
    def _simple_tokenize(self, text: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9']+", text.lower())

    def _window_contains(self, tokens: List[str], idx: int, cues: Set[str]) -> bool:
        lo = max(0, idx - self.window)
        hi = min(len(tokens), idx + self.window + 1)
        span = tokens[lo:hi]
        return any(c in span for c in cues)

    # --- attribute parsing ---
    def _parse_duration_hours(self, text: str) -> Optional[int]:
        m = self.re_duration.search(text)
        if not m:
            return None
        qty = int(m.group(2))
        unit = m.group(3).lower()
        mult = 1
        if unit.startswith('hour') or unit.startswith('hr'):
            mult = 1
        elif unit.startswith('day'):
            mult = 24
        elif unit.startswith('week'):
            mult = 24 * 7
        return qty * mult

    def _parse_severity(self, text: str) -> Tuple[Optional[int], Optional[str]]:
        m = self.re_severity_word.search(text)
        if m:
            level = m.group(1).lower()
            return {"mild": 1, "moderate": 2, "severe": 3}[level], level
        n = None
        # Capture patterns like 7/10 or "pain 6 out of 10"
        m2 = re.search(r"(\d{1,2})\s*(?:/\s*10|out\s*of\s*10)", text)
        if m2:
            try:
                n = int(m2.group(1))
            except Exception:
                n = None
        if n is not None:
            sev = 0 if n <= 0 else 1 if n <= 3 else 2 if n <= 6 else 3
            return sev, f"{n}/10"
        return None, None

    def _parse_onset(self, text: str) -> Optional[str]:
        m = self.re_onset.search(text)
        return m.group(1).lower() if m else None

    def _parse_frequency(self, text: str) -> Optional[str]:
        m = self.re_freq.search(text)
        return m.group(1).lower() if m else None

    # --- main extraction ---
    def extract_rich(self, text: str) -> List[ExtractedSymptom]:
        out: Dict[str, ExtractedSymptom] = {}
        tokens = self._simple_tokenize(text)

        # Find symptoms by regex
        for pat, canon in self.term_regexes:
            for m in pat.finditer(text):
                # estimate token index by counting tokens before match start
                prefix_tokens = self._simple_tokenize(text[:m.start()])
                idx = len(prefix_tokens)
                present = True
                certainty = 1.0
                if self._window_contains(tokens, idx, self.neg_cues):
                    present = False
                if self._window_contains(tokens, idx, self.uncertain_cues):
                    certainty = 0.6
                es = out.get(canon) or ExtractedSymptom(name=canon)
                es.present = present
                es.certainty = min(es.certainty, certainty) if canon in out else certainty
                # parse attributes from full text (cheap but effective)
                es.duration_hours = es.duration_hours or self._parse_duration_hours(text)
                sev, sev_raw = self._parse_severity(text)
                es.severity = es.severity if es.severity is not None else sev
                es.severity_raw = es.severity_raw or sev_raw
                es.onset = es.onset or self._parse_onset(text)
                es.frequency = es.frequency or self._parse_frequency(text)
                out[canon] = es
        return list(out.values())

    def extract(self, text: str) -> Set[str]:  # legacy Stage‑1 interface
        return {e.name for e in self.extract_rich(text) if e.present}


@dataclass
class DiseasePrediction:
    label: str
    probability: float

class DiseaseClassifier:
    def predict(self, canonical_symptoms: List[str], top_k: int = 3) -> List[DiseasePrediction]:
        return []  # Stage 1: stub

# ----------------------------
# State & Dialogue
# ----------------------------

@dataclass
class ConversationState:
    collected: Dict[str, ExtractedSymptom] = field(default_factory=dict)
    turn: int = 0
    phase: str = "collect"  # collect -> clarify -> diagnose -> explain
    k_threshold: int = 3
    asked_queue: deque = field(default_factory=deque)  # record of asked follow-ups

    def add_rich(self, rich: List[ExtractedSymptom]):
        for e in rich:
            if e.name not in self.collected:
                self.collected[e.name] = e
            else:
                # merge conservative: presence false overrides true only if explicit
                cur = self.collected[e.name]
                cur.present = cur.present and e.present
                cur.certainty = min(cur.certainty, e.certainty)
                cur.severity = cur.severity if cur.severity is not None else e.severity
                cur.severity_raw = cur.severity_raw or e.severity_raw
                cur.duration_hours = cur.duration_hours or e.duration_hours
                cur.onset = cur.onset or e.onset
                cur.frequency = cur.frequency or e.frequency

    def positives(self) -> List[str]:
        return sorted([k for k,v in self.collected.items() if v.present])

    def reset(self):
        self.collected.clear()
        self.turn = 0
        self.phase = "collect"
        self.asked_queue.clear()


class RuleBasedChatbot:
    def __init__(self,
                 extractor: Optional[RuleBasedExtractor] = None,
                 classifier: Optional[DiseaseClassifier] = None,
                 k_threshold: int = 3):
        self.extractor = extractor or RuleBasedExtractor()
        self.classifier = classifier or DiseaseClassifier()
        self.state = ConversationState(k_threshold=k_threshold)
        # Load follow-ups & triage/guidance
        self.followups_pack = load_json_maybe('followups.json', FALLBACK_FOLLOWUPS)
        self.triage_pack = load_json_maybe('triage_rules.json', FALLBACK_TRIAGE)
        self.guidance_pack = load_json_maybe('guidance.json', FALLBACK_GUIDANCE)

    # ---------- Safety ----------
    def check_red_flags(self, text: str) -> Optional[str]:
        flags = self.triage_pack.get('red_flags', [])
        for f in flags:
            if re.search(re.escape(f), text, re.I):
                return self.triage_pack.get('message')
        return None

    # ---------- Helpers ----------
    def format_symptom_list(self, items: List[str]) -> str:
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        return ", ".join(items[:-1]) + f" and {items[-1]}"

    def welcome(self) -> str:
        return (
            "Hello! I'm a symptom checker. Tell me what you're feeling. "
            "This is not medical advice. If it's an emergency, seek urgent care."
        )

    def help_text(self) -> str:
        return (
            "Commands: 'done' to finish, 'restart' to reset, 'list' to see collected symptoms, "
            "'remove <symptom>' to delete one, 'help' to see this again."
        )

    # ---------- Follow‑up selection (micro‑tree) ----------
    def _systems_for(self, symptom: str) -> List[str]:
        meta = self.extractor.symptoms_pack.get(symptom, {})
        return meta.get('systems', ['general'])

    def _candidate_followups(self) -> List[str]:
        """Return ordered list of candidate follow‑ups not yet confirmed/asked."""
        positives = set(self.state.positives())
        asked = set(self.state.asked_queue)
        # seed systems from what we have; if none, use general/respiratory first
        systems = set()
        for s in positives:
            systems.update(self._systems_for(s))
        if not systems:
            systems = {"general", "respiratory"}
        # build sequence from micro‑trees
        seq: List[str] = []
        for sysname in systems:
            tree = self.followups_pack.get(sysname, [])
            for node in tree:
                for sym in node:
                    if sym not in positives and sym not in asked:
                        seq.append(sym)
        return seq

    def next_followup_question(self) -> Optional[str]:
        candidates = self._candidate_followups()
        if not candidates:
            return None
        # ask up to two options to keep turn economy high
        opts = candidates[:2]
        self.state.asked_queue.extend(opts)
        if len(opts) == 1:
            return f"Do you also have {opts[0]}?"
        return f"Do you also have {opts[0]} or {opts[1]}?"

    # ---------- Turn processing ----------
    def process_user_input(self, text: str) -> str:
        self.state.turn += 1
        text = text.strip()
        if not text:
            return "Could you describe your symptoms in a bit more detail?"

        # commands
        tl = text.lower()
        if tl == "help":
            return self.help_text()
        if tl == "restart":
            self.state.reset()
            return "Okay, let's start over. What symptoms are you having?"
        if tl == "list":
            pos = self.state.positives()
            if pos:
                return f"So far I have: {self.format_symptom_list(pos)}."
            return "I haven't recorded any symptoms yet."
        if tl.startswith("remove "):
            target = text[7:].strip().lower()
            if target in self.state.collected:
                del self.state.collected[target]
                return f"Removed {target}."
            return f"I couldn't find '{target}' in the list."

        # safety
        warning = self.check_red_flags(text)

        # rich extraction
        rich = self.extractor.extract_rich(text)
        # note: keep only present symptoms for counts
        self.state.add_rich(rich)
        newly_present = [e.name for e in rich if e.present]
        noted = f"Noted {self.format_symptom_list(sorted(set(newly_present)))}." if newly_present else None

        unique_count = len(self.state.positives())
        if unique_count < self.state.k_threshold:
            self.state.phase = "collect"
            q = self.next_followup_question() or "Any other symptoms?"
            pieces = [p for p in [warning, noted, q] if p]
            return " ".join(pieces) if pieces else "Could you tell me a bit more?"

        # move to diagnose
        self.state.phase = "diagnose"
        collected_list = self.state.positives()
        confirm = (
            f"So far I have {self.format_symptom_list(collected_list)}. "
            f"Would you like me to suggest possible next steps? (yes/no)"
        )
        pieces = [p for p in [warning, noted, confirm] if p]
        return " ".join(pieces)

    # ---------- Diagnosis phase (still guidance‑only in Stage 1) ----------
    def process_confirmation(self, text: str) -> str:
        t = text.strip().lower()
        if t in {"yes", "y", "ok", "okay", "sure"}:
            collected_list = self.state.positives()
            preds = self.classifier.predict(collected_list, top_k=3)
            if preds:
                lines = [f"Possible conditions (top {len(preds)}):"]
                for p in preds:
                    lines.append(f"- {p.label}: {p.probability:.1%}")
                lines.append("These are informational only and not a diagnosis.")
                self.state.phase = "explain"
                return "\n".join(lines)
            # Guidance-only path using patterns
            advice_lines: List[str] = []
            key = " + ".join(collected_list)
            for patt, tips in self.guidance_pack.get('patterns', {}).items():
                # very loose match: require all patt tokens to be present
                needed = [p.strip() for p in patt.split('+')]
                if all(n.strip() in collected_list for n in [x.strip() for x in needed]):
                    advice_lines.extend(tips)
            if not advice_lines:
                advice_lines = [
                    "General guidance: rest, hydrate, and consider appropriate OTC relief if suitable.",
                    "Seek professional care if symptoms worsen, persist >3 days, or you develop red‑flag signs."
                ]
            self.state.phase = "explain"
            return (
                f"(Stage 1 guidance)\n"
                f"Symptoms considered: {self.format_symptom_list(collected_list)}.\n"
                + "\n".join(f"- {a}" for a in advice_lines)
            )
        elif t in {"no", "n"}:
            self.state.phase = "collect"
            return "No problem—tell me more about your symptoms, or type 'done' to finish."
        else:
            return "Please reply 'yes' or 'no'."

    # ---------- CLI loop ----------
    def run_cli(self):
        print(self.welcome())
        print(self.help_text())
        while True:
            try:
                user = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye—take care!")
                break

            if not user:
                print("Bot: Could you describe your symptoms?")
                continue

            if user.lower() in {"quit", "exit"}:
                print("Bot: Goodbye—take care!")
                break

            if user.lower() == "done":
                if len(self.state.positives()) >= self.state.k_threshold:
                    print("Bot:", self.process_user_input("done"))
                print("Bot: Wishing you well! If symptoms worsen, seek medical care.")
                break

            if self.state.phase == "diagnose":
                resp = self.process_confirmation(user)
            else:
                resp = self.process_user_input(user)

            print("Bot:", resp)

# ----------------------------
# JSON batch mode (for testing/CI)
# ----------------------------

def run_script_mode_from_json(payload: Dict):
    k = int(payload.get("k", 3))
    bot = RuleBasedChatbot(k_threshold=k)
    outputs = []
    for msg in payload.get("messages", []):
        if bot.state.phase == "diagnose":
            outputs.append(bot.process_confirmation(msg))
        else:
            outputs.append(bot.process_user_input(msg))
    return {
        "responses": outputs,
        "positives": bot.state.positives(),
        "phase": bot.state.phase,
        "rich": {k: bot.state.collected[k].__dict__ for k in bot.state.collected}
    }

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            payload = json.load(f)
        result = run_script_mode_from_json(payload)
        print(json.dumps(result, indent=2))
    else:
        RuleBasedChatbot().run_cli()
