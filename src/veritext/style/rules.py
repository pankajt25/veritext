"""Built-in grammar and style rules for Veritext.

All findings have severity 'info' and never affect similarity or AI scores.
Rules include:
- Passive voice detection
- Repeated words (lexical duplication / stuttering)
- Overlong sentences (> 35 words)
- Wordy filler phrases
- Weak or abrupt transitions
"""

import re
from typing import Any, Dict, List

from veritext.checks.models import Finding

# Common irregular past participles
IRREGULAR_PAST_PARTICIPLES = {
    "been", "begun", "blown", "broken", "brought", "built", "caught", "chosen",
    "done", "drawn", "driven", "eaten", "fallen", "felt", "found", "given",
    "gone", "grown", "heard", "held", "hidden", "kept", "known", "laid",
    "led", "left", "lost", "made", "meant", "met", "paid", "proven", "put",
    "read", "run", "said", "seen", "sent", "set", "shown", "shut", "spoken",
    "spent", "taken", "taught", "told", "thought", "understood", "worn", "won",
    "written"
}

# Forms of 'to be'
BE_VERBS = {"is", "are", "was", "were", "be", "been", "being", "am"}

# Wordy filler phrases and recommended concise alternatives
FILLER_PHRASES = {
    "in order to": "to",
    "due to the fact that": "because",
    "at the present time": "currently or now",
    "for the purpose of": "to",
    "in the event that": "if",
    "with reference to": "regarding",
    "as a matter of fact": "in fact",
    "it should be noted that": "note that",
    "all things considered": "overall",
    "in spite of the fact that": "although",
    "by means of": "by or with",
    "in the near future": "soon",
    "at this point in time": "now",
    "a large number of": "many",
    "take into consideration": "consider",
    "make an assumption": "assume",
    "conduct an investigation into": "investigate",
    "has the ability to": "can",
    "prior to": "before",
    "subsequent to": "after",
}

# Weak transitions or informal sentence openers in academic writing
WEAK_TRANSITION_STARTERS = [
    ("basically,", "Informal starter; consider omitting or using a precise analytical transition."),
    ("obviously,", "Can sound presumptive; allow evidence to demonstrate the point."),
    ("clearly,", "May sound assertive without proof; consider demonstrating directly."),
    ("literally,", "Informal intensifier; typically unnecessary in academic prose."),
    ("and so,", "Informal colloquial link; consider 'Consequently' or 'Therefore'."),
    ("as you can see,", "Informal conversational address; focus on the data/evidence."),
]


def check_passive_voice(text: str) -> List[Finding]:
    """Detects passive voice constructions (be-verb + past participle)."""
    findings: List[Finding] = []
    # Pattern: \b(am|is|are|was|were|be|been|being)\s+([a-z]+ed|[a-z]+en|specific irregulars)\b
    pattern = re.compile(
        r"\b(am|is|are|was|were|be|been|being)\s+(?:[a-zA-Z]+ly\s+)?([a-zA-Z]+)\b",
        re.IGNORECASE,
    )

    for m in pattern.finditer(text):
        aux = m.group(1).lower()
        participle = m.group(2).lower()

        # Check if second word is a past participle
        is_participle = (
            participle in IRREGULAR_PAST_PARTICIPLES
            or participle.endswith("ed")
            or (participle.endswith("en") and len(participle) > 4)
        )

        # Ignore common false positives (e.g. 'is red', 'is bed', 'is dead', 'being open')
        if participle in {"bed", "red", "led", "dead", "open", "even"}:
            is_participle = False

        if is_participle:
            matched_str = m.group(0)
            start, end = m.span()
            findings.append(
                Finding(
                    check="style_passive_voice",
                    severity="info",
                    message=f"Passive voice detected: '{matched_str}'. Consider active voice for directness.",
                    start=start,
                    end=end,
                    details={
                        "rule": "passive_voice",
                        "matched_text": matched_str,
                        "suggestion": "Convert to active voice (e.g., '[Subject] [verb] [object]').",
                    },
                )
            )
    return findings


def check_repeated_words(text: str) -> List[Finding]:
    """Detects adjacent repeated words (e.g. 'the the', 'in in')."""
    findings: List[Finding] = []
    pattern = re.compile(r"\b([A-Za-z]{2,})\s+\1\b", re.IGNORECASE)

    for m in pattern.finditer(text):
        matched_str = m.group(0)
        word = m.group(1)
        # Allow intentional rhetorical repetitions or standard phrases if any
        if word.lower() in {"had", "that"}:
            # 'had had' and 'that that' can occasionally be grammatically valid, but still worth noting
            pass

        start, end = m.span()
        findings.append(
            Finding(
                check="style_repeated_word",
                severity="info",
                message=f"Duplicate consecutive word detected: '{matched_str}'.",
                start=start,
                end=end,
                details={
                    "rule": "repeated_words",
                    "word": word,
                    "matched_text": matched_str,
                    "suggestion": f"Remove redundant '{word}'.",
                },
            )
        )
    return findings


def check_sentence_length(text: str, max_words: int = 35) -> List[Finding]:
    """Flags overly long sentences that may impair reader comprehension."""
    findings: List[Finding] = []
    # Match sentences while capturing their character offsets
    sentence_pattern = re.compile(r"[^.!?\n]+(?:[.!?]|$)", re.MULTILINE)

    for m in sentence_pattern.finditer(text):
        sentence_str = m.group(0).strip()
        if not sentence_str:
            continue
        words = re.findall(r"\b[A-Za-z0-9_\-']+\b", sentence_str)
        if len(words) > max_words:
            start = m.start()
            end = m.end()
            findings.append(
                Finding(
                    check="style_sentence_length",
                    severity="info",
                    message=f"Long sentence ({len(words)} words). Consider dividing into shorter sentences.",
                    start=start,
                    end=end,
                    details={
                        "rule": "sentence_length",
                        "word_count": len(words),
                        "limit": max_words,
                        "suggestion": "Split into two or more focused statements.",
                    },
                )
            )
    return findings


def check_filler_phrases(text: str) -> List[Finding]:
    """Detects wordy filler phrases and offers concise alternatives."""
    findings: List[Finding] = []

    for phrase, replacement in FILLER_PHRASES.items():
        pattern = re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)
        for m in pattern.finditer(text):
            matched_str = m.group(0)
            start, end = m.span()
            findings.append(
                Finding(
                    check="style_filler_phrase",
                    severity="info",
                    message=f"Wordy phrase '{matched_str}'. Consider replacing with '{replacement}'.",
                    start=start,
                    end=end,
                    details={
                        "rule": "filler_phrase",
                        "phrase": phrase,
                        "matched_text": matched_str,
                        "replacement": replacement,
                    },
                )
            )
    return findings


def check_weak_transitions(text: str) -> List[Finding]:
    """Detects informal sentence openers or weak transitions."""
    findings: List[Finding] = []

    for starter, explanation in WEAK_TRANSITION_STARTERS:
        pattern = re.compile(r"(?:^|[.!?\n]\s*)" + re.escape(starter), re.IGNORECASE)
        for m in pattern.finditer(text):
            matched_str = m.group(0).strip()
            start, end = m.span()
            findings.append(
                Finding(
                    check="style_weak_transition",
                    severity="info",
                    message=f"Informal transition '{matched_str}': {explanation}",
                    start=start,
                    end=end,
                    details={
                        "rule": "weak_transition",
                        "matched_text": matched_str,
                        "suggestion": explanation,
                    },
                )
            )
    return findings
