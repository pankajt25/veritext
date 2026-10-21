"""Style and grammar linter coordinating built-in rules and optional LanguageTool."""

import importlib.util
import logging
import shutil
from typing import Any, Dict, List, Optional

from veritext.checks.models import Finding
from veritext.style.metrics import calculate_style_metrics
from veritext.style.models import StyleResult
from veritext.style.rules import (
    check_filler_phrases,
    check_passive_voice,
    check_repeated_words,
    check_sentence_length,
    check_weak_transitions,
)

logger = logging.getLogger(__name__)


def check_languagetool_support() -> Dict[str, Any]:
    """Inspects environment for Java and language-tool-python.

    Never downloads or installs anything.
    """
    has_java = shutil.which("java") is not None
    has_pkg = importlib.util.find_spec("language_tool_python") is not None

    if not has_java and not has_pkg:
        return {
            "available": False,
            "engine": "LanguageTool",
            "reason": "LanguageTool requires a local Java runtime and the language-tool-python package (neither is installed). Built-in rules active.",
        }
    elif not has_java:
        return {
            "available": False,
            "engine": "LanguageTool",
            "reason": "Java runtime was not found in PATH. LanguageTool cannot start without Java. Built-in rules active.",
        }
    elif not has_pkg:
        return {
            "available": False,
            "engine": "LanguageTool",
            "reason": "Optional 'language-tool-python' package is not installed. Built-in rules active.",
        }

    return {
        "available": True,
        "engine": "LanguageTool",
        "reason": "Java runtime and language-tool-python package are available.",
    }


def lint_style(
    text: str,
    enable_languagetool: bool = False,
    max_sentence_words: int = 35,
) -> StyleResult:
    """Runs comprehensive grammar and style checks.

    All style findings are informational ('info' severity) and never affect
    plagiarism similarity or AI probability scores.
    """
    if not text or not text.strip():
        metrics = calculate_style_metrics("")
        return StyleResult(
            findings=[],
            metrics=metrics,
            engine_status={"builtin_rules": True, "languagetool": check_languagetool_support()},
        )

    # 1. Compute readability metrics
    metrics = calculate_style_metrics(text)

    # 2. Run built-in rule checks
    findings: List[Finding] = []
    findings.extend(check_passive_voice(text))
    findings.extend(check_repeated_words(text))
    findings.extend(check_sentence_length(text, max_words=max_sentence_words))
    findings.extend(check_filler_phrases(text))
    findings.extend(check_weak_transitions(text))

    lt_status = check_languagetool_support()

    # 3. Optional LanguageTool check (strictly guarded)
    if enable_languagetool and lt_status.get("available"):
        try:
            import language_tool_python  # type: ignore

            tool = language_tool_python.LanguageTool("en-US")
            matches = tool.check(text)
            for m in matches:
                findings.append(
                    Finding(
                        check="style_languagetool",
                        severity="info",
                        message=f"{m.message} ({m.ruleId})",
                        start=m.offset,
                        end=m.offset + m.errorLength,
                        details={
                            "rule_id": m.ruleId,
                            "replacements": m.replacements[:3],
                            "category": getattr(m, "category", "Grammar"),
                        },
                    )
                )
        except Exception as e:
            logger.warning(f"Error invoking LanguageTool: {e}")
            lt_status["error"] = str(e)

    # Sort all findings chronologically by start offset
    findings.sort(key=lambda f: (f.start, f.end))

    return StyleResult(
        findings=findings,
        metrics=metrics,
        engine_status={
            "builtin_rules": True,
            "languagetool": lt_status,
        },
    )
