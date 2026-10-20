"""Source code similarity check for programming assignments.

Supports:
- Python (.py) using standard library tokenize module
- C, C++, Java, JavaScript (.c, .cpp, .java, .js) using regex tokenizer

Normalizes:
- Comments and whitespace stripped
- Identifiers replaced with placeholder 'ID' (resists variable renaming)
- String literals replaced with placeholder 'STR'
- Number literals replaced with placeholder 'NUM'
- Language keywords and operators preserved

Uses Stage 3 winnowing fingerprinting with token k-grams (default k=7) and supports
an optional boilerplate / template file filter.
"""

from dataclasses import dataclass, field
import io
import json
import keyword
from pathlib import Path
import re
import tokenize
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from veritext.checks.models import CheckResult, Finding
from veritext.fingerprint.hashing import hash_kgram
from veritext.fingerprint.kgrams import generate_kgrams
from veritext.fingerprint.winnow import winnow_hashes

DEFAULT_CODE_K = 7
DEFAULT_CODE_W = 4

SUPPORTED_CODE_EXTENSIONS = {".py", ".java", ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".js"}

# Common keywords across C/C++, Java, and JavaScript
C_CPP_KEYWORDS = {
    "auto", "break", "case", "char", "const", "continue", "default", "do",
    "double", "else", "enum", "extern", "float", "for", "goto", "if", "inline",
    "int", "long", "register", "restrict", "return", "short", "signed", "sizeof",
    "static", "struct", "switch", "typedef", "union", "unsigned", "void",
    "volatile", "while", "class", "namespace", "public", "private", "protected",
    "new", "delete", "template", "typename", "using", "bool", "true", "false",
    "try", "catch", "throw", "virtual", "nullptr", "include", "define",
}

JAVA_KEYWORDS = {
    "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char",
    "class", "const", "continue", "default", "do", "double", "else", "enum",
    "extends", "final", "finally", "float", "for", "goto", "if", "implements",
    "import", "instanceof", "int", "interface", "long", "native", "new", "package",
    "private", "protected", "public", "return", "short", "static", "strictfp",
    "super", "switch", "synchronized", "this", "throw", "throws", "transient",
    "try", "void", "volatile", "while", "true", "false", "null",
}

JS_KEYWORDS = {
    "break", "case", "catch", "class", "const", "continue", "debugger", "default",
    "delete", "do", "else", "export", "extends", "finally", "for", "function",
    "if", "import", "in", "instanceof", "new", "return", "super", "switch",
    "this", "throw", "try", "typeof", "var", "void", "while", "with", "yield",
    "let", "static", "async", "await", "true", "false", "null", "undefined",
    "of", "from",
}


@dataclass
class CodeToken:
    """Represents a normalized source code token."""
    token: str
    line: int


@dataclass
class CodeMatchRange:
    """Represents a matching line span between two files."""
    lines_a: Tuple[int, int]
    lines_b: Tuple[int, int]
    token_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lines_a": list(self.lines_a),
            "lines_b": list(self.lines_b),
            "token_count": self.token_count,
        }


@dataclass
class PairwiseCodeSimilarity:
    """Result of comparing two source code files."""
    file_a: str
    file_b: str
    similarity_pct: float
    matching_ranges: List[CodeMatchRange] = field(default_factory=list)
    tokens_a: int = 0
    tokens_b: int = 0
    shared_fingerprints: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_a": self.file_a,
            "file_b": self.file_b,
            "similarity_pct": round(self.similarity_pct, 2),
            "shared_fingerprints": self.shared_fingerprints,
            "tokens_a": self.tokens_a,
            "tokens_b": self.tokens_b,
            "matching_ranges": [r.to_dict() for r in self.matching_ranges],
        }


def tokenize_python(code_str: str) -> List[CodeToken]:
    """Tokenizes and normalizes Python source code using standard library tokenize."""
    tokens: List[CodeToken] = []
    code_bytes = code_str.encode("utf-8")

    try:
        reader = io.BytesIO(code_bytes).readline
        for tok in tokenize.tokenize(reader):
            tok_type = tok.type
            tok_str = tok.string
            line_no = tok.start[0]

            # Drop comments, whitespace, and structural markers
            if tok_type in (
                tokenize.COMMENT,
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.ENCODING,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.ENDMARKER,
            ):
                continue

            if tok_type == tokenize.NAME:
                if keyword.iskeyword(tok_str):
                    tokens.append(CodeToken(token=f"KW:{tok_str}", line=line_no))
                else:
                    tokens.append(CodeToken(token="ID", line=line_no))
            elif tok_type == tokenize.NUMBER:
                tokens.append(CodeToken(token="NUM", line=line_no))
            elif tok_type == tokenize.STRING:
                tokens.append(CodeToken(token="STR", line=line_no))
            elif tok_type == tokenize.OP:
                tokens.append(CodeToken(token=tok_str, line=line_no))
            else:
                tokens.append(CodeToken(token=tok_str, line=line_no))
    except Exception:
        # Fallback to regex tokenization on error
        return tokenize_generic(code_str, language="py")

    return tokens


def tokenize_generic(code_str: str, language: str = "c") -> List[CodeToken]:
    """Tokenizes and normalizes C, C++, Java, or JavaScript source code."""
    if language in ("java",):
        kw_set = JAVA_KEYWORDS
    elif language in ("js", "javascript"):
        kw_set = JS_KEYWORDS
    else:
        kw_set = C_CPP_KEYWORDS

    tokens: List[CodeToken] = []

    # Tokenizer pattern: comments, strings, numbers, identifiers, operators
    pattern = re.compile(
        r"(?P<LINE_COMMENT>//[^\n]*)|"
        r"(?P<BLOCK_COMMENT>/\*[\s\S]*?\*/)|"
        r"(?P<STRING>\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`)|"
        r"(?P<NUMBER>\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b)|"
        r"(?P<IDENT>[a-zA-Z_$][a-zA-Z0-9_$]*)|"
        r"(?P<MULTI_OP>===|!==|==|!=|<=|>=|&&|\|\||\+\+|--|\+=|-=|\*=|/=|%=|<<|>>|->|::|=>)|"
        r"(?P<OP>[+\-*/%=<>!&|^~?:;,.()\[\]{}])|"
        r"(?P<NEWLINE>\n)|"
        r"(?P<SPACE>[ \t\r]+)|"
        r"(?P<OTHER>[^\s])"
    )

    current_line = 1
    for m in pattern.finditer(code_str):
        kind = m.lastgroup
        val = m.group(0)

        if kind == "NEWLINE":
            current_line += 1
        elif kind == "BLOCK_COMMENT":
            current_line += val.count("\n")
        elif kind in ("LINE_COMMENT", "SPACE"):
            continue
        elif kind == "STRING":
            tokens.append(CodeToken(token="STR", line=current_line))
            current_line += val.count("\n")
        elif kind == "NUMBER":
            tokens.append(CodeToken(token="NUM", line=current_line))
        elif kind == "IDENT":
            if val in kw_set:
                tokens.append(CodeToken(token=f"KW:{val}", line=current_line))
            else:
                tokens.append(CodeToken(token="ID", line=current_line))
        elif kind in ("MULTI_OP", "OP"):
            tokens.append(CodeToken(token=val, line=current_line))
        elif kind == "OTHER":
            tokens.append(CodeToken(token=val, line=current_line))

    return tokens


def tokenize_file(file_path: Union[str, Path]) -> Tuple[List[CodeToken], str]:
    """Loads and tokenizes a source code file according to its extension."""
    path = Path(file_path)
    ext = path.suffix.lower()

    try:
        raw_code = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw_code = path.read_text(encoding="latin-1")

    if ext == ".py":
        return tokenize_python(raw_code), raw_code
    elif ext in (".java",):
        return tokenize_generic(raw_code, language="java"), raw_code
    elif ext in (".js",):
        return tokenize_generic(raw_code, language="js"), raw_code
    else:
        return tokenize_generic(raw_code, language="c"), raw_code


def fingerprint_code(
    tokens: Sequence[CodeToken],
    k: int = DEFAULT_CODE_K,
    w: int = DEFAULT_CODE_W,
) -> List[Tuple[int, int]]:
    """Generates winnowing fingerprints for a sequence of normalized code tokens."""
    if len(tokens) < k:
        return []

    token_strings = [t.token for t in tokens]
    kgrams = generate_kgrams(token_strings, k=k)
    hash_positions = [(hash_kgram(kg), pos) for kg, pos in kgrams]
    return winnow_hashes(hash_positions, w=w)


def compare_code_files(
    file_a: Union[str, Path],
    file_b: Union[str, Path],
    boilerplate_hashes: Optional[Set[int]] = None,
    k: int = DEFAULT_CODE_K,
    w: int = DEFAULT_CODE_W,
) -> PairwiseCodeSimilarity:
    """Computes similarity percentage and matching line ranges between two code files."""
    path_a = Path(file_a)
    path_b = Path(file_b)

    tokens_a, _ = tokenize_file(path_a)
    tokens_b, _ = tokenize_file(path_b)

    fps_a = fingerprint_code(tokens_a, k=k, w=w)
    fps_b = fingerprint_code(tokens_b, k=k, w=w)

    # Filter out boilerplate template hashes if provided
    if boilerplate_hashes:
        fps_a = [fp for fp in fps_a if fp[0] not in boilerplate_hashes]
        fps_b = [fp for fp in fps_b if fp[0] not in boilerplate_hashes]

    if not fps_a or not fps_b:
        return PairwiseCodeSimilarity(
            file_a=path_a.name,
            file_b=path_b.name,
            similarity_pct=0.0,
            matching_ranges=[],
            tokens_a=len(tokens_a),
            tokens_b=len(tokens_b),
            shared_fingerprints=0,
        )

    # Map hashes to positions in B
    b_hash_to_pos: Dict[int, List[int]] = {}
    for h, pos in fps_b:
        b_hash_to_pos.setdefault(h, []).append(pos)

    # Find shared hits: list of (pos_a, pos_b)
    shared_hits: List[Tuple[int, int]] = []
    shared_hashes_set: Set[int] = set()

    for h, pos_a in fps_a:
        if h in b_hash_to_pos:
            shared_hashes_set.add(h)
            for pos_b in b_hash_to_pos[h]:
                shared_hits.append((pos_a, pos_b))

    # Cluster consecutive matching positions into match ranges
    matching_ranges: List[CodeMatchRange] = []
    if shared_hits:
        shared_hits.sort(key=lambda x: (x[0], x[1]))
        clusters: List[List[Tuple[int, int]]] = []
        current_cluster = [shared_hits[0]]

        for next_hit in shared_hits[1:]:
            prev_a, prev_b = current_cluster[-1]
            curr_a, curr_b = next_hit

            # If adjacent or within small gap (w tokens) in both files
            if 0 <= curr_a - prev_a <= w and 0 <= curr_b - prev_b <= w:
                current_cluster.append(next_hit)
            else:
                clusters.append(current_cluster)
                current_cluster = [next_hit]
        if current_cluster:
            clusters.append(current_cluster)

        for cl in clusters:
            start_a = cl[0][0]
            end_a = cl[-1][0] + k - 1
            start_b = cl[0][1]
            end_b = cl[-1][1] + k - 1

            # Map token bounds to line numbers
            line_start_a = tokens_a[min(start_a, len(tokens_a) - 1)].line
            line_end_a = tokens_a[min(end_a, len(tokens_a) - 1)].line
            line_start_b = tokens_b[min(start_b, len(tokens_b) - 1)].line
            line_end_b = tokens_b[min(end_b, len(tokens_b) - 1)].line

            matching_ranges.append(
                CodeMatchRange(
                    lines_a=(line_start_a, line_end_a),
                    lines_b=(line_start_b, line_end_b),
                    token_count=end_a - start_a + 1,
                )
            )

    # Sørensen-Dice similarity coefficient on fingerprints
    total_fps = len(fps_a) + len(fps_b)
    count_shared = sum(1 for h, _ in fps_a if h in b_hash_to_pos)
    sim_pct = (2.0 * count_shared / total_fps) * 100.0 if total_fps > 0 else 0.0
    sim_pct = min(100.0, max(0.0, sim_pct))

    return PairwiseCodeSimilarity(
        file_a=path_a.name,
        file_b=path_b.name,
        similarity_pct=round(sim_pct, 2),
        matching_ranges=matching_ranges,
        tokens_a=len(tokens_a),
        tokens_b=len(tokens_b),
        shared_fingerprints=count_shared,
    )


def compare_code_directory(
    folder: Union[str, Path],
    template_file: Optional[Union[str, Path]] = None,
    k: int = DEFAULT_CODE_K,
    w: int = DEFAULT_CODE_W,
) -> List[PairwiseCodeSimilarity]:
    """Compares all code files in a directory pairwise.

    Args:
        folder: Directory path containing source code files.
        template_file: Optional boilerplate template file to ignore.
        k: k-gram window size.
        w: winnowing window size.

    Returns:
        List of PairwiseCodeSimilarity results for all pairs, sorted by similarity descending.
    """
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Provided path is not a directory: {folder}")

    code_files: List[Path] = [
        p for p in folder_path.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_CODE_EXTENSIONS and not p.name.startswith(".")
    ]

    # Pre-compute template hashes if provided
    boilerplate_hashes: Optional[Set[int]] = None
    if template_file:
        t_path = Path(template_file)
        if t_path.exists():
            t_tokens, _ = tokenize_file(t_path)
            t_fps = fingerprint_code(t_tokens, k=k, w=w)
            boilerplate_hashes = {h for h, _ in t_fps}

    results: List[PairwiseCodeSimilarity] = []
    n = len(code_files)

    for i in range(n):
        for j in range(i + 1, n):
            sim = compare_code_files(
                code_files[i],
                code_files[j],
                boilerplate_hashes=boilerplate_hashes,
                k=k,
                w=w,
            )
            results.append(sim)

    results.sort(key=lambda s: s.similarity_pct, reverse=True)
    return results


def check_code_similarity(
    document: Union[Document, str, Path],
    folder: Optional[Union[str, Path]] = None,
    template_file: Optional[Union[str, Path]] = None,
) -> CheckResult:
    """Checks a single code file against a directory of code files.

    Args:
        document: The target code file Document or path.
        folder: Directory of peer code files (defaults to document's parent dir).
        template_file: Optional boilerplate code file to ignore.

    Returns:
        CheckResult with findings for any peer files exceeding similarity thresholds.
    """
    target_path = Path(document.path if isinstance(document, Document) and document.path else document)
    if not target_path.exists():
        return CheckResult(
            check_name="code_similarity",
            findings=[],
            summary="Target code file does not exist.",
            passed=True,
        )

    cmp_folder = Path(folder) if folder else target_path.parent
    peers = [
        p for p in cmp_folder.iterdir()
        if p.is_file() and p != target_path and p.suffix.lower() in SUPPORTED_CODE_EXTENSIONS and not p.name.startswith(".")
    ]

    if not peers:
        return CheckResult(
            check_name="code_similarity",
            findings=[],
            summary="No peer code files found in directory to compare against.",
            passed=True,
        )

    boilerplate_hashes: Optional[Set[int]] = None
    if template_file:
        t_path = Path(template_file)
        if t_path.exists():
            t_tokens, _ = tokenize_file(t_path)
            t_fps = fingerprint_code(t_tokens)
            boilerplate_hashes = {h for h, _ in t_fps}

    findings: List[Finding] = []
    max_sim = 0.0

    for peer in peers:
        res = compare_code_files(target_path, peer, boilerplate_hashes=boilerplate_hashes)
        if res.similarity_pct > max_sim:
            max_sim = res.similarity_pct

        if res.similarity_pct >= 40.0:
            sev = "high" if res.similarity_pct >= 70.0 else "warning"
            lines_desc = ", ".join(f"L{r.lines_a[0]}-{r.lines_a[1]} with {peer.name}:L{r.lines_b[0]}-{r.lines_b[1]}" for r in res.matching_ranges[:3])
            findings.append(
                Finding(
                    check="code_similarity",
                    severity=sev,
                    message=f"High code similarity ({res.similarity_pct:.1f}%) detected with '{peer.name}' ({lines_desc}).",
                    start=0,
                    end=0,
                    details=res.to_dict(),
                )
            )

    passed = not any(f.severity in ("warning", "high") for f in findings)
    summary = f"Code similarity check: compared against {len(peers)} file(s). Maximum similarity: {max_sim:.1f}%."

    return CheckResult(
        check_name="code_similarity",
        findings=findings,
        summary=summary,
        passed=passed,
        metadata={
            "peer_count": len(peers),
            "max_similarity_pct": max_sim,
        },
    )
