"""Script to generate reproducible Stage 5 demo samples in data/samples/checks_demo/."""

from pathlib import Path
import docx
from docx.shared import Pt, RGBColor

BASE_DIR = Path(__file__).resolve().parent.parent
DEMO_DIR = BASE_DIR / "data" / "samples" / "checks_demo"
CODE_DIR = DEMO_DIR / "code_demo"


def generate_docx_demo(target_path: Path):
    """Creates a .docx file containing white text, tiny font (<2pt), and hidden runs."""
    doc = docx.Document()
    doc.add_heading("Academic Integrity and Automated Verification", level=1)

    p1 = doc.add_paragraph("This document provides a comprehensive analysis of computational methods in text analysis. ")
    p1.add_run("The subsequent evaluation explores empirical evidence gathered across multiple institutions.")

    # Paragraph with white-text run
    p2 = doc.add_paragraph("Modern verification engines rely on probabilistic models to identify anomalies. ")
    white_run = p2.add_run("This white-font sentence was injected maliciously to inflate word counts without visual detection.")
    white_run.font.color.rgb = RGBColor(255, 255, 255)
    p2.add_run(" Furthermore, character-level distributions provide crucial detection signals.")

    # Paragraph with tiny font run (< 2pt)
    p3 = doc.add_paragraph("Standard document formats permit granular typography adjustments. ")
    tiny_run = p3.add_run("Hidden microscopic payload formatted with 1 pt font.")
    tiny_run.font.size = Pt(1.0)
    p3.add_run(" These signals assist in flagging suspicious document formatting.")

    doc.save(target_path)
    print(f"Generated DOCX demo: {target_path}")


def generate_homoglyphs_demo(target_path: Path):
    """Creates a .txt file with Cyrillic homoglyphs and zero-width characters."""
    # Words with Cyrillic substitutions:
    # 'systеm' (Cyrillic \u0435)
    # 'acadеmiс' (Cyrillic \u0435 and \u0441)
    # 'papеr' (Cyrillic \u0435)
    content = (
        "Modern computational linguistics investigates how neural models process written language.\n\n"
        "The automated syst\u0435m evaluates documents for linguistic consistency and acad\u0435mi\u0441 integrity. "
        "In this pap\u0435r, we demonstrate several adversarial evasion techniques.\n\n"
        "Adversaries frequently insert invisible characters such as zero\u200bwidth\u200bspace and "
        "word\u2060joiner formatting marks to break continuous token sequences.\n\n"
        "Additionally, soft\u00adhyphens are embedded to confuse downstream tokenizers while remaining "
        "virtually unnoticeable to human readers.\n"
    )
    target_path.write_text(content, encoding="utf-8")
    print(f"Generated homoglyphs demo: {target_path}")


def generate_citations_demo(target_path: Path):
    """Creates a .txt file with citations, bibliography mismatches, uncited claims, and quotes."""
    content = (
        "Linguistic Patterns in Automated Text Generation\n\n"
        "Recent advances in natural language processing have transformed automated writing systems. "
        "Several foundational principles were established in early machine learning research [1]. "
        "Modern neural network architectures further expanded these generative capabilities [2].\n\n"
        "Studies show that conversational agents improve student productivity by 42% across diverse university cohorts. "
        "The observed gains remained consistent across distinct academic disciplines.\n\n"
        "As discussed by Smith (2020), stylistic variations offer diagnostic clues into machine authorship. "
        "Moreover, specialized transformers achieve superior perplexity scores when benchmarked against human essays [3]. "
        "However, some claims reference experimental studies that are entirely absent from the bibliography [5].\n\n"
        'As the research committee famously concluded: "When language models generate extended multi-paragraph discourse, '
        "they exhibit remarkably uniform burstiness and low token-level perplexity variance compared to human authors writing "
        'under spontaneous conditions."\n\n'
        "References\n\n"
        "[1] Adams, J. (2019). Machine learning fundamentals. Oxford University Press.\n"
        "[2] Baker, R. (2021). Deep learning in production. MIT Press.\n"
        "[3] Clark, T. (2022). Natural language processing. Cambridge University Press.\n"
        "[4] Davis, M. (2018). Statistical foundations of artificial intelligence. Springer.\n"
        "[2] Baker, R. (2021). Deep learning in production. MIT Press.\n"
    )
    target_path.write_text(content, encoding="utf-8")
    print(f"Generated citations demo: {target_path}")


def generate_self_plagiarism_demo(prev_path: Path, new_path: Path):
    """Creates a previous work file and a new submission that reuses a paragraph."""
    reused_paragraph = (
        "Recurrent neural networks and long short-term memory units were historically developed to "
        "address the vanishing gradient problem encountered during the backpropagation through time "
        "algorithm. By incorporating input gates, forget gates, and candidate cell state updates, "
        "these architectures preserved contextual dependencies across sequential information streams."
    )

    prev_content = (
        "Thesis Chapter 2: Historical Evolution of Sequence Modeling\n\n"
        f"{reused_paragraph}\n\n"
        "Subsequent developments replaced recurrent cells with multi-head self-attention mechanisms, "
        "enabling parallelized computation across entire sequences simultaneously."
    )
    prev_path.write_text(prev_content, encoding="utf-8")

    new_content = (
        "Assignment 3: Deep Learning Architectures in Natural Language Processing\n\n"
        "In this assignment, we examine foundational sequential modeling approaches.\n\n"
        f"{reused_paragraph}\n\n"
        "In our practical experiments, we compare this sequential architecture against modern "
        "transformer baselines on classification accuracy."
    )
    new_path.write_text(new_content, encoding="utf-8")
    print(f"Generated self-plagiarism demo pair: {prev_path.name} and {new_path.name}")


def generate_code_demo(code_dir: Path):
    """Creates original, renamed-variable copy, unrelated, and template Python files."""
    code_dir.mkdir(parents=True, exist_ok=True)

    template_code = (
        "# Programming Assignment 2: Graph Pathfinding Template\n"
        "# Student Name: \n"
        "# Student ID: \n\n"
        "class Graph:\n"
        "    def __init__(self):\n"
        "        self.adj = {}\n\n"
        "    def add_edge(self, u, v, weight):\n"
        "        pass\n\n"
        "    def dijkstra(self, start_node):\n"
        "        pass\n"
    )
    (code_dir / "boilerplate_template.py").write_text(template_code, encoding="utf-8")

    original_code = (
        "# Graph pathfinding implementation using Dijkstra's algorithm\n"
        "import heapq\n\n"
        "class Graph:\n"
        "    def __init__(self):\n"
        "        self.adj = {}\n\n"
        "    def add_edge(self, u, v, weight):\n"
        "        if u not in self.adj:\n"
        "            self.adj[u] = []\n"
        "        self.adj[u].append((v, weight))\n\n"
        "    def dijkstra(self, start_node):\n"
        "        distances = {start_node: 0}\n"
        "        priority_queue = [(0, start_node)]\n"
        "        visited = set()\n\n"
        "        while priority_queue:\n"
        "            curr_dist, curr_node = heapq.heappop(priority_queue)\n"
        "            if curr_node in visited:\n"
        "                continue\n"
        "            visited.add(curr_node)\n\n"
        "            for neighbor, weight in self.adj.get(curr_node, []):\n"
        "                distance = curr_dist + weight\n"
        "                if neighbor not in distances or distance < distances[neighbor]:\n"
        "                    distances[neighbor] = distance\n"
        "                    heapq.heappush(priority_queue, (distance, neighbor))\n\n"
        "        return distances\n"
    )
    (code_dir / "original.py").write_text(original_code, encoding="utf-8")

    renamed_code = (
        "# Shortest path calculation on weighted networks\n"
        "# Author: Assignment Submission\n"
        "import heapq\n\n"
        "class Graph:\n"
        "    def __init__(self):\n"
        "        self.adj = {}\n\n"
        "    def add_edge(self, src, dest, cost):\n"
        "        if src not in self.adj:\n"
        "            self.adj[src] = []\n"
        "        self.adj[src].append((dest, cost))\n\n"
        "    def dijkstra(self, root):\n"
        "        min_costs = {root: 0}\n"
        "        pq = [(0, root)]\n"
        "        closed_set = set()\n\n"
        "        while pq:\n"
        "            dist_val, active_v = heapq.heappop(pq)\n"
        "            if active_v in closed_set:\n"
        "                continue\n"
        "            closed_set.add(active_v)\n\n"
        "            for adjacent_v, edge_cost in self.adj.get(active_v, []):\n"
        "                cand_dist = dist_val + edge_cost\n"
        "                if adjacent_v not in min_costs or cand_dist < min_costs[adjacent_v]:\n"
        "                    min_costs[adjacent_v] = cand_dist\n"
        "                    heapq.heappush(pq, (cand_dist, adjacent_v))\n\n"
        "        return min_costs\n"
    )
    (code_dir / "renamed_copy.py").write_text(renamed_code, encoding="utf-8")

    unrelated_code = (
        "# Web scraper and tabular dataset processor\n"
        "import csv\n"
        "import json\n"
        "from pathlib import Path\n\n"
        "def process_csv_records(input_path, output_json):\n"
        "    records = []\n"
        "    with open(input_path, 'r', encoding='utf-8') as f:\n"
        "        reader = csv.DictReader(f)\n"
        "        for row in reader:\n"
        "            clean_row = {k.strip(): v.strip() for k, v in row.items()}\n"
        "            records.append(clean_row)\n\n"
        "    Path(output_json).write_text(json.dumps(records, indent=2))\n"
        "    return len(records)\n"
    )
    (code_dir / "unrelated.py").write_text(unrelated_code, encoding="utf-8")
    print(f"Generated code demo files in: {code_dir}")


def generate_demo_readme(target_path: Path):
    """Writes documentation detailing expected results for each demo."""
    readme_content = """# Veritext Stage 5 Integrity Checks: Demo Dataset

This folder contains verified sample files designed to test and showcase all four Stage 5 integrity modules:

| Demo File | Target Check | Suspicious Features / Expected Findings |
| :--- | :--- | :--- |
| `hidden_content_demo.docx` | **Hidden Content** | 1 white font run (`RGB(255,255,255)`) and 1 tiny font run (`1.0 pt`). Flagged as HIGH severity. |
| `homoglyphs_zero_width_demo.txt` | **Hidden Content** | Cyrillic homoglyph substitutions in Latin words (`systеm`, `acadеmiс`, `papеr`), zero-width spaces (`U+200B`), word joiners (`U+2060`), and soft hyphens (`U+00AD`). Flagged as HIGH severity. |
| `citations_mismatch_demo.txt` | **Citations & Quotes** | (1) Mixed numeric (`[1]`) and author-year (`Smith (2020)`) styles.<br>(2) Missing reference (`[5]` cited in text but not in bibliography).<br>(3) Uncited reference (`[4] Davis (2018)` in bibliography but never cited).<br>(4) Duplicate reference (`[2] Baker (2021)` listed twice).<br>(5) Likely uncited claim ("Studies show that conversational agents improve student productivity by 42%...").<br>(6) Long quote (>25 words) without a citation. |
| `self_plag_previous.txt`<br>`self_plag_new.txt` | **Self-Plagiarism** | Multi-sentence verbatim paragraph on recurrent neural networks reused across both files. High overlap percentage reported. |
| `code_demo/` | **Code Similarity** | Pairwise comparison: `original.py` vs `renamed_copy.py` achieves >80% similarity despite complete identifier renaming; both show ~0% similarity with `unrelated.py`. |

---

### How to Run Each Demo

#### 1. Hidden Content Checks
```bash
# Check Word DOCX for white text and tiny font
python -m veritext.checks data/samples/checks_demo/hidden_content_demo.docx

# Check text file for Cyrillic homoglyphs and zero-width characters
python -m veritext.checks data/samples/checks_demo/homoglyphs_zero_width_demo.txt
```

#### 2. Citation and Quote Checks
```bash
# Check citations, uncited claims, and bibliography consistency
python -m veritext.checks data/samples/checks_demo/citations_mismatch_demo.txt
```

#### 3. Self-Plagiarism Check
```bash
# 1. Register previous work
python -m veritext.checks add-previous data/samples/checks_demo/self_plag_previous.txt

# 2. Check new submission against registered previous work
python -m veritext.checks data/samples/checks_demo/self_plag_new.txt
```

#### 4. Source Code Similarity
```bash
# Run pairwise code similarity across all files in code_demo/
python -m veritext.checks code data/samples/checks_demo/code_demo/

# Run with boilerplate template filtering
python -m veritext.checks code data/samples/checks_demo/code_demo/ --template data/samples/checks_demo/code_demo/boilerplate_template.py
```
"""
    target_path.write_text(readme_content, encoding="utf-8")
    print(f"Generated README: {target_path}")


def main():
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    generate_docx_demo(DEMO_DIR / "hidden_content_demo.docx")
    generate_homoglyphs_demo(DEMO_DIR / "homoglyphs_zero_width_demo.txt")
    generate_citations_demo(DEMO_DIR / "citations_mismatch_demo.txt")
    generate_self_plagiarism_demo(
        DEMO_DIR / "self_plag_previous.txt",
        DEMO_DIR / "self_plag_new.txt",
    )
    generate_code_demo(CODE_DIR)
    generate_demo_readme(DEMO_DIR / "README.md")
    print("All demo samples generated successfully!")


if __name__ == "__main__":
    main()
