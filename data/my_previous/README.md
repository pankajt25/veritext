# Previous Documents Repository (Self-Plagiarism Check)

Place your previously submitted assignments, thesis chapters, published papers, or earlier drafts in this directory.

The self-plagiarism check in Stage 5 will index these files under `origin="self"` using winnowing fingerprints in SQLite and compare new submissions against your past work to detect unintentional overlap or recycled text.

### How to Index Your Previous Work:
You can index files in this directory or anywhere on your machine using:
```bash
python -m veritext.checks add-previous data/my_previous/
# or for a single file:
python -m veritext.checks add-previous path/to/my_old_paper.docx
```
