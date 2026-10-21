# Veritext Local Class Repository

This folder stores documents and winnowing hashes for cross-submission and collusion matching.

## Privacy & Opt-In Consent Notice

- **Local Processing**: All documents added to class repositories are processed completely offline and locally on your machine. No documents are uploaded to cloud servers or third-party APIs.
- **Fingerprints-Only Mode**: When adding submissions with `--fingerprints-only`, Veritext stores solely the mathematical winnowing hashes and character offsets. The underlying student prose is never written to disk, guaranteeing maximum student confidentiality.
- **Anonymised Identification**: During routine similarity scans, submissions are displayed using anonymised identifiers (e.g. `sub_a1b2c3d4`) by default to protect student privacy. Original filenames are only revealed when the `--show-names` flag is explicitly enabled by authorized instructional staff.
- **Data Rights & Purge**: Students and institutions retain all copyright and privacy rights. Any class repository can be deleted at any time via `python -m veritext.classrepo delete <class_name>` or completely purged with `python -m veritext.classrepo purge`.
