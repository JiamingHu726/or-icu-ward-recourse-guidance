# GitHub upload guide

Create a private repository named:

```text
or-icu-ward-recourse-guidance
```

Unzip this staging package, inspect `git status`, and upload the directory contents rather than uploading the ZIP itself.

```bash
git init
git add .
git commit -m "Prepare v0.9.0 load-regime reproducibility staging release"
git branch -M main
git remote add origin https://github.com/REPLACE_WITH_OWNER/or-icu-ward-recourse-guidance.git
git push -u origin main
```

Keep the repository private until author metadata, public URL, and Zenodo information have been finalized. Do not add raw Mannino files, Gurobi licence files, credentials, or local raw result directories.
