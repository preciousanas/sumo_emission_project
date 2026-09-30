# GitHub deployment checklist

Quick guide to push everything in this working directory to
`https://github.com/preciousanas/sumo_emission_project` and connect
Zenodo for automatic DOI-on-release.

---

## 1. One-time GitHub setup (5 min)

```bash
# From the project root (this folder)
cd C:\Users\panavberokhai\Desktop\project\sumo_emission_project

# Initialise git if not already
git init
git config user.name  "Precious Anavberokhai"
git config user.email "panavberokhai@aggies.ncat.edu"

# Add the remote
git remote add origin https://github.com/preciousanas/sumo_emission_project.git
```

If the repo doesn't exist yet on GitHub, create it first at
<https://github.com/new> — leave it empty (no README, no .gitignore),
name it `sumo_emission_project`, owner `preciousanas`, Public.

## 2. Verify what will be committed (1 min)

The `.gitignore` in this folder is designed to keep everything scientifically
important and exclude only build products / caches / very-large raw files.
Preview:

```bash
git status --short | head -100
git status --short | wc -l    # total files that will be committed
```

If you see any file you don't want to publish (e.g. a private key,
`.env`, personal notes), add it to `.gitignore` before the first push.

### Optional: exclude the massive raw per-seed CSVs

Each per-seed `result_seed_<N>.csv` for high-demand volumes is ~100–200 MB, and
GitHub blocks single files >100 MB. Two ways to handle:

**Option A — Git LFS (recommended if you want the raw CSVs in the repo):**
```bash
git lfs install
git lfs track "data/V*/**/result_seed_*.csv"
git lfs track "data/V*/**/tripinfo_seed_*.xml"
git add .gitattributes
```

**Option B — exclude the raw CSVs and only commit summary CSVs:**
Add these lines to `.gitignore`:
```
data/V*/baseline/result_seed_*.csv
data/V*/emission_based/result_seed_*.csv
data/V*/baseline/tripinfo_seed_*.xml
data/V*/emission_based/tripinfo_seed_*.xml
```
Then commit only the aggregated summary CSVs (a few hundred KB total). The
paper's reproducibility statement still holds because `regenerate_cross_volume.py`
can reconstruct any missing summary from the raw sim outputs.

**Recommended for TRIP submission: Option B.** Reviewers rarely need the
100 GB of raw telemetry; they need the summary CSVs and the scripts that
produce them. Point to a Zenodo archive for the full dataset.

## 3. First commit + push (2 min)

```bash
git add README.md LICENSE .gitignore requirements.txt CITATION.cff
git add paper1_TRIP.tex references.bib
git add scripts/ config/
git add experiments/*/results/          # summary CSVs
git add experiments/*/visuals/          # publication figures
git add visuals/                        # publication figures
git add data/capacity_analysis/summary/ # Pipeline A summary CSVs

# Commit
git commit -m "Initial public release (TRIP submission)"

# Push
git branch -M main
git push -u origin main
```

Estimated total repo size after this commit: **~150–300 MB** without Git LFS
(figures + summary CSVs + scripts), or **~2 GB** with LFS-tracked raw CSVs.

## 4. Enable Zenodo integration (5 min)

Zenodo will mint a permanent DOI every time you make a GitHub Release.

1. Go to <https://zenodo.org> and sign in with your GitHub account.
2. Click your username → **GitHub** in the top-right menu.
3. Find `preciousanas/sumo_emission_project` in the repo list and toggle it **ON**.
4. Back on GitHub, go to
   `https://github.com/preciousanas/sumo_emission_project/releases/new`.
5. Click **Choose a tag** → type `v1.0.0` → **Create new tag on publish**.
6. **Release title:** `TRIP submission v1.0.0`.
7. **Description:** paste the abstract from the paper.
8. Click **Publish release**.

Within ~1 minute Zenodo will pick up the release and mint a DOI. The DOI
will appear in the Zenodo badge on your repo's Zenodo page — copy it and
update:
- The `\section*{Data and Code Availability}` section in `paper1_TRIP.tex`
  (currently says "DOI to be assigned upon manuscript acceptance").
- `CITATION.cff` (add a `doi:` field near the top).

## 5. Optional: post the preprint to arXiv

TRIP allows preprints. To post to arXiv:

1. Compile `paper1_TRIP.tex` locally: `pdflatex → bibtex → pdflatex → pdflatex`.
2. Create an arXiv account at <https://arxiv.org/user/register> if you don't have one
   (you'll need endorsement from an existing arXiv author in your field — one of your
   co-authors likely qualifies).
3. Submit at <https://arxiv.org/submit>.
4. Suggested primary category: **eess.SY** (Systems and Control) or **cs.CE**
   (Computational Engineering). Secondary: **cs.SY**, **stat.AP**.
5. Add the arXiv ID to `README.md` and to the paper's `\section*{Data and Code Availability}`.

## 6. Verify everything is reachable

After push + Zenodo release, visit each of these URLs and confirm they load:

- [ ] `https://github.com/preciousanas/sumo_emission_project` — repo homepage renders README
- [ ] `https://github.com/preciousanas/sumo_emission_project/blob/main/paper1_TRIP.tex` — manuscript
- [ ] `https://github.com/preciousanas/sumo_emission_project/blob/main/scripts/publication_figures_ext.py` — one of the analysis scripts
- [ ] Zenodo page for the release (linked from the "Zenodo" badge in your README)

## 7. Update the paper

Once the Zenodo DOI is minted, edit two lines in `paper1_TRIP.tex`:

```latex
%% In the "Data and Code Availability" section:
%%   ... permanently archived on Zenodo (DOI to be assigned upon manuscript acceptance).
%% Change to:
%%   ... permanently archived on Zenodo, DOI: 10.5281/zenodo.XXXXXXX.
```

Rerun `pdflatex/bibtex/pdflatex/pdflatex` and re-upload the PDF to arXiv
(and later to the TRIP submission system).

---

## Notes on protecting sensitive paths

The `.gitignore` in this folder already prevents virtual-env directories
(`paper1env/`, `venv/`) and personal outputs (`outputs/`, `tmp/`) from being
committed. If your local scripts reference **any absolute paths on your
personal machine** (e.g. `C:\Users\panavberokhai\...`), scan the codebase
once before pushing:

```bash
grep -rn "C:\\\\Users\|/home/[a-z]" scripts/ experiments/ --include="*.py" | head
```

If any matches appear, replace with relative paths or environment variables
(`Path(__file__).resolve().parents[1]` is the standard idiom used elsewhere
in the codebase).
