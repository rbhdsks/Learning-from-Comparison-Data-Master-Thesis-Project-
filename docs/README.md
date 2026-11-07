# Documentation

This directory contains the thesis source, research notes, references,
figures, and documentation assets used by the project.

## Directory structure

| Path | Purpose |
| --- | --- |
| `thesis/Notes.tex` | Main thesis document |
| `thesis/weekly-log.tex` | Weekly research log |
| `thesis/references.bib` | Bibliographic references |
| `thesis/images/` | Figures included in the thesis |
| `thesis/tables/` | Generated tables for thesis experiments |
| `thesis/papers/` | Supporting papers and reading material |
| `thesis/archive/` | Earlier versions preserved for reference |
| `assets/readme/` | Images displayed in the repository README |

The contents of `docs/thesis/` are synchronized with Overleaf.
The files under `docs/assets/` are used by the GitHub documentation and
are not pushed to Overleaf.

## Experiment figures and tables

Figures are grouped under `thesis/images/experiments/` and tables under
`thesis/tables/experiments/`. Current output groups include:

| Output group | Purpose |
| --- | --- |
| `context_dependent_model/` | Full-CDM recovery, cross-model fits, unified comparisons, and resource scaling |
| `context_dependent_model/foundations/` | BTL–CDM probability comparisons, the constructed cyclic-preference example, and parameter growth |
| `cdm_paper_reproduction/01_synthetic_recovery/` | Synthetic recovery figures and tables, including extended observation-count runs |
| `cdm_paper_reproduction/02_synthetic_iia_tests/` | IIA pilot figures and diagnostic outputs, where generated |

Numerical results, settings, and fit diagnostics are maintained separately
under `results/experiments/` at the repository root. They are not part of
the `docs/thesis/` Overleaf subtree. Some experiments write additional
summary CSV files alongside thesis tables.

The cyclic-preference foundation figures demonstrate exact model
relationships. The cycle-detection notebook currently contains a design
and exact checks; noisy-data detection results have not yet been generated.
Paper-reproduction IIA outputs are pilot results and should be described
with their optimization and separation limitations.

See the [notebook guide](../notebooks/README.md) for the corresponding
experiments and their current scope.

## Overleaf documentation

The thesis document is maintained in the following Overleaf project:

[Open the thesis project in Overleaf](https://www.overleaf.com/project/6a192de01f9877e35710e1d1)

To avoid conflicting histories, changes should normally be made in
either VS Code or Overleaf and synchronized before continuing in the
other editor.

## Sync from VS Code to Overleaf

After editing files inside `docs/thesis/`, first confirm that the
document compiles locally. Then copy and run:

```bash
git add docs/thesis

git commit -m ".."

git push

git subtree push --prefix=docs/thesis overleaf main
```

The first push saves the commit to the current GitHub branch. The
subtree command sends only the contents of `docs/thesis/` to the
Overleaf project.

After the push finishes, open Overleaf and recompile the document if
automatic compilation does not start.

## Sync from Overleaf to VS Code

If changes were made directly in the Overleaf editor, first make sure
that the local working tree is clean:

```bash
git status --short
```

No output means that there are no uncommitted local changes.

Then copy and run:

```bash
git subtree pull --prefix=docs/thesis overleaf main --squash

git push
```

The subtree pull imports the latest Overleaf revision into
`docs/thesis/` as a squashed commit. The final command pushes that
commit to the current GitHub branch.

## If both copies were edited

Do not immediately push one copy over the other. First commit or safely
store the local changes, then pull from Overleaf:

```bash
git status

git add docs/thesis

git commit -m "docs: save local thesis changes"

git subtree pull --prefix=docs/thesis overleaf main --squash
```

If Git reports a conflict, resolve the affected files before pushing to
GitHub or Overleaf.

## Tracked thesis files

The repository should track:

- LaTeX source files;
- bibliography files;
- required logos and images;
- final experiment figures used by the thesis.

Temporary LaTeX build files such as `.aux`, `.log`, `.fls`, `.toc`,
`.synctex.gz`, and locally compiled thesis PDFs should not be committed.

Authentication tokens used for Overleaf Git access must never be added
to the repository.
