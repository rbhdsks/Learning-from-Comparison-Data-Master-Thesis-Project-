# Learning from Comparison Data

Research code and [notes](https://www.overleaf.com/project/6a192de01f9877e35710e1d1) for my M.Tech thesis on learning preference models from observed choice data. 
My project begun with understanding pairwise comparisons where now I am developing a good understanding of the underlying probability and optimization methods from the first principles.


## Repository structure

| Path                              | Purpose                                          |
| --------------------------------- | ------------------------------------------------ |
| `notebooks/foundations/`          | Mathematical derivations and small examples      |
| `notebooks/practice/choix/`       | Practice notebooks using and from Choix library  |
| `notebooks/experiments/`          | Main thesis experiments                          |
| `src/bradley_terry/`              | Not started yet                                  |
| `docs/`                           | Research notes and report material               |
| `references/paper_code/cdm_icml/` | Code accompanying the main reference paper.      |
| `references/libraries/choix/`     | Local reference copy of the Choix library        |

Code under `references/` belongs to its original authors. It is kept separate from the implementation developed in this project.

The notebooks in `notebooks/experiments/` are numbered in their intended reading order.


## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## References

* Arjun Seshadri, Alex Peysakhovich, and Johan Ugander. [Discovering Context Effects from Raw Choice Data](https://proceedings.mlr.press/v97/seshadri19a.html). ICML, 2019.
* [Choix](https://github.com/lucasmaystre/choix): inference tools for models based on Luce's choice axiom.

## Status

This repository is under active development as part of my code for M.Tech thesis. Implementations, experiments, and documentation will change as the research progresses.

## Project
Supervisor: Prof. Suryanarayana Sankagiri
Name : **Nitesh Kumar Shah**
M.Tech Project
ID25M806
Department of Data Science & AI.
IIT Madras.
