# Context Effects in Discrete Choice Models

M.Tech Project (I2MP DSAI), IIT Madras
Advisor: Prof. Suryanarayana Sankagiri

Studying the Contextual Deviation Model (CDM) from Seshadri et al. (2019), building up from MLE and MNL, and testing IIA violations on real choice data...

## Structure

```
.
├── src/                 code and experiments
├── cdm-icml/             reference paper and notes
├── choix-practice/       notebooks for the choix library
├── choix/                local copy of the choix library
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy pandas choix --break-system-packages
```


## References

- Seshadri, Peysakhovich, Ugander (2019). *Discovering Context Effects from Raw Choice Data.* ICML.
