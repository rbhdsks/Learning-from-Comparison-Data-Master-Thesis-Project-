# Context Effects in Discrete Choice Models

MTP, M.Tech I2MP DSAI, IIT Madras. Advisor: Prof. Suryanarayana Sankagiri.

Most choice models assume that adding an irrelevant option to the menu doesn't
change how you feel about the rest. People don't actually work that way. This
repo follows Seshadri, Singh and Ugander (ICML 2019) on models that drop that
assumption.

`bradley-terry.py` generates synthetic pairwise comparisons from known item
strengths and fits them back by maximum likelihood, as a sanity check on the
basic machinery. The paper and reading notes are in `cdm-icml/`.

```bash
pip install -r requirements.txt
python bradley-terry.py
```

Work in progress.
