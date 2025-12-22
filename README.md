# MULTI-AUTHOR WRITING STYLE ANALYSIS

This is the repo for Group 9's exam project for the course Advanced Natural Language Processing and Deep Learning at ITU (2025/2026).

The project is based on the PAN (2025) shared task and uses the same training and validation data for the monolingual classification task.
We experiment with various configurations of adapter-based fine-tuning of the multilingual modern BERT model for authorship-change detection, and we evaluate multilingual (zero-shot) transfer without training on the target languages.

## Repository structure

- Root folder contains training scripts used for different fine-tuning configurations (Jupyter notebooks) as well as `test.ipynb` used for evaluation.
- `Data/` holds the PAN shared-task training/validation data and multilingual evaluation data (scraped from language-specific subreddits).
- `predictions/` holds prediction files produced by the different model configurations.
- `Evaluation/` holds validation data and evaluation scripts.
- `results/` holds `.txt` files generated from the evaluation script (organized by model configuration).
- `trained_adapters/` stores trained adapters for reproducibility.

Naming convention example:
`mmbert_lora_lang_2lcls` = mmBERT + LoRA + 2-layer head, trained on top of a previously trained language adapter.

---

## Analysis notebook (custom scripts)

The notebook `analysis_stats.ipynb` contains custom scripts used for the Analysis section of the report and Supplementary Information D.
The notebook is located at:
- `scripts/analysis_stats.ipynb`

To reproduce the results from this notebook, all required data files (both PAN data and the newly compiled Reddit datasets) must be placed in the same folder as `analysis_stats.ipynb`, **or** the data directory paths inside the notebook must be updated accordingly.

---

## Reddit scraping (multilingual evaluation data)

We generate multilingual evaluation datasets (Danish, Italian, Polish) by scraping Reddit and saving the data in PAN-style format:
- `problem-<id>.txt` (one sentence per line)
- `truth-problem-<id>.json` (fields: `authors`, `changes`)

The custom scraping script is placed at:
- `scripts/scrape_reddit_panstyle.py`

---

### Credit
Scraping uses YARS: [datavorous/yars](https://github.com/datavorous/yars)

The analysis notebook uses HurtLex: [valeriobasile/hurtlex](https://github.com/valeriobasile/hurtlex)

---

### Install YARS dependencies
Recommended:
```bash
pip install -r requirements.txt



