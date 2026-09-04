# SIH26165 — OIL SIF Precursor NLP

This repository contains a small prototype for classifying SIF potential and
discovering recurring safety precursors from narrative safety reports.

The included `data/synthetic_reports.csv` file is generated for demonstration
only. It is **synthetic data and is NOT official OIL data**. It must not be
presented as a real OIL safety-report dataset.

Generate or regenerate the dataset with:

```bash
python src/data_generator.py
```

Train the saved TF-IDF + Logistic Regression classifier:

```bash
python src/sif_classifier.py
```

Run the Streamlit dashboard:

```bash
python -m streamlit run app.py
```

The demo CSV includes report text, site/location, activity, SIF label and
confidence, Life-Saving Rule, precursor, hazard, barrier information, and a
priority indicator. Reviewer decisions are stored locally in
`data/review_feedback.csv` and do not overwrite AI outputs.

git clone https://github.com/abera-hiluf/SIH-hackaton-NLP-.git
cd SIH-hackaton-NLP-

python -m venv .venv
source .venv/Scripts/activate

python -m pip install -r requirements.txt
python src/data_generator.py
python src/sif_classifier.py

python -m streamlit run app.py
