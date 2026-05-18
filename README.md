# Email Spam Detection System
**Olarinoye Omowumi Racheal | Matric No: 2022/43935**  
*Osun State University, Osogbo – B.Sc. Computer Science Final Year Project*  
*Supervisor: Dr. P. Ozoh*

---

## Overview

A machine learning-based email spam detection system implementing and comparing five supervised learning algorithms (Naïve Bayes, Decision Tree, Random Forest, SVM, Logistic Regression) with TF-IDF feature extraction, as described in Chapter 3 of the project report.

---

## Project Structure

```
email_spam_detection/
│
├── main.py                         ← End-to-end pipeline entry point
├── requirements.txt                ← Python dependencies
├── README.md
│
├── data/
│   └── spam.csv                    ← Place dataset here (UCI/Kaggle)
│
├── diagrams/
│   └── chapter3_uml_diagrams.xml   ← Draw.io UML diagrams (import this)
│
├── models/
│   └── saved/                      ← Fitted models (.pkl) saved here
│
├── reports/
│   └── figures/                    ← All generated plots saved here
│
├── src/
│   ├── config.py                   ← Central configuration (paths, hyperparams)
│   │
│   ├── data/
│   │   └── data_loader.py          ← DataLoader class (Section 3.2)
│   │
│   ├── features/
│   │   ├── preprocessor.py         ← TextPreprocessor class (Section 3.3)
│   │   └── feature_extractor.py    ← FeatureExtractor / TF-IDF (Section 3.3)
│   │
│   ├── models/
│   │   ├── base_classifier.py      ← Abstract BaseClassifier
│   │   ├── classifiers.py          ← 5 concrete classifiers (Section 3.4)
│   │   └── model_trainer.py        ← ModelTrainer / CV / GridSearch (Section 3.5)
│   │
│   ├── evaluation/
│   │   └── evaluator.py            ← ModelEvaluator / metrics / plots (Section 3.6)
│   │
│   └── utils/
│       └── eda.py                  ← EDAAnalyser (Section 3.2 / Objectives)
│
└── tests/
    └── test_suite.py               ← Unit tests (run with pytest)
```

---

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the Dataset

**Option A – UCI SMS Spam Collection (recommended)**
```
https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
```
Save the CSV as `data/spam.csv`.

**Option B – Kaggle Email Spam Dataset**
```
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
```

The `DataLoader` auto-detects common column names (`v1`/`v2`, `label`/`message`, etc.).

---

## Running the Pipeline

### Full Pipeline (default hyperparameters)
```bash
python main.py
```

### Full Pipeline + Hyperparameter Optimisation
```bash
python main.py --optimise
```

### EDA Only
```bash
python main.py --eda-only
```

### Single Email Prediction
```bash
python main.py --predict "Congratulations! You have won a free prize."
```

---

## Running the Tests
```bash
pytest tests/test_suite.py -v
```

---

## UML Diagrams (Draw.io)

Open `diagrams/chapter3_uml_diagrams.xml` in [draw.io](https://app.diagrams.net/):

1. Go to **File → Import from → Device**
2. Select `chapter3_uml_diagrams.xml`
3. The file contains **four diagrams** (scroll vertically):
   - **Use Case Diagram** – system actors and use cases
   - **Activity Diagram** – end-to-end pipeline flow
   - **Class Diagram** – all classes, attributes, methods, and relationships
   - **Sequence Diagram** – email classification interaction flow

---

## Methodology Summary (Chapter 3)

| Stage | Component | Key Choices |
|-------|-----------|-------------|
| Data Collection | `DataLoader` | UCI Spam Dataset, auto-column detection |
| Preprocessing | `TextPreprocessor` | Lowercase, punctuation removal, stop-words, Porter stemming |
| Feature Extraction | `FeatureExtractor` | TF-IDF (max 5,000 terms, unigrams+bigrams), chi-squared selection (top 3,000) |
| Model Selection | `classifiers.py` | NB, DT, RF, SVM (linear), LR |
| Training | `ModelTrainer` | 80/20 stratified split, 10-fold CV, optional GridSearchCV |
| Evaluation | `ModelEvaluator` | Accuracy, Precision, Recall, F1, AUC; Confusion Matrix; ROC Curve |

---

## Expected Results

Based on the literature reviewed in Chapter 2, expected performance ranges are:

| Metric | Expected Range |
|--------|---------------|
| Accuracy | 95–99% |
| Precision | 93–99% |
| Recall | 90–98% |
| F1-Score | 92–98% |
| AUC | 0.97–0.999 |

---

## Configuration

All key settings are centralised in `src/config.py`:

```python
MAX_FEATURES  = 5000    # TF-IDF vocabulary size
NGRAM_RANGE   = (1, 2)  # unigrams + bigrams
CHI2_K        = 3000    # features retained after chi-squared selection
TEST_SIZE     = 0.20    # 80/20 train/test split
CV_FOLDS      = 10      # stratified k-fold cross-validation
SCORING_METRIC= "f1"    # optimisation target
```
