# 🔍 Quora Question Pair — Duplicate Question Detector

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NLP](https://img.shields.io/badge/NLP-FuzzyWuzzy%20%7C%20NLTK-green)](https://pypi.org/project/fuzzywuzzy/)


> A machine learning–powered NLP application that detects whether two Quora questions are semantically **duplicate** or **not**. Built with advanced feature engineering, trained on **404,290** question pairs, and deployed as an interactive **Streamlit** web app.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Feature Engineering](#-feature-engineering)
- [Model](#-model)
- [Streamlit Web App](#-streamlit-web-app)
- [Installation](#-installation)
- [Usage](#-usage)
- [Tech Stack](#-tech-stack)
- [Results](#-results)
- [Future Improvements](#-future-improvements)

---

## 🧠 Overview

Quora receives thousands of questions every day — many of which are semantically identical but phrased differently. This project addresses that challenge by building a **binary text classification** system that:

- Preprocesses raw question pairs using NLP techniques
- Engineers **22+ handcrafted features** (token-based, length-based, and fuzzy features)
- Combines them with **Bag-of-Words (BoW)** representations
- Predicts whether a pair of questions is **duplicate (1)** or **not duplicate (0)**

---

## 📊 Dataset

| Property             | Value               |
|----------------------|---------------------|
| Source               | [Quora Question Pairs (Kaggle)](https://www.kaggle.com/c/quora-question-pairs) |
| Total Pairs          | 404,290             |
| Duplicate Pairs      | 149,263 (~36.9%)    |
| Non-Duplicate Pairs  | 255,027 (~63.1%)    |
| Missing Values       | 3 rows (negligible) |
| Duplicate Rows       | 0                   |

**Dataset Columns:**

| Column         | Description                                 |
|----------------|---------------------------------------------|
| `id`           | Row ID                                      |
| `qid1`         | Unique ID of Question 1                     |
| `qid2`         | Unique ID of Question 2                     |
| `question1`    | Full text of Question 1                     |
| `question2`    | Full text of Question 2                     |
| `is_duplicate` | 1 = Duplicate, 0 = Not Duplicate (target)   |

---

## 📁 Project Structure

```
Quora-Question-Pair/
│
├── eda.ipynb                          # Exploratory Data Analysis
├── bow.ipynb                          # Bag-of-Words baseline model
├── bow_with_basic_features.ipynb      # BoW + basic feature engineering
├── Adv_feat_with_preprocessing.ipynb  # Advanced features + full preprocessing pipeline
│
├── train.csv                          # Raw training dataset
├── model.pkl                          # Trained ML model (serialized)
├── cv.pkl                             # CountVectorizer (serialized)
│
└── streamlit-app/
    ├── app.py                         # Streamlit web application
    ├── helper.py                      # Feature extraction & preprocessing logic
    ├── model.pkl                      # Model (copy for deployment)
    └── cv.pkl                         # Vectorizer (copy for deployment)
```

---

## ⚙️ Feature Engineering

The feature set is carefully crafted across **three categories**:

### 1. 🔤 Basic Features
| Feature                  | Description                                      |
|--------------------------|--------------------------------------------------|
| `q1_len`                 | Character length of Question 1                  |
| `q2_len`                 | Character length of Question 2                  |
| `q1_num_words`           | Word count of Question 1                        |
| `q2_num_words`           | Word count of Question 2                        |
| `common_words`           | Number of words common to both questions        |
| `total_words`            | Total words in both questions                   |
| `word_share`             | Ratio of common words to total words            |

### 2. 🧩 Token Features
| Feature                     | Description                                                |
|-----------------------------|------------------------------------------------------------|
| `cwc_min`                   | Common non-stopwords / min(q1 words, q2 words)            |
| `cwc_max`                   | Common non-stopwords / max(q1 words, q2 words)            |
| `csc_min`                   | Common stopwords / min(q1 stops, q2 stops)                |
| `csc_max`                   | Common stopwords / max(q1 stops, q2 stops)                |
| `ctc_min`                   | Common tokens / min(q1 tokens, q2 tokens)                 |
| `ctc_max`                   | Common tokens / max(q1 tokens, q2 tokens)                 |
| `last_word_eq`              | 1 if last words of both questions match                   |
| `first_word_eq`             | 1 if first words of both questions match                  |

### 3. 📏 Length Features
| Feature                  | Description                                           |
|--------------------------|-------------------------------------------------------|
| `abs_len_diff`           | Absolute difference in token lengths                 |
| `mean_len`               | Average token length of both questions               |
| `longest_substr_ratio`   | Length of longest common substring / min(len(q1), len(q2)) |

### 4. 🔀 Fuzzy Features
| Feature              | Description                                    |
|----------------------|------------------------------------------------|
| `fuzz_ratio`         | Simple ratio of character-level similarity    |
| `fuzz_partial_ratio` | Best partial match ratio                      |
| `token_sort_ratio`   | Ratio after sorting tokens alphabetically     |
| `token_set_ratio`    | Ratio using set intersection of tokens        |

### 5. 📦 Bag-of-Words (BoW) Features
- Separate BoW vectors for **Question 1** and **Question 2** using `CountVectorizer`
- Combined with the 22 handcrafted features into a single feature vector

---

## 🤖 Model

- **Algorithm:** Random Forest Classifier (or equivalent — serialized in `model.pkl`)
- **Vectorizer:** `CountVectorizer` (serialized in `cv.pkl`)
- **Input:** Combined feature vector (22 engineered features + BoW for Q1 + BoW for Q2)
- **Output:** `1` (Duplicate) or `0` (Not Duplicate)

### Preprocessing Pipeline
The `preprocess()` function in `helper.py` handles:
- Lowercasing and stripping whitespace
- Replacing special characters (`%`, `$`, `₹`, `€`, `@`)
- Expanding number abbreviations (e.g., `1,000` → `1k`)
- Decontracting English contractions (`can't` → `can not`)
- Removing HTML tags using `BeautifulSoup`
- Removing punctuation

---

## 🌐 Streamlit Web App

An interactive web application that lets users check if two questions are duplicates in real time.

### How it works:
1. User enters **Question 1** and **Question 2**
2. The app preprocesses the inputs and extracts all features
3. The trained model predicts: **"Duplicate"** or **"Not Duplicate"**

### Run the app locally:
```bash
cd streamlit-app
streamlit run app.py
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.9+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Divyansh-Parihar/Quora-Question-pair.git
cd Quora-Question-Pair

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
cd streamlit-app
streamlit run app.py
```

### Required Libraries

```
streamlit
scikit-learn
numpy
pandas
matplotlib
seaborn
nltk
beautifulsoup4
fuzzywuzzy
python-Levenshtein
distance
```

> **Note:** Download NLTK stopwords before running:
> ```python
> import nltk
> nltk.download('stopwords')
> ```

---

## 🚀 Usage

### Option 1: Streamlit App
```bash
cd streamlit-app
streamlit run app.py
```
Navigate to `http://localhost:8501` and enter two questions to check for duplicates.

### Option 2: Notebooks

| Notebook | Description |
|----------|-------------|
| `eda.ipynb` | Explore the dataset — distribution, missing values, repeated questions |
| `bow.ipynb` | Baseline BoW model |
| `bow_with_basic_features.ipynb` | BoW + basic features model |
| `Adv_feat_with_preprocessing.ipynb` | Full pipeline with advanced feature engineering |

---

## 🧰 Tech Stack

| Tool / Library   | Purpose                              |
|------------------|--------------------------------------|
| Python           | Core programming language            |
| Pandas & NumPy   | Data manipulation                    |
| Matplotlib & Seaborn | EDA & visualization              |
| NLTK             | Stopwords, tokenization              |
| BeautifulSoup4   | HTML tag removal                     |
| FuzzyWuzzy       | Fuzzy string matching features       |
| Distance         | Longest common substring computation |
| scikit-learn     | ML model & CountVectorizer           |
| Streamlit        | Web app deployment                   |
| Pickle           | Model serialization                  |

---

## 📈 Results

### Dataset Statistics

<img width="901" height="770" alt="image" src="https://github.com/user-attachments/assets/c9285ffd-5e08-48c3-9376-13785379035c" />

| Metric             | Value          |
|--------------------|----------------|
| Dataset Size       | 404,290 pairs  |
| Unique Questions   | 537,933        |
| Repeated Questions | 111,780        |
| Features Used      | 22 + BoW       |
| Test Set Size      | 9,000 samples  |

### Model Performance (on 9,000-sample test set)

| Model              | Accuracy  | Weighted F1 |
|--------------------|-----------|-------------|
| Random Forest      | **78.87%**| **0.79**    |
| XGBoost            | **78.80%**| —           |

### Random Forest — Classification Report

| Class             | Precision | Recall | F1-Score | Support |
|-------------------|-----------|--------|----------|---------|
| 0 (Not Duplicate) | 0.81      | 0.86   | 0.84     | 5,707   |
| 1 (Duplicate)     | 0.74      | 0.66   | 0.70     | 3,293   |
| **Weighted Avg**  | **0.79**  | **0.79**| **0.79**| 9,000   |

### Confusion Matrix (Random Forest)

|                    | Predicted: Not Dup | Predicted: Dup |
|--------------------|--------------------|----------------|
| **Actual: Not Dup**| 4,926              | 781            |
| **Actual: Dup**    | 1,121              | 2,172          |

---

## 🔮 Future Improvements

- [ ] Add **TF-IDF** features alongside BoW
- [ ] Experiment with **BERT / Sentence Transformers** for semantic similarity
- [ ] Implement **cross-validation** and hyperparameter tuning
- [ ] Deploy on **Streamlit Cloud** or **Hugging Face Spaces**
- [ ] Add evaluation metrics dashboard in the app
- [ ] Add **WMD (Word Mover's Distance)** features

---

## 🙋‍♂️ Author

**Divyansh Parihar**  
📧 [Email]divyanshparihar6505@gmail.com
🔗 [LinkedIn]https://www.linkedin.com/in/divyansh-parihar-836765334/  
🐙 [GitHub]https://github.com/Divyansh-Parihar

---

> ⭐ If you found this project useful, please consider giving it a star!
