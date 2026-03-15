# Masked Intent Inference Challenge – Text Analysis & Feature Extraction

## Overview

This project explores the **Masked Intent Inference Challenge** dataset, which contains social media sentences labeled across **49 intent categories**. The main goal is to perform text preprocessing, exploratory data analysis, and extract **category-specific keywords** using **TF-IDF**, along with visualizing them via **word clouds**.

---

## Dataset

The dataset contains three main CSV files:

| File                    | Description                                                 |
| ----------------------- | ----------------------------------------------------------- |
| `train.csv`             | Training set with `id`, `sentence`, and `category` columns. |
| `test.csv`              | Test set with `id` and `sentence` columns.                  |
| `sample_submission.csv` | Example submission file with `id` and `prediction` columns. |

---

## Project Structure

```text
/project-folder
│
├── Solution.ipynb  # Jupyter notebook with TF-IDF workflow
└── README.md                 # Project documentation
```

---

## Dependencies

The project uses **Python 3** and the following libraries:

* `numpy`
* `pandas`
* `matplotlib`
* `scikit-learn` (`TfidfVectorizer`, `ENGLISH_STOP_WORDS`)
* `wordcloud`
* `re` (regular expressions)
* `os` (file handling)

Install missing dependencies via:

```bash
pip install numpy pandas matplotlib scikit-learn wordcloud
```

---

## Workflow

1. **Data Loading**
   Load `train.csv` and `test.csv` using `pandas`.

2. **Exploratory Data Analysis**

   * Check the number of categories and their distribution.
   * Inspect rare and common categories.
   * Sample sentences from each category for inspection.

3. **Text Preprocessing**

   * Lowercase conversion.
   * Tokenization using regex.
   * Removal of English stopwords (`sklearn.feature_extraction.text.ENGLISH_STOP_WORDS`).

4. **Keyword Extraction with TF-IDF**

   * Create a TF-IDF vectorizer (`ngram_range=(1,2)`, `max_features=5000`).
   * Compute TF-IDF for combined sentences per category.
   * Extract top N (default 15) keywords for each category.

5. **Visualization**

   * Word count distribution histograms per category.
   * **Word clouds** for top keywords.

---

## Generating Word Clouds

To visualize the top keywords for each category:

```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

for category, keywords in category_keywords.items():
    text = " ".join(keywords)
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Word Cloud for Category: {category}")
    plt.show()
```

* This will create a word cloud for each category using the extracted **top TF-IDF keywords**.
* Larger words represent higher importance in the category.

---

## Example: Top Keywords for Selected Categories

| Category      | Top Keywords                                                                                                                                    |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| APR           | `['faggot', 'tranny', 'im', 'like', 'just', 'dont', 'people', 'trannies', 'youre', 'know', 'trans', 'word', 'say', 'uj', 'gay']`                |
| Remorse       | `['sorry', 'im', 'im sorry', 'regret', 'dont', 'oh', 'didnt', 'just', 'like', 'bad', 'meant', 'apologies', 'feel', 'right', 'youre']`           |
| Embarrassment | `['im', 'awkward', 'just', 'bad', 'oh', 'like', 'dont', 'embarrassment', 'weird', 'ashamed', 'feel', 'embarrassed', 'thats', 'youre', 'sorry']` |

---

## Example Plot:
![APR sample plot](https://github.com/tltommu/Masked-Intent-Inference-Challenge/blob/main/Screenshot%202026-03-16%20035433.png)

## Notes

* Some sentences contain multilingual or special characters that may produce warnings in Matplotlib.
* Rare categories have very few examples (e.g., `idk/skip` has only 52 sentences), which may affect keyword extraction.

---

---


