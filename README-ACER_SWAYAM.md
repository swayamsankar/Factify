# Factify – AI Fake News & Fake Email Detection System

A Unified Multilingual, Multimodal, Explainable & Robust Misinformation Detection Framework

Fake news and phishing emails are two of the biggest digital threats today. Factify solves both using advanced AI models, deep learning, language models, metadata analysis, and interpretable ML techniques.

This system can detect:

- Fake vs Real News Articles
- Phishing vs Legit Emails

Factify achieves high accuracy through a hybrid architecture that combines linguistic, metadata, and multimodal signals.

---

## Proposed Multilingual & Multimodal Approach

Factify uses a multilingual and multimodal AI framework to ensure reliability, explainability, and robustness against manipulation.

### We use these technologies to solve the problem:

**Multilingual Embeddings**  
mBERT, XLM-R for detecting fake news in multiple languages.

**Multimodal Fusion**  
Combines text, images, email metadata, and URLs.

**Explainability (XAI)**  
LIME, SHAP, attention heatmaps.

**Robustness Techniques**  
Adversarial training, noise handling, obfuscation resistance.

---

# System Architecture (End-to-End Pipeline)

Factify follows a unified system pipeline to classify news & emails.  
This architecture is used directly inside the project.

### 1. Input
- News articles (True/Fake datasets, RSS feeds)
- Emails (SMTP sender, subject, body, metadata)

### 2. Preprocessing
- Text cleaning
- Translation & normalization
- Header parsing (for emails)
- URL extraction
- Tokenization

### 3. Feature Extraction
- TF-IDF
- Word embeddings
- Transformer embeddings (mBERT / XLM-R)
- Email metadata (MX/SPF, link count, suspicious keywords)

### 4. Fusion Layer
Combines text, metadata, images, and URL features  
Unified representation for classification.

### 5. Classification
- News: Fake / Real
- Email: Legit / Likely Fake / Fake

Hybrid DL model (LSTM–GRU) + rule-based security checks

### 6. XAI Module
- SHAP
- LIME
- Attention visualization

Provides transparent, interpretable decisions.

---

# Key Features

## Fake News Detector
- 98–99% accuracy
- LSTM–GRU hybrid model
- Transformer-enhanced multilingual support
- EDA & visualization pipeline
- Model versioning and performance monitoring

## Fake Email / Phishing Detector
- MX record check
- SPF record analysis
- Disposable domain blacklist
- Suspicious keyword scoring
- URL & metadata analysis
- Optional ML-based phishing classifier
- HTML interface to check suspicious emails

## Explainability (XAI)
- Word-highlight explanations
- Decision reasoning & score breakdown
- Transparency for users

## Full Web Application
- Flask backend
- HTML + CSS frontend
- API endpoints for integration
- JSON results for automation

## CI/CD + Deployment
- Dockerfile included
- GitHub Actions pipeline
- Render / Railway deployment-ready

---

# Tech Stack

| Component | Technology |
|----------|------------|
| Language | Python 3.9+ |
| ML Framework | TensorFlow / Keras |
| NLP | NLTK, TF-IDF, mBERT, XLM-R |
| Email Security | DNS/MX/SPF, URL parsing |
| Backend | Flask |
| Deployment | Docker, GitHub Actions |
| Visualization | Matplotlib, Seaborn |
| Frontend | HTML, CSS |

---

---

# Model Architecture (Best Performing)

```python
Sequential([
    Embedding(10000, 100),
    LSTM(100, return_sequences=True),
    GRU(100),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
```

Supports multiple variants:
- LSTM–GRU  
- BiLSTM  
- CNN-LSTM  
- Transformer-based embeddings  

---

# Performance Evaluation

```
precision    recall  f1-score   support

0       0.99      0.99      0.99
1       0.99      0.99      0.99

Accuracy: 99%
```

---

# Future Enhancements

- Multilingual news & email support  
- Transformer-only architecture (BERT, RoBERTa)  
- Browser extension integration  
- Real-time phishing detection  
- Explainable AI dashboards  
- Few-shot learning for unseen fake patterns  

---

# Contact

Swayam Sankar  
Email: swayamsankar898@gmail.com  
GitHub: https://github.com/swayamsankar
