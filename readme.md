# TriSent – Multiclass Sentiment Analysis System

## 1. Project Overview

**TriSent** is a lightweight, production-style sentiment analysis system that classifies any given text into **three sentiment categories**:

- 😊 Positive  
- 😞 Negative  
- 😐 Mixed / Neutral

TriSent is a production-style, human-in-the-loop sentiment analysis system that classifies text into positive, negative, and mixed categories using classical machine learning.

The system follows a **human-in-the-loop learning approach**, where user feedback is collected during prediction and later used to improve the model in a controlled and safe manner.

---

## 2. Purpose of the Project

The goals of this project are:

- To understand how **real machine learning systems are designed**, not just trained once
- To demonstrate the separation between **training, prediction, and retraining**
- To build a **simple, explainable, CPU-only ML system** suitable for low-end machines
- To design a sentiment system that can be integrated into other backend projects (e.g., comment analysis)

This project prioritizes **system design and workflow correctness** over chasing high accuracy scores.

---

## 3. Key Features

- Multiclass sentiment classification (Positive / Negative / Mixed)
- Works on any general text (comments, posts, feedback)
- Fully offline and CPU-only (runs smoothly on i3 systems)
- Fast predictions using a pre-trained model
- Human feedback collection for continuous improvement
- Clear separation of concerns (training vs inference)
- Production-inspired ML workflow

---

## 4. Technology Stack

- **Language:** Python  
- **ML Library:** scikit-learn  
- **Text Representation:** TF-IDF Vectorization  
- **Model:** Logistic Regression (multiclass)  
- **Model Persistence:** joblib  
- **Data Format:** Plain text (`.txt`) with pipe (`|`) separation

No deep learning frameworks are used to keep the system lightweight and explainable.

---

## 5. Project Structure

```
TriSent/
│
├── data/
│   ├── sentiment.txt        # Trusted, admin-labeled training data
│   └── feedback.txt         # User feedback (untrusted, temporary)
│
├── models/
│   ├── vectorizer.joblib    # Saved TF-IDF vectorizer
│   └── sentiment_model.joblib  # Trained ML model
│
├── train.py                 # Initial / base training script
├── predict.py               # Fast prediction + feedback collection
├── retrain.py               # Controlled retraining with feedback
└── README.md                # Project documentation
```

---

## 6. High-Level System Flow

```
Admin Data → Training → Model Ready
                      ↓
                User Prediction
                      ↓
                Feedback Collected
                      ↓
              Controlled Retraining
```

The model **does not learn automatically** during prediction. Learning only happens when retraining is explicitly triggered.

---

## 7. Data Handling Strategy

### 7.1 Sentiment Categories

- **Positive:** Clear appreciation or liking  
- **Negative:** Clear dissatisfaction or dislike  
- **Mixed:** Ambiguous, neutral, or mixed emotions

Example:
```
positive|i really like this
negative|this was disappointing
mixed|good idea but poor execution
```

---

### 7.2 Two Types of Data

#### 1. Trusted Data (Admin-Labeled)
- Stored in `data/sentiment.txt`
- Manually reviewed and labeled by the admin
- Used directly for training and retraining

#### 2. Untrusted Data (User Feedback)
- Stored in `data/feedback.txt`
- Collected during prediction
- Not used for training until explicitly merged

This separation prevents noisy or incorrect user input from corrupting the model.

---

## 8. Script Responsibilities

### 8.1 `train.py` – Initial Training

**Purpose:**
Creates the base model using trusted data.

**What it does:**
- Reads `sentiment.txt`
- Converts text into numerical features using TF-IDF
- Trains a multiclass Logistic Regression model
- Saves the trained model and vectorizer to disk

**When to run:**
- First-time project setup
- After manually adding clean data to `sentiment.txt`

---

### 8.2 `predict.py` – Prediction & Feedback Collection

**Purpose:**
Used for daily prediction and demo.

**What it does:**
- Loads the pre-trained model and vectorizer
- Predicts sentiment instantly (no retraining)
- Allows the user to correct the prediction using numeric input
- Stores verified feedback in `feedback.txt`

**Important:**
The model does **not** learn during prediction.

---

### 8.3 `retrain.py` – Controlled Retraining

**Purpose:**
Safely updates the model using collected feedback.

**What it does:**
- Merges `feedback.txt` into `sentiment.txt`
- Clears the feedback buffer
- Calls `train.py` to retrain the model
- Saves updated model artifacts

**Why controlled retraining matters:**
- Prevents noisy learning
- Ensures only reviewed feedback affects the model

---

## 9. Correct Usage Workflow

| Scenario | Action |
|--------|--------|
| First-time setup | Add data to `sentiment.txt` → run `train.py` |
| Normal usage / demo | Run `predict.py` |
| Collect user feedback | Happens automatically via `predict.py` |
| Improve model | Run `retrain.py` |

---

## 10. Why This Design Is Production-Inspired

Real-world ML systems follow these principles:

- Training is expensive → done offline
- Prediction must be fast → model is pre-loaded
- User feedback is noisy → stored separately
- Retraining is intentional → not automatic

TriSent applies these principles at a small, educational scale.

---

## 11. Limitations

- The model depends heavily on the quality of labeled data
- Very short or ambiguous text may be classified as mixed
- Accuracy is not the primary focus; **system design is**

These limitations are expected in classical ML systems.

---

## 12. Future Improvements

- Confidence scores for predictions
- REST API for backend integration
- Hinglish / multilingual support
- Model versioning (v1, v2, v3)
- Integration with other backend projects

---

## 13. Conclusion

TriSent is not just a sentiment classifier; it is a **complete machine learning workflow** demonstrating:

- Proper separation of concerns
- Human-in-the-loop learning
- Production-style ML thinking
- Practical real-world constraints

The project emphasizes **how ML systems should be built**, not just how models are trained.

---
