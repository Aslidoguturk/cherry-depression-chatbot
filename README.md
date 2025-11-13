# 🍒 Cherry: Depression Screening Chatbot

Cherry is a **Streamlit-based chatbot** designed as a research prototype for **early depression screening**.  
It combines **Natural Language Processing (NLP)** techniques, **sentiment analysis**, and the **PHQ-8 questionnaire** to analyze user text responses.

> ⚠️ *This project is for educational and research purposes only. It is **not a medical or diagnostic tool.***


## 🚀 Features
- Chatbot interface built with **Streamlit**
- Text preprocessing and sentiment extraction using **VADER**
- Depression classification using a **Bi-LSTM** model with **GloVe embeddings**
- Integration with **PHQ-8 questionnaire** scores
- Modular design with separate preprocessing, training, and testing scripts
- Unit tests for major components



## 🧠 Tech Stack
- **Language:** Python  
- **Frameworks/Libraries:** TensorFlow, Streamlit, NumPy, Pandas, Scikit-learn, NLTK, VADER Sentiment  
- **Model:** Bi-LSTM with GloVe 100d embeddings  
- **Dataset:** DAIC-WOZ / AVEC 2017 (used with permission — not included)



## ⚙️ How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/Aslidoguturk/cherry-depression-chatbot.git
cd cherry-depression-chatbot


📊 Dataset Information

The project uses the DAIC-WOZ / AVEC 2017 dataset, which requires permission for access due to privacy constraints.
Therefore, the dataset and model weights are not included in this repository.

