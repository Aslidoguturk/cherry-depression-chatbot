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


This project was developed as a research prototype using **restricted datasets**.

The code in this repository expects several files that are **not stored on GitHub**, for privacy and licensing reasons:

- **Original datasets** (multiple sources) that I combined into a new dataset.
- The **final combined dataset** (e.g. `X_combined.npy`, `y.npy`, CSV files).
- The **trained model weights** (e.g. `.h5` / `.keras` file).
- The **tokenizer** object used during training (e.g. `tokenizer.pickle`).
- The **GloVe embeddings file** (e.g. `glove.6B.100d.txt`).

Because of this, someone who clones the repository will **not be able to fully train or reproduce the model immediately.**  
However, the code is provided so others can see the preprocessing steps, model architecture, and testing approach.

### How to (approximately) reproduce the setup

To run the full pipeline, a user would need to:

1. **Obtain the original datasets** from their official sources (they require permission).
2. **Follow the preprocessing steps** in `data_preprocessing.py` / `preprocessing_utils.py`  
   to build their own combined dataset in the expected format.
3. Download **GloVe 6B 100d embeddings** and place the file (e.g. `glove.6B.100d.txt`)  
   in the project directory.
4. Run the training script (e.g. `training_model.py`) to create:
   - a new trained model file  
   - a new tokenizer file  

After that, they can run the Streamlit app (`cherryapp.py`) using their own data and model.


