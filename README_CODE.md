# 🍒 Cherry – Depression Detection Chatbot

A multimodal mental health support tool combining sentiment analysis, user text, and PHQ-8 scoring to help detect early signs of depression.

---

## 🗂️ Repository Structure

### 📁 Root Files

| File                                                                       | Description |
|----------------------------------------------------------------------------|-------------|
| `cherryapp.py`                                                             | **Main app** – Streamlit-based chatbot UI that handles conversation, PHQ-8 form, and result display. |
| `data_preprocessing.py`                                                    | Cleans, augments, and processes text + sentiment + PHQ-8 data. Saves combined arrays. |
| `training_model.py`                                                        | Trains the Bi-LSTM model using GloVe embeddings and multi-input (text + sentiment + PHQ). |
| `preprocessing_utils.py`                                                   | Helper functions: text cleaning, synonym augmentation, etc. |
| `1combinetrainanddev.py`                                                   | Combines DAIC-WOZ `train` and `dev` CSVs into a single set. |
| `2prepare_dataset.py`                                                      | Prepares and filters transcript text into training format. |
| `glove.6B.100d.txt`                                                        | GloVe embeddings (100D) – used to initialise embedding layer. |
| `tokenizer.pickle`                                                         | Tokenizer used to encode text consistently. |
| `lstm_depression_model_with_phq.keras`                                     | Final trained model saved in Keras format. |
| `X_combined.npy`, `y.npy`                                                  | Preprocessed feature matrix and labels for model training. |
| `Lastt_training_data_combined_extended.csv`                                | Final processed dataset with all combined fields. |
| `train_split_Depression_AVEC2017.csv`, `dev_split_Depression_AVEC2017.csv` | Source data from DAIC-WOZ corpus. |
| `combined_user_text_try.csv`, `combined_sorted_Depression.csv`             | Intermediate CSVs generated during cleaning. |
| `README_CODE.md`                                                           | Technical overview and instructions. |

### 📁 tests

Contains unit and integration tests for core logic:

- `test_model_prediction.py`: Verifies predictions fall within 0–1 range.
- `test_preprocessing.py`: Checks data cleaning and augmentation functions.
- `test_integration_pipeline.py`: End-to-end input → prediction pipeline.
- `test_chatbot_extra.py`: Covers additional chatbot components.
- `test_ui.py`: UI load test using Selenium.
- `test_phq_label.py`: Ensures PHQ score-to-label mapping is accurate.

---

## ▶️ How to Run Cherry Locally

1. **Install dependencies**  
   (Recommended: Use virtualenv or conda)  
   ```bash
   pip install -r requirements.txt
   
   ``bash
    py -m pip install streamlit

    .\venv\Scripts\python.exe -m streamlit run cherryapp.py

