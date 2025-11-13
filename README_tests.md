# Cherry Chatbot - Test Suite Overview

This document describes the tests written for the Cherry mental health chatbot project. It includes unit tests, integration tests, and optional UI testing using Selenium.
Note: This test suite focuses on core logic such as preprocessing, model prediction, and scoring. 
Some UI and user interaction features (e.g., feedback forms, music embeds, and chat display logic) were not tested, 
as they are difficult to cover with standard unit testing tools and are not critical for the main functionality.

---

## Test Files Overview

| File Name                      | Type               | Description                                                                                |
|--------------------------------|--------------------|--------------------------------------------------------------------------------------------|
| `test_chatbot_extra.py`        | Unit + Integration | Tests unusual inputs (e.g., empty text, emojis, long strings) and checks saving chat logs. |
| `test_integration_pipeline.py` | Integration        | Simulates the full text prediction pipeline from cleaning to model output.                 |
| `test_model_prediction.py`     | Unit               | Verifies that the model produces valid output (0–1) for sample inputs.                     |
| `test_phq_label.py`            | Unit               | Tests the scoring function that maps PHQ-8 scores to depression severity labels.           |
| `test_preprocessing.py`        | Unit               | Tests custom text cleaning and synonym augmentation.                                       |
| `test_ui.py`                   | End-to-End         | Launches the Streamlit app and confirms the web interface loads correctly.                 |

---


## How to Run All Tests

Open a terminal and run:

```bash
pytest tests/




