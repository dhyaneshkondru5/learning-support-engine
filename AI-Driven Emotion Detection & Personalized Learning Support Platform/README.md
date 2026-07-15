# GenAI Project — AI-Driven Emotion Detection & Personalized Learning Support Platform

Detects a student's emotional state (Bored, Confident, Confused, Curious, Frustrated)
from free-text input using a BiLSTM classifier (and optionally a fine-tuned BERT model),
then generates empathetic, field-aware guidance via Gemini — with graceful fallback
templates if Gemini is unavailable.

## ⚠️ Important: use a short folder path on Windows

Windows has a 260-character path limit. TensorFlow/PyTorch install deep, nested file
trees, so if this project sits somewhere like
`C:\Users\you\Downloads\genai_project (1)\genai_project\venv\...` the install **will fail**
partway through with an `OSError: No such file or directory`.

**Move the project folder to a short path before doing anything else**, e.g.:
```
C:\GenAI_Project
```
So `app.py` ends up at `C:\GenAI_Project\app.py`, not several folders deep in Downloads.

## Quick start (from scratch)

Open a terminal **inside the short-path folder** (`C:\GenAI_Project` or similar) and run:

```bash
cd C:\GenAI_Project

python -m venv venv
venv\Scripts\activate          # cmd.exe
.\venv\Scripts\Activate.ps1    # PowerShell — if blocked, run:
                                #   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

pip install -r requirements.txt

copy .env.example .env         # then edit .env and paste your real GEMINI_API_KEY

streamlit run app.py
```

Your terminal prompt must show `(venv)` at the start before running `pip install` or
`streamlit run` — if it doesn't, the venv isn't activated and packages will install to
the wrong place.

Get a Gemini API key at https://aistudio.google.com/ (Get API Key → Create API Key).

The BiLSTM model is already trained and included in `models/bilstm/` — no training step
needed to get started. Just install dependencies and run.


## Project structure

```
emotion_learning_assistant/
├── app.py                          # Streamlit UI
├── requirements.txt
├── .env.example
├── utils/
│   ├── preprocessing.py            # text cleaning + keyword boost
│   ├── predict.py                  # BiLSTM / BERT inference, unified schema
│   ├── gemini_helper.py            # prompt building + Gemini call + fallback
│   └── logger.py                   # CSV logging + analytics data load
├── scripts/
│   ├── generate_dataset.py         # synthetic training data generator
│   └── train_bilstm.py             # trains + exports the BiLSTM model
├── models/
│   ├── bilstm/                     # trained BiLSTM weights + tokenizer (generated)
│   └── bert_emotion_model_final/   # optional — drop in your fine-tuned BERT here
└── data/
    ├── emotion_dataset.csv         # generated training data
    └── emotion_logs.csv            # generated interaction logs
```

## Notes on the BiLSTM baseline

The included `scripts/generate_dataset.py` builds a synthetic, template-based dataset
so the app is runnable out of the box without an external dataset. It gets very high
accuracy because the templates are clean and repetitive — for real-world robustness,
replace `data/emotion_dataset.csv` with genuine labeled student text (see the Kaggle
training notes in the project documentation) and retrain.

## Adding BERT

The BERT tab activates automatically once `models/bert_emotion_model_final/` contains
a valid HuggingFace `config.json`, weights, and tokenizer files (fine-tune on Kaggle
with GPU, then download and drop the files in). Until then, the app runs in
BiLSTM-only mode with no errors.

## Emotion classes

`Bored`, `Confident`, `Confused`, `Curious`, `Frustrated`
