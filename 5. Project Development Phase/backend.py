"""
backend.py
----------
Backend logic for the AI-Powered Emotion Detection & Personalized
Learning Support Platform.

Handles:
  - Loading the BiLSTM emotion classification model
  - Keyword-blend hybrid classification (model + rule-based keywords)
  - Gemini API integration for personalized learning suggestions
  - CSV logging of interactions
  - Data prep for the Plotly analytics dashboard
"""

import os
import csv
import datetime
import numpy as np

# ----------------------------------------------------------------------
# Optional imports - wrap in try/except so this file can be inspected
# even if dependencies aren't installed yet.
# ----------------------------------------------------------------------
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except ImportError:
    tf = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
MODEL_PATH = "models/bilstm_emotion_model.h5"
TOKENIZER_PATH = "models/tokenizer.pkl"
LOG_FILE = "logs/interaction_log.csv"
MAX_SEQUENCE_LENGTH = 100

EMOTION_LABELS = ["happy", "sad", "angry", "anxious", "neutral", "confused", "frustrated"]

# Keyword blend dictionary - boosts/overrides model prediction
KEYWORD_MAP = {
    "confused": ["confused", "don't understand", "lost", "stuck"],
    "frustrated": ["frustrated", "annoyed", "give up", "hate this"],
    "anxious": ["anxious", "worried", "nervous", "scared"],
    "sad": ["sad", "down", "depressed", "unmotivated"],
    "happy": ["happy", "excited", "great", "love this"],
    "angry": ["angry", "mad", "furious"],
}

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


# ----------------------------------------------------------------------
# MODEL LOADING
# ----------------------------------------------------------------------
_model = None
_tokenizer = None


def load_resources():
    """Load the BiLSTM model and tokenizer once (lazy singleton)."""
    global _model, _tokenizer
    if _model is None and tf is not None and os.path.exists(MODEL_PATH):
        _model = load_model(MODEL_PATH)
    if _tokenizer is None and os.path.exists(TOKENIZER_PATH):
        import pickle
        with open(TOKENIZER_PATH, "rb") as f:
            _tokenizer = pickle.load(f)
    return _model, _tokenizer


# ----------------------------------------------------------------------
# HYBRID EMOTION CLASSIFICATION
# ----------------------------------------------------------------------
def keyword_scan(text: str):
    """Return a keyword-based emotion guess, or None if no match."""
    text_lower = text.lower()
    for emotion, keywords in KEYWORD_MAP.items():
        if any(kw in text_lower for kw in keywords):
            return emotion
    return None


def model_predict(text: str):
    """Run the BiLSTM model on the input text and return predicted label."""
    model, tokenizer = load_resources()
    if model is None or tokenizer is None:
        return "neutral", 0.0  # fallback if model isn't available

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    preds = model.predict(padded, verbose=0)[0]
    idx = int(np.argmax(preds))
    confidence = float(preds[idx])
    return EMOTION_LABELS[idx], confidence


def classify_emotion(text: str):
    """
    Hybrid classifier: blends keyword rules with BiLSTM model output.
    Keyword match takes priority when confidence from the model is low.
    """
    model_label, confidence = model_predict(text)
    keyword_label = keyword_scan(text)

    if keyword_label and confidence < 0.6:
        final_label = keyword_label
        source = "keyword"
    else:
        final_label = model_label
        source = "model"

    return {
        "text": text,
        "final_emotion": final_label,
        "model_emotion": model_label,
        "model_confidence": round(confidence, 3),
        "keyword_emotion": keyword_label,
        "source": source,
    }


# ----------------------------------------------------------------------
# GEMINI API - PERSONALIZED LEARNING SUPPORT
# ----------------------------------------------------------------------
def get_learning_suggestion(text: str, emotion: str) -> str:
    """
    Call Gemini API to generate a personalized learning support
    message based on the student's detected emotion and input text.
    """
    if genai is None or not GEMINI_API_KEY:
        return _fallback_suggestion(emotion)

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = (
            f"A student wrote: \"{text}\"\n"
            f"Detected emotion: {emotion}\n\n"
            "As a supportive learning coach, give a short (2-3 sentence) "
            "personalized, encouraging response that helps them with their "
            "study situation given this emotional state."
        )
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return _fallback_suggestion(emotion)


def _fallback_suggestion(emotion: str) -> str:
    fallback = {
        "confused": "It's okay to feel confused — try breaking the topic into smaller steps and revisit the basics.",
        "frustrated": "Take a short break, then come back with fresh eyes. Frustration usually means you're close to a breakthrough.",
        "anxious": "Try a quick breathing exercise, then tackle one small task at a time.",
        "sad": "Be kind to yourself today. Even a little progress counts.",
        "happy": "Great energy! This is a good time to tackle something challenging.",
        "angry": "Pause for a moment before continuing — a short walk can help reset focus.",
        "neutral": "Let's keep the momentum going — pick one small goal for this session.",
    }
    return fallback.get(emotion, fallback["neutral"])


# ----------------------------------------------------------------------
# CSV LOGGING
# ----------------------------------------------------------------------
def log_interaction(result: dict, suggestion: str):
    """Append an interaction record to the CSV log for analytics."""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    file_exists = os.path.exists(LOG_FILE)

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "text", "final_emotion", "model_emotion",
                "model_confidence", "keyword_emotion", "source", "suggestion"
            ])
        writer.writerow([
            datetime.datetime.now().isoformat(),
            result["text"],
            result["final_emotion"],
            result["model_emotion"],
            result["model_confidence"],
            result["keyword_emotion"],
            result["source"],
            suggestion,
        ])


def load_log_dataframe():
    """Load the CSV log into a pandas DataFrame for the dashboard."""
    import pandas as pd
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame(columns=[
            "timestamp", "text", "final_emotion", "model_emotion",
            "model_confidence", "keyword_emotion", "source", "suggestion"
        ])
    return pd.read_csv(LOG_FILE)


# ----------------------------------------------------------------------
# MAIN PIPELINE FUNCTION (called by frontend)
# ----------------------------------------------------------------------
def process_student_input(text: str):
    """
    Full pipeline: classify emotion, get a personalized suggestion,
    log the interaction, and return everything to the frontend.
    """
    result = classify_emotion(text)
    suggestion = get_learning_suggestion(text, result["final_emotion"])
    log_interaction(result, suggestion)
    result["suggestion"] = suggestion
    return result
