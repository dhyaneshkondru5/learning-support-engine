"""
app.py
AI-Driven Emotion Detection & Personalized Learning Support Platform
Run with: streamlit run app.py
"""
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from utils.predict import load_bilstm, load_bert, predict_bilstm, predict_bert
from utils.gemini_helper import get_ai_response
from utils.logger import log_interaction, load_logs
from utils.preprocessing import EMOTIONS

st.set_page_config(page_title="Emotion-Aware Learning Assistant", layout="wide", page_icon="🎓")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for key, default in {
    "current_result": None,
    "ai_response": None,
    "response_type": None,
    "history": [],
    "field": "Computer Science",
    "problem_text": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🎓 Emotion-Aware Learning Assistant")
st.caption("Describe what you're struggling with — the assistant detects your emotional state and gives tailored, empathetic guidance.")

tab_assistant, tab_analytics = st.tabs(["💬 Assistant", "📊 Analytics"])

# ---------------------------------------------------------------------------
# Assistant tab
# ---------------------------------------------------------------------------
with tab_assistant:
    col1, col2 = st.columns([1, 1.4])

    with col1:
        st.subheader("Tell us what's going on")
        field = st.selectbox(
            "Field of study",
            ["Computer Science", "Mathematics", "Statistics", "Physics", "Data Structures", "Other"],
            key="field",
        )
        problem_text = st.text_area(
            "Describe your problem or how you're feeling about it",
            key="problem_text",
            height=150,
            placeholder="e.g. I don't understand how recursion works, I keep getting stuck...",
        )

        run_col, clear_col = st.columns(2)
        run_clicked = run_col.button("Analyze", type="primary", use_container_width=True)
        clear_clicked = clear_col.button("Clear", use_container_width=True)

        if clear_clicked:
            st.session_state["current_result"] = None
            st.session_state["ai_response"] = None
            st.session_state["response_type"] = None
            st.rerun()

        if run_clicked:
            if not problem_text.strip():
                st.error("Please describe your problem before analyzing.")
            else:
                with st.spinner("Analyzing emotion and generating guidance..."):
                    bilstm_assets = load_bilstm()
                    bert_assets = load_bert()

                    results = {}
                    if bilstm_assets:
                        try:
                            results["BiLSTM"] = predict_bilstm(problem_text, bilstm_assets)
                        except Exception as e:
                            st.error(f"BiLSTM prediction failed: {e}")
                    if bert_assets:
                        try:
                            results["BERT"] = predict_bert(problem_text, bert_assets)
                        except Exception as e:
                            st.warning(f"BERT prediction failed, continuing with BiLSTM only: {e}")

                    if not results:
                        st.error("No trained models found. Run `python scripts/train_bilstm.py` first.")
                    else:
                        primary_model = "BERT" if "BERT" in results else "BiLSTM"
                        chosen = results[primary_model]
                        chosen["field"] = field

                        ai = get_ai_response(
                            field, problem_text,
                            chosen["predicted_emotion"], chosen["secondary_emotion"],
                            chosen["confidence_score"],
                        )

                        st.session_state["current_result"] = results
                        st.session_state["primary_model"] = primary_model
                        st.session_state["ai_response"] = ai["response"]
                        st.session_state["response_type"] = ai["response_type"]

                        log_interaction(chosen, ai["response"], ai["response_type"])
                        st.session_state["history"].append({
                            "field": field, "text": problem_text,
                            "emotion": chosen["predicted_emotion"], "response": ai["response"],
                        })

    with col2:
        st.subheader("Results")
        if not st.session_state["current_result"]:
            st.info("Run an analysis to see emotion predictions and AI guidance here.")
        else:
            results = st.session_state["current_result"]
            tabs = st.tabs(list(results.keys()))
            for tab, model_name in zip(tabs, results.keys()):
                with tab:
                    r = results[model_name]
                    st.metric("Predicted Emotion", r["predicted_emotion"], f"{r['confidence_score']:.0%} confidence")
                    if r["secondary_emotion"]:
                        st.caption(f"Also detected: **{r['secondary_emotion']}**")
                    st.progress(r["confidence_score"])

                    st.markdown("**Score breakdown**")
                    for emo in EMOTIONS:
                        st.progress(r["emotion_scores"][emo], text=f"{emo}: {r['emotion_scores'][emo]:.0%}")

            st.divider()
            st.subheader("💡 AI Guidance")
            if st.session_state["response_type"] == "fallback":
                st.caption("⚠️ Gemini unavailable — showing a fallback response.")
            st.write(st.session_state["ai_response"])

            if st.button("🔄 Regenerate Response"):
                chosen = results[st.session_state["primary_model"]]
                ai = get_ai_response(
                    st.session_state["field"], st.session_state["problem_text"],
                    chosen["predicted_emotion"], chosen["secondary_emotion"],
                    chosen["confidence_score"],
                )
                st.session_state["ai_response"] = ai["response"]
                st.session_state["response_type"] = ai["response_type"]
                log_interaction(chosen, ai["response"], ai["response_type"])
                st.rerun()

    if st.session_state["history"]:
        with st.expander(f"📜 Session history ({len(st.session_state['history'])})"):
            for i, h in enumerate(reversed(st.session_state["history"]), 1):
                st.markdown(f"**{i}. [{h['field']}] {h['emotion']}** — {h['text']}")
                st.caption(h["response"])
                st.divider()

# ---------------------------------------------------------------------------
# Analytics tab
# ---------------------------------------------------------------------------
with tab_analytics:
    st.subheader("📊 Emotion Analytics Dashboard")
    df = load_logs()
    if df is None or df.empty:
        st.info("No logged interactions yet. Run an analysis in the Assistant tab first.")
    else:
        import plotly.express as px

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Interactions", len(df))
        c2.metric("Most Common Emotion", df["predicted_emotion"].mode()[0])
        c3.metric("Avg. Confidence", f"{df['confidence_score'].mean():.0%}")

        fig1 = px.histogram(
            df, x="predicted_emotion", color="model_used", barmode="group",
            title="Emotion Distribution by Model",
            category_orders={"predicted_emotion": EMOTIONS},
        )
        st.plotly_chart(fig1, use_container_width=True)

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        fig2 = px.line(
            df, x="timestamp", y="confidence_score", color="predicted_emotion",
            title="Confidence Trend Over Time", markers=True,
        )
        st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.pie(df, names="field", title="Interactions by Field")
        st.plotly_chart(fig3, use_container_width=True)

        with st.expander("Raw log data"):
            st.dataframe(df, use_container_width=True)
