"""
app.py
------
Frontend for the AI-Powered Emotion Detection & Personalized
Learning Support Platform.

Built with Streamlit. Talks to backend.py for all model / logic /
logging work, and renders a Plotly analytics dashboard.
"""

import streamlit as st
import plotly.express as px

import backend

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="AI Learning Support Platform",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 AI-Powered Emotion Detection & Learning Support")
st.caption("Tell me how you're feeling about your studies, and I'll help you out.")

# ----------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------------
page = st.sidebar.radio("Navigate", ["Check In", "Analytics Dashboard"])

# ----------------------------------------------------------------------
# PAGE 1: CHECK IN
# ----------------------------------------------------------------------
if page == "Check In":
    st.subheader("How are you feeling right now?")

    user_text = st.text_area(
        "Describe how you're feeling about your studies today:",
        placeholder="e.g. I'm really confused about this topic and feel like giving up...",
        height=120,
    )

    if st.button("Analyze & Get Support", type="primary"):
        if not user_text.strip():
            st.warning("Please enter how you're feeling first.")
        else:
            with st.spinner("Analyzing your input..."):
                result = backend.process_student_input(user_text)

            st.success(f"Detected emotion: **{result['final_emotion'].capitalize()}**")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Model prediction", result["model_emotion"])
                st.metric("Model confidence", f"{result['model_confidence']*100:.1f}%")
            with col2:
                st.metric("Keyword match", result["keyword_emotion"] or "None")
                st.metric("Final source used", result["source"])

            st.markdown("### 💡 Personalized Suggestion")
            st.info(result["suggestion"])

# ----------------------------------------------------------------------
# PAGE 2: ANALYTICS DASHBOARD
# ----------------------------------------------------------------------
elif page == "Analytics Dashboard":
    st.subheader("📊 Emotion Analytics Dashboard")

    df = backend.load_log_dataframe()

    if df.empty:
        st.info("No interactions logged yet. Go to 'Check In' to get started.")
    else:
        st.write(f"Total interactions logged: **{len(df)}**")

        # Emotion distribution
        emotion_counts = df["final_emotion"].value_counts().reset_index()
        emotion_counts.columns = ["emotion", "count"]

        fig1 = px.bar(
            emotion_counts, x="emotion", y="count",
            title="Emotion Distribution", color="emotion",
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Emotion over time
        df["timestamp"] = pd_to_datetime = __import__("pandas").to_datetime(df["timestamp"])
        fig2 = px.scatter(
            df, x="timestamp", y="final_emotion", color="final_emotion",
            title="Emotion Over Time",
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Raw data
        with st.expander("View raw log data"):
            st.dataframe(df, use_container_width=True)
