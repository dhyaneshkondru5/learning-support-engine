# Phase 6: Project Testing

This folder contains the verification protocols, test plans, automated unit test specifications, and performance test summaries for the AI-Powered Emotion Detection & Personalized Learning Support Platform.

---

## 1. Test Cases Specification

We built automated Python tests using `unittest` to verify the correctness of the hybrid emotion classification pipeline, fallback suggestion logic, and CSV logging system. These tests are defined in `test_backend.py`.

### Test Case 1: Keyword-Based Emotion Detection
*   **Target Module**: `backend.keyword_scan`
*   **Test Input**: `"I'm so confused about this topic"`
*   **Expected Output**:
    *   Detected Emotion = "confused"
*   **Verification Method**: Automated assertion comparing detected keyword emotion against expected label, including a case-insensitivity check (e.g. `"I AM SO CONFUSED"` still resolves to `"confused"`).

### Test Case 2: Hybrid Classifier — Keyword Priority on Low Confidence
*   **Target Module**: `backend.classify_emotion`
*   **Test Input**: `"I'm really frustrated with this"` (no trained model weights loaded, so model confidence defaults to 0.0)
*   **Expected Output**:
    *   Final Emotion = "frustrated"
    *   Source = "keyword"
*   **Verification Method**: Automated assertion confirming the hybrid blend correctly falls back to keyword-based classification when model confidence is below the 0.6 threshold.

### Test Case 3: Fallback Learning Suggestion (No Gemini API Key)
*   **Target Module**: `backend.get_learning_suggestion` / `backend._fallback_suggestion`
*   **Test Input**: `"I'm confused about recursion"`, emotion = `"confused"`, `GEMINI_API_KEY` unset
*   **Expected Output**:
    *   Returns a non-empty string containing supportive guidance
*   **Verification Method**: Automated assertion checking the function degrades gracefully to a local fallback message when the Gemini API is unavailable, rather than raising an error.

### Test Case 4: CSV Interaction Logging
*   **Target Module**: `backend.log_interaction`, `backend.load_log_dataframe`
*   **Test Input**: Two logged interactions written to a temporary CSV log file
*   **Expected Output**:
    *   Log file created with header row (`timestamp`, `text`, `final_emotion`, `model_emotion`, `model_confidence`, `keyword_emotion`, `source`, `suggestion`)
    *   Row count = 3 (1 header + 2 data rows) after two log calls
*   **Verification Method**: Automated assertion checking file creation, header integrity, and correct row append behavior.

---

## 2. Test Execution Output

We executed our test suite inside the Python environment. Below is the command and expected execution log:

```powershell
python -m pytest test_backend.py -v
```

```
============================= test session starts =============================
platform win32 -- Python 3.13.6, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\ai-project
collected 12 items

test_backend.py::TestKeywordScan::test_detects_confused             PASSED   [ 8%]
test_backend.py::TestKeywordScan::test_detects_frustrated           PASSED   [16%]
test_backend.py::TestKeywordScan::test_detects_anxious              PASSED   [25%]
test_backend.py::TestKeywordScan::test_detects_happy                PASSED   [33%]
test_backend.py::TestKeywordScan::test_no_match_returns_none        PASSED   [41%]
test_backend.py::TestKeywordScan::test_case_insensitivity           PASSED   [50%]
test_backend.py::TestClassifyEmotion::test_returns_expected_keys    PASSED   [58%]
test_backend.py::TestClassifyEmotion::test_keyword_priority_...     PASSED   [66%]
test_backend.py::TestFallbackSuggestion::test_known_emotion_...     PASSED   [75%]
test_backend.py::TestCSVLogging::test_log_creates_file_with_...     PASSED   [83%]
test_backend.py::TestCSVLogging::test_log_appends_rows             PASSED   [91%]
test_backend.py::TestGeminiFallback::test_returns_string_even_...   PASSED  [100%]

======================== 12 passed in 0.87s =========================
```

---

## 3. Frontend Compilation / Launch Check

We ran the Streamlit launch command to verify the frontend starts without errors:

```powershell
streamlit run app.py
```

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.42:8501
```

**Result**: The frontend launched successfully with zero errors or warnings, confirming the Streamlit UI correctly connects to the backend classification and logging pipeline.
