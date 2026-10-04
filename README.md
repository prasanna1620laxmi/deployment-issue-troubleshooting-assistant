# IT Application Maintenance: Deployment Issue Troubleshooting Assistant

Upload a deployment log (.log, .txt, .csv) and get a suggested error category, severity, probable root cause,
evidence, prioritized troubleshooting steps and a validation checklist. No Docker or Plotly needed.

**Important:** the output is a set of *hypotheses* for a human to verify. The classifier is trained on a small
synthetic dataset and is **not production-accurate**. Never upload confidential production logs without authorization.
Secret masking is regex-based and best-effort: it does not catch every secret.

## 1. Setup (Windows, VS Code terminal)

1. Install **Python 3.11 or 3.12** from python.org (tick "Add Python to PATH"). TensorFlow does not support the newest Python versions, so avoid 3.13+.
2. Unzip the project, then in VS Code use **File > Open Folder** and pick `deployment_troubleshooter`.
3. Open a terminal (**Terminal > New Terminal**) and run:

```
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python verify_setup.py
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use the Command Prompt.
Linux/macOS: `python3 -m venv venv` then `source venv/bin/activate`; the other commands are the same.

`verify_setup.py` should print `OK` for every package.

## 2. Gemini API key (optional, enables the AI agent)

1. Create a key at https://aistudio.google.com/app/apikey (free tiers have quotas and terms that can change).
2. Copy `.env.example` to `.env` and replace `your_api_key_here` with your key. Never share or commit `.env`.
3. Check it: `python check_gemini.py`. If the model name is rejected, set `GEMINI_MODEL` in `.env` to a current model listed in AI Studio.

Without a key, the app still works using rule-based fallback text from the knowledge base.

## 3. Train the TensorFlow model

```
python train_model.py
```
This creates `models/error_classifier.keras`, `vocab.json`, `labels.json` and `metrics.json`, and prints per-category
precision, recall, F1 and a confusion matrix on a held-out test set. Scores on this template-based data will look very
high; real logs will score lower.

## 4. Run the tests and the app

```
pytest
streamlit run app.py
```
Open the address shown (usually http://localhost:8501) and upload a file from `sample_uploads/`.

## 5. Project layout

| File | Purpose |
|---|---|
| `config.py` | Paths, limits, categories, `.env` loading, logging |
| `log_parser.py` | File validation, redaction, parsing, multiline errors, duplicates |
| `classifier.py` / `train_model.py` | Keras classifier: data checks, training, saving, prediction, low-confidence handling |
| `knowledge_base.py` + `data/troubleshooting.json` | Known causes, checks, actions; recommendation validator |
| `prompts.py` | Gemini instructions |
| `agent.py` | Single LangChain agent with 4 read-only tools + the full pipeline and fallbacks |
| `report_generator.py` | Report assembly and Markdown export |
| `database.py` | SQLite history and feedback (summaries only, parameterized SQL) |
| `app.py` | Streamlit dashboard |
| `tests/` | Automated tests |

How it flows: upload -> redact secrets -> parse -> (agent: Log Analyzer, Classifier, Knowledge Base, Validator)
-> safety validation -> report. If AI or the model is unavailable, rules take over and a warning is shown.
The agent has no tools that run commands, restart services, or touch databases.

## 6. Evaluation procedure (targets are goals, not results)

- **Per-category classification:** use the report printed by `train_model.py` (precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = harmonic mean). For a meaningful test, collect 20+ *real, authorized, labeled* logs per category and evaluate on those only.
- **Root-cause accuracy:** on a labeled set of real incidents, accuracy = cases where an engineer agrees the top cause was correct / total cases.
- **Recommendation usefulness / user satisfaction:** the Helpful / Not Helpful buttons store votes in `history.db`; rate = helpful / total votes. Also have engineers score steps 1-5 in a small review.
- **MTTR improvement:** record time-to-resolution for comparable incidents with and without the tool; improvement = (old mean - new mean) / old mean.
- **Response time:** shown per analysis in the dashboard (`analysis_seconds`); average it over many runs, with and without AI.

## 7. Known limitations

- Synthetic training data: it recognizes familiar phrasings and may mislabel unusual or mixed errors; improve it by adding real labeled logs to `data/sample_logs.csv` and retraining.
- Only the first lines of the most important errors are classified; a log with several unrelated failures gets one category.
- Redaction patterns are basic; review logs before using AI.
- Gemini free-tier quotas and available model names change; the app falls back to rules when calls fail.
- The AI could still produce wrong text; the validator removes risky or unrelated steps but cannot prove correctness.

## 8. Common errors

| Problem | Fix |
|---|---|
| `'python' is not recognized` | Reinstall Python with "Add to PATH", reopen the terminal |
| TensorFlow install fails | Check `python --version` is 3.11 or 3.12 and the venv is active |
| `ModuleNotFoundError` | Activate the venv, then `pip install -r requirements.txt` |
| "model not trained" warning | Run `python train_model.py` |
| Quota / rate-limit warning | Wait, or untick "Use Gemini AI agent" |
| Model not found | Update `GEMINI_MODEL` in `.env` |
