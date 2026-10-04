"""app.py - Streamlit dashboard.  Run: streamlit run app.py"""
import pandas as pd
import streamlit as st

import agent
import config
import database
import log_parser
import report_generator

config.setup_logging()
database.init_db()

st.set_page_config(page_title="Deployment Troubleshooting Assistant", page_icon="🛠️", layout="wide")
st.title("🛠️ Deployment Issue Troubleshooting Assistant")
st.caption("Upload a deployment log to get a suggested category, probable cause and troubleshooting steps. "
           "Suggestions are hypotheses for a human to verify.")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    use_ai = st.checkbox("Use Gemini AI agent", value=True,
                         help="If off, or if the API is unavailable, rule-based analysis is used.")
    save_history = st.checkbox("Save summary to history", value=True,
                               help="Stores a summary only, never the raw log.")
    st.write("Gemini key:", "✅ found" if config.get_api_key() else "❌ not set (see .env.example)")
    st.write("TensorFlow model:", "✅ trained" if config.MODEL_PATH.exists() else "❌ not trained (python train_model.py)")
    st.warning("Only upload logs you are authorized to share. Secret masking is best-effort and can miss things.")


def on_feedback(value: str):
    database.save_feedback(st.session_state["analysis_id"], value)
    st.session_state["feedback_given"] = value


tab_report, tab_history = st.tabs(["Analysis", "History & charts"])

# ------------------------------------------------------------------ analysis tab
with tab_report:
    uploaded = st.file_uploader("Upload a log file", type=["log", "txt", "csv"])
    text = None
    if uploaded is not None:
        try:
            text, read_warnings = log_parser.read_log_file(uploaded.name, uploaded.getvalue())
            for w in read_warnings:
                st.warning(w)
            preview, _ = log_parser.redact_sensitive(text)
            st.subheader("Preview (first 40 lines, secrets masked)")
            st.code("\n".join(preview.splitlines()[:40]), language="text")
        except log_parser.LogValidationError as exc:
            st.error(str(exc))

    if text is not None and st.button("Analyze Deployment", type="primary"):
        with st.spinner("Analyzing the log... this can take up to a minute when AI is used."):
            report = agent.run_analysis(uploaded.name, text, use_ai=use_ai)
        st.session_state["report"] = report
        st.session_state["feedback_given"] = None
        st.session_state["analysis_id"] = database.save_analysis(report) if save_history else None

    report = st.session_state.get("report")
    if report:
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Error category", report["category"])
        c2.metric("Severity", report["severity"])
        c3.metric("Classifier confidence", report["confidence"])
        st.caption(f"Method: {report['source']} | Time: {report['analysis_seconds']}s | "
                   f"Tools used: {', '.join(report['tools_used']) or 'none'}")

        for w in report["warnings"]:
            st.warning(w)

        st.subheader("Summary")
        st.write(report["executive_summary"])
        st.caption(f"Severity rationale: {report['severity_rationale']}")

        st.subheader("Probable root cause (hypothesis)")
        st.info(report["probable_root_cause"])
        if report["alternative_causes"]:
            with st.expander("Alternative possible causes"):
                for item in report["alternative_causes"]:
                    st.write(f"- {item}")

        st.subheader("Supporting evidence")
        st.code("\n".join(report["evidence"]) or "No error lines found.", language="text")

        st.subheader("Recommended corrective actions (priority order)")
        for n, step in enumerate(report["recommended_steps"], 1):
            st.write(f"{n}. {step}")

        st.subheader("Validation checklist")
        for i, step in enumerate(report["validation_steps"]):
            st.checkbox(step, key=f"val_{report['generated_at']}_{i}")

        st.subheader("Confidence and limitations")
        st.write(report["confidence_explanation"])
        for item in report["limitations"]:
            st.write(f"- {item}")

        st.download_button("Download report (Markdown)", report_generator.render_markdown(report),
                           file_name="troubleshooting_report.md", mime="text/markdown")

        if st.session_state.get("analysis_id"):
            st.write("Was this analysis helpful?")
            b1, b2 = st.columns([1, 8])
            b1.button("👍 Helpful", on_click=on_feedback, args=("helpful",))
            b2.button("👎 Not helpful", on_click=on_feedback, args=("not_helpful",))
            if st.session_state.get("feedback_given"):
                st.success("Thanks, your feedback was saved.")

# ------------------------------------------------------------------ history tab
with tab_history:
    history = database.get_history()
    if not history:
        st.info("No saved analyses yet.")
    else:
        counts = database.category_counts()
        st.subheader("Analyses by category")
        st.bar_chart(pd.Series(counts, name="analyses"))
        stats = database.feedback_stats()
        total = sum(stats.values())
        if total:
            st.metric("Helpful rate (from feedback)", f"{100 * stats.get('helpful', 0) / total:.0f}%", help=f"{total} votes")
        st.subheader("Recent analyses")
        st.dataframe(pd.DataFrame(history)[["created_at", "file_name", "category", "severity", "confidence", "method", "seconds", "feedback"]],
                     width='stretch')
