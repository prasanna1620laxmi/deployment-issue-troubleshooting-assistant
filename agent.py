"""agent.py - the single tool-using LangChain agent plus the full analysis pipeline.

run_analysis() is the one function the Streamlit app calls. Steps:
  1. redact secrets  2. parse the log  3. (optional) let the Gemini agent use the 4 approved tools
  4. fall back to rules if AI is unavailable  5. validate and build the report

The tools are READ-ONLY. The agent cannot run commands or change anything.
"""
import json
import logging
import re
import time
from dataclasses import dataclass, field

import classifier
import config
import knowledge_base
import log_parser
import prompts
import report_generator

logger = logging.getLogger(__name__)


@dataclass
class AnalysisContext:
    """Holds the data the tools work on, so the AI never has to copy the whole log."""
    parsed: dict
    focus: str
    classification: dict = None
    kb_matches: list = field(default_factory=list)
    tools_used: list = field(default_factory=list)


def classify_with_fallback(text: str):
    """Use the TensorFlow model; if unavailable, use keyword rules. Returns (classification, warning)."""
    try:
        return classifier.predict(text), None
    except classifier.ModelNotTrainedError:
        warning = "TensorFlow model not trained yet (run train_model.py); keyword rules were used instead."
    except Exception as exc:  # noqa: BLE001
        logger.warning("Classifier failed: %s", type(exc).__name__)
        warning = f"Classifier error ({type(exc).__name__}); keyword rules were used instead."
    category, score = knowledge_base.keyword_category(text)
    confidence = min(0.9, 0.4 + 0.15 * score) if score else 0.0
    return {"category": category, "best_guess": category, "confidence": round(confidence, 3),
            "low_confidence": score == 0, "probabilities": {}}, warning


def build_tools(ctx: AnalysisContext) -> list:
    """Create the four approved tools. Each one only reads data held in ctx."""
    from langchain_core.tools import tool

    @tool
    def analyze_log() -> str:
        """Summarize the uploaded deployment log: counts, time range, key errors, warnings and stack traces."""
        ctx.tools_used.append("Log Analyzer")
        return log_parser.summarize_for_prompt(ctx.parsed)

    @tool
    def classify_error() -> str:
        """Predict the most likely error category for the uploaded log using the trained TensorFlow model."""
        ctx.tools_used.append("TensorFlow Error Classifier")
        ctx.classification, _ = classify_with_fallback(ctx.focus)
        c = ctx.classification
        return json.dumps({"category": c["category"], "best_guess": c["best_guess"],
                           "confidence": c["confidence"], "low_confidence": c["low_confidence"]})

    @tool
    def search_knowledge_base(query: str) -> str:
        """Look up known causes, checks, corrective actions and validation steps.
        query: an error category name (preferred) or keywords from the error message."""
        ctx.tools_used.append("Knowledge Base Search")
        ctx.kb_matches = knowledge_base.search_kb(query) or knowledge_base.search_kb(ctx.focus)
        slim = [{k: e[k] for k in ("category", "probable_causes", "diagnostic_checks", "corrective_actions",
                                    "validation_steps")} for e in ctx.kb_matches]
        return json.dumps(slim) if slim else "No matching knowledge-base entry found."

    @tool
    def validate_recommendations(recommendations_json: str) -> str:
        """Check draft recommendations for safety and relevance.
        recommendations_json: a JSON list of recommendation strings. Returns approved and rejected items."""
        ctx.tools_used.append("Recommendation Validator")
        try:
            recs = json.loads(recommendations_json)
            if not isinstance(recs, list):
                raise ValueError
        except ValueError:
            return "Invalid input: pass a JSON list of strings."
        result = knowledge_base.validate_recommendations(recs, ctx.kb_matches, ctx.focus)
        return json.dumps(result)

    return [analyze_log, classify_error, search_knowledge_base, validate_recommendations]


def friendly_error(exc: Exception) -> str:
    """Translate common Gemini/LangChain failures into plain messages. Never includes the API key."""
    text = f"{type(exc).__name__} {exc}".lower()
    if "429" in text or "quota" in text or "resource_exhausted" in text or "rate" in text and "limit" in text:
        return "Gemini quota or rate limit reached. Wait a while or check your plan. Rule-based analysis was used."
    if "api key" in text or "api_key" in text or "permission_denied" in text or "401" in text or "403" in text:
        return "Gemini rejected the API key or access. Check GOOGLE_API_KEY in your .env file. Rule-based analysis was used."
    if "404" in text or "not found" in text or "model" in text and "support" in text:
        return "The Gemini model name was not found or is not available to your account. Check GEMINI_MODEL. Rule-based analysis was used."
    if "connect" in text or "network" in text or "timeout" in text or "dns" in text:
        return "Could not reach Gemini (network problem). Rule-based analysis was used."
    if isinstance(exc, ImportError):
        return "LangChain packages are missing. Run: pip install -r requirements.txt. Rule-based analysis was used."
        detail = str(exc).replace(config.get_api_key() or "\0", "[key]")[:250]
    return f"AI analysis failed ({type(exc).__name__}): {detail}. Rule-based analysis was used."


def _message_text(content) -> str:
    """Gemini may return a string or a list of content parts."""
    if isinstance(content, str):
        return content
    parts = []
    for part in content or []:
        parts.append(part if isinstance(part, str) else part.get("text", ""))
    return "".join(parts)


def parse_ai_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("The AI response did not contain a JSON object.")
    data = json.loads(text[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("The AI response JSON was not an object.")
    return data


def make_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=config.get_model_name(), google_api_key=config.get_api_key(),
                                  max_retries=2)


def run_gemini_agent(ctx: AnalysisContext) -> dict:
    """Run the single tool-using agent and return its parsed JSON fields."""
    from langchain.agents import create_agent

    agent = create_agent(model=make_llm(), tools=build_tools(ctx), system_prompt=prompts.AGENT_SYSTEM_PROMPT)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": prompts.AGENT_TASK_PROMPT}]},
        config={"recursion_limit": 15},
    )
    return parse_ai_json(_message_text(result["messages"][-1].content))


def run_analysis(file_name: str, raw_text: str, use_ai: bool = True) -> dict:
    """Full pipeline. Returns the report dict (see report_generator.build_report)."""
    started = time.time()
    warnings = []
    sanitized, redactions = log_parser.redact_sensitive(raw_text)   # secrets masked BEFORE any AI use
    parsed = log_parser.parse_log(sanitized)
    ctx = AnalysisContext(parsed=parsed, focus=log_parser.focus_text(parsed))

    ai_fields = None
    if use_ai:
        if not config.get_api_key():
            warnings.append("No Gemini API key found in .env; rule-based analysis was used.")
        elif parsed["error_count"] == 0 and parsed["warning_count"] == 0:
            warnings.append("No errors or warnings found, so AI analysis was skipped.")
        else:
            try:
                ai_fields = run_gemini_agent(ctx)
                if not ctx.tools_used:
                    warnings.append("The agent did not call any tools; treat the AI text with extra caution.")
            except Exception as exc:  # noqa: BLE001  (we deliberately catch everything and fall back)
                logger.warning("Gemini agent failed: %s", type(exc).__name__)
                warnings.append(friendly_error(exc))

    if ctx.classification is None:
        ctx.classification, warning = classify_with_fallback(ctx.focus)
        if warning:
            warnings.append(warning)
    if not ctx.kb_matches:
        category = ctx.classification["category"]
        ctx.kb_matches = knowledge_base.search_kb(category) if category != config.UNKNOWN_CATEGORY else []
        if not ctx.kb_matches and ctx.classification.get("best_guess") and not ctx.classification["low_confidence"]:
            ctx.kb_matches = knowledge_base.search_kb(ctx.focus)

    return report_generator.build_report(
        file_name, parsed, ctx.classification, ctx.kb_matches, ai_fields, warnings,
        tools_used=ctx.tools_used, redactions=redactions, seconds=time.time() - started)
