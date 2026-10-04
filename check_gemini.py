"""check_gemini.py - Phase 5 test of your API key. Run: python check_gemini.py"""
import agent
import config
import prompts

if not config.get_api_key():
    raise SystemExit("No GOOGLE_API_KEY found. Copy .env.example to .env and add your key.")
print("Model:", config.get_model_name())
try:
    reply = agent.make_llm().invoke(prompts.CONNECTION_TEST_PROMPT)
    print("Reply:", agent._message_text(reply.content))
except Exception as exc:  # noqa: BLE001
    print("FAILED:", agent.friendly_error(exc))
