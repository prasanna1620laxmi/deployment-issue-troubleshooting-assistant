"""verify_setup.py - Phase 1 check. Run: python verify_setup.py"""
import sys

print("Python:", sys.version.split()[0])
for name in ["pandas", "numpy", "sklearn", "streamlit", "tensorflow", "langchain", "langchain_google_genai", "dotenv"]:
    try:
        module = __import__(name)
        print(f"OK      {name} {getattr(module, '__version__', '')}")
    except Exception as exc:  # noqa: BLE001
        print(f"MISSING {name}: {exc}")
