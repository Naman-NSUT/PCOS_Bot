"""
manual_infer_check.py — manual smoke test for the chat model.

NOT a pytest test. This makes a real, billed API call, so it is deliberately
named outside pytest's `test_*.py` collection glob and guarded by __main__.

Run explicitly:
    venv/bin/python manual_infer_check.py
"""
import sys

sys.path.append('.')

from src.core.session import Session
from src.core.llm_client import build_system_prompt, build_user_prompt, generate_response


def main() -> None:
    session = Session()
    sys_prompt = build_system_prompt(session)
    user_prompt = build_user_prompt(
        session, "Hi, I am experiencing irregular periods and I'm very worried."
    )
    print("Querying LLM…")
    print("Response:\n", generate_response(sys_prompt, user_prompt))


if __name__ == "__main__":
    main()
