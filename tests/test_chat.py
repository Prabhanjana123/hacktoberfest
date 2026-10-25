import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stockpulse import gemma  # noqa: E402


class FakeClient:
    def __init__(self, fail_first=False):
        self.trace = []
        self.fail_first = fail_first
        self.calls = []

    def generate(self, model, prompt, **kwargs):
        self.calls.append((model, prompt, kwargs))
        if self.fail_first and len(self.calls) == 1:
            raise RuntimeError("temporary model error")
        self.trace.append({"purpose": kwargs.get("purpose"), "model": model, "latency_s": 0.01})
        return "A test reply from Gemma."


def test_chat_reply_uses_fast_model_and_session_context():
    client = FakeClient()
    result = gemma.chat_reply(
        client,
        [{"role": "user", "content": "Explain my backtest."}],
        context="QuantProof OOS return: -0.92%", depth="quick",
    )
    assert result["reply"] == "A test reply from Gemma."
    assert result["model"] == gemma.FAST_MODEL
    assert result["used_fallback"] is False
    assert "QuantProof OOS return: -0.92%" in client.calls[0][1]
    assert "You are StockPulse Chat" in client.calls[0][2]["system"]


def test_chat_reply_falls_back_to_alternate_model():
    client = FakeClient(fail_first=True)
    result = gemma.chat_reply(
        client, [{"role": "user", "content": "Explain Sharpe ratio."}], depth="quick"
    )
    assert result["used_fallback"] is True
    assert result["model"] == gemma.DEEP_MODEL
    assert len(client.calls) == 2
    assert "alternate configured Gemma model" not in result["reply"]


def test_chat_reply_rejects_invalid_turn_order():
    client = FakeClient()
    with pytest.raises(ValueError, match="last chat message must be from the user"):
        gemma.chat_reply(client, [{"role": "assistant", "content": "Hello"}])
