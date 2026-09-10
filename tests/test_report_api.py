"""
tests/test_report_api.py — HTTP-level coverage of POST /chat/report.

The upload endpoint is the only production path that reaches the vision model
and it previously had ZERO test coverage — no test in the suite called
record_report_turn or the route, and there were no FastAPI TestClient tests at
all. Everything here drives the real router; only the paid boundaries are mocked.
"""
import io
import pytest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from src.agents import conversation_agent as agent
from src.api.routes import chat as chat_route
from src.memory import store
from src.core.session import reset_store


def _jpeg(w=400, h=300) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, "JPEG")
    return buf.getvalue()


PANEL = {
    "analytes": [
        {"name": "Total Testosterone", "value": 85, "unit": "ng/dL"},
        {"name": "AMH", "value": 7.2, "unit": "ng/mL"},
    ],
    "notes": [], "unreadable": [], "error": None,
}


@pytest.fixture
def client():
    store.reset_store_for_tests("sqlite://")
    reset_store()
    chat_route._report_calls.clear()
    app = FastAPI()
    app.include_router(chat_route.router)
    with patch.object(agent, "generate_response", return_value="ok"), \
         patch.object(agent, "_run_crag", return_value=("", [])), \
         patch("src.agents.report_agent.transcribe_images", return_value=PANEL), \
         patch("src.agents.report_agent._retrieve", return_value=("", [])), \
         patch("src.agents.report_agent.synthesize_verdict",
               return_value={"verdict": "VERDICT", "disclaimer": "DISCLAIMER"}):
        yield TestClient(app)


class TestUploadHappyPath:
    def test_a_photo_is_analysed(self, client):
        r = client.post("/chat/report",
                        files={"files": ("r.jpg", _jpeg(), "image/jpeg")})
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert "Total Testosterone" in body["parsed_values"]
        assert body["user_token"]

    def test_response_carries_identity_for_a_first_time_caller(self, client):
        r = client.post("/chat/report", files={"files": ("r.jpg", _jpeg(), "image/jpeg")})
        assert r.json()["user_token"].count(".") == 1


class TestUploadGuards:
    def test_missing_content_type_is_rejected(self, client):
        """`if f.content_type and ...` skipped the check when the header was absent."""
        r = client.post("/chat/report", files={"files": ("r.jpg", _jpeg(), "")})
        assert r.status_code == 415

    def test_wrong_content_type_is_rejected(self, client):
        r = client.post("/chat/report",
                        files={"files": ("r.pdf", b"%PDF-1.4", "application/pdf")})
        assert r.status_code == 415

    def test_oversized_file_is_rejected(self, client):
        big = b"\xff\xd8\xff" + b"\x00" * (9 * 1024 * 1024)
        r = client.post("/chat/report", files={"files": ("r.jpg", big, "image/jpeg")})
        assert r.status_code == 413

    def test_aggregate_size_is_capped(self, client):
        blob = b"\xff\xd8\xff" + b"\x00" * (7 * 1024 * 1024)
        files = [("files", (f"p{i}.jpg", blob, "image/jpeg")) for i in range(4)]
        r = client.post("/chat/report", files=files)
        assert r.status_code == 413

    def test_no_files_is_rejected(self, client):
        r = client.post("/chat/report", files={})
        assert r.status_code in (400, 422)

    def test_non_image_bytes_never_reach_the_model(self, client):
        """Content-Type is attacker-controlled; the bytes must be validated."""
        from src.nodes.transcribe_report import transcribe_images
        out = transcribe_images([b"<html>not an image</html>"])
        assert out["analytes"] == []
        assert out["error"] == "no usable images"

    def test_extra_pages_are_reported_not_silently_dropped(self, client):
        files = [("files", (f"p{i}.jpg", _jpeg(), "image/jpeg")) for i in range(8)]
        r = client.post("/chat/report", files=files)
        assert r.status_code == 200
        body = r.json()
        assert body["pages_dropped"] == 2
        assert "weren't included" in body["answer"]


class TestRateLimit:
    def test_repeated_uploads_are_limited(self, client):
        codes = []
        for _ in range(chat_route.REPORT_RATE_LIMIT + 2):
            codes.append(client.post(
                "/chat/report",
                files={"files": ("r.jpg", _jpeg(), "image/jpeg")}).status_code)
        assert 429 in codes, f"never rate limited: {codes}"

    def test_the_limit_response_says_when_to_retry(self, client):
        for _ in range(chat_route.REPORT_RATE_LIMIT + 2):
            r = client.post("/chat/report",
                            files={"files": ("r.jpg", _jpeg(), "image/jpeg")})
            if r.status_code == 429:
                assert "Retry-After" in r.headers
                return
        pytest.fail("never rate limited")


class TestUploadOwnership:
    def test_a_stranger_cannot_upload_into_someone_elses_session(self, client):
        first = client.post("/chat/report",
                            files={"files": ("r.jpg", _jpeg(), "image/jpeg")}).json()
        chat_route._report_calls.clear()
        second = client.post(
            "/chat/report",
            files={"files": ("r.jpg", _jpeg(), "image/jpeg")},
            data={"session_id": first["session_id"]},      # someone else's session
        ).json()
        assert second["session_id"] != first["session_id"]
