from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("lang,name", [("fra", "French"), ("rus", "Russian"), ("zho", "Simplified Chinese")])
def test_translate_consigne_names_the_language_and_forbids_answering(lang, name):
    text = (REPO_ROOT / f"consigne/translate-v1-{lang}.md").read_text(encoding="utf-8")
    assert name in text and "translation only" in text and "<prompt" not in text


def test_same_request_consigne_fixes_the_json_shape():
    text = (REPO_ROOT / "consigne/check-same-request-v1.md").read_text(encoding="utf-8")
    assert '"same_request"' in text and '"reason"' in text and "<prompt" not in text
