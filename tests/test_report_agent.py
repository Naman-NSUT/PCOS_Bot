"""
tests/test_report_agent.py — Report analyser subagent.

The design property these tests protect: the vision model TRANSCRIBES, it never
JUDGES. Every clinical conclusion comes from the deterministic parse + flag
engine and the computed concordance, so the tests assert on those and mock the
two model calls at the ends of the pipeline.

Zero network calls.
"""
import pytest
from unittest.mock import patch

from src.agents import report_agent
from src.nodes.synthesize_verdict import build_concordance
from src.nodes.transcribe_report import _clean_analytes, to_report_text
from src.memory import store
from src.memory.models import Predicate
from src.core.session import Session, reset_store, update_symptoms


PANEL = [
    {"name": "LH", "value": 12.4, "unit": "mIU/mL"},
    {"name": "FSH", "value": 5.1, "unit": "mIU/mL"},
    {"name": "AMH", "value": 7.2, "unit": "ng/mL"},
    {"name": "Testosterone", "value": 85, "unit": "ng/dL"},
    {"name": "Fasting Glucose", "value": 104, "unit": "mg/dL"},
    {"name": "Fasting Insulin", "value": 24.0, "unit": "uIU/mL"},
    {"name": "TSH", "value": 2.1, "unit": "mIU/L"},
]


@pytest.fixture(autouse=True)
def _offline():
    store.reset_store_for_tests("sqlite://")
    reset_store()
    # NOTE: patching "src.nodes.synthesize_verdict.synthesize_verdict" is a
    # NO-OP here — report_agent binds the name at import time
    # (`from src.nodes.synthesize_verdict import synthesize_verdict`), so the
    # module-global it actually calls is report_agent.synthesize_verdict.
    # Patching the source module left the real function live, which would have
    # made a paid LLM call if _run() had not patched the right target.
    with patch.object(report_agent, "_retrieve", return_value=("", [])), \
         patch.object(report_agent, "synthesize_verdict",
                      return_value={"verdict": "V", "disclaimer": "D"}):
        yield


def _run(session=None, analytes=None, notes=None):
    payload = {"analytes": analytes if analytes is not None else PANEL,
               "notes": notes or [], "unreadable": [], "error": None}
    with patch.object(report_agent, "transcribe_images", return_value=payload), \
         patch.object(report_agent, "synthesize_verdict",
                      return_value={"verdict": "VERDICT", "disclaimer": "DISCLAIMER"}):
        return report_agent.analyse_report([b"fake-jpeg"], session=session)


class TestTranscriptionIsOnlyOCR:
    def test_malformed_rows_are_dropped_not_guessed(self):
        raw = [
            {"name": "LH", "value": 12.4, "unit": "mIU/mL"},
            {"name": "", "value": 5.0, "unit": "x"},        # no name
            {"name": "FSH", "value": "not a number"},        # unparseable
            "a bare string",                                  # wrong type
            None,
        ]
        assert _clean_analytes(raw) == [{"name": "LH", "value": 12.4, "unit": "mIU/mL"}]

    def test_non_list_payload_is_survivable(self):
        assert _clean_analytes({"analytes": []}) == []
        assert _clean_analytes(None) == []

    def test_rendered_text_feeds_the_existing_regex_parser(self):
        text = to_report_text(PANEL)
        assert "LH: 12.4 mIU/mL" in text
        from src.nodes.parse_report import parse_report_node
        parsed = parse_report_node({"report_text": text})["parsed_values"]
        assert parsed["LH"]["value"] == 12.4
        assert parsed["AMH"]["status"] == "critical_high"


class TestDeterministicJudgement:
    """The rule engine, not the model, decides what is abnormal."""

    def test_flags_come_from_rules(self):
        out = _run()
        f = out["diagnostic_flags"]
        assert f["hyperandrogenism"] is True
        assert f["insulin_resistance"] is True
        assert f["thyroid_flag"] is False           # TSH 2.1 is normal

    def test_derived_values_are_computed(self):
        out = _run()
        assert out["parsed_values"]["LH/FSH Ratio"]["value"] == 2.43
        assert out["parsed_values"]["HOMA-IR"]["status"] == "critical_high"


class TestConcordance:
    """Labs and conversation cover DIFFERENT criteria — that is the point."""

    def test_labs_alone_cannot_reach_ovulatory_dysfunction(self):
        c = build_concordance(
            {"hyperandrogenism": True, "hyperandrogenism_evidence": ["T high"]},
            symptom_list=[], report_notes=[])
        assert c["criteria"]["ovulatory_dysfunction"]["clinical_support"] == []
        assert "ovulatory_dysfunction" in c["not_assessable"]

    def test_conversation_supplies_what_bloodwork_cannot(self):
        c = build_concordance(
            {"hyperandrogenism": True, "hyperandrogenism_evidence": ["T high"]},
            symptom_list=["irregular_periods"], report_notes=[])
        assert c["criteria"]["ovulatory_dysfunction"]["clinical_support"] == ["irregular_periods"]
        assert "ovulatory_dysfunction" in c["conversation_only"]
        assert c["feature_count"] == 2

    def test_corroboration_is_identified(self):
        c = build_concordance(
            {"hyperandrogenism": True, "hyperandrogenism_evidence": ["T high"]},
            symptom_list=["hirsutism", "acne"], report_notes=[])
        assert "hyperandrogenism" in c["corroborated"]

    def test_labs_only_when_conversation_is_silent(self):
        c = build_concordance(
            {"hyperandrogenism": True, "hyperandrogenism_evidence": ["T high"]},
            symptom_list=["fatigue"], report_notes=[])
        assert "hyperandrogenism" in c["labs_only"]

    def test_amh_is_a_proxy_not_morphology(self):
        """AMH must never be presented as equivalent to an ultrasound."""
        c = build_concordance(
            {"polycystic_morphology_evidence": ["AMH elevated: 7.2 ng/mL"]},
            symptom_list=[], report_notes=[])
        pcm = c["criteria"]["polycystic_morphology"]
        assert pcm["lab_support"] is False
        assert pcm["lab_proxy"] == ["AMH elevated: 7.2 ng/mL"]

    def test_printed_ultrasound_remark_is_recognised(self):
        c = build_concordance(
            {}, symptom_list=[],
            report_notes=["Ultrasound: multiple small follicles noted, both ovaries."])
        assert c["criteria"]["polycystic_morphology"]["imaging_mentioned"] is True
        assert "polycystic_morphology" in c["features_with_support"]


class TestSessionIntegration:
    def test_conversation_symptoms_reach_the_concordance(self):
        s = Session()
        update_symptoms(s, ["irregular_periods", "hirsutism"])
        out = _run(session=s)
        assert "hyperandrogenism" in out["concordance"]["corroborated"]

    def test_report_never_inflates_symptom_count(self):
        """
        Lab findings are not self-reported symptoms. symptom_count gates the PCOS
        disclosure and lab-test rules, so a report must not move it.
        """
        s = Session()
        update_symptoms(s, ["acne"])
        before = s.symptom_count
        _run(session=s)
        assert s.symptom_count == before

    def test_abnormal_marker_names_are_remembered_without_values(self):
        s = Session(user_id="u1")
        _run(session=s)
        remembered = {f["object"] for f in store.get_facts("u1", Predicate.SUGGESTED_TEST)}
        assert "AMH" in remembered
        assert not any(any(ch.isdigit() for ch in r) for r in remembered)

    def test_anonymous_session_records_nothing(self):
        s = Session()                       # no user_id
        _run(session=s)
        assert store.get_facts("u1") == []


class TestFailureModes:
    def test_unreadable_image_asks_for_a_better_photo(self):
        out = _run(analytes=[])
        assert out["ok"] is False
        assert "photo" in out["verdict"].lower()

    def test_vision_error_is_not_fatal(self):
        payload = {"analytes": [], "notes": [], "unreadable": [], "error": "429 rate limited"}
        with patch.object(report_agent, "transcribe_images", return_value=payload):
            out = report_agent.analyse_report([b"x"], session=None)
        assert out["ok"] is False

    def test_unrecognised_analytes_do_not_crash(self):
        out = _run(analytes=[{"name": "Widget Count", "value": 5, "unit": "u"}])
        assert out["ok"] is False

    def test_no_images_short_circuits(self):
        out = report_agent.analyse_report([], session=None)
        assert out["ok"] is False


class TestAnalyteNameNormalisation:
    """
    Labs print names their own way. The vision step transcribes verbatim (right),
    so the mapping onto canonical names has to happen in code.

    This silently dropped BOTH androgens in a live run — "Testosterone, Total"
    and "DHEA-Sulphate" never reached the parser — which made hyperandrogenism
    look unsupported by the bloodwork when it was the strongest signal on the page.
    """

    @pytest.mark.parametrize("printed,canonical", [
        ("Testosterone, Total",              "Total Testosterone"),
        ("Total Testosterone",               "Total Testosterone"),
        ("Testosterone (Total)",             "Total Testosterone"),
        ("DHEA-Sulphate",                    "DHEAS"),
        ("DHEA-S",                           "DHEAS"),
        ("Luteinizing Hormone (LH)",         "LH"),
        ("Follicle Stimulating Hormone (FSH)", "FSH"),
        ("Anti-Mullerian Hormone (AMH)",     "AMH"),
        ("HDL Cholesterol",                  "HDL"),
        ("Thyroid Stimulating Hormone",      "TSH"),
        ("Prolactin",                        "Prolactin"),
    ])
    def test_printed_names_map_to_canonical(self, printed, canonical):
        from src.nodes.transcribe_report import normalise_analyte_name
        assert normalise_analyte_name(printed) == canonical

    def test_unknown_analyte_passes_through_unchanged(self):
        from src.nodes.transcribe_report import normalise_analyte_name
        assert normalise_analyte_name("Widget Count") == "Widget Count"

    def test_both_androgens_survive_the_round_trip(self):
        from src.nodes.parse_report import parse_report_node
        analytes = [
            {"name": "Testosterone, Total", "value": 85, "unit": "ng/dL"},
            {"name": "DHEA-Sulphate", "value": 460, "unit": "ug/dL"},
        ]
        parsed = parse_report_node({"report_text": to_report_text(analytes)})["parsed_values"]
        assert parsed["Total Testosterone"]["status"] == "high"
        assert parsed["DHEAS"]["status"] == "high"

    def test_hyperandrogenism_is_corroborated_not_conversation_only(self):
        """The end-to-end consequence of the bug."""
        from src.nodes.flag_indicators import flag_indicators_node
        from src.nodes.parse_report import parse_report_node
        analytes = [{"name": "Testosterone, Total", "value": 85, "unit": "ng/dL"}]
        parsed = parse_report_node({"report_text": to_report_text(analytes)})["parsed_values"]
        flags = flag_indicators_node({"parsed_values": parsed})["diagnostic_flags"]
        c = build_concordance(flags, ["hirsutism"], [])
        assert "hyperandrogenism" in c["corroborated"]
