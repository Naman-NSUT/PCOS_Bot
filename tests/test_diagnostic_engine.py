"""
tests/test_diagnostic_engine.py
Unit tests for the rule-based diagnostic engine.
Zero API calls — purely local logic.
"""
import pytest
from src.analysis.report_parser import ParsedReport, BiomarkerValue
from src.analysis.diagnostic_engine import run_diagnostic_engine


def _make_report(**kwargs) -> ParsedReport:
    """Helper to build a ParsedReport with specific biomarkers."""
    report = ParsedReport()
    for name, (value, unit) in kwargs.items():
        report.biomarkers[name] = BiomarkerValue(value=value, unit=unit, raw="")
    return report


class TestHyperandrogenism:
    def test_elevated_testosterone_flags_hyperandrogenism(self):
        report = _make_report(**{"Total Testosterone": (90.0, "ng/dL")})
        flags = run_diagnostic_engine(report)
        assert flags.hyperandrogenism is True
        assert len(flags.hyperandrogenism_evidence) >= 1

    def test_normal_testosterone_no_flag(self):
        report = _make_report(**{"Total Testosterone": (45.0, "ng/dL")})
        flags = run_diagnostic_engine(report)
        assert flags.hyperandrogenism is False

    def test_elevated_dheas_flags_hyperandrogenism(self):
        report = _make_report(**{"DHEAS": (500.0, "µg/dL")})
        flags = run_diagnostic_engine(report)
        assert flags.hyperandrogenism is True


class TestInsulinResistance:
    def test_high_homa_ir_flags_ir(self):
        report = _make_report(**{"HOMA-IR": (3.5, "ratio")})
        flags = run_diagnostic_engine(report)
        assert flags.insulin_resistance is True

    def test_low_homa_ir_no_flag(self):
        report = _make_report(**{"HOMA-IR": (1.8, "ratio")})
        flags = run_diagnostic_engine(report)
        assert flags.insulin_resistance is False

    def test_elevated_fasting_insulin_flags_ir(self):
        report = _make_report(**{"Fasting Insulin": (25.0, "µIU/mL")})
        flags = run_diagnostic_engine(report)
        assert flags.insulin_resistance is True

    def test_prediabetic_glucose_flags_ir(self):
        report = _make_report(**{"Fasting Glucose": (110.0, "mg/dL")})
        flags = run_diagnostic_engine(report)
        assert flags.insulin_resistance is True


class TestPolycysticMorphologyProxy:
    def test_high_amh_flags_morphology(self):
        report = _make_report(**{"AMH": (5.0, "ng/mL")})
        flags = run_diagnostic_engine(report)
        assert flags.polycystic_morphology_proxy is True

    def test_normal_amh_no_flag(self):
        report = _make_report(**{"AMH": (2.5, "ng/mL")})
        flags = run_diagnostic_engine(report)
        assert flags.polycystic_morphology_proxy is False

    def test_high_lh_fsh_ratio_flags_morphology(self):
        report = _make_report(**{"LH/FSH Ratio": (2.5, "ratio")})
        flags = run_diagnostic_engine(report)
        assert flags.polycystic_morphology_proxy is True

    def test_normal_lh_fsh_ratio_no_flag(self):
        report = _make_report(**{"LH/FSH Ratio": (1.5, "ratio")})
        flags = run_diagnostic_engine(report)
        assert flags.polycystic_morphology_proxy is False


class TestThyroidAndProlactin:
    def test_elevated_tsh_flags_thyroid(self):
        report = _make_report(**{"TSH": (7.0, "µIU/mL")})
        flags = run_diagnostic_engine(report)
        assert flags.thyroid_flag is True

    def test_low_tsh_flags_thyroid(self):
        report = _make_report(**{"TSH": (0.1, "µIU/mL")})
        flags = run_diagnostic_engine(report)
        assert flags.thyroid_flag is True

    def test_elevated_prolactin_flags_prolactin(self):
        report = _make_report(**{"Prolactin": (45.0, "ng/mL")})
        flags = run_diagnostic_engine(report)
        assert flags.prolactin_flag is True


class TestMetabolicRisk:
    def test_multiple_metabolic_factors_increase_score(self):
        report = _make_report(**{
            "Total Cholesterol":  (250.0, "mg/dL"),
            "Triglycerides":      (220.0, "mg/dL"),
            "HDL":                (35.0,  "mg/dL"),
            "Waist Circumference":(92.0,  "cm"),
        })
        flags = run_diagnostic_engine(report)
        assert flags.metabolic_risk_score >= 3

    def test_no_metabolic_factors_zero_score(self):
        report = _make_report(**{"AMH": (2.0, "ng/mL")})
        flags = run_diagnostic_engine(report)
        assert flags.metabolic_risk_score == 0


class TestCriticalFlags:
    def test_critical_testosterone_flagged(self):
        report = _make_report(**{"Total Testosterone": (200.0, "ng/dL")})  # > 82 * 1.5
        flags = run_diagnostic_engine(report)
        assert len(flags.critical_flags) >= 1

    def test_empty_report_no_flags(self):
        report = ParsedReport()
        flags = run_diagnostic_engine(report)
        assert flags.hyperandrogenism is False
        assert flags.insulin_resistance is False
        assert flags.metabolic_risk_score == 0
        assert flags.critical_flags == []
