"""
tests/test_report_parser.py
Unit tests for the regex-based lab report parser.
These tests run without a Gemini API key or ChromaDB — purely local.
"""
import pytest
from src.analysis.report_parser import parse_report, BiomarkerValue


SAMPLE_REPORT_1 = """
Patient Lab Results
-------------------
LH: 12.4 mIU/mL
FSH: 5.1 mIU/mL
Total Testosterone: 85 ng/dL
Free Testosterone: 2.3 pg/mL
AMH: 7.2 ng/mL
Prolactin: 18 ng/mL
TSH: 3.2 µIU/mL
Fasting Glucose: 102 mg/dL
Fasting Insulin: 22 µIU/mL
Total Cholesterol: 210 mg/dL
HDL: 45 mg/dL
LDL: 140 mg/dL
Triglycerides: 180 mg/dL
BMI: 27.3 kg/m²
"""

SAMPLE_REPORT_2 = """
Anti Mullerian Hormone (AMH): 4.5 ng/mL
Luteinizing Hormone (LH) = 14.0 mIU/mL
Follicle Stimulating Hormone (FSH) = 6.2 IU/L
DHEAS: 380 ug/dL
"""

SAMPLE_REPORT_MINIMAL = """
Testosterone 90 ng/dL
Glucose 95 mg/dL
"""


class TestParseReport:
    def test_basic_extraction(self):
        result = parse_report(SAMPLE_REPORT_1)
        assert "LH" in result.biomarkers
        assert result.biomarkers["LH"].value == 12.4

    def test_fsh_extracted(self):
        result = parse_report(SAMPLE_REPORT_1)
        assert "FSH" in result.biomarkers
        assert result.biomarkers["FSH"].value == 5.1

    def test_testosterone_extracted(self):
        result = parse_report(SAMPLE_REPORT_1)
        assert "Total Testosterone" in result.biomarkers
        assert result.biomarkers["Total Testosterone"].value == 85.0

    def test_amh_extracted(self):
        result = parse_report(SAMPLE_REPORT_1)
        assert "AMH" in result.biomarkers
        assert result.biomarkers["AMH"].value == 7.2

    def test_lh_fsh_ratio_computed(self):
        """LH/FSH ratio should be computed automatically."""
        result = parse_report(SAMPLE_REPORT_1)
        assert "LH/FSH Ratio" in result.biomarkers
        expected = round(12.4 / 5.1, 2)
        assert result.biomarkers["LH/FSH Ratio"].value == expected

    def test_homa_ir_computed(self):
        """HOMA-IR should be computed from glucose + insulin."""
        result = parse_report(SAMPLE_REPORT_1)
        assert "HOMA-IR" in result.biomarkers
        # (102/18 * 22) / 22.5 ≈ 5.55
        assert result.biomarkers["HOMA-IR"].value > 0

    def test_alternative_name_amh(self):
        """Parser should handle 'Anti Mullerian Hormone' alias."""
        result = parse_report(SAMPLE_REPORT_2)
        assert "AMH" in result.biomarkers
        assert result.biomarkers["AMH"].value == 4.5

    def test_alternative_format_lh(self):
        """Parser should handle 'LH = 14.0' format."""
        result = parse_report(SAMPLE_REPORT_2)
        assert "LH" in result.biomarkers
        assert result.biomarkers["LH"].value == 14.0

    def test_minimal_report(self):
        result = parse_report(SAMPLE_REPORT_MINIMAL)
        assert "Total Testosterone" in result.biomarkers
        assert result.biomarkers["Total Testosterone"].value == 90.0

    def test_empty_report(self):
        result = parse_report("No lab values here.")
        assert len(result.biomarkers) == 0

    def test_metabolic_markers(self):
        result = parse_report(SAMPLE_REPORT_1)
        assert "Total Cholesterol" in result.biomarkers
        assert result.biomarkers["Total Cholesterol"].value == 210.0
        assert "HDL" in result.biomarkers
        assert result.biomarkers["Triglycerides"].value == 180.0
