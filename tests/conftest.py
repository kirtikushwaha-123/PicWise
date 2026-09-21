import pytest


SLOW_TEST_MODULES = {
    "test_broad_real_world_validation.py",
    "test_ml_error_analysis.py",
    "test_ml_evaluation.py",
    "test_ml_improvement_experiments.py",
    "test_ocr_integration.py",
    "test_personal_care_inference.py",
    "test_personal_care_real_world_validation.py",
    "test_real_world_defect_remediation.py",
}


def pytest_collection_modifyitems(items):
    slow_marker = pytest.mark.slow

    for item in items:
        if item.path.name in SLOW_TEST_MODULES:
            item.add_marker(slow_marker)
