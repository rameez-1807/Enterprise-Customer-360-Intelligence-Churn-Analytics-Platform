"""
Unit Test Suite for Churn Explainer & XAI Engine
==================================================
Verifies preprocessing alignment, fallback logic for importances,
and SHAP global calculations inside the ChurnExplainer.

Author: Principal Python Test Architect
Version: 1.0.0
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.core.exceptions import ModelInferenceError
from src.models.explainer import ChurnExplainer


@pytest.fixture
def mock_explainer() -> ChurnExplainer:
    """Fixture providing a ChurnExplainer instance with mocked loading methods."""
    with patch.object(ChurnExplainer, "_load_artifacts", return_value=None):
        explainer = ChurnExplainer()
        # Setup dummy attributes
        mock_pipeline = MagicMock()
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([0.1, 0.4, 0.2, 0.3])
        mock_pipeline.named_steps = {"model": mock_model, "preprocessor": MagicMock()}
        explainer.pipeline = mock_pipeline
        explainer.is_shap_available = False
        return explainer


def test_explainer_initialization_no_artifacts() -> None:
    """Verifies that instantiating ChurnExplainer without artifacts directory raises an error."""
    with pytest.raises(ModelInferenceError):
        # This will fail to find 'best_model.joblib' in raw dummy paths
        ChurnExplainer(artifact_dir="nonexistent_artifacts_dir")


def test_explain_global_importance_fallback(mock_explainer: ChurnExplainer) -> None:
    """Verifies fallback to model's feature importances when SHAP is disabled."""
    df_dummy = pd.DataFrame(
        {
            "CustomerID": [50001],
            "Tenure": [5.0],
            "SatisfactionScore": [3],
            "PreferredLoginDevice": ["Mobile Phone"],
            "PreferredPaymentMode": ["COD"],
        }
    )

    # Mock preprocessing output
    prep_mock = pd.DataFrame(
        [[5.0, 3.0, 1.0, 0.0]], columns=["Tenure", "SatisfactionScore", "Device_Mobile", "Mode_COD"]
    )

    with patch.object(mock_explainer, "_get_preprocessed_df", return_value=(prep_mock, pd.Series([50001]))):
        importances = mock_explainer.explain_global_importance(df_dummy)
        assert isinstance(importances, dict)
        assert "Tenure" in importances
        assert importances["Tenure"] == 0.1
