"""
Unit Test Suite for Centralized Dashboard Data Service
======================================================
Verifies singleton instantiation patterns, caching behaviors, and data
query operations on the DashboardDataService.

Author: Principal Python Test Architect
Version: 1.0.0
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.dashboard.data_service import DashboardDataService


@pytest.fixture
def mock_pipeline_components():
    """Mocks all data and model components needed by DashboardDataService."""
    with (
        patch("src.dashboard.data_service.ingest_dataset") as mock_ingest,
        patch("src.dashboard.data_service.clean_dataset") as mock_clean,
        patch("src.dashboard.data_service.build_features") as mock_features,
        patch("src.dashboard.data_service.analyze_rfm") as mock_rfm,
        patch("src.dashboard.data_service.segment_customers") as mock_segment,
        patch("src.dashboard.data_service.compute_customer_health") as mock_health,
        patch("src.dashboard.data_service.ChurnPredictor") as mock_pred,
        patch("src.dashboard.data_service.ChurnExplainer") as mock_exp,
        patch("src.analytics.kpi_engine.calculate_kpis") as mock_kpi,
    ):

        # Set up mock returns
        df_dummy = pd.DataFrame({"CustomerID": [50001], "Churn_Prediction": [0]})
        mock_ingest.return_value = (df_dummy, {})
        mock_clean.return_value = (df_dummy, {})
        mock_features.return_value = (df_dummy, {})
        mock_rfm.return_value = (df_dummy, {"top_cohort": {"segment": "Champions"}})
        mock_segment.return_value = (df_dummy, {})
        mock_health.return_value = (df_dummy, {"average_health_score": 85.0})

        prediction_meta_dummy = {"risk_distribution": {"Critical": {"count": 1}, "High": {"count": 2}}}
        mock_pred.return_value.predict.return_value = (df_dummy, prediction_meta_dummy)
        mock_exp.return_value.explain_global_importance.return_value = {}

        mock_kpi.return_value = {
            "customer_base": {
                "total_customers": 2,
                "active_customers": 1,
                "churned_customers": 1,
                "churn_rate_pct": 50.0,
                "retention_rate_pct": 50.0,
            },
            "health_indices": {},
            "engagement": {"avg_satisfaction_score": 3.0, "complaint_rate_pct": 20.0},
            "recommendations": ["Mock recommendation"],
        }

        # Reset the singleton class attribute
        DashboardDataService._instance = None

        yield {
            "ingest": mock_ingest,
            "clean": mock_clean,
            "features": mock_features,
            "rfm": mock_rfm,
            "segment": mock_segment,
            "health": mock_health,
        }


def test_dashboard_service_singleton(mock_pipeline_components) -> None:
    """Verifies that the DashboardDataService conforms to the Singleton pattern."""
    service1 = DashboardDataService(filepath="dummy_path.xlsx")
    service2 = DashboardDataService(filepath="dummy_path.xlsx")

    assert service1 is service2
    assert service1.filepath == "dummy_path.xlsx"


def test_get_dashboard_summary(mock_pipeline_components) -> None:
    """Verifies retrieval of consolidated dashboard metrics."""
    service = DashboardDataService(filepath="dummy_path.xlsx")
    # Manually populate processed df if needed
    service.df_processed = pd.DataFrame(
        {
            "CustomerID": [50001, 50002],
            "Churn_Prediction": [1, 0],
            "Customer_Health_Score": [30.0, 85.0],
            "SatisfactionScore": [1, 5],
        }
    )

    service.health_meta["average_health_score"] = 57.5
    summary = service.get_dashboard_summary()
    assert summary["total_customers"] == 2
    assert summary["churn_rate_pct"] == 50.0
    assert summary["average_health_score"] == 57.5
    assert summary["average_satisfaction"] == 3.0
