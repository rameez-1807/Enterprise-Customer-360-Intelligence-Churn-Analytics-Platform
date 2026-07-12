"""
Centralized Dashboard Data Service
===================================
 central data facade for the presentation layer (Streamlit dashboard).
Loads, cleans, and runs the entire analytics and machine learning pipeline
once, caching the fully-processed data mart in memory to guarantee O(1)
responses to dashboard visual requests.

Integrates the KPI Engine, Customer Health Engine, Segmentation Engine,
Behavioral RFM Engine, Prediction Engine, and Explainable AI Engine.

Author: Principal Backend Architect
Version: 1.0.0
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from src.analytics.customer_health import compute_customer_health
from src.analytics.segmentation import segment_customers
from src.core.exceptions import KPICalculationError, ModelInferenceError
from src.core.logger import get_logger
from src.data.cleaning import clean_dataset
from src.data.ingestion import ingest_dataset
from src.features.builder import build_features
from src.features.rfm import analyze_rfm
from src.models.explainer import ChurnExplainer
from src.models.predictor import ChurnPredictor

logger = get_logger(__name__)


class DashboardDataService:
    """Centralized facade service managing the dashboard data pipeline."""

    _instance: Optional["DashboardDataService"] = None
    _initialized: bool = False

    def __new__(cls, *args: Any, **kwargs: Any) -> "DashboardDataService":
        """Implements Singleton pattern to prevent repeated pipeline execution."""
        if not cls._instance:
            cls._instance = super(DashboardDataService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, filepath: Optional[str] = None) -> None:
        """Initializes the data service, running the pipeline if not yet loaded.

        Args:
            filepath: Absolute path to the Excel file. Defaults to config settings.
        """
        if self._initialized:
            return

        logger.info("Initializing Centralized Dashboard Data Service...")

        # Load configs and paths
        from config.settings import RAW_DATASET_PATH

        self.filepath = filepath or str(RAW_DATASET_PATH)

        # In-memory caches
        self.df_processed: pd.DataFrame = pd.DataFrame()

        self.kpi_meta: Dict[str, Any] = {}
        self.health_meta: Dict[str, Any] = {}
        self.segmentation_meta: Dict[str, Any] = {}
        self.rfm_meta: Dict[str, Any] = {}
        self.prediction_meta: Dict[str, Any] = {}
        self.global_xai_meta: Dict[str, float] = {}

        self.predictor: ChurnPredictor = None  # type: ignore
        self.explainer: ChurnExplainer = None  # type: ignore

        self._run_data_pipeline()
        self._initialized = True

    def _run_data_pipeline(self) -> None:
        """Executes the full ingestion-cleaning-features-inference pipeline."""
        logger.info("Executing Dashboard Data Pipeline:")

        # 1. Ingest raw data
        logger.info("  - Loading data ingestion engine...")
        df_raw, _ = ingest_dataset(filepath=self.filepath, optimize_memory=False)

        # 2. Clean data
        logger.info("  - Executing data cleaning and imputation...")
        df_clean, _ = clean_dataset(df_raw, primary_key="CustomerID")

        # 3. Build features
        logger.info("  - Executing domain feature construction...")
        df_feat, _ = build_features(df_clean)

        # 4. Behavioral RFM analysis
        logger.info("  - Computing customer RFM segments...")
        df_rfm, self.rfm_meta = analyze_rfm(df_feat)

        # 5. Customer Segmentation
        logger.info("  - Running customer flag segmentation rules...")
        df_segmented, self.segmentation_meta = segment_customers(df_rfm)

        # 6. Customer Health scoring
        logger.info("  - Computing customer health scores...")
        df_health, self.health_meta = compute_customer_health(df_segmented)

        # 7. Initialize ML predictors and explainers
        logger.info("  - Initializing ML Predictor & Explainers...")
        self.predictor = ChurnPredictor()
        self.explainer = ChurnExplainer()

        # 8. Score customer base for predictions
        logger.info("  - Running ML churn batch inference...")
        self.df_processed, self.prediction_meta = self.predictor.predict(df_health)

        # 9. Compute global SHAP values
        logger.info("  - Computing global SHAP explainability metrics...")
        self.global_xai_meta = self.explainer.explain_global_importance(self.df_processed)

        # 10. Run KPI calculations
        logger.info("  - Running descriptive BI KPI Engine...")
        from src.analytics.kpi_engine import calculate_kpis

        self.kpi_meta = calculate_kpis(self.df_processed)

        logger.info("Dashboard Data Pipeline execution completed successfully.")

    # =============================================================================
    # PUBLIC API ENDPOINTS (READ-ONLY GETTERS)
    # =============================================================================

    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Returns a high-level summary of the entire customer base."""
        cb = self.kpi_meta["customer_base"]
        hi = self.kpi_meta["health_indices"]

        return {
            "total_customers": cb["total_customers"],
            "active_customers": cb["active_customers"],
            "churned_customers": cb["churned_customers"],
            "churn_rate_pct": cb["churn_rate_pct"],
            "retention_rate_pct": cb["retention_rate_pct"],
            "average_health_score": self.health_meta["average_health_score"],
            "average_satisfaction": self.kpi_meta["engagement"]["avg_satisfaction_score"],
            "complaint_rate_pct": self.kpi_meta["engagement"]["complaint_rate_pct"],
            "critical_risk_count": self.prediction_meta["risk_distribution"]["Critical"]["count"],
            "high_risk_count": self.prediction_meta["risk_distribution"]["High"]["count"],
        }

    def get_kpi_summary(self) -> Dict[str, Any]:
        """Returns all computed KPIs from the KPI engine."""
        return self.kpi_meta

    def get_customer_health_summary(self) -> Dict[str, Any]:
        """Returns customer health distribution and statistics."""
        return self.health_meta

    def get_segmentation_summary(self) -> Dict[str, Any]:
        """Returns customer cohort flag distribution metrics."""
        return self.segmentation_meta

    def get_rfm_summary(self) -> Dict[str, Any]:
        """Returns behavioral RFM summaries and cohort metrics."""
        return self.rfm_meta

    def get_prediction_summary(self) -> Dict[str, Any]:
        """Returns batch inference results and risk statistics."""
        return self.prediction_meta

    def get_xai_summary(self) -> Dict[str, float]:
        """Returns global feature importance weights (SHAP)."""
        return self.global_xai_meta

    def get_customer_360(self, customer_id: int) -> Dict[str, Any]:
        """Returns a 360-degree analytical and ML profile for a single customer.

        Args:
            customer_id: CustomerID of the target customer.

        Returns:
            A dictionary containing core demographic information, engineered metrics,
            RFM scores, health metrics, predictive scores, and local explanations.
        """
        if self.df_processed is None:
            raise KPICalculationError("Processed data mart is not loaded.")

        row = self.df_processed[self.df_processed["CustomerID"] == customer_id]
        if row.empty:
            raise KPICalculationError(f"Customer with ID {customer_id} not found in the processed store.")

        # Extract values
        cust_row = row.iloc[0]

        # 1. Profile information
        profile = {
            "customer_id": int(cust_row["CustomerID"]),
            "gender": str(cust_row["Gender"]),
            "marital_status": str(cust_row["MaritalStatus"]),
            "city_tier": int(cust_row["CityTier"]),
            "preferred_login_device": str(cust_row["PreferredLoginDevice"]),
            "preferred_payment_mode": str(cust_row["PreferredPaymentMode"]),
            "preferred_order_category": str(cust_row["PreferedOrderCat"]),
        }

        # 2. Activity metrics
        activity = {
            "tenure_months": float(cust_row["Tenure"]),
            "days_since_last_order": float(cust_row["DaySinceLastOrder"]),
            "order_count": float(cust_row["OrderCount"]),
            "hour_spend_on_app": float(cust_row["HourSpendOnApp"]),
            "number_of_device_registered": int(cust_row["NumberOfDeviceRegistered"]),
            "number_of_address": int(cust_row["NumberOfAddress"]),
            "complain_logged": int(cust_row["Complain"]),
            "satisfaction_score": int(cust_row["SatisfactionScore"]),
            "cashback_amount": float(cust_row["CashbackAmount"]),
            "coupon_used": float(cust_row["CouponUsed"]),
        }

        # 3. RFM Metrics
        rfm = {
            "recency_score": int(cust_row["Recency_Score"]),
            "frequency_score": int(cust_row["Frequency_Score"]),
            "monetary_score": int(cust_row["Monetary_Score"]),
            "composite_rfm_score": str(cust_row["RFM_Score"]),
            "rfm_sum": int(cust_row["RFM_Sum"]),
            "rfm_segment": str(cust_row["RFM_Segment"]),
            "crm_recommendation": str(cust_row["RFM_Recommendation"]),
        }

        # 4. Health Metrics
        health = {
            "health_score": float(cust_row["CustomerHealthScore"]),
            "health_category": str(cust_row["Health_Category"]),
            "health_grade": str(cust_row["Health_Grade"]),
            "risk_level": str(cust_row["Risk_Level"]),
            "service_quality_score": float(cust_row["Service_Quality_Score"]),
            "purchase_momentum_score": float(cust_row["Purchase_Momentum_Score"]),
            "app_engagement_score": float(cust_row["App_Engagement_Score"]),
            "tenure_loyalty_score": float(cust_row["Tenure_Loyalty_Score"]),
        }

        # 5. Prediction Metrics
        predictions = {
            "predicted_churn": int(cust_row["Churn_Prediction"]),
            "churn_probability": float(cust_row["Churn_Probability"]),
            "confidence_score": float(cust_row["Prediction_Confidence"]),
            "prediction_risk_tier": str(cust_row["Prediction_Risk_Level"]),
            "crm_retention_action": str(cust_row["Prediction_CRM_Action"]),
            "model_version": str(cust_row["Prediction_Model_Version"]),
            "prediction_timestamp": str(cust_row["Prediction_Timestamp"]),
        }

        # 6. Local SHAP explanations
        logger.info(f"Generating local SHAP explainability values for customer: {customer_id}...")
        local_xai = self.explainer.explain_local_customer(self.df_processed, customer_id)

        return {
            "profile": profile,
            "activity_metrics": activity,
            "rfm": rfm,
            "health": health,
            "predictions": predictions,
            "explainability": local_xai,
        }

    def get_dashboard_metadata(self) -> Dict[str, Any]:
        """Returns service health and runtime parameters."""
        return {
            "service_status": "ONLINE",
            "pipeline_last_run_timestamp": datetime.now(timezone.utc).isoformat(),
            "data_source_filepath": self.filepath,
            "processed_records_count": len(self.df_processed) if self.df_processed is not None else 0,
        }
