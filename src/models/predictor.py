"""
Enterprise Churn Prediction Inference Engine
=============================================
Loads the serialized production model artifact (`best_model.joblib`)
and executes inference on customer records. Classifies prediction results
into risk tiers, generates custom customer retention action plans,
and compiles aggregate batch prediction metrics.

Author: Principal AI Platform Engineer
Version: 1.0.0
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.core.exceptions import ModelInferenceError
from src.core.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# OPERATIONAL CRM ENGAGEMENT MAP
# =============================================================================
# Action recommendations mapped to customer risk levels
# =============================================================================

CRM_PLAYBOOK_MAP: Dict[str, str] = {
    "Critical": "Immediate Retention Call: Dispatch account manager or customer success specialist to resolve complaints.",
    "High": "Loyalty Campaign: Deploy high-incentive target cashback vouchers and satisfaction recovery surveys.",
    "Medium": "Premium Offer: Send standard loyalty upgrade offers and category cross-selling incentives.",
    "Low": "No Action Required: Continue standard baseline newsletter and support services.",
}


# =============================================================================
# MODEL UTILITY & LOADER
# =============================================================================


class ChurnPredictor:
    """Manages model loading, schema compatibility validation, and inference routing."""

    def __init__(self, artifact_dir: Optional[Union[str, Path]] = None) -> None:
        """Initializes the predictor, loading the best model and metadata.

        Args:
            artifact_dir: Directory containing 'best_model.joblib' and 'model_metadata.json'.
                          Falls back to settings.py path if None.
        """
        if artifact_dir is None:
            from config.settings import MODEL_ARTIFACTS_DIR

            self.artifact_path = Path(MODEL_ARTIFACTS_DIR)
        else:
            self.artifact_path = Path(artifact_dir)

        self.model_file = self.artifact_path / "best_model.joblib"
        self.meta_file = self.artifact_path / "model_metadata.json"

        self.model: Optional[Pipeline] = None
        self.metadata: Dict[str, Any] = {}
        self.model_version: str = "1.0.0"
        self.algorithm_name: str = "Unknown"

        self._load_model_artifacts()

    def _load_model_artifacts(self) -> None:
        """Loads and validates model and metadata files from disk.

        Raises:
            ModelInferenceError: If model files are missing or corrupted.
        """
        if not self.model_file.exists():
            error_msg = f"Model artifact file not found at: {self.model_file}"
            logger.error(error_msg)
            raise ModelInferenceError(error_msg)

        if not self.meta_file.exists():
            logger.warning(f"Model metadata file missing at: {self.meta_file}. Proceeding with default config.")
        else:
            try:
                with open(self.meta_file, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                    self.algorithm_name = self.metadata.get("best_model", {}).get("algorithm", "Unknown")
            except Exception as e:
                logger.warning(f"Failed to read model metadata: {e}. Proceeding with defaults.")

        try:
            self.model = joblib.load(self.model_file)
            logger.info(f"Successfully loaded production model: '{self.algorithm_name}' from {self.model_file.name}")
        except Exception as e:
            error_msg = f"Failed to deserialize model joblib artifact: {e}"
            logger.error(error_msg)
            raise ModelInferenceError(error_msg) from e

    def predict(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Executes classification scoring and risk tiering for input records.

        Args:
            df: Customer DataFrame containing required input features.

        Returns:
            A tuple containing:
                - DataFrame enriched with predictions, probabilities, grades,
                  risk levels, and action recommendations.
                - A dictionary containing batch performance stats.

        Raises:
            ModelInferenceError: If model is not loaded or feature schema is incompatible.
        """
        if self.model is None:
            raise ModelInferenceError("Model pipeline not loaded. Inference cannot proceed.")

        if df.empty:
            raise ModelInferenceError("Cannot execute predictions on an empty DataFrame.")

        logger.info(f"Running inference on {len(df)} customer records.")
        df_pred = df.copy()

        # Step 1: Execute predictions and probabilities
        try:
            # We predict on the features, mapping through pipeline preprocessing
            # Target variable (Churn) is dropped if present to prevent target leak warnings
            features = df_pred.drop(columns=["Churn"], errors="ignore")
            # If CustomerID is present, drop it before feeding to pipeline
            features = features.drop(columns=["CustomerID"], errors="ignore")

            # Prune extra operational columns not in our training preprocessor features
            # Keep only the features that were explicitly expected by theColumnTransformer
            # (which matches standard columns list defined in trainer.py)
            from src.models.trainer import CATEGORICAL_FEATURES, NUMERIC_FEATURES

            model_inputs = features[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

            predictions = self.model.predict(model_inputs)
            probabilities = self.model.predict_proba(model_inputs)[:, 1]

        except Exception as e:
            error_msg = f"Feature schema incompatibility or preprocessing failure: {e}"
            logger.error(error_msg)
            raise ModelInferenceError(error_msg) from e

        # Step 2: Enrich DataFrame
        df_pred["Churn_Prediction"] = predictions
        df_pred["Churn_Probability"] = probabilities.round(4)

        # Confidence Score: if predicted churn (1), confidence is probability.
        # If predicted active (0), confidence is (1 - probability)
        df_pred["Prediction_Confidence"] = np.where(predictions == 1, probabilities, 1.0 - probabilities).round(4)

        # Step 3: Classify Risk Levels
        # Prob >= 0.90 -> Critical, 0.75-0.89 -> High, 0.50-0.74 -> Medium, < 0.50 -> Low
        bins = [-0.01, 0.4999, 0.7499, 0.8999, 1.01]
        labels = ["Low", "Medium", "High", "Critical"]
        df_pred["Prediction_Risk_Level"] = pd.cut(probabilities, bins=bins, labels=labels)
        df_pred["Prediction_Risk_Level"] = df_pred["Prediction_Risk_Level"].astype(str)

        # Step 4: Map CRM Action recommendations
        df_pred["Prediction_CRM_Action"] = df_pred["Prediction_Risk_Level"].map(CRM_PLAYBOOK_MAP)

        # Append execution metadata columns
        df_pred["Prediction_Timestamp"] = datetime.now(timezone.utc).isoformat()
        df_pred["Prediction_Model_Version"] = self.model_version

        # Step 5: Compile Batch stats
        total_records = len(df_pred)
        avg_prob = float(probabilities.mean())
        risk_counts = df_pred["Prediction_Risk_Level"].value_counts()

        critical_count = int(risk_counts.get("Critical", 0))
        high_count = int(risk_counts.get("High", 0))
        medium_count = int(risk_counts.get("Medium", 0))
        low_count = int(risk_counts.get("Low", 0))

        batch_stats = {
            "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
            "model_version": self.model_version,
            "algorithm_used": self.algorithm_name,
            "total_predictions": total_records,
            "average_churn_probability": round(avg_prob, 4),
            "risk_distribution": {
                "Critical": {"count": critical_count, "percentage": round((critical_count / total_records) * 100, 2)},
                "High": {"count": high_count, "percentage": round((high_count / total_records) * 100, 2)},
                "Medium": {"count": medium_count, "percentage": round((medium_count / total_records) * 100, 2)},
                "Low": {"count": low_count, "percentage": round((low_count / total_records) * 100, 2)},
            },
        }

        logger.info(f"Batch prediction completed: Average churn probability={avg_prob:.4f}")
        return df_pred, batch_stats


def print_batch_prediction_report(metadata: Dict[str, Any]) -> None:
    """Prints a professional, human-readable summary of batch predictions.

    Args:
        metadata: Dict containing prediction summary stats generated by predict().
    """
    border = "=" * 80
    section_divider = "-" * 80

    print(f"\n{border}")
    print("                 ENTERPRISE CHURN PREDICTION BATCH INFERENCE REPORT")
    print(border)
    print(f"Timestamp (UTC): {metadata['prediction_timestamp']}")
    print(f"Model Version:   {metadata['model_version']} (Algorithm: '{metadata['algorithm_used']}')")
    print(f"Total Scored:    {metadata['total_predictions']:,} customer records")
    print(f"Average Prob:    {metadata['average_churn_probability']:.4%}")
    print(border)

    print("\nRISK DISTRIBUTION ANALYSIS")
    print(section_divider)

    dist = metadata["risk_distribution"]
    for risk_tier, info in dist.items():
        print(f"  - Risk Tier: {risk_tier:<10} | Count: {info['count']:,} ({info['percentage']}%)")

    print("\nCRM ENGAGEMENT PLAYBOOK ACTION SUMMARY")
    print(section_divider)
    for risk_tier in ["Critical", "High", "Medium", "Low"]:
        count = dist[risk_tier]["count"]
        if count > 0:
            print(f"\n* PLAYBOOK FOR {risk_tier.upper()} RISK COHORT ({count:,} customers)")
            print(f"  Action: {CRM_PLAYBOOK_MAP[risk_tier]}")

    print(f"\n{border}")
    print("                 END OF BATCH PREDICTION INFERENCE REPORT")
    print(f"{border}\n")
