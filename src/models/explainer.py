"""
Enterprise Explainable AI (XAI) Engine
========================================
Implements post-hoc explainability layers for the customer churn classifier,
utilizing SHAP (SHapley Additive exPlanations) values. Computes global feature
attributions and generates natural-language customer-level local explanations
detailing the specific positive and negative drivers behind a customer's
churn prediction.

This module resolves pipeline preprocessors and maps SHAP attributions back to
their post-encoded categorical columns.

Author: Principal Explainable AI Engineer
Version: 1.0.0
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src.core.exceptions import ModelInferenceError
from src.core.logger import get_logger
from src.models.trainer import CATEGORICAL_FEATURES, NUMERIC_FEATURES, get_feature_names

logger = get_logger(__name__)


class ChurnExplainer:
    """Manages global and local model explainability calculations using SHAP."""

    def __init__(self, artifact_dir: Optional[Union[str, Path]] = None) -> None:
        """Initializes the explainer, loading the model and metadata.

        Args:
            artifact_dir: Directory containing serialized model artifacts.
        """
        if artifact_dir is None:
            from config.settings import MODEL_ARTIFACTS_DIR

            self.artifact_path = Path(MODEL_ARTIFACTS_DIR)
        else:
            self.artifact_path = Path(artifact_dir)

        self.model_file = self.artifact_path / "best_model.joblib"
        self.meta_file = self.artifact_path / "model_metadata.json"

        self.pipeline: Pipeline = None  # type: ignore
        self.metadata: Dict[str, Any] = {}
        self.explainer: Optional[shap.TreeExplainer] = None
        self.is_shap_available: bool = False

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Deserializes artifacts and initializes the SHAP TreeExplainer."""
        if not self.model_file.exists():
            raise ModelInferenceError(f"Model file not found at: {self.model_file}")

        try:
            self.pipeline = joblib.load(self.model_file)
            logger.info("Pipeline loaded successfully inside ChurnExplainer.")

            # Load metadata
            if self.meta_file.exists():
                with open(self.meta_file, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)

            # Initialize SHAP TreeExplainer on the fitted model step
            fitted_model = self.pipeline.named_steps["model"]
            try:
                # TreeExplainer is fast and optimized for gradient boosted trees
                self.explainer = shap.TreeExplainer(fitted_model)
                self.is_shap_available = True
                logger.info("SHAP TreeExplainer successfully initialized.")
            except Exception as e:
                logger.warning(f"Failed to initialize SHAP TreeExplainer: {e}. Falling back to default importance.")
                self.is_shap_available = False

        except Exception as e:
            raise ModelInferenceError(f"Failed to load explainer artifacts: {e}") from e

    def _get_preprocessed_df(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocesses input DataFrame to match model post-OHE columns.

        Args:
            df: Input customer DataFrame.

        Returns:
            A tuple containing:
                - preprocessed_df: DataFrame with post-OHE features.
                - customer_ids: Series containing CustomerIDs for alignment.
        """
        df_copy = df.copy()

        # Save CustomerID if present
        customer_ids = df_copy["CustomerID"] if "CustomerID" in df_copy.columns else df_copy.index.to_series()

        # Drop label and operational keys
        from src.models.trainer import EXCLUDED_COLUMNS

        cols_to_drop = ["Churn", "CustomerID"] + [c for c in EXCLUDED_COLUMNS if c in df_copy.columns]
        features = df_copy.drop(columns=cols_to_drop, errors="ignore")

        # Select only expected columns
        model_inputs = features[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

        # Apply pipeline preprocessing transformer
        preprocessor = self.pipeline.named_steps["preprocessor"]
        X_preprocessed = preprocessor.transform(model_inputs)

        # Use native scikit-learn feature name extraction to ensure shape alignment
        raw_names = preprocessor.get_feature_names_out()
        feature_names = [name.replace("num__", "").replace("cat__", "") for name in raw_names]

        preprocessed_df = pd.DataFrame(X_preprocessed, columns=feature_names)
        return preprocessed_df, customer_ids

    def explain_global_importance(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculates global feature contributions using SHAP mean absolute values.

        Falls back to Gini/LGBM split importance if SHAP is disabled.

        Args:
            df: Customer DataFrame.

        Returns:
            Dictionary mapping feature names to importance scores.
        """
        X_prep, _ = self._get_preprocessed_df(df)

        if not self.is_shap_available or self.explainer is None:
            logger.warning("SHAP unavailable. Retrieving model internal feature importances.")
            fitted_model = self.pipeline.named_steps["model"]
            feature_names = X_prep.columns.tolist()
            importances = fitted_model.feature_importances_
            return {feature_names[i]: float(importances[i]) for i in range(len(feature_names))}

        try:
            logger.info("Computing global SHAP values...")
            # Compute SHAP values for the batch
            # We use the explainer check for probability outputs or raw margins
            shap_values = self.explainer.shap_values(X_prep)

            # For binary classification, shap_values can be a list [class_0, class_1]
            # or a single 2D array depending on shap package/model type.
            if isinstance(shap_values, list) and len(shap_values) == 2:
                # Class 1 (churn) SHAP values
                shap_values_class = shap_values[1]
            else:
                shap_values_class = shap_values

            # Mean absolute SHAP value represents global impact
            mean_abs_shap = np.abs(shap_values_class).mean(axis=0)
            importance_dict = {X_prep.columns[i]: float(mean_abs_shap[i]) for i in range(len(X_prep.columns))}
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

        except Exception as e:
            logger.error(f"Failed to calculate global SHAP values: {e}. Falling back to internal importances.")
            fitted_model = self.pipeline.named_steps["model"]
            feature_names = X_prep.columns.tolist()
            importances = fitted_model.feature_importances_
            return {feature_names[i]: float(importances[i]) for i in range(len(feature_names))}

    def explain_local_customer(self, df: pd.DataFrame, customer_id: int) -> Dict[str, Any]:
        """Generates a local explanation for a specific customer.

        Args:
            df: Customer DataFrame containing the target customer record.
            customer_id: CustomerID value of the customer to explain.

        Returns:
            A dictionary containing SHAP values, risk scores, positive/negative
            drivers, and an executive natural-language explanation.
        """
        # Filter for customer row
        if "CustomerID" not in df.columns:
            raise ModelInferenceError("CustomerID column missing from input DataFrame.")

        customer_row = df[df["CustomerID"] == customer_id]
        if customer_row.empty:
            raise ModelInferenceError(f"Customer with ID {customer_id} not found in the dataset.")

        X_prep, c_ids = self._get_preprocessed_df(customer_row)

        # Get raw probability
        features_to_drop = ["Churn", "CustomerID"] + [
            c
            for c in ["RFM_Segment", "RFM_Score", "Health_Category", "Health_Grade", "Risk_Level"]
            if c in customer_row.columns
        ]
        raw_features = customer_row.drop(columns=features_to_drop, errors="ignore")[
            NUMERIC_FEATURES + CATEGORICAL_FEATURES
        ]
        prob = float(self.pipeline.predict_proba(raw_features)[0, 1])

        explanation: Dict[str, Any] = {
            "customer_id": customer_id,
            "churn_probability": round(prob, 4),
            "risk_level": (
                "Critical" if prob >= 0.90 else "High" if prob >= 0.75 else "Medium" if prob >= 0.50 else "Low"
            ),
            "drivers": {"positive_churn_drivers": [], "negative_churn_drivers": []},
            "executive_summary": "",
        }

        if not self.is_shap_available or self.explainer is None:
            explanation["executive_summary"] = (
                "Explanation fallback: Model predictions are based on default global feature importances."
            )
            return explanation

        try:
            # Calculate local SHAP values
            shap_values = self.explainer.shap_values(X_prep)
            if isinstance(shap_values, list) and len(shap_values) == 2:
                local_shap = shap_values[1][0]  # First row (only row) of class 1
            else:
                local_shap = shap_values[0]

            feature_names = X_prep.columns.tolist()
            feature_values = X_prep.iloc[0].tolist()

            drivers = []
            for i in range(len(feature_names)):
                drivers.append(
                    {"feature": feature_names[i], "value": feature_values[i], "shap_contribution": float(local_shap[i])}
                )

            # Sort by SHAP contributions
            drivers_sorted = sorted(drivers, key=lambda x: x["shap_contribution"], reverse=True)

            # Positive drivers (push score towards churn)
            pos_drivers = [d for d in drivers_sorted if d["shap_contribution"] > 0.001][:5]
            # Negative drivers (push score away from churn/keep customer active)
            neg_drivers = [d for d in reversed(drivers_sorted) if d["shap_contribution"] < -0.001][:5]

            explanation["drivers"]["positive_churn_drivers"] = pos_drivers
            explanation["drivers"]["negative_churn_drivers"] = neg_drivers

            # Generate natural-language explanation
            summary_bullets = []
            if prob >= 0.50:
                summary_bullets.append(
                    f"Customer {customer_id} is flagged at Churn Risk ({prob:.1%}) driven primarily by:"
                )
                for d in pos_drivers[:3]:
                    # Format feature text logically
                    feat_desc = d["feature"].replace("_", " ")
                    summary_bullets.append(f"  - High {feat_desc} (Impact: +{d['shap_contribution']:.2f})")
            else:
                summary_bullets.append(
                    f"Customer {customer_id} is classified as Low Churn Risk ({prob:.1%}). Active indicators include:"
                )
                for d in neg_drivers[:3]:
                    feat_desc = d["feature"].replace("_", " ")
                    summary_bullets.append(f"  - Positive {feat_desc} (Impact: {d['shap_contribution']:.2f})")

            explanation["executive_summary"] = "\n".join(summary_bullets)

        except Exception as e:
            logger.error(f"Local SHAP explanation calculation failed: {e}")
            explanation["executive_summary"] = f"Error generating explainability reports: {e}"

        return explanation
