"""
Enterprise Churn Prediction Training Engine
=============================================
Orchestrates the preprocessing, split, cross-validation, and training of
machine learning models for customer churn prediction. Supports Logistic
Regression, Random Forest, XGBoost, and LightGBM. Automatically handles
categorical encoding, class imbalance, evaluates performance using
business-aligned metrics, selects the best model based on F1-Score,
and serializes artifacts.

Author: Principal Machine Learning Engineer
Version: 1.0.0
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.core.exceptions import ModelTrainingError
from src.core.logger import get_logger

# Optional LightGBM & XGBoost imports
try:
    import lightgbm as lgb

    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

logger = get_logger(__name__)


# =============================================================================
# COLUMN CONTRACT & PREPROCESSING SCHEMA
# =============================================================================

# Categorical features requiring One-Hot Encoding
CATEGORICAL_FEATURES: List[str] = [
    "PreferredLoginDevice",
    "PreferredPaymentMode",
    "Gender",
    "PreferedOrderCat",
    "MaritalStatus",
    "TenureGroup",
    "WarehouseDistanceBucket",
    "OrderFrequencyTier",
]

# Numeric features requiring scaling for linear models
NUMERIC_FEATURES: List[str] = [
    "Tenure",
    "WarehouseToHome",
    "HourSpendOnApp",
    "NumberOfDeviceRegistered",
    "SatisfactionScore",
    "NumberOfAddress",
    "Complain",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "OrderCount",
    "DaySinceLastOrder",
    "CashbackAmount",
    "CityTier",
    "OrderVelocity",
    "AppExposure",
    "CashbackEfficiency",
    "ComplaintFrictionIndex",
    "CustomerLoyaltyScore",
    "AddressStabilityIndex",
    "TenureOrderRatio",
    "RawHealthIndex",
    "RuleBasedChurnIndicator",
]

EXCLUDED_COLUMNS: List[str] = [
    "CustomerID",
    "RFM_Segment",
    "RFM_Recommendation",
    "RFM_Score",
    "Health_Category",
    "Health_Grade",
    "Risk_Level",
]


def prepare_features_and_target(df: pd.DataFrame, target_col: str = "Churn") -> Tuple[pd.DataFrame, pd.Series]:
    """Isolates the target variable and prunes structural columns.

    Args:
        df: Customer DataFrame containing clean and engineered features.
        target_col: Name of the classification label.

    Returns:
        A tuple of (X, y) where X is the feature space and y is the labels.
    """
    if target_col not in df.columns:
        raise ModelTrainingError(f"Target column '{target_col}' not found in DataFrame.")

    # Drop label and non-predictive columns
    cols_to_drop = [target_col] + [c for c in EXCLUDED_COLUMNS if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = df[target_col]

    logger.info(f"Feature matrix isolated: {X.shape[0]} samples, {X.shape[1]} features.")
    return X, y


def build_preprocessors() -> Tuple[ColumnTransformer, ColumnTransformer]:
    """Constructs column transformers for scaling and encoding.

    Returns:
        A tuple containing:
            - lr_preprocessor: Pipeline containing scaling for Logistic Regression.
            - tree_preprocessor: Preprocessor with only categorical encoding.
    """
    # 1. Preprocessor for Linear Models (Scaler + OHE)
    lr_preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    # 2. Preprocessor for Tree-based Models (Only OHE, Scale-invariant)
    tree_preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return lr_preprocessor, tree_preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Extracts post-preprocessing feature names from a ColumnTransformer.

    Allows mapping feature importances and coefficients back to their OHE columns.

    Args:
        preprocessor: Fitted ColumnTransformer.

    Returns:
        List of strings containing post-preprocessed feature names.
    """
    feature_names = []

    for name, trans, cols in preprocessor.transformers_:
        if trans == "drop" or name == "remainder":
            continue
        elif trans == "passthrough":
            feature_names.extend(cols)
        elif isinstance(trans, StandardScaler):
            feature_names.extend(cols)
        elif isinstance(trans, OneHotEncoder):
            # Extract categories from OHE
            cats = trans.categories_
            for i, col in enumerate(cols):
                feature_names.extend([f"{col}_{cat}" for cat in cats[i]])

    return feature_names


def train_churn_models(
    df: pd.DataFrame, target_column: str = "Churn", test_size: float = 0.2, random_state: int = 42
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Trains, cross-validates, and evaluates churn classification models.

    Automatically handles stratified splitting, preprocesses variables, evaluates
    performance metrics, leaderboard rankings, and outputs the winning pipeline.

    Args:
        df: Customer data mart.
        target_column: Name of the classification label.
        test_size: Ratio of train-test split.
        random_state: Random seed.

    Returns:
        A tuple containing:
            - The winning trained scikit-learn Pipeline object.
            - A metadata dictionary detailing comparison logs and leaderboard metrics.
    """
    # 1. Isolate feature space and labels
    X, y = prepare_features_and_target(df, target_column)

    # 2. Stratified Train-Test Split (Imbalanced target preservation)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    logger.info(f"Stratified split complete: Train={len(X_train)} samples, Test={len(X_test)} samples.")

    # 3. Calculate class weight imbalance scale
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    imbalance_ratio = neg_count / pos_count if pos_count > 0 else 1.0
    logger.info(f"Class distribution: Negatives={neg_count}, Positives={pos_count} (Ratio={imbalance_ratio:.2f})")

    # 4. Construct preprocessors
    lr_preprocessor, tree_preprocessor = build_preprocessors()

    # 5. Define candidates registry
    candidates: Dict[str, Dict[str, Any]] = {
        "LogisticRegression": {
            "preprocessor": lr_preprocessor,
            "estimator": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state),
        },
        "RandomForest": {
            "preprocessor": tree_preprocessor,
            "estimator": RandomForestClassifier(
                class_weight="balanced", n_estimators=300, random_state=random_state, n_jobs=-1
            ),
        },
    }

    if XGB_AVAILABLE:
        candidates["XGBoost"] = {
            "preprocessor": tree_preprocessor,
            "estimator": xgb.XGBClassifier(
                scale_pos_weight=imbalance_ratio,
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1,
            ),
        }
    else:
        logger.warning("XGBoost is not installed. Skipping XGBoost from trainer registry.")

    if LGBM_AVAILABLE:
        candidates["LightGBM"] = {
            "preprocessor": tree_preprocessor,
            "estimator": lgb.LGBMClassifier(
                scale_pos_weight=imbalance_ratio,
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                random_state=random_state,
                n_jobs=-1,
                verbose=-1,
            ),
        }
    else:
        logger.warning("LightGBM is not installed. Skipping LightGBM from trainer registry.")

    # 6. Evaluation loop
    leaderboard = {}
    trained_pipelines = {}
    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    for name, conf in candidates.items():
        logger.info(f"Training pipeline: {name}...")
        pipeline = Pipeline([("preprocessor", conf["preprocessor"]), ("model", conf["estimator"])])

        start_time = time.time()

        # Fit pipeline
        pipeline.fit(X_train, y_train)
        training_duration = time.time() - start_time

        # Out-of-fold Cross Validation score on training set
        logger.info(f"  - Executing 5-fold cross-validation for {name}...")
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_splitter, scoring="f1")
        cv_f1_mean = float(cv_scores.mean())
        cv_f1_std = float(cv_scores.std())

        # Test set predictions
        y_pred = pipeline.predict(X_test)
        if hasattr(pipeline.named_steps["model"], "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred.astype(float)

        # Performance calculations
        accuracy = float(accuracy_score(y_test, y_pred))
        precision = float(precision_score(y_test, y_pred, zero_division=0))
        recall = float(recall_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        roc_auc = float(roc_auc_score(y_test, y_prob))
        conf_mat = confusion_matrix(y_test, y_pred).tolist()

        logger.info(f"  - Metrics: F1={f1:.4f} | Recall={recall:.4f} | ROC AUC={roc_auc:.4f}")

        leaderboard[name] = {
            "test_accuracy": round(accuracy, 4),
            "test_precision": round(precision, 4),
            "test_recall": round(recall, 4),
            "test_f1_score": round(f1, 4),
            "test_roc_auc": round(roc_auc, 4),
            "cv_f1_mean": round(cv_f1_mean, 4),
            "cv_f1_std": round(cv_f1_std, 4),
            "training_time_seconds": round(training_duration, 4),
            "confusion_matrix": conf_mat,
        }
        trained_pipelines[name] = pipeline

    # 7. Select Winner based on test set F1-Score (Standard for imbalanced classifiers)
    winner_name = max(leaderboard, key=lambda k: leaderboard[k]["test_f1_score"])
    winner_pipeline = trained_pipelines[winner_name]
    logger.info(f"Winning model selected: {winner_name} with F1-Score={leaderboard[winner_name]['test_f1_score']:.4f}")

    # 8. Extract feature importances or coefficients from winner
    fitted_preprocessor = winner_pipeline.named_steps["preprocessor"]
    fitted_model = winner_pipeline.named_steps["model"]
    feature_names = get_feature_names(fitted_preprocessor)

    importances = {}
    if hasattr(fitted_model, "feature_importances_"):
        raw_imp = fitted_model.feature_importances_
        importances = {feature_names[i]: float(raw_imp[i]) for i in range(len(feature_names))}
    elif hasattr(fitted_model, "coef_"):
        raw_coef = fitted_model.coef_[0]
        importances = {feature_names[i]: float(raw_coef[i]) for i in range(len(feature_names))}

    # Sort importances descending
    sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    # 9. Build and serialize metadata
    metadata = {
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "leaderboard": leaderboard,
        "best_model": {
            "algorithm": winner_name,
            "metrics": leaderboard[winner_name],
            "feature_importance": [
                {"feature": item[0], "importance": round(item[1], 6)} for item in sorted_importances[:20]
            ],
        },
    }

    # Save best model to disk
    from config.settings import MODEL_ARTIFACTS_DIR

    MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_ARTIFACTS_DIR / "best_model.joblib"
    meta_path = MODEL_ARTIFACTS_DIR / "model_metadata.json"

    joblib.dump(winner_pipeline, model_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"Best model serialized to {model_path}")
    logger.info(f"Model metadata serialized to {meta_path}")

    return winner_pipeline, metadata


def print_training_report(metadata: Dict[str, Any]) -> None:
    """Prints a professional, human-readable training evaluation report.

    Args:
        metadata: Dict containing training metadata from train_churn_models().
    """
    border = "=" * 80
    section_divider = "-" * 80

    print(f"\n{border}")
    print("                 ENTERPRISE CHURN PREDICTION LEADERBOARD")
    print(border)
    print(f"Timestamp (UTC): {metadata['training_timestamp']}")
    print(border)

    print(f"\nLEADERBOARD RANKINGS (Sorted by F1-Score)")
    print(section_divider)

    leaderboard = metadata["leaderboard"]
    sorted_leader = sorted(leaderboard.items(), key=lambda x: x[1]["test_f1_score"], reverse=True)

    for rank, (model_name, metrics) in enumerate(sorted_leader, 1):
        print(f"\nRank {rank}: {model_name}")
        print(
            f"  Test F1-Score:    {metrics['test_f1_score']:.4f} (CV F1-Mean = {metrics['cv_f1_mean']:.4f} +/- {metrics['cv_f1_std']:.4f})"
        )
        print(f"  Test Recall:      {metrics['test_recall']:.4f}")
        print(f"  Test Precision:   {metrics['test_precision']:.4f}")
        print(f"  Test ROC AUC:     {metrics['test_roc_auc']:.4f}")
        print(f"  Accuracy:         {metrics['test_accuracy']:.4f}")
        print(
            f"  Training Time:    {metrics['training_duration_seconds'] if 'training_duration_seconds' in metrics else metrics['training_time_seconds']:.4f}s"
        )
        print(
            f"  Confusion Matrix: TN={metrics['confusion_matrix'][0][0]} | FP={metrics['confusion_matrix'][0][1]} | FN={metrics['confusion_matrix'][1][0]} | TP={metrics['confusion_matrix'][1][1]}"
        )

    print(f"\n{border}")
    print("                 WINNING MODEL & EXPLANATION")
    print(border)
    best = metadata["best_model"]
    print(f"Winning Algorithm: {best['algorithm']}")
    print(f"Top 10 Feature Drivers:")
    for idx, item in enumerate(best["feature_importance"][:10], 1):
        print(f"  {idx:2d}. {item['feature']:<35} : {item['importance']:.6f}")

    print(f"\n{border}")
    print("                 END OF CHURN TRAINING REPORT")
    print(f"{border}\n")
