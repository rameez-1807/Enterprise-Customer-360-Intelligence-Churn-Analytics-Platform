"""
Enterprise Customer 360 Pipeline Orchestration Entry Point
============================================================
Primary execution controller for the analytics and machine learning pipeline.
Orchestrates the sequential execution of:
    1. Data Ingestion & Validation
    2. Data Cleaning & Outlier Capping
    3. Feature Engineering & RFM Scoring
    4. Customer flag Segmentation & Health Grading
    5. Machine Learning Training & Leaderboard Comparisons
    6. Batch Inference and MLOps serialization

Author: Principal Python Engineer
Version: 1.0.0
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from src.core.exceptions import Customer360BaseError
from src.core.logger import get_logger
from src.data.ingestion import ingest_dataset
from src.data.cleaning import clean_dataset
from src.features.builder import build_features
from src.features.rfm import analyze_rfm
from src.analytics.segmentation import segment_customers
from src.analytics.customer_health import compute_customer_health
from src.models.trainer import train_churn_models, print_training_report
from src.models.predictor import ChurnPredictor, print_batch_prediction_report

logger = get_logger(__name__)


def run_data_pipeline(filepath: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Runs ingestion, cleaning, features, RFM, segmentation, and health engines.

    Args:
        filepath: Path to the Excel dataset.

    Returns:
        A tuple containing:
            - Fully processed and graded customer DataFrame.
            - Combined dictionary containing metadata logs.
    """
    logger.info("Executing Enterprise Data Engineering & Analytics Pipeline:")
    
    # 1. Ingestion
    logger.info("  - Ingesting dataset...")
    df_raw, ingest_meta = ingest_dataset(filepath=filepath, optimize_memory=False)
    
    # 2. Cleaning
    logger.info("  - Running cleaning and imputation...")
    df_clean, cleaning_meta = clean_dataset(df_raw, primary_key="CustomerID")
    
    # 3. Features
    logger.info("  - Building derived features...")
    df_feat, feature_meta = build_features(df_clean)
    
    # 4. Behavioral RFM
    logger.info("  - Scoring RFM cohorts...")
    df_rfm, rfm_meta = analyze_rfm(df_feat)
    
    # 5. Customer Segmentation
    logger.info("  - Running flags segmentation...")
    df_seg, seg_meta = segment_customers(df_rfm)
    
    # 6. Customer Health
    logger.info("  - Scoring customer health indexes...")
    df_health, health_meta = compute_customer_health(df_seg)
    
    logger.info(f"Data pipeline complete. Scored mart shape: {df_health.shape[0]} rows × {df_health.shape[1]} columns.")
    
    combined_meta = {
        "ingest": ingest_meta,
        "cleaning": cleaning_meta,
        "features": feature_meta,
        "rfm": rfm_meta,
        "segmentation": seg_meta,
        "health": health_meta
    }
    
    return df_health, combined_meta


def main() -> None:
    """Entry point parsing command line arguments and routing execution."""
    from config.settings import RAW_DATASET_PATH
    
    parser = argparse.ArgumentParser(description="Enterprise Customer 360 Analytics CLI Entry Point")
    parser.add_argument(
        "--pipeline",
        choices=["full", "data-only", "predict-only"],
        default="full",
        help="Specify the pipeline execution mode (default: full)."
    )
    parser.add_argument(
        "--file",
        type=str,
        default=str(RAW_DATASET_PATH),
        help="Path to the Excel dataset."
    )
    
    args = parser.parse_args()
    
    logger.info(f"Starting pipeline orchestrator in Mode: '{args.pipeline.upper()}'")
    logger.info(f"Target File: {args.file}")

    try:
        if args.pipeline == "full":
            # Run data pipeline
            df_graded, _ = run_data_pipeline(args.file)
            
            # Train and rank model classifiers
            logger.info("  - Starting ML training comparison loop...")
            _, train_meta = train_churn_models(df_graded, target_column="Churn")
            print_training_report(train_meta)
            
        elif args.pipeline == "data-only":
            # Run data pipeline only
            df_graded, _ = run_data_pipeline(args.file)
            # Save intermediate staging dataset
            out_dir = Path("data/processed")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "graded_customers.csv"
            df_graded.to_csv(out_path, index=False)
            logger.info(f"Staged processed data mart saved to {out_path}")
            
        elif args.pipeline == "predict-only":
            # Ingest and clean, then run predictor inference
            df_graded, _ = run_data_pipeline(args.file)
            
            logger.info("  - Initializing ML Predictor and running inference...")
            predictor = ChurnPredictor()
            _, pred_meta = predictor.predict(df_graded)
            print_batch_prediction_report(pred_meta)

    except Exception as e:
        logger.error(f"Pipeline execution encountered a critical error: {e}")
        sys.exit(1)

    logger.info("Platform pipeline run completed successfully.")


if __name__ == "__main__":
    main()
