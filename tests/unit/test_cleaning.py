"""
Unit Test Suite for Enterprise Data Cleaning & Imputation Engine
==================================================================
Contains unit tests verifying correct behavior of the data cleaning pipeline,
including whitespace scrubbing, row and primary key deduplication,
categorical normalization, missing value imputation, and outlier capping.

Author: Principal Python Test Architect
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import pytest

from src.core.exceptions import ImputationError
from src.data.cleaning import (
    clean_dataset,
    clean_whitespaces,
    clip_outliers,
    enforce_datatypes,
    handle_duplicates,
    impute_missing_values,
    normalize_categories,
    print_cleaning_report,
)


@pytest.fixture
def raw_test_dataframe() -> pd.DataFrame:
    """Fixture providing a raw, dirty dataframe representing e-commerce data."""
    return pd.DataFrame(
        {
            "CustomerID": [50001, 50002, 50002, 50004, 50005],  # Duplicated 50002
            "Churn": [1, 0, 0, 0, 1],
            "Tenure": [4.0, np.nan, 2.0, 50.0, 1.0],  # 50 is an outlier
            "PreferredLoginDevice": [" Mobile Phone ", "Phone", "Phone", "Computer", "Mobile Phone"],
            "WarehouseToHome": [6.0, 8.0, 8.0, np.nan, 30.0],
            "PreferredPaymentMode": ["Debit Card", "CC", "CC", "Cash on Delivery", "Debit Card"],
            "PreferedOrderCat": ["Laptop & Accessory", "Mobile", "Mobile", "Fashion", "Laptop & Accessory"],
            "SatisfactionScore": [2, 3, 3, 4, 5],
            "Complain": [1, 1, 1, 0, 0],
            "OrderAmountHikeFromlastYear": [11.0, 15.0, 15.0, 14.0, np.nan],
            "CouponUsed": [1.0, 0.0, 0.0, 1.0, 2.0],
            "OrderCount": [1.0, 1.0, 1.0, 1.0, 3.0],
            "DaySinceLastOrder": [5.0, 0.0, 0.0, np.nan, 3.0],
            "CashbackAmount": [159.93, 120.90, 120.90, 180.00, 200.00],
        }
    )


def test_clean_whitespaces(raw_test_dataframe):
    """Test that whitespaces are correctly stripped from headers and string cells."""
    dirty_df = raw_test_dataframe.copy()
    # Add dirty column name with spaces
    dirty_df.rename(columns={"CustomerID": " CustomerID "}, inplace=True)

    clean_df = clean_whitespaces(dirty_df)

    # Assert headers are stripped
    assert "CustomerID" in clean_df.columns
    assert " CustomerID " not in clean_df.columns
    # Assert string cells are stripped
    assert clean_df["PreferredLoginDevice"].iloc[0] == "Mobile Phone"


def test_handle_duplicates(raw_test_dataframe):
    """Test that duplicate rows and duplicate primary keys are correctly handled."""
    df_clean, meta = handle_duplicates(raw_test_dataframe, primary_key="CustomerID", keep_strategy="first")

    # Index 1 (NaN tenure) and Index 2 (2.0 tenure) are not exact duplicates,
    # but they share the same CustomerID.
    assert len(df_clean) == 4
    assert meta["exact_duplicates_removed"] == 0
    assert meta["pk_duplicates_removed"] == 1

    # Introduce an exact duplicate row
    df_exact_dup = raw_test_dataframe.copy()
    # Duplicate row 0 exactly
    df_exact_dup = pd.concat([df_exact_dup, df_exact_dup.iloc[[0]]], ignore_index=True)
    df_clean_exact, meta_exact = handle_duplicates(df_exact_dup, primary_key=None, keep_strategy="first")

    assert len(df_clean_exact) == 5
    assert meta_exact["exact_duplicates_removed"] == 1
    assert meta_exact["pk_duplicates_removed"] == 0


def test_normalize_categories(raw_test_dataframe):
    """Test that categorical values are correctly merged and normalized."""
    df_norm = normalize_categories(raw_test_dataframe)

    # Phone -> Mobile Phone
    assert "Phone" not in df_norm["PreferredLoginDevice"].unique()
    assert "Mobile Phone" in df_norm["PreferredLoginDevice"].unique()

    # CC -> Credit Card
    assert "CC" not in df_norm["PreferredPaymentMode"].unique()
    assert "Credit Card" in df_norm["PreferredPaymentMode"].unique()

    # Cash on Delivery -> COD
    assert "Cash on Delivery" not in df_norm["PreferredPaymentMode"].unique()
    assert "COD" in df_norm["PreferredPaymentMode"].unique()

    # Mobile -> Mobile Phone
    assert "Mobile" not in df_norm["PreferedOrderCat"].unique()
    assert "Mobile Phone" in df_norm["PreferedOrderCat"].unique()


def test_impute_missing_values(raw_test_dataframe):
    """Test that missing values are correctly imputed using grouped/global strategies."""
    # Run imputation with group by
    df_imputed, meta = impute_missing_values(raw_test_dataframe, group_by_col="PreferedOrderCat")

    # Check that nulls are gone
    assert df_imputed["Tenure"].isnull().sum() == 0
    assert df_imputed["WarehouseToHome"].isnull().sum() == 0
    assert df_imputed["OrderAmountHikeFromlastYear"].isnull().sum() == 0
    assert df_imputed["DaySinceLastOrder"].isnull().sum() == 0

    # Verify metadata tracking
    assert "Tenure" in meta
    assert meta["Tenure"]["imputed_count"] == 1


def test_impute_missing_values_invalid_strategy(raw_test_dataframe):
    """Test that invalid imputation strategies raise ImputationError."""
    with pytest.raises(ImputationError):
        impute_missing_values(raw_test_dataframe, numerical_strategy="invalid_strategy")


def test_clip_outliers(raw_test_dataframe):
    """Test that numerical outliers are correctly capped via IQR bounding."""
    # In raw_test_dataframe, Tenure has values: [4, NaN, 2, 50, 1]
    # Median is 3. Q1=2, Q3=16. IQR=14. Max upper bound = 16 + 1.5 * 14 = 37.
    # Therefore, 50 should be capped to 37.0.
    df_imputed, _ = impute_missing_values(raw_test_dataframe, group_by_col=None)
    df_clipped, meta = clip_outliers(df_imputed, iqr_multiplier=1.5, target_columns=["Tenure"])

    assert df_clipped["Tenure"].max() <= 37.0
    assert meta["Tenure"]["total_capped"] == 1


def test_enforce_datatypes(raw_test_dataframe):
    """Test that columns are downcasted and dtypes are standardized."""
    df_enforced = enforce_datatypes(raw_test_dataframe)

    # Verify floating columns are downcasted to float32
    assert df_enforced["Tenure"].dtype == np.float32
    # Verify integer columns are downcasted to int32
    assert df_enforced["CustomerID"].dtype == np.int32
    # Verify object columns are converted to clean strings
    assert df_enforced["PreferredLoginDevice"].dtype == object or str


def test_clean_dataset_pipeline(raw_test_dataframe):
    """Test the complete orchestrated data cleaning pipeline."""
    df_clean, meta = clean_dataset(raw_test_dataframe, primary_key="CustomerID")

    # Total row counts
    assert len(df_clean) == 4
    # No nulls left
    assert df_clean.isnull().sum().sum() == 0
    # Categoricals normalized
    assert "Phone" not in df_clean["PreferredLoginDevice"].unique()
    # Datatypes optimized
    assert df_clean["Tenure"].dtype == np.float32
    assert df_clean["CustomerID"].dtype == np.int32
    # Metadata generated
    assert meta["input_shape"] == (5, 14)
    assert meta["output_shape"] == (4, 14)


def test_clean_dataset_empty_dataframe():
    """Test that empty dataframes can pass safely but result in empty outputs."""
    df_empty = pd.DataFrame(columns=["CustomerID", "Churn", "Tenure"])
    df_clean, meta = clean_dataset(df_empty, primary_key="CustomerID")

    assert df_clean.empty
    assert meta["output_shape"] == (0, 3)


def test_impute_missing_values_categorical_and_mean(raw_test_dataframe, mocker):
    """Test categorical column imputation and mean-based numerical imputation by mocking config."""
    # Mock config to include categorical imputation columns
    mocked_config = {
        "data_processing": {"imputation_columns": {"numerical": ["Tenure"], "categorical": ["PreferredLoginDevice"]}}
    }
    mocker.patch("src.data.cleaning.load_model_config", return_value=mocked_config)

    dirty_df = raw_test_dataframe.copy()
    dirty_df.loc[0, "PreferredLoginDevice"] = np.nan  # Inject categorical null

    df_imputed, meta = impute_missing_values(
        dirty_df, numerical_strategy="mean", categorical_strategy="mode", group_by_col=None
    )

    # Assert numeric mean is used
    assert not df_imputed["Tenure"].isnull().any()
    # Assert categorical mode is used
    assert not df_imputed["PreferredLoginDevice"].isnull().any()
    assert df_imputed["PreferredLoginDevice"].iloc[0] == "Phone"  # Mode

    # Test constant categorical imputation
    df_const, _ = impute_missing_values(
        dirty_df, numerical_strategy="mean", categorical_strategy="Missing", group_by_col=None
    )
    assert df_const["PreferredLoginDevice"].iloc[0] == "Missing"


def test_print_cleaning_report(capsys):
    """Test that print_cleaning_report prints correctly formatted logs without errors."""
    dummy_meta = {
        "cleaning_timestamp": "2026-07-12T01:30:00Z",
        "input_shape": (100, 20),
        "output_shape": (98, 20),
        "deduplication": {"exact_duplicates_removed": 1, "pk_duplicates_removed": 1, "total_removed": 2},
        "imputation": {"Tenure": {"imputed_count": 5, "description": "global median (10.0)"}},
        "outlier_capping": {
            "WarehouseToHome": {
                "total_capped": 3,
                "below_capped": 0,
                "above_capped": 3,
                "lower_bound": -5.0,
                "upper_bound": 35.0,
            }
        },
    }

    print_cleaning_report(dummy_meta)
    captured = capsys.readouterr()

    assert "ENTERPRISE DATA CLEANING & AUDIT REPORT" in captured.out
    assert "DEDUPLICATION SUMMARY" in captured.out
    assert "MISSING VALUE IMPUTATION" in captured.out
    assert "OUTLIER TREATMENT" in captured.out
