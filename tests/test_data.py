"""
A src/data.py modul tesztjei.

Ezek a tesztek ellenőrzik:
- Az adat betöltését
- A feature oszlopok azonosítását
- A unit-szintű split helyességét (nincs data leakage)
- Az X/y szétválasztást
"""

from pathlib import Path

import pandas as pd
import pytest

from src.data import (
    get_feature_columns,
    load_raw_data,
    prepare_xy,
    split_by_unit,
)


# --- Fixture-ök ---

@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    """Egy kis minta adathalmaz a tesztekhez (nem a teljes)."""
    data_path = Path("data/raw/sensor_data.csv")
    if not data_path.exists():
        pytest.skip(f"Adatfájl nem található: {data_path}. Futtasd: python -m src.generate_data")
    return load_raw_data(data_path)


# --- Tesztek ---

def test_load_raw_data_returns_dataframe(sample_df: pd.DataFrame) -> None:
    """A load_raw_data DataFrame-et ad vissza."""
    assert isinstance(sample_df, pd.DataFrame)
    assert len(sample_df) > 0


def test_expected_columns(sample_df: pd.DataFrame) -> None:
    """Az adathalmaz tartalmazza a várt oszlopokat."""
    expected = {"unit", "cycle", "anomaly"}
    assert expected.issubset(set(sample_df.columns))


def test_get_feature_columns(sample_df: pd.DataFrame) -> None:
    """A feature oszlopok s1..s21."""
    feature_cols = get_feature_columns(sample_df)
    assert len(feature_cols) == 21
    assert all(c.startswith("s") for c in feature_cols)


def test_split_by_unit_no_overlap(sample_df: pd.DataFrame) -> None:
    """A unit-szintű split nem tartalmaz átfedést (nincs data leakage)."""
    train_df, val_df, test_df = split_by_unit(sample_df)

    train_units = set(train_df["unit"].unique())
    val_units = set(val_df["unit"].unique())
    test_units = set(test_df["unit"].unique())

    assert train_units.isdisjoint(val_units), "Train és Val átfed!"
    assert train_units.isdisjoint(test_units), "Train és Test átfed!"
    assert val_units.isdisjoint(test_units), "Val és Test átfed!"


def test_split_proportions(sample_df: pd.DataFrame) -> None:
    """A split arányai közel a 70/15/15-höz."""
    train_df, val_df, test_df = split_by_unit(sample_df)
    total = len(sample_df)

    assert 0.60 < len(train_df) / total < 0.80
    assert 0.10 < len(val_df) / total < 0.20
    assert 0.10 < len(test_df) / total < 0.20


def test_prepare_xy(sample_df: pd.DataFrame) -> None:
    """Az X a feature-öket, az y a címkét tartalmazza."""
    X, y = prepare_xy(sample_df)

    assert X.shape[1] == 21  # 21 szenzor
    assert len(X) == len(y)
    assert set(y.unique()).issubset({0, 1})
