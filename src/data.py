"""
Adatkezelő modul.

Felelősség:
- A nyers szenzoradat betöltése
- Train/val/test split **unit szinten** (data leakage elkerülése)
- X (feature-ök) és y (célváltozó) szétválasztása

Ez a modul a teljes pipeline alapja: a training, evaluation és monitoring
is ezt használja, így a split konzisztens marad.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


# --- Konfiguráció ---

DEFAULT_DATA_PATH = Path("data/raw/sensor_data.csv")
DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.15
DEFAULT_VAL_SIZE = 0.15  # a maradék 70% a train


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """A szenzor oszlopok nevei (s1, s2, ..., s21)."""
    return [c for c in df.columns if c.startswith("s") and c[1:].isdigit()]


def load_raw_data(path: Path | str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """A nyers szenzoradat betöltése CSV-ből."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Az adatfájl nem található: {path}\n"
            f"Futtasd először: python src/generate_data.py"
        )
    return pd.read_csv(path)


def split_by_unit(
    df: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Train/val/test split **unit szinten**.

    A split unit szinten történik, hogy egy egység teljes idősora
    csak az egyik halmazba kerüljön. Ez elkerüli a data leakage-et,
    ami akkor jelentkezne, ha ugyanannak az egységnek a sorai
    mindkét halmazba kerülnének.

    Visszaad: (train_df, val_df, test_df)
    """
    units = df["unit"].unique()

    # Először: test kiválasztása
    splitter_test = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_val_idx, test_idx = next(
        splitter_test.split(df, groups=df["unit"])
    )
    train_val_df = df.iloc[train_val_idx]
    test_df = df.iloc[test_idx]

    # Másodszor: val kiválasztása a maradékból
    # A val_size a teljes adathalmaz arányában értendő, ezért át kell számolni
    val_size_adjusted = val_size / (1 - test_size)
    splitter_val = GroupShuffleSplit(
        n_splits=1, test_size=val_size_adjusted, random_state=random_state
    )
    train_idx, val_idx = next(
        splitter_val.split(train_val_df, groups=train_val_df["unit"])
    )
    train_df = train_val_df.iloc[train_idx]
    val_df = train_val_df.iloc[val_idx]

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """X (szenzor feature-ök) és y (anomaly címke) szétválasztása."""
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df["anomaly"]
    return X, y


def main() -> None:
    """Gyors ellenőrzés: betölti az adatot és kiírja a split méreteket."""
    df = load_raw_data()
    train_df, val_df, test_df = split_by_unit(df)

    print(f"Teljes:  {len(df):>6,} sor, {df['unit'].nunique():>3} egység")
    print(f"Train:   {len(train_df):>6,} sor, {train_df['unit'].nunique():>3} egység")
    print(f"Val:     {len(val_df):>6,} sor, {val_df['unit'].nunique():>3} egység")
    print(f"Test:    {len(test_df):>6,} sor, {test_df['unit'].nunique():>3} egység")

    # Ellenőrzés: nincs átfedés az egységek között
    train_units = set(train_df["unit"].unique())
    val_units = set(val_df["unit"].unique())
    test_units = set(test_df["unit"].unique())

    assert train_units.isdisjoint(val_units), "Train és Val egységek átfednek!"
    assert train_units.isdisjoint(test_units), "Train és Test egységek átfednek!"
    assert val_units.isdisjoint(test_units), "Val és Test egységek átfednek!"
    print("\n✓ Nincs átfedés az egységek között (nincs data leakage)")


if __name__ == "__main__":
    main()
