"""
Szintetikus ipari szenzoradat generátor.

A NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation)
struktúráját utánozza: több "hajtómű" (unit), mindegyik egy idősor
21 szenzor csatornával, degradációs szakasszal a végén.

Ez az MVP adatforrása. A pipeline architektúrája identikus egy valódi
NASA CMAPSS adathalmazzal, csak az adatforrás cserélhető.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# --- Konfiguráció ---

N_UNITS = 100            # hány "hajtómű"
N_SENSORS = 21           # szenzor csatornák száma
MIN_CYCLES = 150         # minimum élettartam
MAX_CYCLES = 350         # maximum élettartam
DEGRADATION_FRACTION = 0.25  # az életciklus hány százaléka degradációs
RANDOM_SEED = 42         # reprodukálhatóság


def generate_unit(unit_id: int, rng: np.random.Generator) -> pd.DataFrame:
    """Egy hajtómű idősorának generálása."""
    n_cycles = rng.integers(MIN_CYCLES, MAX_CYCLES)
    cycles = np.arange(1, n_cycles + 1)

    # Minden szenzornak van egy alap szintje és egy zajszintje
    base_levels = rng.uniform(0.3, 0.7, size=N_SENSORS)
    noise_levels = rng.uniform(0.01, 0.05, size=N_SENSORS)

    # Ciklikus komponens (üzemi ciklus)
    period = rng.uniform(20, 40)
    cycle_component = np.sin(2 * np.pi * cycles / period)

    # Degradációs szakasz kezdete
    degradation_start = int(n_cycles * (1 - DEGRADATION_FRACTION))

    data = {"unit": unit_id, "cycle": cycles}

    for s in range(N_SENSORS):
        # Alapérték + ciklikus + zaj
        signal = (
            base_levels[s]
            + 0.1 * cycle_component
            + rng.normal(0, noise_levels[s], size=n_cycles)
        )

        # Degradáció: a szakasz végén a jel eltolódik
        degradation = np.zeros(n_cycles)
        if degradation_start < n_cycles:
            n_deg = n_cycles - degradation_start
            degradation[degradation_start:] = np.linspace(
                0, rng.uniform(0.1, 0.3), n_deg
            )
        signal += degradation

        data[f"s{s + 1}"] = signal

    # Anomaly címke: 1 a degradációs szakaszban, 0 egyébként
    anomaly = np.zeros(n_cycles, dtype=int)
    anomaly[degradation_start:] = 1
    data["anomaly"] = anomaly

    return pd.DataFrame(data)


def generate_dataset(n_units: int = N_UNITS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """A teljes adathalmaz generálása N darab egységre."""
    rng = np.random.default_rng(seed)
    frames = [generate_unit(unit_id=u + 1, rng=rng) for u in range(n_units)]
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Szintetikus szenzoradat generátor")
    parser.add_argument("--n-units", type=int, default=N_UNITS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/sensor_data.csv"),
    )
    args = parser.parse_args()

    # Kimeneti mappa létrehozása
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generálás: {args.n_units} egység, seed={args.seed}")
    df = generate_dataset(n_units=args.n_units, seed=args.seed)

    df.to_csv(args.output, index=False)

    # Összefoglaló
    n_units = df["unit"].nunique()
    n_rows = len(df)
    n_anomaly = int(df["anomaly"].sum())
    anomaly_pct = 100 * n_anomaly / n_rows

    print(f"Kész: {args.output}")
    print(f"  Egységek:        {n_units}")
    print(f"  Sorok:           {n_rows:,}")
    print(f"  Anomalia sorok:  {n_anomaly:,} ({anomaly_pct:.1f}%)")
    print(f"  Oszlopok:        {list(df.columns)}")


if __name__ == "__main__":
    main()
