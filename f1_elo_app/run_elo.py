"""Run the full ELO pipeline from the command line.

    python run_elo.py                 # uses f1elo/config.py paths
    python run_elo.py --data ./data   # override the input folder

It loads the data, computes ratings for every race, writes the enriched
results to output/, and prints a few leaderboards.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from f1elo import EloEngine, analysis, build_race_frame, load_data, config


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute F1 ELO ratings.")
    parser.add_argument("--data", type=str, default=None,
                        help="Folder containing the Kaggle F1 CSVs.")
    parser.add_argument("--out", type=str, default=None,
                        help="Folder to write outputs into.")
    args = parser.parse_args()

    data_dir = Path(args.data) if args.data else config.DATA_DIR
    out_dir = Path(args.out) if args.out else config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_dir} ...")
    raw = load_data(data_dir)
    df = build_race_frame(data_dir)
    print(f"  {len(df):,} result rows across {df['raceId'].nunique():,} races")

    print("Computing ELO ratings ...")
    engine = EloEngine()
    df = engine.run(df, progress=True)

    # Persist the enriched results and final rating tables.
    enriched_path = out_dir / "results_with_elo.csv"
    df.to_csv(enriched_path, index=False)
    print(f"Wrote {enriched_path}")

    # --- A few summaries to confirm it worked ---
    print("\nTop 10 constructors by peak ELO:")
    print(analysis.top_constructors_by_peak(df, raw["constructors"], n=10)
          [["constructor", "EloRating", "race"]].to_string())

    print("\nTop 10 drivers by peak ELO:")
    print(analysis.top_drivers_by_peak(df, raw["drivers"], n=10)
          [["driverName", "DriverEloRating", "race"]].to_string())

    print("\nDone.")


if __name__ == "__main__":
    main()
