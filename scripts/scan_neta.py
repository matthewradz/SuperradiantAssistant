"""Walk all historical HDF5 shots, extract Neta_2 + Atom Loading globals."""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

# Make repo root importable when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from superradiant_assistant.config import CONFIG
from superradiant_assistant.interfaces.hdf5_reader import list_shots, read_shot


def main() -> None:
    shots = list_shots(CONFIG.historical_data_root)
    print(f"Found {len(shots)} HDF5 shots under {CONFIG.historical_data_root}")

    rows = []
    for i, p in enumerate(shots):
        try:
            sig = read_shot(p)
        except Exception as e:
            print(f"[{i+1}/{len(shots)}] FAIL {p.name}: {e}")
            continue

        row = {
            "shot_id": sig.shot_id,
            "Neta_1": sig.Neta_1,
            "Neta_2": sig.Neta_2,
            "Neta_3": sig.Neta_3,
            "r_sq_2": sig.r_sq_2,
            "chi_square_2": sig.chi_square_2,
        }
        # Pull a few key Atom Loading globals if present
        for k in (
            "green_mot_frequency",
            "LS_blue_mot_intensity",
            "green_molasses_intensity",
            "spin_b_field_z",
            "lattice_loading_frequency",
        ):
            row[k] = sig.atom_loading_globals.get(k)
        rows.append(row)

    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parent.parent / "scan_neta_summary.csv"
    df.to_csv(out, index=False)

    print("\n=== Summary ===")
    if "Neta_2" in df.columns and df["Neta_2"].notna().any():
        s = df["Neta_2"].dropna()
        print(f"  Neta_2 valid:  {len(s)} / {len(df)}")
        print(f"  Neta_2 mean:   {s.mean():.1f}")
        print(f"  Neta_2 max:    {s.max():.1f}")
        print(f"  Neta_2 min:    {s.min():.1f}")
    print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()