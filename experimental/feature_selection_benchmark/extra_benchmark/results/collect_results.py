from pathlib import Path
import pandas as pd

script_dir = Path(__file__).resolve().parent

stability_files = sorted(script_dir.glob("stability*.csv"))
validity_files = sorted(script_dir.glob("validity*.csv"))

def combine_csvs(files, output_name):
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        df["source_file"] = file.name
        dfs.append(df)

    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        out_path = script_dir / output_name
        combined.to_csv(out_path, index=False)
        print(f"Saved {len(combined)} rows to {out_path}")
    else:
        print(f"No files found for {output_name}")

combine_csvs(stability_files, "stability_results.csv")
combine_csvs(validity_files, "validity_results.csv")