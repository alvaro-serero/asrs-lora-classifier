"""Load the raw ASRS CSV exports into one clean DataFrame."""

from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw" / "nasa"

COLUMNS = {
    "ACN": "acn",
    "Time | Date": "date",
    "Events | Anomaly": "anomaly",
    "Report 1 | Narrative": "narrative",
}

ORPHANS = {"Critical", "Less Severe"}
MIN_YEAR = 2010


def parse_anomaly(s: str) -> list[str]:
    """Split an anomaly string on ';' and re-attach severity fragments."""
    labels = []
    for part in s.split(";"):
        part = part.strip()
        if part == "":
            continue
        if part in ORPHANS:
            assert labels[-1] == "Conflict Ground Conflict", (s, part)
            labels[-1] = f"{labels[-1]} {part}"
        else:
            labels.append(part)
    return labels


def load_file(path: Path) -> pd.DataFrame:
    """Read one export, merge the two header rows, keep our 4 columns."""
    header = pd.read_csv(path, header=None, dtype=str, nrows=2)
    groups = header.iloc[0].str.strip()
    fields = header.iloc[1].str.strip()
    names = [f if g == "" else f"{g} | {f}" for g, f in zip(groups, fields)]

    df = pd.read_csv(path, header=None, skiprows=2, names=names, dtype=str)
    df = df[list(COLUMNS)].rename(columns=COLUMNS)
    df["file"] = path.name
    return df


def load_all(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load every CSV in raw_dir and return one clean table."""
    paths = sorted(raw_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files in {raw_dir}")

    df = pd.concat([load_file(p) for p in paths], ignore_index=True)
    assert not df["acn"].duplicated().any(), "duplicate ACNs across files"

    df = df.dropna(subset=["anomaly", "narrative"])
    df["labels"] = df["anomaly"].apply(parse_anomaly)
    df["year"] = df["date"].str[:4].astype(int)
    df = df[df["year"] >= MIN_YEAR].reset_index(drop=True)
    return df
