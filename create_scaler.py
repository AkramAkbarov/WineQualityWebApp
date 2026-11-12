import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

FEATURE_COLUMNS_UNDERSCORE = [c.replace(" ", "_") for c in FEATURE_COLUMNS]


def find_csv() -> Path | None:
    folder = Path(__file__).resolve().parent
    patterns = ["*.csv", "data/*.csv"]
    for pattern in patterns:
        for path in folder.glob(pattern):
            return path
    return None


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else find_csv()
    if csv_path is None or not csv_path.is_file():
        print("CSV tapılmadı. İstifadə: python create_scaler.py wine_quality.csv")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    cols = FEATURE_COLUMNS if FEATURE_COLUMNS[0] in df.columns else FEATURE_COLUMNS_UNDERSCORE
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"CSV-də sütunlar tapılmadı: {missing}")
        sys.exit(1)

    X = df[cols].values
    scaler = StandardScaler()
    scaler.fit(X)

    out = Path(__file__).resolve().parent / "scaler.pkl"
    joblib.dump(scaler, out)
    print(f"Yazıldı: {out} ({len(X)} sətir üzrə fit edildi)")


if __name__ == "__main__":
    main()
