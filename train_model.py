"""
Wine Quality (6 sinif) üçün model + scaler yenidən təlim.

Standart UCI / Kaggle CSV: 11 feature + "quality" (adətən 3-8, 6 sinif -> 0-5).

Istifade:
  python train_model.py
  python train_model.py path/to/wine.csv

Sonra serveri yeniden baslatin: uvicorn main:app --reload
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import Winequality

FEATURE_SPACE = [
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
FEATURE_UNDERSCORE = [c.replace(" ", "_") for c in FEATURE_SPACE]


def _read_csv(path: Path) -> pd.DataFrame:
    for sep in (";", ",", "\t"):
        try:
            df = pd.read_csv(path, sep=sep)
            if len(df.columns) >= 12:
                return df
        except Exception:
            continue
    return pd.read_csv(path)


def _pick_feature_columns(df: pd.DataFrame) -> list[str]:
    if FEATURE_SPACE[0] in df.columns:
        return FEATURE_SPACE
    if FEATURE_UNDERSCORE[0] in df.columns:
        return FEATURE_UNDERSCORE
    raise ValueError(
        "CSV-de 11 feature sütunu tapilmadi (fixed acidity ve ya fixed_acidity)."
    )


def _pick_quality_column(df: pd.DataFrame) -> str:
    for name in ("quality", "Wine Quality", "wine_quality", "target"):
        if name in df.columns:
            return name
    raise ValueError("CSV-de 'quality' sütunu tapilmadi.")


def map_quality_to_6_classes(y: np.ndarray) -> np.ndarray:
    """quality -> 0..5 (6 sinif). UCI/Kaggle adeten quality 3-8 -> 0-5."""
    y = y.astype(np.int64)
    u = np.sort(np.unique(y))

    if len(u) == 6 and u[0] == 0 and u[-1] == 5:
        return y

    if u.min() >= 0 and u.max() <= 5:
        return y

    # UCI red/white: quality tam eded 3..8 (cox vaxt 6 ferqli deyer)
    if u.min() >= 3 and u.max() <= 10:
        shifted = y - 3
        if shifted.max() <= 5:
            return shifted

    if len(u) == 6:
        mapping = {v: i for i, v in enumerate(u.tolist())}
        return np.array([mapping[v] for v in y], dtype=np.int64)

    ranks = pd.Series(y).rank(method="average")
    bins = pd.qcut(ranks, q=6, labels=False, duplicates="drop")
    arr = bins.astype(np.int64).to_numpy()
    uniq = np.sort(np.unique(arr))
    remap = {v: i for i, v in enumerate(uniq.tolist())}
    return np.array([remap[v] for v in arr], dtype=np.int64)


def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes)
    counts = np.maximum(counts, 1)
    w = len(y) / (num_classes * counts.astype(np.float64))
    w = w / w.sum() * num_classes
    return torch.tensor(w, dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv",
        nargs="?",
        default=None,
        help="Wine CSV (Kaggle / UCI). Bos buraxilsa qovluqda *.csv axtarilir.",
    )
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    csv_path: Path | None
    if args.csv:
        csv_path = Path(args.csv)
    else:
        csv_path = None
        for p in sorted(base.glob("*.csv")):
            csv_path = p
            break
        if csv_path is None:
            for p in sorted((base / "data").glob("*.csv")) if (base / "data").is_dir() else []:
                csv_path = p
                break

    if csv_path is None or not csv_path.is_file():
        print("CSV tapilmadi. Kaggle CSV-ni bu qovluqa atin ve ya:")
        print("  python train_model.py wine_quality.csv")
        sys.exit(1)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = _read_csv(csv_path)
    feat_cols = _pick_feature_columns(df)
    qcol = _pick_quality_column(df)

    X = df[feat_cols].values.astype(np.float32)
    y = map_quality_to_6_classes(df[qcol].values)

    if y.min() < 0 or y.max() > 5:
        raise ValueError(
            f"Label araligi duzgun deyil: min={y.min()}, max={y.max()}. "
            "CSV ve quality sütununu yoxlayin."
        )

    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xt = torch.tensor(X_train_s, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.long, device=device)
    Xv = torch.tensor(X_test_s, dtype=torch.float32, device=device)
    yv = torch.tensor(y_test, dtype=torch.long, device=device)

    loader = DataLoader(
        TensorDataset(Xt, yt), batch_size=args.batch_size, shuffle=True, drop_last=False
    )

    model = Winequality().to(device)
    weights = compute_class_weights(y_train, 6).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="max", factor=0.5, patience=15
    )

    best_state = None
    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        for xb, yb in loader:
            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optim.step()

        model.eval()
        with torch.inference_mode():
            pred = model(Xv).argmax(dim=1)
            acc = (pred == yv).float().mean().item() * 100.0
        sched.step(acc)

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 50 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:4d}  val_acc={acc:.2f}%  lr={optim.param_groups[0]['lr']:.6f}")

    if best_state is None:
        best_state = model.state_dict()

    model_path = base / "model.pth"
    scaler_path = base / "scaler.pkl"
    torch.save(best_state, model_path)
    joblib.dump(scaler, scaler_path)

    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        pred = model(Xv).argmax(dim=1)
        final_acc = (pred == yv).float().mean().item() * 100.0

    print(f"\nBest val accuracy (saved checkpoint): {best_acc:.2f}%")
    print(f"Final val accuracy (loaded best): {final_acc:.2f}%")
    print(f"Saved: {model_path}")
    print(f"Saved: {scaler_path}")


if __name__ == "__main__":
    main()
