# UI_test.py  (Single-file: UI + ANN backend + LSTM backend)
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import os

# Quiet TF noise (optional, safe)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import joblib
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# ANN BACKEND (integrated)
# ============================================================
def infer_task(y: pd.Series) -> str:
    y_no_na = y.dropna()
    if y_no_na.empty:
        return "unknown"

    if not pd.api.types.is_numeric_dtype(y_no_na):
        return "classification"

    n = len(y_no_na)
    nunique = int(y_no_na.nunique(dropna=True))
    unique_ratio = nunique / max(1, n)

    is_integer_like = pd.api.types.is_integer_dtype(y_no_na) or np.allclose(
        y_no_na.astype(float).values, np.round(y_no_na.astype(float).values), atol=1e-9
    )

    if is_integer_like and nunique <= 20 and unique_ratio <= 0.2:
        return "classification"

    return "regression"


def _expand_datetime_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Convert datetime columns into numeric features to avoid one-hot explosion.
    - If a column is datetime dtype -> expand.
    - If an object column looks like datetime (parsable) -> expand.
    """
    X = X.copy()

    def add_parts(colname: str, dt: pd.Series):
        X[f"{colname}__year"] = dt.dt.year
        X[f"{colname}__month"] = dt.dt.month
        X[f"{colname}__day"] = dt.dt.day
        X[f"{colname}__dayofweek"] = dt.dt.dayofweek
        X[f"{colname}__hour"] = dt.dt.hour

    dt_cols = X.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    for c in dt_cols:
        dt = pd.to_datetime(X[c], errors="coerce")
        add_parts(c, dt)
        X.drop(columns=[c], inplace=True)

    obj_cols = X.select_dtypes(include=["object"]).columns.tolist()
    for c in obj_cols:
        s = X[c]
        sample = s.dropna().astype(str).head(50)
        if sample.empty:
            continue
        parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
        if parsed.notna().mean() >= 0.8:
            full = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
            add_parts(c, full)
            X.drop(columns=[c], inplace=True)

    return X


def _make_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_cols, categorical_cols


def train_ann_from_df(
        df: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
        task_choice: str = "auto",
        test_size: float = 0.2,
        epochs: int = 30,
        batch_size: int = 32,
        lr: float = 0.001,
        early_stop: bool = True,
        patience: int = 5,
        seed: int = 42,
) -> Tuple[tf.keras.Model, ColumnTransformer, Optional[LabelEncoder], Dict[str, Any], Dict[str, List[float]]]:
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all").copy()

    if target_col not in df.columns:
        raise ValueError(f"Target column not found: {target_col}")
    for c in feature_cols:
        if c not in df.columns:
            raise ValueError(f"Feature column not found: {c}")

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    X = _expand_datetime_features(X)

    task = infer_task(y) if task_choice == "auto" else task_choice
    if task == "unknown":
        raise ValueError("Target column is empty after removing NA.")

    label_encoder: Optional[LabelEncoder] = None

    if task == "classification":
        label_encoder = LabelEncoder()
        y_enc = label_encoder.fit_transform(y.astype(str).fillna("NA"))
        n_classes = int(len(label_encoder.classes_))

        stratify = y_enc if n_classes >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=test_size, random_state=seed, shuffle=True, stratify=stratify
        )
    else:
        y_enc = pd.to_numeric(y, errors="coerce").astype(float).values
        keep = ~np.isnan(y_enc)
        X = X.loc[keep].reset_index(drop=True)
        y_enc = y_enc[keep]
        n_classes = 1

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=test_size, random_state=seed, shuffle=True
        )

    preprocessor, numeric_cols, categorical_cols = _make_preprocessor(X)

    X_train_np = preprocessor.fit_transform(X_train)
    X_test_np = preprocessor.transform(X_test)

    model = tf.keras.Sequential([
        layers.Input(shape=(X_train_np.shape[1],)),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    if task == "classification":
        if n_classes == 2:
            model.add(layers.Dense(1, activation="sigmoid"))
            model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
        else:
            model.add(layers.Dense(n_classes, activation="softmax"))
            model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    else:
        model.add(layers.Dense(1, activation="linear"))
        model.compile(optimizer=optimizer, loss="mse")

    callbacks = []
    if early_stop:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        ))

    history_obj = model.fit(
        X_train_np,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks,
    )
    history = history_obj.history

    results: Dict[str, Any] = {
        "task": task,
        "n_features_after_encoding": int(X_train_np.shape[1]),
        "numeric_features": int(len(numeric_cols)),
        "categorical_features": int(len(categorical_cols)),
    }

    if task == "classification":
        probs = model.predict(X_test_np, verbose=0)
        if n_classes == 2:
            y_pred = (probs.reshape(-1) > 0.5).astype(int)
        else:
            y_pred = np.argmax(probs, axis=1)

        results.update({
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classes": int(n_classes),
        })
    else:
        preds = model.predict(X_test_np, verbose=0).reshape(-1)
        results.update({
            "mae": float(mean_absolute_error(y_test, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
            "r2_score": float(r2_score(y_test, preds)),
        })

    return model, preprocessor, label_encoder, results, {
        "loss": [float(x) for x in history.get("loss", [])],
        "val_loss": [float(x) for x in history.get("val_loss", [])],
    }


def predict_from_df(
        df_features: pd.DataFrame,
        model: tf.keras.Model,
        preprocessor: ColumnTransformer,
        task: str,
        label_encoder: Optional[LabelEncoder] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    X = _expand_datetime_features(df_features)
    X_np = preprocessor.transform(X)

    if task == "classification":
        probs = model.predict(X_np, verbose=0)
        if probs.ndim == 2 and probs.shape[1] > 1:
            pred_idx = np.argmax(probs, axis=1)
            conf = np.max(probs, axis=1)
        else:
            p = probs.reshape(-1)
            pred_idx = (p > 0.5).astype(int)
            conf = p

        if label_encoder is not None:
            decoded = label_encoder.inverse_transform(pred_idx.astype(int))
            return decoded, conf
        return pred_idx, conf

    preds = model.predict(X_np, verbose=0).reshape(-1)
    return preds, None


# ============================================================
# LSTM BACKEND (integrated)
# ============================================================
@dataclass
class LSTMConfig:
    target_col: str
    date_col: Optional[str] = None

    lookback: int = 10
    test_size: float = 0.2

    lstm_units: int = 50
    dropout: float = 0.2
    epochs: int = 100
    batch_size: int = 32

    patience: int = 10
    val_split: float = 0.1
    seed: int = 42


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _clean_series(df: pd.DataFrame, target_col: str, date_col: Optional[str]) -> Tuple[pd.DataFrame, np.ndarray]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in the dataset.")

    work = df.copy()

    # 1. Robust Target Cleaning
    # If target is object/string, try to clean it (remove $, commas, etc.)
    if work[target_col].dtype == "object":
        work[target_col] = work[target_col].astype(str).str.replace(r'[^\d.-]', '', regex=True)

    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")

    # Initial check: if everything is NaN, we can't proceed
    if work[target_col].isna().all():
        sample_vals = df[target_col].dropna().head(5).tolist()
        raise ValueError(
            f"Target column '{target_col}' contains no numeric values. "
            f"Sample values found: {sample_vals}. LSTM requires numbers to forecast."
        )

    # 2. Robust Date Cleaning
    if date_col and date_col in work.columns:
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        # If date conversion failed for many rows, we might want to warn, but let's just drop invalid dates
        work = work.dropna(subset=[date_col])
        work = work.sort_values(date_col)

    # 3. Final NaN handling
    # Fill small gaps, then drop remaining NaNs
    work[target_col] = work[target_col].ffill().bfill()
    work = work.dropna(subset=[target_col])

    if len(work) < 2:
        raise ValueError(
            f"After cleaning, only {len(work)} valid numeric rows remain. "
            "LSTM needs at least 2-3 points to function."
        )

    series = work[target_col].to_numpy().reshape(-1, 1)
    return work, series


def make_sequences(scaled: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    if len(scaled) <= lookback:
        raise ValueError(f"Not enough data. Need more than lookback={lookback} rows.")

    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback: i, 0])
        y.append(scaled[i, 0])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y


def build_lstm(lookback: int, lstm_units: int, dropout: float) -> tf.keras.Model:
    model = Sequential(
        [
            layers.Input(shape=(lookback, 1)),
            layers.LSTM(lstm_units, return_sequences=True),
            layers.Dropout(dropout),
            layers.LSTM(lstm_units),
            layers.Dropout(dropout),
            layers.Dense(1),
        ]
    )
    # Use a smaller learning rate and clipnorm to prevent NaNs from exploding gradients
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss="mse")
    return model


def train_lstm_from_df(
        df: pd.DataFrame,
        cfg: LSTMConfig
) -> Tuple[tf.keras.Model, MinMaxScaler, Dict[str, list], Dict[str, float], Dict[str, np.ndarray]]:
    _set_seed(cfg.seed)

    work, values = _clean_series(df, cfg.target_col, cfg.date_col)

    # Dynamically adjust lookback if data is too small
    actual_lookback = cfg.lookback
    if len(values) <= actual_lookback:
        actual_lookback = max(1, len(values) - 2)
        # We don't update cfg.lookback here because it's a dataclass passed by value/ref
        # but we need to use actual_lookback for the rest of the function

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    try:
        X, y = make_sequences(scaled, actual_lookback)
    except ValueError as e:
        raise ValueError(f"Data too small for LSTM: {e}")

    # Ensure we have at least 1 sample for training and 1 for testing if possible
    if len(X) < 2:
        # If only 1 sequence possible, use it for training and test
        X_train, y_train = X, y
        X_test, y_test = X, y
    else:
        split_idx = int(len(X) * (1 - cfg.test_size))
        if split_idx < 1: split_idx = 1
        if split_idx >= len(X): split_idx = len(X) - 1

        X_train, y_train = X[:split_idx], y[:split_idx]
        X_test, y_test = X[split_idx:], y[split_idx:]

    model = build_lstm(actual_lookback, cfg.lstm_units, cfg.dropout)

    cb = EarlyStopping(monitor="val_loss", patience=cfg.patience, restore_best_weights=True)
    hist = model.fit(
        X_train,
        y_train,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        validation_split=cfg.val_split,
        callbacks=[cb],
        verbose=0,
    )
    history = {k: [float(x) for x in v] for k, v in hist.history.items()}

    train_pred = model.predict(X_train, verbose=0)
    test_pred = model.predict(X_test, verbose=0)

    y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    train_pred_actual = scaler.inverse_transform(train_pred)
    test_pred_actual = scaler.inverse_transform(test_pred)

    # --- SAFETY: remove NaNs before metrics ---
    y_test_actual_flat = np.asarray(y_test_actual).reshape(-1)
    test_pred_actual_flat = np.asarray(test_pred_actual).reshape(-1)
    mask = np.isfinite(y_test_actual_flat) & np.isfinite(test_pred_actual_flat)

    if mask.sum() == 0:
        # Fallback if all predictions are NaN
        rmse, mae, r2 = 0.0, 0.0, 0.0
    else:
        y_test_clean = y_test_actual_flat[mask]
        test_pred_clean = test_pred_actual_flat[mask]
        rmse = float(np.sqrt(mean_squared_error(y_test_clean, test_pred_clean)))
        mae = float(mean_absolute_error(y_test_clean, test_pred_clean))
        try:
            r2 = float(r2_score(y_test_clean, test_pred_clean))
        except:
            r2 = 0.0

    metrics = {"rmse": rmse, "mae": mae, "r2_score": r2, "task": "regression"}

    outputs = {
        "y_train_actual": y_train_actual.astype(np.float32),
        "y_test_actual": y_test_actual.astype(np.float32),
        "train_pred_actual": train_pred_actual.astype(np.float32),
        "test_pred_actual": test_pred_actual.astype(np.float32),
        "scaled_full": scaled.astype(np.float32),
        "lookback": np.array([actual_lookback], dtype=np.int32),
    }

    return model, scaler, history, metrics, outputs


def forecast_future(
        model: tf.keras.Model,
        scaler: MinMaxScaler,
        scaled_full: np.ndarray,
        lookback: int,
        n_steps: int,
) -> np.ndarray:
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if len(scaled_full) <= lookback:
        raise ValueError("Not enough history to forecast. Increase data or reduce lookback.")

    last_seq = scaled_full[-lookback:].reshape(1, lookback, 1).astype(np.float32)

    preds = []
    curr = last_seq.copy()
    for _ in range(n_steps):
        p = model.predict(curr, verbose=0)
        preds.append(float(p[0, 0]))
        curr = np.append(curr[:, 1:, :], p.reshape(1, 1, 1), axis=1)

    preds = np.array(preds, dtype=np.float32).reshape(-1, 1)
    preds_inv = scaler.inverse_transform(preds)
    return preds_inv


# ============================================================
# Storage (projects persistence)
# ============================================================
DATA_DIR = Path(".neural_studio")
DATA_DIR.mkdir(exist_ok=True)
PROJECTS_FILE = DATA_DIR / "projects.json"
CURRENT_FILE = DATA_DIR / "current_project.json"


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def get_projects() -> List[Dict[str, Any]]:
    return _read_json(PROJECTS_FILE, [])


def save_projects(projects: List[Dict[str, Any]]) -> None:
    _write_json(PROJECTS_FILE, projects)


def get_current_project() -> Optional[Dict[str, Any]]:
    return _read_json(CURRENT_FILE, None)


def set_current_project(project: Dict[str, Any]) -> None:
    _write_json(CURRENT_FILE, project)


def upsert_project(project: Dict[str, Any]) -> None:
    projects = get_projects()
    idx = next((i for i, p in enumerate(projects) if p.get("id") == project.get("id")), None)
    if idx is None:
        projects.insert(0, project)
    else:
        projects[idx] = project
    save_projects(projects)
    set_current_project(project)


# ============================================================
# UI helpers (unchanged styling)
# ============================================================
def inject_css():
    st.markdown(
        """
        <style>
          .block-container { padding-top: 1rem; padding-bottom: 6rem; max-width: 1200px; }
          @media (min-width: 1024px) { .block-container { padding-bottom: 2rem; } }

          .hero {
            border-radius: 18px;
            padding: 28px;
            color: white;
            background: linear-gradient(135deg, #0f172a 0%, #312e81 45%, #0891b2 100%);
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.10);
          }
          .hero .blur1 {
            position:absolute; top:-90px; right:-90px;
            width:300px; height:300px; border-radius:999px;
            background: rgba(34, 211, 238, 0.18);
            filter: blur(40px);
          }
          .hero .blur2 {
            position:absolute; bottom:-90px; left:-90px;
            width:300px; height:300px; border-radius:999px;
            background: rgba(99, 102, 241, 0.18);
            filter: blur(40px);
          }
          .hero h1 { margin: 0.25rem 0 0.25rem 0; font-size: 2.0rem; }
          @media (min-width: 768px){ .hero h1{ font-size: 2.8rem; } }

          .ns-card {
            border-radius: 16px;
            padding: 18px;
            border: 1px solid #e2e8f0;
            background: white;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
          }
          .muted { color: #64748b; }
          .badge {
            display:inline-block; padding: 4px 10px; border-radius:999px;
            font-size: 12px; border: 1px solid #e2e8f0; background: #f8fafc; color:#0f172a;
          }
          .badge.good { background: #ecfdf5; border-color:#a7f3d0; color:#065f46; }
          .badge.warn { background: #fffbeb; border-color:#fde68a; color:#92400e; }
          .btn-grad > button {
            background: linear-gradient(90deg, #06b6d4 0%, #6366f1 100%) !important;
            color: white !important;
            border: 0 !important;
          }
          .btn-emerald > button {
            background: linear-gradient(90deg, #10b981 0%, #14b8a6 100%) !important;
            color: white !important;
            border: 0 !important;
          }

          .bottom-nav {
            position: fixed;
            left: 0; right: 0; bottom: 0;
            background: rgba(255,255,255,0.92);
            border-top: 1px solid #e2e8f0;
            padding: 10px 12px;
            z-index: 9999;
          }
          @media (min-width: 1024px){ .bottom-nav { display: none; } }
          .bottom-nav .row {
            max-width: 1200px; margin: 0 auto;
            display: grid; grid-template-columns: repeat(5, 1fr);
            gap: 8px;
          }
          .bottom-nav a {
            text-decoration: none; color: #64748b;
            font-size: 12px; text-align:center;
            padding: 8px 6px; border-radius: 14px;
            border: 1px solid transparent;
          }
          .bottom-nav a.active {
            color: #4f46e5;
            border-color: #c7d2fe;
            background: #eef2ff;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def status_bar(current: str, processing: bool = False):
    steps = ["data_loaded", "preprocessed", "configured", "trained", "evaluated"]
    labels = {
        "data_loaded": "Data Loaded",
        "preprocessed": "Preprocessed",
        "configured": "Model Configured",
        "trained": "Trained",
        "evaluated": "Evaluated",
    }
    current_idx = steps.index(current) if current in steps else -1

    st.markdown('<div class="ns-card">', unsafe_allow_html=True)
    cols = st.columns(5)
    for i, s in enumerate(steps):
        done = i <= current_idx
        with cols[i]:
            st.markdown(
                f"""
                <div style="display:flex; gap:10px; align-items:center;">
                  <div style="
                    width:32px;height:32px;border-radius:999px;
                    display:flex;align-items:center;justify-content:center;
                    font-weight:700; font-size:14px;
                    color:{'white' if done else '#64748b'};
                    background:{'linear-gradient(90deg,#06b6d4,#6366f1)' if done else '#f1f5f9'};
                    border:1px solid {'rgba(99,102,241,0.35)' if done else '#e2e8f0'};
                  ">{i + 1}</div>
                  <div style="font-size:13px; font-weight:600; color:{'#0f172a' if done else '#64748b'};">
                    {labels[s]}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    if processing:
        st.caption("Processing…")
    st.markdown("</div>", unsafe_allow_html=True)


def bottom_nav(active: str):
    items = [
        ("home", "Home"),
        ("data", "Data"),
        ("model", "Model"),
        ("train", "Train"),
        ("predict", "Predict"),
    ]
    links = []
    for key, label in items:
        cls = "active" if key == active else ""
        links.append(f'<a class="{cls}" href="?page={key}">{label}</a>')
    st.markdown(
        f"""
        <div class="bottom-nav">
          <div class="row">
            {''.join(links)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def new_project() -> Dict[str, Any]:
    p = {
        "id": str(uuid.uuid4()),
        "name": f"Project {time.strftime('%Y-%m-%d')}",
        "task_type": "classification",  # classification | regression | auto-detect
        "model_type": "ann",  # ann | lstm
        "status": "data_loaded",
        "dataset": {"filename": "—", "rows": 0, "cols": 0, "missing": 0, "path": None, "file_type": None,
                    "sheet": None},
        "columns": {"target": None, "time": None, "features": []},
        "preprocess": {
            "missing_strategy": "Drop rows",
            "scaling": "Standard Scaler",
            "encoding": "One-Hot Encoding",
            "split": 0.8,
            "seed": 42,
            "lookback": 20,
            "horizon": 1,
        },
        "train_config": {"epochs": 20, "batch_size": 32, "lr": 0.001, "early_stop": True, "patience": 5},
        "train_logs": [],
        "history": {"loss": [], "val_loss": []},
        "evaluation_metrics": None,
        "artifacts": None,
        "feature_meta": {},
    }
    upsert_project(p)
    return p


def ensure_current_project() -> Optional[Dict[str, Any]]:
    return get_current_project()


def project_badge(p: Dict[str, Any]) -> str:
    model = p.get("model_type", "—").upper()
    task = p.get("task_type", "—")
    return f"{model} • {task}"


# ============================================================
# Caching
# ============================================================
@st.cache_data(show_spinner=False)
def cached_load_dataset(path: str, file_type: str, sheet: Optional[str]):
    if file_type == "csv":
        return pd.read_csv(path)
    return pd.read_excel(path, sheet_name=sheet)


@st.cache_resource(show_spinner=False)
def cached_load_ann_artifacts(model_path: str, preprocessor_path: str, label_encoder_path: str):
    model = tf.keras.models.load_model(model_path)
    preprocessor = joblib.load(preprocessor_path)
    label_encoder = joblib.load(label_encoder_path)
    return model, preprocessor, label_encoder


@st.cache_resource(show_spinner=False)
def cached_load_lstm_artifacts(model_path: str, scaler_path: str, outputs_path: str):
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    outputs = np.load(outputs_path, allow_pickle=False)
    return model, scaler, outputs


# ============================================================
# Data loading helpers
# ============================================================
def load_project_dataset(p: Dict[str, Any]) -> pd.DataFrame:
    ds = p.get("dataset", {})
    path = ds.get("path")
    if not path:
        raise FileNotFoundError("Dataset file not saved. Upload again in Data page.")
    return cached_load_dataset(path, ds.get("file_type"), ds.get("sheet"))


def build_feature_meta(df: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    for f in features:
        if f not in df.columns:
            continue
        s = df[f]
        is_num = pd.api.types.is_numeric_dtype(s)
        meta[f] = {"type": "numeric" if is_num else "categorical"}
        if not is_num:
            vals = s.dropna().astype(str).unique().tolist()
            meta[f]["options"] = vals[:50]
    return meta


# ============================================================
# Pages
# ============================================================
def page_home():
    st.markdown(
        """
        <div class="hero">
          <div class="blur1"></div>
          <div class="blur2"></div>
          <div style="position:relative;">
            <div style="font-weight:600; color: rgba(255,255,255,0.85);">✨ Neural Studio</div>
            <h1>Build, Train, Evaluate & Deploy</h1>
            <div style="max-width: 800px; color: rgba(255,255,255,0.80); font-size: 15px;">
              A guided, mobile-first machine learning workflow UI — from upload to predictions — with a premium modern design.
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown('<div class="btn-grad">', unsafe_allow_html=True)
        if st.button("➕ New Project", use_container_width=True):
            new_project()
            st.success("New project created. Go to Data Upload.")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        if st.button("📂 Open Current Project", use_container_width=True):
            if ensure_current_project() is None:
                st.warning("No current project found. Create one first.")

    st.write("")
    st.subheader("Features")
    feats = [
        ("📤", "Upload Data", "CSV/Excel upload with preview & column selection"),
        ("🧹", "Preprocess", "Split settings are used in training"),
        ("🧠", "Design Model", "ANN + LSTM configuration"),
        ("🏋️", "Train", "Real training + saved model artifacts"),
        ("✅", "Evaluate", "Real metrics + confusion matrix (ANN classification)"),
        ("🔮", "Predict", "ANN manual/batch • LSTM future forecast"),
    ]
    cols = st.columns(3)
    for i, (ico, title, desc) in enumerate(feats):
        with cols[i % 3]:
            st.markdown(
                f"""
                <div class="ns-card" style="min-height: 140px;">
                  <div style="font-size:28px; margin-bottom:6px;">{ico}</div>
                  <div style="font-weight:800; font-size:16px;">{title}</div>
                  <div class="muted" style="margin-top:4px; font-size:13px;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    st.subheader("Recent Projects")
    projects = get_projects()
    if not projects:
        st.info("No saved projects yet.")
    else:
        grid = st.columns(3)
        for i, p in enumerate(projects[:6]):
            with grid[i % 3]:
                st.markdown('<div class="ns-card">', unsafe_allow_html=True)
                st.markdown(f"**{p.get('name', 'Untitled')}**")
                st.caption(project_badge(p))
                ds = p.get("dataset", {})
                st.write(f"Dataset: {ds.get('filename', '—')}")
                st.write(f"Rows: {ds.get('rows', 0)} • Cols: {ds.get('cols', 0)} • Missing: {ds.get('missing', 0)}")
                if st.button("Open Project", key=f"open_{p.get('id')}"):
                    set_current_project(p)
                    st.success("Project opened as current.")
                st.markdown("</div>", unsafe_allow_html=True)


def page_data():
    p = ensure_current_project()
    if p is None:
        st.warning("No current project. Go to Home and create a new project.")
        return

    st.title("Data Upload")
    status_bar(p.get("status", "data_loaded"))

    st.markdown('<div class="ns-card">', unsafe_allow_html=True)
    st.write("### Upload Dataset (CSV or Excel)")
    uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])

    df = None
    sheet_name = None
    file_type = None

    if uploaded is not None:
        name = uploaded.name.lower()
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded)
            file_type = "csv"
        else:
            xls = pd.ExcelFile(uploaded)
            sheet_name = st.selectbox("Select sheet", xls.sheet_names, index=0)
            df = pd.read_excel(uploaded, sheet_name=sheet_name)
            file_type = "excel"

    st.markdown("</div>", unsafe_allow_html=True)

    if df is not None:
        df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
        missing = int(df.isna().sum().sum())

        DATASETS_DIR = DATA_DIR / "datasets"
        DATASETS_DIR.mkdir(exist_ok=True)

        suffix = ".csv" if file_type == "csv" else ".xlsx"
        dataset_path = DATASETS_DIR / f"{p['id']}{suffix}"
        with open(dataset_path, "wb") as f:
            f.write(uploaded.getbuffer())

        try:
            cached_load_dataset.clear()
        except Exception:
            pass

        p["dataset"] = {
            "filename": uploaded.name,
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "missing": missing,
            "path": str(dataset_path),
            "file_type": file_type,
            "sheet": sheet_name,
        }
        p["raw_preview"] = df.head(10).to_dict(orient="records")
        p["raw_columns"] = df.columns.tolist()
        p["status"] = "data_loaded"
        p["evaluation_metrics"] = None
        p["artifacts"] = None
        p["history"] = {"loss": [], "val_loss": []}
        upsert_project(p)

    ds = p.get("dataset", {})
    st.write("")
    st.markdown('<div class="ns-card">', unsafe_allow_html=True)
    st.write("### Dataset Information")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filename", ds.get("filename", "—"))
    c2.metric("Rows", ds.get("rows", 0))
    c3.metric("Columns", ds.get("cols", 0))
    mv = ds.get("missing", 0)
    c4.metric("Missing Values", mv)
    if mv and mv > 0:
        st.warning("Missing values detected — handled automatically during training.")
    st.markdown("</div>", unsafe_allow_html=True)

    cols = p.get("raw_columns", [])
    if cols:
        st.write("")
        st.markdown('<div class="ns-card">', unsafe_allow_html=True)
        st.write("### Column Configuration")
        target = st.selectbox(
            "Target Column (what to predict)",
            cols,
            index=cols.index(p["columns"]["target"]) if p["columns"]["target"] in cols else 0
        )

        time_col = st.selectbox(
            "Time Column (optional, for time series)",
            ["(None)"] + cols,
            index=(cols.index(p["columns"]["time"]) + 1) if p["columns"]["time"] in cols else 0
        )

        features = st.multiselect(
            "Feature Columns (inputs)",
            [c for c in cols if c != target],
            default=p["columns"]["features"] if p["columns"]["features"] else [c for c in cols if c != target][
                                                                              : min(6, max(0, len(cols) - 1))],
        )

        p["columns"]["target"] = target
        p["columns"]["time"] = None if time_col == "(None)" else time_col
        p["columns"]["features"] = features

        try:
            df_full = load_project_dataset(p)
            p["feature_meta"] = build_feature_meta(df_full, features)
        except Exception:
            p["feature_meta"] = {}

        upsert_project(p)

        st.caption(f"Selected features: {len(features)}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="ns-card">', unsafe_allow_html=True)
        st.write("### Data Preview (first 10 rows)")
        preview = p.get("raw_preview", [])
        if preview:
            st.dataframe(pd.DataFrame(preview), use_container_width=True, height=320)
        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="btn-grad">', unsafe_allow_html=True)
        if st.button("Continue to Preprocessing ➜", use_container_width=True):
            st.query_params["page"] = "preprocess"
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Upload a CSV/Excel to enable column selection and preview.")


def page_preprocess():
    p = ensure_current_project()
    if p is None:
        st.warning("No current project.")
        return

    st.title("Preprocess")
    status_bar(p.get("status", "data_loaded"))

    ds = p.get("dataset", {})
    cols_cfg = p.get("columns", {})

    st.markdown('<div class="ns-card">', unsafe_allow_html=True)
    st.write("### Dataset Summary")
    st.write(f"**Project:** {p.get('name')}")
    st.write(
        f"**Rows:** {ds.get('rows', 0)} • **Features:** {len(cols_cfg.get('features', []))} • **Target:** {cols_cfg.get('target', '—')}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="ns-card">', unsafe_allow_html=True)
    st.write("### Missing Values")
    p["preprocess"]["missing_strategy"] = st.selectbox(
        "Strategy",
        ["Drop rows", "Fill with mean/median", "Forward fill (time-series)"],
        index=["Drop rows", "Fill with mean/median", "Forward fill (time-series)"].index(
            p["preprocess"]["missing_strategy"]),
    )
    st.caption("ANN uses imputers automatically; LSTM uses forward/back fill on the target series.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="ns-card">', unsafe_allow_html=True)
    st.write("### Train/Test Split")
    split = st.slider("Train %", 50, 95, int(p["preprocess"]["split"] * 100))
    p["preprocess"]["split"] = split / 100.0
    p["preprocess"]["seed"] = st.number_input("Random Seed", value=int(p["preprocess"]["seed"]), step=1)
    st.caption(f"Train: {split}% • Test: {100 - split}%")
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="ns-card" style="border-color:#c4b5fd; background:#faf5ff;">', unsafe_allow_html=True)
    st.write("### LSTM / Time Series Settings")
    p["preprocess"]["lookback"] = st.number_input("Lookback window", value=int(p["preprocess"]["lookback"]),
                                                  min_value=1, step=1)
    p["preprocess"]["horizon"] = st.number_input("Forecast horizon", value=int(p["preprocess"]["horizon"]), min_value=1,
                                                 step=1)
    st.caption("Horizon is used in the Predict page for future steps.")
    st.markdown("</div>", unsafe_allow_html=True)

    upsert_project(p)

    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("⬅ Back to Data", use_container_width=True):
            st.query_params["page"] = "data"
    with c2:
        st.markdown('<div class="btn-grad">', unsafe_allow_html=True)
        if st.button("Continue to Model Configuration ➜", use_container_width=True):
            p["status"] = "preprocessed"
            upsert_project(p)
            st.query_params["page"] = "model"
        st.markdown("</div>", unsafe_allow_html=True)


def page_model():
    p = ensure_current_project()
    if p is None:
        st.warning("No current project.")
        return

    st.title("Model Architecture")
    status_bar(p.get("status", "data_loaded"))

    st.markdown('<div class="ns-card">', unsafe_allow_html=True)
    st.write("### Model Type")
    model_type = st.radio("Choose", ["ann", "lstm"], index=0 if p.get("model_type") == "ann" else 1, horizontal=True)
    p["model_type"] = model_type
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="ns-card">', unsafe_allow_html=True)
    st.write("### Task Type")
    task_type = st.selectbox(
        "Task",
        ["auto-detect", "classification", "regression"],
        index=["auto-detect", "classification", "regression"].index(p.get("task_type", "classification"))
    )
    p["task_type"] = task_type
    st.caption("For LSTM, task is always regression (time-series forecasting).")
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    if model_type == "ann":
        st.markdown('<div class="ns-card" style="background:#faf5ff; border-color:#ddd6fe;">', unsafe_allow_html=True)
        st.write("### ANN Configuration")
        p.setdefault("ann_config", {})
        p["ann_config"]["hidden_layers"] = st.number_input("Hidden layers", min_value=1, max_value=6,
                                                           value=int(p["ann_config"].get("hidden_layers", 3)))
        hl = int(p["ann_config"]["hidden_layers"])
        neurons = p["ann_config"].get("neurons", [256, 128, 64])
        neurons = (neurons + [64] * hl)[:hl]
        new_neurons = []
        for i in range(hl):
            new_neurons.append(
                int(st.number_input(f"Neurons in layer {i + 1}", min_value=1, max_value=1024, value=int(neurons[i]))))
        p["ann_config"]["neurons"] = new_neurons
        p["ann_config"]["activation"] = st.selectbox("Activation", ["ReLU", "Tanh", "Sigmoid"],
                                                     index=["ReLU", "Tanh", "Sigmoid"].index(
                                                         p["ann_config"].get("activation", "ReLU")))
        p["ann_config"]["output_activation"] = st.selectbox("Output activation",
                                                            ["Auto", "Linear", "Sigmoid", "Softmax"],
                                                            index=["Auto", "Linear", "Sigmoid", "Softmax"].index(
                                                                p["ann_config"].get("output_activation", "Auto")))
        st.caption("Backend uses strong ANN defaults; these UI knobs are stored but not mapped layer-by-layer yet.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown('<div class="ns-card" style="background:#ecfeff; border-color:#a5f3fc;">', unsafe_allow_html=True)
        st.write("### LSTM Configuration")
        p.setdefault("lstm_config", {})
        p["lstm_config"]["units"] = st.number_input("LSTM units", min_value=1, max_value=1024,
                                                    value=int(p["lstm_config"].get("units", 64)))
        p["lstm_config"]["layers"] = st.number_input("Number of LSTM layers", min_value=1, max_value=6,
                                                     value=int(p["lstm_config"].get("layers", 2)))
        p["lstm_config"]["dropout"] = st.slider("Dropout rate", 0.0, 0.8, float(p["lstm_config"].get("dropout", 0.2)))
        p["lstm_config"]["bidirectional"] = st.checkbox("Bidirectional",
                                                        value=bool(p["lstm_config"].get("bidirectional", False)))
        st.caption("This implementation trains a strong 2-layer LSTM (as in your backend). Extra UI fields are stored.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="btn-grad">', unsafe_allow_html=True)
    if st.button("Continue to Training ➜", use_container_width=True):
        p["status"] = "configured"
        upsert_project(p)
        st.query_params["page"] = "train"
    st.markdown("</div>", unsafe_allow_html=True)

    upsert_project(p)


def page_train():
    p = ensure_current_project()
    if p is None:
        st.warning("No current project.")
        return

    st.title("Train Model")
    status_bar(p.get("status", "data_loaded"))

    ds = p.get("dataset", {})
    features = p.get("columns", {}).get("features", [])

    st.markdown('<div class="ns-card">', unsafe_allow_html=True)
    st.write("### Model Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Architecture", p.get("model_type", "—").upper())
    c2.metric("Task", p.get("task_type", "—"))
    c3.metric("Features", len(features))
    train_samples = int(ds.get("rows", 0) * float(p["preprocess"]["split"])) if ds.get("rows", 0) else 0
    c4.metric("Training Samples", train_samples)
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="ns-card">', unsafe_allow_html=True)
    st.write("### Training Configuration")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        p["train_config"]["epochs"] = int(
            st.number_input("Epochs", min_value=1, max_value=500, value=int(p["train_config"]["epochs"])))
    with c2:
        p["train_config"]["batch_size"] = int(
            st.number_input("Batch size", min_value=1, max_value=2048, value=int(p["train_config"]["batch_size"])))
    with c3:
        p["train_config"]["lr"] = float(st.number_input("Learning rate (ANN only)", min_value=1e-6, max_value=1.0,
                                                        value=float(p["train_config"]["lr"]), format="%.6f"))
    with c4:
        p["train_config"]["patience"] = int(
            st.number_input("Early stop patience", min_value=1, max_value=50, value=int(p["train_config"]["patience"])))
    p["train_config"]["early_stop"] = st.checkbox("Enable early stopping", value=bool(p["train_config"]["early_stop"]))
    st.markdown("</div>", unsafe_allow_html=True)

    upsert_project(p)

    st.write("")
    st.markdown('<div class="ns-card">', unsafe_allow_html=True)
    st.write("### Training Controls")
    st.markdown('<div class="btn-emerald">', unsafe_allow_html=True)
    run = st.button("▶ Start Training", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if run:
        dataset_path = ds.get("path")
        if not dataset_path:
            st.error("No dataset saved. Go to Data page and upload a file.")
            return

        target = p.get("columns", {}).get("target")
        time_col = p.get("columns", {}).get("time")
        feature_cols = p.get("columns", {}).get("features", [])

        if not target:
            st.error("Select target in Data page first.")
            return

        try:
            df = load_project_dataset(p)
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")
            return

        epochs = int(p["train_config"]["epochs"])
        batch_size = int(p["train_config"]["batch_size"])
        lr = float(p["train_config"]["lr"])
        early_stop = bool(p["train_config"]["early_stop"])
        patience = int(p["train_config"]["patience"])
        seed = int(p["preprocess"]["seed"])
        test_size = 1.0 - float(p["preprocess"]["split"])

        MODELS_DIR = DATA_DIR / "models" / p["id"]
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        if p.get("model_type") == "ann":
            if not feature_cols:
                st.error("Select feature columns in Data page first (ANN needs features).")
                return

            p["feature_meta"] = build_feature_meta(df, feature_cols)

            ui_task = p.get("task_type", "classification")
            task_choice = "auto" if ui_task == "auto-detect" else ui_task

            with st.spinner("Training model (real ANN)..."):
                model, preprocessor, label_encoder, results, history = train_ann_from_df(
                    df=df,
                    target_col=target,
                    feature_cols=feature_cols,
                    task_choice=task_choice,
                    test_size=test_size,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    early_stop=early_stop,
                    patience=patience,
                    seed=seed,
                )

            model_path = MODELS_DIR / "model.keras"
            prep_path = MODELS_DIR / "preprocessor.joblib"
            le_path = MODELS_DIR / "label_encoder.joblib"

            model.save(model_path)
            joblib.dump(preprocessor, prep_path)
            joblib.dump(label_encoder, le_path)

            try:
                cached_load_ann_artifacts.clear()
            except Exception:
                pass

            p["artifacts"] = {
                "kind": "ann",
                "model_path": str(model_path),
                "preprocessor_path": str(prep_path),
                "label_encoder_path": str(le_path),
            }
            p["history"] = history
            p["evaluation_metrics"] = results
            p["status"] = "trained"
            upsert_project(p)

            st.success("Training finished (ANN). You can now Evaluate / Predict / Visualize.")

        else:
            if feature_cols:
                st.info("Note: LSTM uses ONLY the target series (features are ignored in this version).")

            cfg = LSTMConfig(
                target_col=target,
                date_col=time_col,
                lookback=int(p["preprocess"]["lookback"]),
                test_size=float(test_size),
                lstm_units=int(p.get("lstm_config", {}).get("units", 64)),
                dropout=float(p.get("lstm_config", {}).get("dropout", 0.2)),
                epochs=int(epochs),
                batch_size=int(batch_size),
                patience=int(patience),
                seed=int(seed),
                val_split=0.1,
            )

            with st.spinner("Training model (real LSTM)..."):
                model, scaler, history, metrics, outputs = train_lstm_from_df(df, cfg)

            model_path = MODELS_DIR / "lstm_model.keras"
            scaler_path = MODELS_DIR / "lstm_scaler.joblib"
            outputs_path = MODELS_DIR / "lstm_outputs.npz"

            model.save(model_path)
            joblib.dump(scaler, scaler_path)
            np.savez_compressed(outputs_path, **outputs)

            try:
                cached_load_lstm_artifacts.clear()
            except Exception:
                pass

            # Get the actual lookback used (it might have been adjusted)
            used_lookback = int(outputs["lookback"][0]) if "lookback" in outputs else cfg.lookback

            p["artifacts"] = {
                "kind": "lstm",
                "model_path": str(model_path),
                "scaler_path": str(scaler_path),
                "outputs_path": str(outputs_path),
                "lookback": used_lookback,
            }
            p["history"] = {
                "loss": history.get("loss", []),
                "val_loss": history.get("val_loss", []),
            }
            p["evaluation_metrics"] = metrics
            p["status"] = "trained"
            upsert_project(p)

            st.success("Training finished (LSTM). You can now Evaluate / Predict / Visualize.")

    st.write("")
    st.markdown('<div class="ns-card">', unsafe_allow_html=True)
    st.write("### Training Logs")
    logs = p.get("train_logs", [])
    if logs:
        st.code("\n".join(logs[-30:]), language="text")
    else:
        st.caption("Logs are minimal in this version (Keras training is silent).")
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="btn-grad">', unsafe_allow_html=True)
    if st.button("⚡ Evaluate Model", use_container_width=True):
        st.query_params["page"] = "evaluate"
    st.markdown("</div>", unsafe_allow_html=True)


def page_evaluate():
    p = ensure_current_project()
    if p is None:
        st.warning("No current project.")
        return

    st.title("Evaluate Model")
    status_bar(p.get("status", "data_loaded"))

    if p.get("status") not in ["trained", "evaluated"]:
        st.warning("Please train a model first.")
        return

    metrics = p.get("evaluation_metrics")
    if metrics is None:
        st.info("No evaluation metrics found. Train the model first.")
        return

    task = metrics.get("task", "classification")
    is_cls = (task == "classification")

    if is_cls:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{metrics.get('accuracy', 0.0) * 100:.1f}%")
        c2.metric("Precision", f"{metrics.get('precision', 0.0) * 100:.1f}%")
        c3.metric("Recall", f"{metrics.get('recall', 0.0) * 100:.1f}%")
        c4.metric("F1 Score", f"{metrics.get('f1_score', 0.0) * 100:.1f}%")

        st.write("")
        st.markdown('<div class="ns-card">', unsafe_allow_html=True)
        st.write("### Confusion Matrix")
        cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
        st.dataframe(pd.DataFrame(cm), use_container_width=True)
        st.caption("Rows = actual, columns = predicted.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{metrics.get('mae', 0.0):.4f}")
        c2.metric("RMSE", f"{metrics.get('rmse', 0.0):.4f}")
        c3.metric("R² Score", f"{metrics.get('r2_score', 0.0) * 100:.1f}%")

        st.write("")
        st.markdown('<div class="ns-card" style="border-color:#a7f3d0; background:#ecfdf5;">', unsafe_allow_html=True)
        st.write("### Model Performance Summary")
        st.write(f"Your model explains **{metrics.get('r2_score', 0.0) * 100:.1f}%** of the variance.")
        st.markdown("</div>", unsafe_allow_html=True)

    p["status"] = "evaluated"
    upsert_project(p)

    st.write("")
    st.markdown('<div class="btn-grad">', unsafe_allow_html=True)
    if st.button("Make Predictions ➜", use_container_width=True):
        st.query_params["page"] = "predict"
    st.markdown("</div>", unsafe_allow_html=True)


def page_predict():
    p = ensure_current_project()
    if p is None:
        st.warning("No current project.")
        return

    st.title("Predict")
    status_bar(p.get("status", "data_loaded"))

    artifacts = p.get("artifacts")
    if not artifacts:
        st.warning("No trained model found. Train the model first.")
        return

    kind = artifacts.get("kind", "ann")

    if kind == "ann":
        features = p.get("columns", {}).get("features", [])
        if not features:
            st.warning("No features selected. Go to Data Upload and select feature columns.")
            return

        try:
            model, preprocessor, label_encoder = cached_load_ann_artifacts(
                artifacts["model_path"],
                artifacts["preprocessor_path"],
                artifacts["label_encoder_path"],
            )
        except Exception as e:
            st.error(f"Failed to load trained model artifacts: {e}")
            return

        task = (p.get("evaluation_metrics") or {}).get("task", "classification")

        tab1, tab2 = st.tabs(["Manual Input", "Upload File"])

        with tab1:
            st.markdown('<div class="ns-card">', unsafe_allow_html=True)
            st.write("### Input Features")

            meta = p.get("feature_meta", {}) or {}
            vals: Dict[str, Any] = {}
            cols = st.columns(2)

            for i, f in enumerate(features):
                with cols[i % 2]:
                    fmeta = meta.get(f, {"type": "numeric"})
                    if fmeta.get("type") == "categorical":
                        options = fmeta.get("options") or []
                        if options:
                            vals[f] = st.selectbox(f, options=options, index=0)
                        else:
                            vals[f] = st.text_input(f, value="")
                    else:
                        vals[f] = st.number_input(f, value=0.0, format="%.6f")

            st.markdown("</div>", unsafe_allow_html=True)

            st.write("")
            st.markdown('<div class="btn-emerald">', unsafe_allow_html=True)
            do = st.button("🔮 Predict", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            if do:
                row = pd.DataFrame([vals], columns=features)
                pred, conf = predict_from_df(row, model, preprocessor, task=task, label_encoder=label_encoder)

                st.markdown(
                    '<div class="ns-card" style="background: linear-gradient(90deg,#06b6d4,#6366f1); color:white;">',
                    unsafe_allow_html=True)
                st.write("### Prediction Result")

                if task == "classification":
                    st.write(f"**Predicted Class:** {pred[0]}")
                    if conf is not None:
                        st.write(f"**Confidence:** {float(conf[0]) * 100:.1f}%")
                else:
                    st.write(f"**Predicted Value:** {float(pred[0]):.6f}")

                st.markdown("</div>", unsafe_allow_html=True)

        with tab2:
            st.markdown('<div class="ns-card">', unsafe_allow_html=True)
            up = st.file_uploader("Upload CSV or Excel for batch predictions", type=["csv", "xlsx", "xls"],
                                  key="pred_upload")
            st.caption("File must include the same feature columns.")
            sheet = None
            df_up = None

            if up is not None:
                if up.name.lower().endswith(".csv"):
                    df_up = pd.read_csv(up)
                else:
                    xls = pd.ExcelFile(up)
                    sheet = st.selectbox("Select sheet", xls.sheet_names, index=0, key="pred_sheet")
                    df_up = pd.read_excel(up, sheet_name=sheet)

            st.markdown("</div>", unsafe_allow_html=True)

            if df_up is not None:
                missing_cols = [c for c in features if c not in df_up.columns]
                if missing_cols:
                    st.error(f"Missing required columns: {missing_cols}")
                else:
                    st.markdown('<div class="btn-emerald">', unsafe_allow_html=True)
                    gen = st.button("Generate Predictions", use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                    if gen:
                        X_feat = df_up[features].copy()
                        preds, conf = predict_from_df(X_feat, model, preprocessor, task=task,
                                                      label_encoder=label_encoder)

                        out = df_up.copy()
                        out["prediction"] = preds

                        if task == "classification" and conf is not None:
                            out["confidence"] = np.round(conf.astype(float), 6)

                        st.success("Predictions generated (REAL).")
                        st.dataframe(out.head(20), use_container_width=True)

                        csv_bytes = out.to_csv(index=False).encode("utf-8")
                        st.download_button("Download Results", data=csv_bytes, file_name="predictions.csv",
                                           mime="text/csv")

    else:
        # LSTM Predict = Future Forecast
        try:
            model, scaler, outputs = cached_load_lstm_artifacts(
                artifacts["model_path"],
                artifacts["scaler_path"],
                artifacts["outputs_path"],
            )
        except Exception as e:
            st.error(f"Failed to load LSTM artifacts: {e}")
            return

        st.markdown('<div class="ns-card">', unsafe_allow_html=True)
        st.write("### Future Forecast (LSTM)")
        horizon_default = int(p.get("preprocess", {}).get("horizon", 1))
        n_future = st.number_input("Steps to forecast", 1, 2000, int(horizon_default))
        st.caption("Forecast steps are sequential (one-step ahead repeated).")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="btn-emerald">', unsafe_allow_html=True)
        do = st.button("🔮 Forecast", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if do:
            try:
                scaled_full = outputs["scaled_full"]
                lookback = int(artifacts.get("lookback", int(outputs["lookback"][0]) if "lookback" in outputs else 10))
                future = forecast_future(model, scaler, scaled_full, lookback=lookback, n_steps=int(n_future))
                out_df = pd.DataFrame(future, columns=["forecast"])
                st.success("Forecast generated.")
                st.dataframe(out_df, use_container_width=True)

                csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download Forecast CSV", data=csv_bytes, file_name="lstm_forecast.csv",
                                   mime="text/csv")
            except Exception as e:
                st.error(f"Forecast error: {e}")


def page_save_load():
    p = ensure_current_project()
    if p is None:
        st.warning("No current project.")
        return

    st.title("Save / Load")
    status_bar(p.get("status", "data_loaded"))

    st.markdown('<div class="ns-card">', unsafe_allow_html=True)
    st.write("### Current Model Info")
    st.write(f"**Project:** {p.get('name')}")
    st.write(
        f"**Model:** {p.get('model_type', '—').upper()} • **Task:** {(p.get('evaluation_metrics') or {}).get('task', p.get('task_type', '—'))}")
    st.write(f"**Status:** {p.get('status', '—')}")
    st.write("**Features:** " + ", ".join(p.get("columns", {}).get("features", [])[:12]) + (
        " ..." if len(p.get("columns", {}).get("features", [])) > 12 else ""))
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="ns-card" style="border-color:#a7f3d0;">', unsafe_allow_html=True)
    st.write("### Save Project Package (JSON)")
    st.caption("Saves UI settings + history + paths. (Model files are stored under .neural_studio/models/)")
    json_bytes = json.dumps(p, indent=2).encode("utf-8")
    st.download_button("💾 Download JSON", data=json_bytes, file_name="neural_studio_project.json",
                       mime="application/json")
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="ns-card" style="border-color:#c7d2fe;">', unsafe_allow_html=True)
    st.write("### Load Project Package (JSON)")
    up = st.file_uploader("Upload saved JSON", type=["json"])
    if up is not None:
        try:
            loaded = json.loads(up.read().decode("utf-8"))
            if "id" not in loaded:
                loaded["id"] = str(uuid.uuid4())
            upsert_project(loaded)
            st.success("Loaded and set as current project.")
        except Exception as e:
            st.error(f"Failed to load JSON: {e}")
    st.markdown("</div>", unsafe_allow_html=True)


def page_visualize():
    p = ensure_current_project()
    if p is None:
        st.warning("No current project.")
        return

    st.title("Visualize")
    status_bar(p.get("status", "data_loaded"))

    loss = p.get("history", {}).get("loss", [])
    val_loss = p.get("history", {}).get("val_loss", [])

    st.markdown('<div class="ns-card">', unsafe_allow_html=True)
    st.write("### Training Loss Chart")
    if loss:
        fig = plt.figure()
        plt.plot(range(1, len(loss) + 1), loss, label="Training Loss")
        if val_loss:
            plt.plot(range(1, len(val_loss) + 1), val_loss, label="Validation Loss")
        plt.xlabel("Epoch (training rounds)")
        plt.ylabel("Loss (prediction error)")
        plt.legend()
        st.pyplot(fig, clear_figure=True)
        st.caption("X-axis: Epoch. Y-axis: Loss. Lower is better.")
    else:
        st.info("No training history yet. Train the model first.")
    st.markdown("</div>", unsafe_allow_html=True)

    artifacts = p.get("artifacts") or {}
    if artifacts.get("kind") == "lstm":
        st.write("")
        st.markdown('<div class="ns-card">', unsafe_allow_html=True)
        st.write("### LSTM Predictions vs Actual")
        try:
            model, scaler, outputs = cached_load_lstm_artifacts(
                artifacts["model_path"],
                artifacts["scaler_path"],
                artifacts["outputs_path"],
            )

            y_train_actual = outputs["y_train_actual"]
            y_test_actual = outputs["y_test_actual"]
            train_pred_actual = outputs["train_pred_actual"]
            test_pred_actual = outputs["test_pred_actual"]
            lookback = int(artifacts.get("lookback", int(outputs["lookback"][0]) if "lookback" in outputs else 10))

            fig2 = plt.figure(figsize=(12, 6))
            train_idx = range(lookback, lookback + len(y_train_actual))
            test_idx = range(lookback + len(y_train_actual), lookback + len(y_train_actual) + len(y_test_actual))

            plt.plot(train_idx, y_train_actual, label="Actual (Train)", alpha=0.6)
            plt.plot(train_idx, train_pred_actual, label="Predicted (Train)", alpha=0.6, linestyle="--")
            plt.plot(test_idx, y_test_actual, label="Actual (Test)", alpha=0.7)
            plt.plot(test_idx, test_pred_actual, label="Predicted (Test)", alpha=0.8, linestyle="--")
            plt.axvline(x=lookback + len(y_train_actual), linestyle=":", label="Train/Test Split")
            plt.legend()
            st.pyplot(fig2, clear_figure=True)
        except Exception as e:
            st.info(f"LSTM plot not available: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    metrics = p.get("evaluation_metrics") or {}
    if metrics.get("task") == "regression" and artifacts.get("kind") == "ann":
        st.write("")
        st.markdown('<div class="ns-card">', unsafe_allow_html=True)
        st.write("### Residual Plot (sample)")
        st.caption("For a good model, residuals should scatter around 0.")

        try:
            df = load_project_dataset(p)
            features = p.get("columns", {}).get("features", [])
            target = p.get("columns", {}).get("target")
            if features and target:
                model, preprocessor, le = cached_load_ann_artifacts(
                    p["artifacts"]["model_path"],
                    p["artifacts"]["preprocessor_path"],
                    p["artifacts"]["label_encoder_path"],
                )

                X = df[features].copy()
                y = pd.to_numeric(df[target], errors="coerce").astype(float)
                keep = ~np.isnan(y.values)
                X = X.loc[keep]
                y = y.loc[keep].values

                preds, _ = predict_from_df(X, model, preprocessor, task="regression", label_encoder=le)
                residuals = y - preds

                fig = plt.figure()
                plt.scatter(preds, residuals, alpha=0.6)
                plt.axhline(0, linewidth=1)
                plt.xlabel("Predicted")
                plt.ylabel("Residual (actual - predicted)")
                st.pyplot(fig, clear_figure=True)
        except Exception as e:
            st.info(f"Residual plot not available: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="ns-card" style="border-color:#a7f3d0; background:#ecfdf5;">', unsafe_allow_html=True)
    st.write("### Evaluation Metrics")
    m = p.get("evaluation_metrics")
    if not m:
        st.caption("No evaluation metrics yet.")
    else:
        st.json(m)
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# Routing
# ============================================================
PAGES = {
    "home": ("Home", page_home),
    "data": ("Data Upload", page_data),
    "preprocess": ("Preprocess", page_preprocess),
    "model": ("Model", page_model),
    "train": ("Train", page_train),
    "evaluate": ("Evaluate", page_evaluate),
    "predict": ("Predict", page_predict),
    "save": ("Save/Load", page_save_load),
    "visualize": ("Visualize", page_visualize),
}


def main():
    st.set_page_config(page_title="Neural Studio", layout="wide")
    inject_css()

    with st.sidebar:
        st.markdown("## Neural Studio")
        st.caption("ML workflow builder (Python UI)")

        cur = get_current_project()
        if cur:
            st.markdown(f"**Current:** {cur.get('name', 'Untitled')}")
            st.caption(project_badge(cur))
        else:
            st.info("No current project.")

        st.write("")

        page_keys = list(PAGES.keys())
        labels = [PAGES[k][0] for k in page_keys]

        active_key = st.query_params.get("page", "home")
        if active_key not in PAGES:
            active_key = "home"

        idx = page_keys.index(active_key)
        chosen = st.radio("Navigate", labels, index=idx)

        chosen_key = page_keys[labels.index(chosen)]
        if chosen_key != active_key:
            st.query_params["page"] = chosen_key
            st.rerun()

        st.write("---")
        if st.button("➕ New Project", use_container_width=True):
            new_project()
            st.success("Created new project.")
            st.query_params["page"] = "data"
            st.rerun()

        if st.button("🗑️ Clear Current Project", use_container_width=True):
            if CURRENT_FILE.exists():
                CURRENT_FILE.unlink()
            st.success("Cleared current project.")
            st.query_params["page"] = "home"
            st.rerun()

    active_key = st.query_params.get("page", "home")
    if active_key not in PAGES:
        active_key = "home"

    bottom_nav(active_key)

    _, renderer = PAGES[active_key]
    renderer()


if __name__ == "__main__":
    main()
