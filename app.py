# UI_test.py (Single-file: UI + ANN backend + MULTIVARIATE LSTM backend)
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
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# Small utilities
# ============================================================
def _safe_index(options: List[Any], value: Any, default: int = 0) -> int:
    try:
        return options.index(value)
    except Exception:
        return default


def _clamp_int(x: Any, lo: int, hi: int, default: int) -> int:
    try:
        v = int(x)
        return max(lo, min(hi, v))
    except Exception:
        return default


def _clamp_float(x: Any, lo: float, hi: float, default: float) -> float:
    try:
        v = float(x)
        return max(lo, min(hi, v))
    except Exception:
        return default


# ============================================================
# Navigation (Fixes the “double click” issue)
# ============================================================
def request_nav(page: str) -> None:
    st.session_state["_nav_to"] = page


def goto(page: str) -> None:
    st.query_params["page"] = page
    st.rerun()


def apply_pending_nav() -> None:
    nav_to = st.session_state.pop("_nav_to", None)
    if nav_to:
        goto(nav_to)


# ============================================================
# ANN BACKEND
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


def _map_activation(name: str) -> str:
    name = (name or "").strip().lower()
    if name in ["relu", "re-lu"]:
        return "relu"
    if name == "tanh":
        return "tanh"
    if name == "sigmoid":
        return "sigmoid"
    return "relu"


def _build_ann_model(
    input_dim: int,
    task: str,
    n_classes: int,
    ann_config: Optional[Dict[str, Any]],
    lr: float,
) -> tf.keras.Model:
    ann_config = ann_config or {}
    hidden_layers = _clamp_int(ann_config.get("hidden_layers", 3), 1, 12, 3)

    neurons = ann_config.get("neurons", [256, 128, 64])
    if not isinstance(neurons, list):
        neurons = [256, 128, 64]
    neurons = (neurons + [64] * hidden_layers)[:hidden_layers]
    neurons = [max(1, int(n)) for n in neurons]

    act = _map_activation(ann_config.get("activation", "ReLU"))
    out_choice = (ann_config.get("output_activation", "Auto") or "Auto").strip().lower()

    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    for n in neurons:
        model.add(layers.Dense(n, activation=act))
        model.add(layers.Dropout(0.2))

    optimizer = tf.keras.optimizers.Adam(learning_rate=float(lr))

    if task == "classification":
        if n_classes == 2:
            out_units = 1
            default_out_act = "sigmoid"
            loss = "binary_crossentropy"
        else:
            out_units = n_classes
            default_out_act = "softmax"
            loss = "sparse_categorical_crossentropy"

        out_act = default_out_act
        if out_choice in ["sigmoid", "softmax", "linear"]:
            out_act = out_choice

        model.add(layers.Dense(out_units, activation=out_act))
        model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    else:
        model.add(layers.Dense(1, activation="linear"))
        model.compile(optimizer=optimizer, loss="mse")

    return model


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
    ann_config: Optional[Dict[str, Any]] = None,
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

    model = _build_ann_model(
        input_dim=int(X_train_np.shape[1]),
        task=task,
        n_classes=int(n_classes),
        ann_config=ann_config,
        lr=float(lr),
    )

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
            "class_labels": label_encoder.classes_.tolist() if label_encoder is not None else None,
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
# MULTIVARIATE LSTM BACKEND
# ============================================================
@dataclass
class LSTMConfig:
    target_col: str
    feature_cols: List[str]
    date_col: Optional[str] = None

    lookback: int = 10
    horizon: int = 1
    test_size: float = 0.2

    lstm_units: int = 64
    dropout: float = 0.2
    epochs: int = 100
    batch_size: int = 32

    patience: int = 10
    val_split: float = 0.1
    seed: int = 42


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    if s.dtype == "object":
        s = s.astype(str).str.replace(r"[^\d.-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce").astype(float)


def _prepare_multivariate_frame(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    date_col: Optional[str],
) -> Tuple[pd.DataFrame, List[str]]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    for c in feature_cols:
        if c not in df.columns:
            raise ValueError(f"Feature column '{c}' not found.")

    work = df.copy()

    if date_col and date_col in work.columns:
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        work = work.dropna(subset=[date_col])
        work = work.sort_values(date_col)

    X = work[feature_cols].copy()
    X = _expand_datetime_features(X)

    X = pd.get_dummies(X, dummy_na=True)

    y = _coerce_numeric_series(work[target_col])

    aligned = X.copy()
    aligned["_target_"] = y.values

    aligned = aligned.replace([np.inf, -np.inf], np.nan)
    aligned = aligned.ffill().bfill()
    aligned = aligned.dropna(axis=0, how="any")

    if len(aligned) < 10:
        raise ValueError(f"After cleaning, only {len(aligned)} usable rows remain. Need more data for LSTM.")

    input_cols = [c for c in aligned.columns if c != "_target_"] + ["_target_"]
    return aligned, input_cols


def make_sequences_multivariate(arr: np.ndarray, lookback: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if len(arr) <= lookback + horizon - 1:
        raise ValueError("Not enough data for lookback+horizon.")

    X, y = [], []
    target_idx = arr.shape[1] - 1
    for i in range(lookback, len(arr) - horizon + 1):
        X.append(arr[i - lookback:i, :])
        y.append(arr[i + horizon - 1, target_idx])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y


def build_lstm_multivariate(lookback: int, n_channels: int, lstm_units: int, dropout: float) -> tf.keras.Model:
    model = Sequential(
        [
            layers.Input(shape=(lookback, n_channels)),
            layers.LSTM(lstm_units, return_sequences=True),
            layers.Dropout(dropout),
            layers.LSTM(lstm_units),
            layers.Dropout(dropout),
            layers.Dense(1),
        ]
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss="mse")
    return model


def train_lstm_from_df(
    df: pd.DataFrame,
    cfg: LSTMConfig
) -> Tuple[tf.keras.Model, Dict[str, Any], Dict[str, list], Dict[str, float], Dict[str, np.ndarray]]:
    _set_seed(cfg.seed)

    aligned, input_cols = _prepare_multivariate_frame(df, cfg.target_col, cfg.feature_cols, cfg.date_col)

    Xy = aligned[[c for c in input_cols if c != "_target_"]].to_numpy(dtype=np.float32)
    y = aligned["_target_"].to_numpy(dtype=np.float32).reshape(-1, 1)
    full = np.concatenate([Xy, y], axis=1)

    split_idx = int(len(full) * (1 - cfg.test_size))
    split_idx = max(2, min(split_idx, len(full) - 1))

    train_full = full[:split_idx]
    test_full = full[split_idx:]

    scaler_all = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    train_scaled = scaler_all.fit_transform(train_full)
    test_scaled = scaler_all.transform(test_full)

    scaler_y.fit(train_full[:, [-1]])

    scaled_full = np.vstack([train_scaled, test_scaled]).astype(np.float32)

    lookback = int(cfg.lookback)
    horizon = int(cfg.horizon)
    if len(scaled_full) <= lookback + horizon:
        lookback = max(1, len(scaled_full) - horizon - 1)

    X_seq, y_seq = make_sequences_multivariate(scaled_full, lookback=lookback, horizon=horizon)

    seq_split = int(len(X_seq) * (1 - cfg.test_size))
    seq_split = max(1, min(seq_split, len(X_seq) - 1))

    X_train, y_train = X_seq[:seq_split], y_seq[:seq_split]
    X_test, y_test = X_seq[seq_split:], y_seq[seq_split:]

    n_channels = int(X_train.shape[2])
    model = build_lstm_multivariate(lookback, n_channels, cfg.lstm_units, cfg.dropout)

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

    train_pred = model.predict(X_train, verbose=0).reshape(-1, 1)
    test_pred = model.predict(X_test, verbose=0).reshape(-1, 1)

    y_train_actual = scaler_y.inverse_transform(y_train.reshape(-1, 1))
    y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    train_pred_actual = scaler_y.inverse_transform(train_pred)
    test_pred_actual = scaler_y.inverse_transform(test_pred)

    y_test_flat = y_test_actual.reshape(-1)
    pred_flat = test_pred_actual.reshape(-1)
    mask = np.isfinite(y_test_flat) & np.isfinite(pred_flat)
    if mask.sum() == 0:
        rmse = mae = r2 = 0.0
    else:
        rmse = float(np.sqrt(mean_squared_error(y_test_flat[mask], pred_flat[mask])))
        mae = float(mean_absolute_error(y_test_flat[mask], pred_flat[mask]))
        try:
            r2 = float(r2_score(y_test_flat[mask], pred_flat[mask]))
        except Exception:
            r2 = 0.0

    metrics = {
        "task": "regression",
        "rmse": rmse,
        "mae": mae,
        "r2_score": r2,
        "lookback_used": int(lookback),
        "horizon_used": int(horizon),
        "channels_used": int(n_channels),
        "rows_used": int(len(aligned)),
        "train_sequences": int(len(X_train)),
        "test_sequences": int(len(X_test)),
    }

    pack = {
        "scaler_all": scaler_all,
        "scaler_y": scaler_y,
        "input_feature_columns": [c for c in input_cols if c != "_target_"],
        "target_column": cfg.target_col,
    }

    last_feature_scaled = scaled_full[-1, :-1].astype(np.float32)

    outputs = {
        "scaled_full": scaled_full.astype(np.float32),
        "lookback": np.array([lookback], dtype=np.int32),
        "horizon": np.array([horizon], dtype=np.int32),
        "last_feature_scaled": last_feature_scaled.astype(np.float32),
        "y_train_actual": y_train_actual.astype(np.float32),
        "y_test_actual": y_test_actual.astype(np.float32),
        "train_pred_actual": train_pred_actual.astype(np.float32),
        "test_pred_actual": test_pred_actual.astype(np.float32),
    }

    return model, pack, history, metrics, outputs


def forecast_future_multivariate(
    model: tf.keras.Model,
    scaler_y: MinMaxScaler,
    scaled_full: np.ndarray,
    lookback: int,
    n_steps: int,
    last_feature_scaled: np.ndarray,
) -> np.ndarray:
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if len(scaled_full) <= lookback:
        raise ValueError("Not enough history to forecast. Increase data or reduce lookback.")

    n_channels = scaled_full.shape[1]
    curr = scaled_full[-lookback:, :].reshape(1, lookback, n_channels).astype(np.float32)

    preds_scaled = []
    for _ in range(n_steps):
        p = model.predict(curr, verbose=0).reshape(-1)[0]
        preds_scaled.append(p)

        next_row = np.concatenate([last_feature_scaled, np.array([p], dtype=np.float32)], axis=0).reshape(1, 1, n_channels)
        curr = np.concatenate([curr[:, 1:, :], next_row], axis=1)

    preds_scaled = np.array(preds_scaled, dtype=np.float32).reshape(-1, 1)
    preds_inv = scaler_y.inverse_transform(preds_scaled)
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
# UI helpers (styling)
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
        "task_type": "classification",
        "model_type": "ann",
        "status": "data_loaded",
        "dataset": {"filename": "—", "rows": 0, "cols": 0, "missing": 0, "path": None, "file_type": None, "sheet": None},
        "columns": {"target": None, "time": None, "features": []},
        "preprocess": {
            "missing_strategy": "Drop rows",
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
        "viz_cache": {},
        "ann_config": {"hidden_layers": 3, "neurons": [256, 128, 64], "activation": "ReLU", "output_activation": "Auto"},
        "lstm_config": {"units": 64, "layers": 2, "dropout": 0.2, "bidirectional": False},
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
def cached_load_lstm_artifacts(model_path: str, pack_path: str, outputs_path: str):
    model = tf.keras.models.load_model(model_path)
    pack = joblib.load(pack_path)
    outputs = np.load(outputs_path, allow_pickle=False)
    return model, pack, outputs


@st.cache_data(show_spinner=False)
def cached_feature_meta(dataset_path: str, file_type: str, sheet: Optional[str], features: Tuple[str, ...]) -> Dict[str, Any]:
    if file_type == "csv":
        df_full = pd.read_csv(dataset_path)
    else:
        df_full = pd.read_excel(dataset_path, sheet_name=sheet)
    return build_feature_meta(df_full, list(features))


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
        st.button("➕ New Project", use_container_width=True, on_click=lambda: (new_project(), request_nav("data")))
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
        ("🧠", "Design Model", "ANN + Multivariate LSTM configuration"),
        ("🏋️", "Train", "Real training + saved model artifacts"),
        ("✅", "Evaluate", "Real metrics + labeled confusion matrix (ANN classification)"),
        ("🔮", "Predict", "ANN manual/batch • LSTM future forecast (features held constant if unknown)"),
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
                    request_nav("data")
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
            cached_feature_meta.clear()
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
        p.setdefault("viz_cache", {})
        p["viz_cache"].pop("ann_regression", None)

        # reset selections
        p["columns"]["target"] = None
        p["columns"]["time"] = None
        p["columns"]["features"] = []
        p["feature_meta"] = {}

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

        saved_target = p.get("columns", {}).get("target")
        target = st.selectbox("Target Column (what to predict)", cols, index=_safe_index(cols, saved_target, 0))

        time_options = ["(None)"] + [c for c in cols if c != target]
        saved_time = p.get("columns", {}).get("time")
        time_default = "(None)" if (saved_time is None or saved_time == target or saved_time not in time_options) else saved_time
        time_col = st.selectbox("Time Column (optional, for time series)", time_options, index=_safe_index(time_options, time_default, 0))

        feature_options = [c for c in cols if c != target]
        saved_features = p.get("columns", {}).get("features", []) or []
        safe_defaults = [c for c in saved_features if c in feature_options]
        if not safe_defaults:
            safe_defaults = feature_options[: min(6, len(feature_options))]

        features = st.multiselect("Feature Columns (inputs)", feature_options, default=safe_defaults)

        p["columns"]["target"] = target
        p["columns"]["time"] = None if time_col == "(None)" else time_col
        p["columns"]["features"] = [c for c in features if c != target]

        try:
            ds_path = p.get("dataset", {}).get("path")
            ds_type = p.get("dataset", {}).get("file_type")
            ds_sheet = p.get("dataset", {}).get("sheet")
            if ds_path and p["columns"]["features"]:
                p["feature_meta"] = cached_feature_meta(ds_path, ds_type, ds_sheet, tuple(p["columns"]["features"]))
            else:
                p["feature_meta"] = {}
        except Exception:
            p["feature_meta"] = {}

        upsert_project(p)

        st.caption(f"Selected features: {len(p['columns']['features'])}")
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
        st.button("Continue to Preprocessing ➜", use_container_width=True, on_click=request_nav, args=("preprocess",))
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
    st.write(f"**Rows:** {ds.get('rows', 0)} • **Features:** {len(cols_cfg.get('features', []))} • **Target:** {cols_cfg.get('target', '—')}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="ns-card">', unsafe_allow_html=True)
    st.write("### Missing Values")
    p["preprocess"]["missing_strategy"] = st.selectbox(
        "Strategy",
        ["Drop rows", "Fill with mean/median", "Forward fill (time-series)"],
        index=_safe_index(["Drop rows", "Fill with mean/median", "Forward fill (time-series)"], p["preprocess"].get("missing_strategy"), 0),
    )
    st.caption("ANN uses imputers automatically; LSTM cleans & forward/back fills after encoding.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="ns-card">', unsafe_allow_html=True)
    st.write("### Train/Test Split")
    split = st.slider("Train %", 50, 95, int(float(p["preprocess"].get("split", 0.8)) * 100))
    p["preprocess"]["split"] = split / 100.0
    p["preprocess"]["seed"] = st.number_input("Random Seed", value=int(p["preprocess"].get("seed", 42)), step=1)
    st.caption(f"Train: {split}% • Test: {100 - split}%")
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="ns-card" style="border-color:#c4b5fd; background:#faf5ff;">', unsafe_allow_html=True)
    st.write("### LSTM / Time Series Settings")
    p["preprocess"]["lookback"] = st.number_input("Lookback window", value=int(p["preprocess"].get("lookback", 20)), min_value=1, step=1)
    p["preprocess"]["horizon"] = st.number_input("Forecast horizon (steps ahead)", value=int(p["preprocess"].get("horizon", 1)), min_value=1, step=1)
    st.caption("Multivariate LSTM will use: past lookback rows of (features + target) to predict target at horizon.")
    st.markdown("</div>", unsafe_allow_html=True)

    upsert_project(p)

    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        st.button("⬅ Back to Data", use_container_width=True, on_click=request_nav, args=("data",))
    with c2:
        st.markdown('<div class="btn-grad">', unsafe_allow_html=True)
        st.button(
            "Continue to Model Configuration ➜",
            use_container_width=True,
            on_click=lambda: (p.__setitem__("status", "preprocessed"), upsert_project(p), request_nav("model")),
        )
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
        index=_safe_index(["auto-detect", "classification", "regression"], p.get("task_type", "classification"), 1),
    )
    p["task_type"] = task_type
    st.caption("For LSTM, task is always regression (time-series forecasting).")
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    if model_type == "ann":
        st.markdown('<div class="ns-card" style="background:#faf5ff; border-color:#ddd6fe;">', unsafe_allow_html=True)
        st.write("### ANN Configuration (Applied for real ✅)")
        p.setdefault("ann_config", {})
        p["ann_config"]["hidden_layers"] = st.number_input("Hidden layers", min_value=1, max_value=12, value=int(p["ann_config"].get("hidden_layers", 3)))
        hl = int(p["ann_config"]["hidden_layers"])
        neurons = p["ann_config"].get("neurons", [256, 128, 64])
        if not isinstance(neurons, list):
            neurons = [256, 128, 64]
        neurons = (neurons + [64] * hl)[:hl]
        new_neurons = []
        for i in range(hl):
            new_neurons.append(int(st.number_input(f"Neurons in layer {i + 1}", min_value=1, max_value=2048, value=int(neurons[i]))))
        p["ann_config"]["neurons"] = new_neurons
        p["ann_config"]["activation"] = st.selectbox("Activation", ["ReLU", "Tanh", "Sigmoid"], index=_safe_index(["ReLU", "Tanh", "Sigmoid"], p["ann_config"].get("activation", "ReLU"), 0))
        p["ann_config"]["output_activation"] = st.selectbox("Output activation", ["Auto", "Linear", "Sigmoid", "Softmax"], index=_safe_index(["Auto", "Linear", "Sigmoid", "Softmax"], p["ann_config"].get("output_activation", "Auto"), 0))
        st.caption("These settings are used during ANN training.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown('<div class="ns-card" style="background:#ecfeff; border-color:#a5f3fc;">', unsafe_allow_html=True)
        st.write("### LSTM Configuration (Multivariate ✅)")
        p.setdefault("lstm_config", {})
        p["lstm_config"]["units"] = st.number_input("LSTM units", min_value=1, max_value=2048, value=int(p["lstm_config"].get("units", 64)))
        p["lstm_config"]["layers"] = st.number_input("Number of LSTM layers (stored)", min_value=1, max_value=6, value=int(p["lstm_config"].get("layers", 2)))
        p["lstm_config"]["dropout"] = st.slider("Dropout rate", 0.0, 0.8, float(p["lstm_config"].get("dropout", 0.2)))
        p["lstm_config"]["bidirectional"] = st.checkbox("Bidirectional (stored)", value=bool(p["lstm_config"].get("bidirectional", False)))
        st.caption("This LSTM uses selected FEATURES + past TARGET to predict future TARGET.")
        st.markdown("</div>", unsafe_allow_html=True)

    upsert_project(p)

    st.write("")
    st.markdown('<div class="btn-grad">', unsafe_allow_html=True)
    st.button(
        "Continue to Training ➜",
        use_container_width=True,
        on_click=lambda: (p.__setitem__("status", "configured"), upsert_project(p), request_nav("train")),
    )
    st.markdown("</div>", unsafe_allow_html=True)


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
        p["train_config"]["epochs"] = int(st.number_input("Epochs", min_value=1, max_value=500, value=int(p["train_config"]["epochs"])))
    with c2:
        p["train_config"]["batch_size"] = int(st.number_input("Batch size", min_value=1, max_value=2048, value=int(p["train_config"]["batch_size"])))
    with c3:
        p["train_config"]["lr"] = float(st.number_input("Learning rate (ANN only)", min_value=1e-6, max_value=1.0, value=float(p["train_config"]["lr"]), format="%.6f"))
    with c4:
        p["train_config"]["patience"] = int(st.number_input("Early stop patience", min_value=1, max_value=50, value=int(p["train_config"]["patience"])))
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
                    ann_config=p.get("ann_config", {}),
                )

            # -----------------------------
            # Save residual plot data for ANN regression (so Visualize can plot it)
            # -----------------------------
            try:
                if results.get("task") == "regression":
                    X_full = df[feature_cols].copy()
                    y_full = pd.to_numeric(df[target], errors="coerce").astype(float).values

                    keep = ~np.isnan(y_full)
                    X_full = X_full.loc[keep].reset_index(drop=True)
                    y_full = y_full[keep]

                    X_full = _expand_datetime_features(X_full)

                    X_train, X_test, y_train, y_test = train_test_split(
                        X_full,
                        y_full,
                        test_size=test_size,
                        random_state=seed,
                        shuffle=True,
                    )

                    X_test_np = preprocessor.transform(X_test)
                    y_pred = model.predict(X_test_np, verbose=0).reshape(-1)

                    p.setdefault("viz_cache", {})
                    p["viz_cache"]["ann_regression"] = {
                        "y_test": y_test.tolist(),
                        "y_pred": y_pred.tolist(),
                    }
                else:
                    p.setdefault("viz_cache", {})
                    p["viz_cache"].pop("ann_regression", None)
            except Exception:
                p.setdefault("viz_cache", {})
                p["viz_cache"].pop("ann_regression", None)

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
            if not feature_cols:
                st.error("For MULTIVARIATE LSTM, please select feature columns in Data page.")
                return

            cfg = LSTMConfig(
                target_col=target,
                feature_cols=feature_cols,
                date_col=time_col,
                lookback=int(p["preprocess"]["lookback"]),
                horizon=int(p["preprocess"]["horizon"]),
                test_size=float(test_size),
                lstm_units=int(p.get("lstm_config", {}).get("units", 64)),
                dropout=float(p.get("lstm_config", {}).get("dropout", 0.2)),
                epochs=int(epochs),
                batch_size=int(batch_size),
                patience=int(patience),
                seed=int(seed),
                val_split=0.1,
            )

            with st.spinner("Training model (real MULTIVARIATE LSTM)..."):
                model, pack, history, metrics, outputs = train_lstm_from_df(df, cfg)

            model_path = MODELS_DIR / "lstm_model.keras"
            pack_path = MODELS_DIR / "lstm_pack.joblib"
            outputs_path = MODELS_DIR / "lstm_outputs.npz"

            model.save(model_path)
            joblib.dump(pack, pack_path)
            np.savez_compressed(outputs_path, **outputs)

            try:
                cached_load_lstm_artifacts.clear()
            except Exception:
                pass

            used_lookback = int(outputs["lookback"][0]) if "lookback" in outputs else cfg.lookback

            p["artifacts"] = {
                "kind": "lstm",
                "model_path": str(model_path),
                "pack_path": str(pack_path),
                "outputs_path": str(outputs_path),
                "lookback": used_lookback,
            }
            p["history"] = {"loss": history.get("loss", []), "val_loss": history.get("val_loss", [])}
            p["evaluation_metrics"] = metrics
            p["status"] = "trained"
            upsert_project(p)

            st.success("Training finished (Multivariate LSTM). You can now Evaluate / Predict / Visualize.")

    st.write("")
    st.markdown('<div class="btn-grad">', unsafe_allow_html=True)
    st.button("⚡ Evaluate Model", use_container_width=True, on_click=request_nav, args=("evaluate",))
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
        st.write("### Confusion Matrix (labeled)")
        cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
        labels = metrics.get("class_labels") or None
        cm_df = pd.DataFrame(cm)
        if labels and len(labels) == cm_df.shape[0] == cm_df.shape[1]:
            cm_df.index = [f"Actual: {x}" for x in labels]
            cm_df.columns = [f"Pred: {x}" for x in labels]
        st.dataframe(cm_df, use_container_width=True)
        st.caption("Rows = actual, columns = predicted.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{metrics.get('mae', 0.0):.4f}")
        c2.metric("RMSE", f"{metrics.get('rmse', 0.0):.4f}")
        r2v = float(metrics.get("r2_score", 0.0))
        c3.metric("R² Score", f"{r2v:.4f}")

        if "channels_used" in metrics:
            st.write("")
            st.markdown('<div class="ns-card">', unsafe_allow_html=True)
            st.write("### LSTM Details")
            st.write(f"Channels used: **{metrics.get('channels_used')}** (features + target)")
            st.write(f"Lookback: **{metrics.get('lookback_used')}** • Horizon: **{metrics.get('horizon_used')}**")
            st.write(f"Rows used: **{metrics.get('rows_used')}**")
            st.write(f"Train sequences: **{metrics.get('train_sequences')}** • Test sequences: **{metrics.get('test_sequences')}**")
            st.markdown("</div>", unsafe_allow_html=True)

    p["status"] = "evaluated"
    upsert_project(p)

    st.write("")
    st.markdown('<div class="btn-grad">', unsafe_allow_html=True)
    st.button("Make Predictions ➜", use_container_width=True, on_click=request_nav, args=("predict",))
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
            cols2 = st.columns(2)

            for i, f in enumerate(features):
                with cols2[i % 2]:
                    fmeta = meta.get(f, {"type": "numeric"})
                    if fmeta.get("type") == "categorical":
                        options = fmeta.get("options") or []
                        vals[f] = st.selectbox(f, options=options, index=0) if options else st.text_input(f, value="")
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

                st.markdown('<div class="ns-card" style="background: linear-gradient(90deg,#06b6d4,#6366f1); color:white;">', unsafe_allow_html=True)
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
            up = st.file_uploader("Upload CSV or Excel for batch predictions", type=["csv", "xlsx", "xls"], key="pred_upload")
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
                        preds, conf = predict_from_df(X_feat, model, preprocessor, task=task, label_encoder=label_encoder)

                        out = df_up.copy()
                        out["prediction"] = preds
                        if task == "classification" and conf is not None:
                            out["confidence"] = np.round(conf.astype(float), 6)

                        st.success("Predictions generated (REAL).")
                        st.dataframe(out.head(20), use_container_width=True)

                        csv_bytes = out.to_csv(index=False).encode("utf-8")
                        st.download_button("Download Results", data=csv_bytes, file_name="predictions.csv", mime="text/csv")

    else:
        try:
            model, pack, outputs = cached_load_lstm_artifacts(
                artifacts["model_path"],
                artifacts["pack_path"],
                artifacts["outputs_path"],
            )
        except Exception as e:
            st.error(f"Failed to load LSTM artifacts: {e}")
            return

        st.markdown('<div class="ns-card">', unsafe_allow_html=True)
        st.write("### Future Forecast (Multivariate LSTM)")
        horizon_default = int(p.get("preprocess", {}).get("horizon", 1))
        n_future = st.number_input("Steps to forecast", 1, 2000, int(horizon_default))
        st.caption("If future feature values are unknown, we hold features constant at the last observed values.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="btn-emerald">', unsafe_allow_html=True)
        do = st.button("🔮 Forecast", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if do:
            try:
                scaled_full = outputs["scaled_full"]
                lookback = int(artifacts.get("lookback", int(outputs["lookback"][0]) if "lookback" in outputs else 10))
                last_feat = outputs["last_feature_scaled"]
                scaler_y = pack["scaler_y"]

                future = forecast_future_multivariate(
                    model=model,
                    scaler_y=scaler_y,
                    scaled_full=scaled_full,
                    lookback=lookback,
                    n_steps=int(n_future),
                    last_feature_scaled=last_feat,
                )
                out_df = pd.DataFrame(future, columns=["forecast"])
                st.success("Forecast generated.")
                st.dataframe(out_df, use_container_width=True)

                csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download Forecast CSV", data=csv_bytes, file_name="lstm_forecast.csv", mime="text/csv")
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
    st.write(f"**Model:** {p.get('model_type', '—').upper()} • **Task:** {(p.get('evaluation_metrics') or {}).get('task', p.get('task_type', '—'))}")
    st.write(f"**Status:** {p.get('status', '—')}")
    st.write("**Features:** " + ", ".join(p.get("columns", {}).get("features", [])[:12]) + (" ..." if len(p.get("columns", {}).get("features", [])) > 12 else ""))
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="ns-card" style="border-color:#a7f3d0;">', unsafe_allow_html=True)
    st.write("### Save Project Package (JSON)")
    st.caption("Saves UI settings + history + paths. (Model files are stored under .neural_studio/models/)")
    json_bytes = json.dumps(p, indent=2).encode("utf-8")
    st.download_button("💾 Download JSON", data=json_bytes, file_name="neural_studio_project.json", mime="application/json")
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
            loaded.setdefault("columns", {"target": None, "time": None, "features": []})
            loaded.setdefault("dataset", {"filename": "—", "rows": 0, "cols": 0, "missing": 0, "path": None, "file_type": None, "sheet": None})
            loaded.setdefault("preprocess", {"split": 0.8, "seed": 42, "lookback": 20, "horizon": 1, "missing_strategy": "Drop rows"})
            loaded.setdefault("train_config", {"epochs": 20, "batch_size": 32, "lr": 0.001, "early_stop": True, "patience": 5})
            loaded.setdefault("ann_config", {"hidden_layers": 3, "neurons": [256, 128, 64], "activation": "ReLU", "output_activation": "Auto"})
            loaded.setdefault("lstm_config", {"units": 64, "layers": 2, "dropout": 0.2, "bidirectional": False})
            loaded.setdefault("viz_cache", {})
            upsert_project(loaded)
            request_nav("data")
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

    # -----------------------------
    # Residual Plot (ANN Regression) - restored ✅
    # -----------------------------
    viz = (p.get("viz_cache") or {}).get("ann_regression")
    if viz:
        try:
            y_test = np.array(viz["y_test"], dtype=float)
            y_pred = np.array(viz["y_pred"], dtype=float)
            residuals = y_test - y_pred

            st.write("")
            st.markdown('<div class="ns-card">', unsafe_allow_html=True)
            st.write("### Residual Plot (ANN Regression)")
            fig_r = plt.figure()
            plt.scatter(y_pred, residuals, alpha=0.6)
            plt.axhline(0)
            plt.xlabel("Predicted")
            plt.ylabel("Residual (actual - predicted)")
            st.pyplot(fig_r, clear_figure=True)
            st.caption("Good model: residuals scattered around 0 without a clear pattern.")
            st.markdown("</div>", unsafe_allow_html=True)
        except Exception:
            pass

    artifacts = p.get("artifacts") or {}
    if artifacts.get("kind") == "lstm":
        st.write("")
        st.markdown('<div class="ns-card">', unsafe_allow_html=True)
        st.write("### LSTM Predictions vs Actual")
        try:
            model, pack, outputs = cached_load_lstm_artifacts(
                artifacts["model_path"],
                artifacts["pack_path"],
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

        # LSTM residual plot (bonus)
        try:
            y_test_l = outputs["y_test_actual"].reshape(-1)
            y_pred_l = outputs["test_pred_actual"].reshape(-1)
            residuals_l = y_test_l - y_pred_l

            st.write("")
            st.markdown('<div class="ns-card">', unsafe_allow_html=True)
            st.write("### Residual Plot (LSTM Test)")
            fig_rl = plt.figure()
            plt.scatter(y_pred_l, residuals_l, alpha=0.6)
            plt.axhline(0)
            plt.xlabel("Predicted")
            plt.ylabel("Residual (actual - predicted)")
            st.pyplot(fig_rl, clear_figure=True)
            st.caption("Based on the saved LSTM test predictions.")
            st.markdown("</div>", unsafe_allow_html=True)
        except Exception:
            pass

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

    # Apply pending navigation at the start of each run (fixes double-click)
    apply_pending_nav()

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
            request_nav(chosen_key)

        st.write("---")
        if st.button("➕ New Project", use_container_width=True):
            new_project()
            request_nav("data")

        if st.button("🗑️ Clear Current Project", use_container_width=True):
            if CURRENT_FILE.exists():
                CURRENT_FILE.unlink()
            request_nav("home")

    # Apply nav requests from sidebar/buttons
    apply_pending_nav()

    active_key = st.query_params.get("page", "home")
    if active_key not in PAGES:
        active_key = "home"

    bottom_nav(active_key)

    _, renderer = PAGES[active_key]
    renderer()


if __name__ == "__main__":
    main()
