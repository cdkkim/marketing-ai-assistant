#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Early Warning System — selectable methods
LightGBM (classification) + Cox Time-Varying (lifelines) + XGBoost AFT (survival)

Usage examples:
  python early_warning_methods.py --method lgbm --info big_data_set1.csv --kpi big_data_set2.csv --cust big_data_set3.csv --outdir ./out
  python early_warning_methods.py --method cox  --info ... --kpi ... --cust ... --outdir ./out
  python early_warning_methods.py --method aft  --info ... --kpi ... --cust ... --outdir ./out
  python early_warning_methods.py --method all  --info ... --kpi ... --cust ... --outdir ./out

Install:
  pip install numpy pandas scikit-learn lightgbm lifelines xgboost
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from lightgbm import LGBMClassifier
from lifelines import CoxTimeVaryingFitter
import xgboost as xgb


# ------------------------
# Utility parsing/cleaning
# ------------------------

SPECIAL_MISSING = {-999999.9, -999999.0, -99999.9, -99999.0}

BUCKET_COLS = [
    "MCT_OPE_MS_CN",
    "RC_M1_SAA",
    "RC_M1_TO_UE_CT",
    "RC_M1_UE_CUS_CN",
    "RC_M1_AV_NP_AT",
    "APV_CE_RAT",
]

RATE_COLS_0_100 = [
    "DLV_SAA_RAT",
    "M1_SME_RY_SAA_RAT",
    "M1_SME_RY_CNT_RAT",
    "M12_SME_RY_SAA_PCE_RT",
    "M12_SME_BZN_SAA_PCE_RT",
    "M12_SME_RY_ME_MCT_RAT",
    "M12_SME_BZN_ME_MCT_RAT",
    "M12_MAL_1020_RAT","M12_MAL_30_RAT","M12_MAL_40_RAT","M12_MAL_50_RAT","M12_MAL_60_RAT",
    "M12_FME_1020_RAT","M12_FME_30_RAT","M12_FME_40_RAT","M12_FME_50_RAT","M12_FME_60_RAT",
    "MCT_UE_CLN_REU_RAT","MCT_UE_CLN_NEW_RAT",
    "RC_M1_SHC_RSD_UE_CLN_RAT","RC_M1_SHC_WP_UE_CLN_RAT","RC_M1_SHC_FLP_UE_CLN_RAT",
]


def parse_bucket(s: Optional[str]) -> Tuple[Optional[int], Optional[float]]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None, None
    s = str(s).strip()
    m = re.match(r"^(\d+)_", s)
    ordinal = int(m.group(1)) if m else None

    m2 = re.search(r"(\d+)\s*-\s*(\d+)\s*%?", s)
    if m2:
        lo = float(m2.group(1)) / 100.0
        hi = float(m2.group(2)) / 100.0
        return ordinal, float((lo + hi) / 2.0)

    if "90%초과" in s or "90% 초과" in s:
        return ordinal, 0.95
    if "75-90%" in s:
        return ordinal, 0.825
    if "50-75%" in s:
        return ordinal, 0.625
    if "25-50%" in s:
        return ordinal, 0.375
    if "10-25%" in s:
        return ordinal, 0.175
    if "10%이하" in s or "10% 이하" in s or "1구간" in s:
        return ordinal, 0.05

    m3 = re.search(r"(\d+(\.\d+)?)\s*%?", s)
    if m3:
        val = float(m3.group(1)) / 100.0
        return ordinal, val

    return ordinal, None


def add_bucket_features(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            continue
        ordinals, mids = [], []
        for v in df[c].astype("string"):
            o, m = parse_bucket(v)
            ordinals.append(o)
            mids.append(m)
        df[c + "_ORD"] = ordinals
        df[c + "_MID"] = mids
    return df


def to_period_month(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype(str), errors="coerce", format="%Y%m").dt.to_period("M")


def standardize_rates(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df[c].isin(SPECIAL_MISSING), c] = np.nan
        df[c] = df[c] / 100.0
    return df


def replace_special_missing(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df.loc[df[col].isin(SPECIAL_MISSING), col] = np.nan
    return df


def build_peer_zscores(df: pd.DataFrame,
                       peer_keys: Tuple[str, str] = ("MCT_SIGUNGU_NM", "HPSN_MCT_BZN_CD_NM"),
                       value_cols: Optional[List[str]] = None) -> pd.DataFrame:
    if value_cols is None:
        value_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    group_cols = list(peer_keys) + ["TA_YM"]
    g = df.groupby(group_cols)
    for c in value_cols:
        mean = g[c].transform("mean")
        std = g[c].transform("std")
        z = (df[c] - mean) / std.replace({0.0: np.nan})
        df[f"{c}__PEER_Z"] = z
    return df


def add_rolling_features(df: pd.DataFrame,
                         id_col: str = "ENCODED_MCT",
                         time_col: str = "TA_YM",
                         windows: Tuple[int, ...] = (3, 6, 12)) -> pd.DataFrame:
    df = df.sort_values([id_col, time_col]).copy()
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    for w in windows:
        rolled = df.groupby(id_col)[num_cols].rolling(window=w, min_periods=1).mean().reset_index(level=0, drop=True)
        rolled.columns = [f"{c}__MA{w}" for c in rolled.columns]
        df = pd.concat([df, rolled], axis=1)

        if w == windows[0]:
            pctchg = df.groupby(id_col)[num_cols].pct_change(periods=1)
            pctchg.columns = [f"{c}__PCT1" for c in pctchg.columns]
            df = pd.concat([df, pctchg], axis=1)

        vol = df.groupby(id_col)[num_cols].rolling(window=w, min_periods=2).std().reset_index(level=0, drop=True)
        vol.columns = [f"{c}__VOL{w}" for c in vol.columns]
        df = pd.concat([df, vol], axis=1)

    return df


def make_labels(df: pd.DataFrame,
                id_col: str,
                time_col: str,
                drop_horizons: List[int],
                drop_thresh: float,
                close_horizon: int) -> pd.DataFrame:
    df = df.sort_values([id_col, time_col]).copy()

    kpi_candidates = [c for c in ["RC_M1_SAA_MID","RC_M1_TO_UE_CT_MID","RC_M1_UE_CUS_CN_MID","RC_M1_AV_NP_AT_MID"]
                      if c in df.columns]
    if not kpi_candidates:
        raise ValueError("No KPI *_MID columns found for drop labeling.")
    kpi = df[kpi_candidates].ffill(axis=1).bfill(axis=1).iloc[:,0]
    df["KPI_PROXY"] = kpi

    ma3 = df.groupby(id_col)["KPI_PROXY"].rolling(window=3, min_periods=2).mean().reset_index(level=0, drop=True)
    df["KPI_PROXY_MA3"] = ma3

    for H in drop_horizons:
        future = df.groupby(id_col)["KPI_PROXY"].shift(-H)
        ratio = (future - df["KPI_PROXY_MA3"]) / df["KPI_PROXY_MA3"]
        lab = (ratio <= drop_thresh).astype("Int8")
        lab[(future.isna()) | (df["KPI_PROXY_MA3"].isna())] = pd.NA
        df[f"y_drop_h{H}"] = lab

    if "MCT_ME_D" in df.columns:
        close_month = pd.to_datetime(df["MCT_ME_D"], errors="coerce").dt.to_period("M")
        df["__CLOSE_MONTH"] = close_month
        y_close = pd.Series(np.zeros(len(df), dtype="Int8"), index=df.index)
        mask = df["__CLOSE_MONTH"].notna()
        dist = (df.loc[mask, "__CLOSE_MONTH"].astype("int") - df.loc[mask, time_col].astype("int"))
        y_close_mask = (dist >= 1) & (dist <= close_horizon)
        y_close.loc[df.loc[mask].index[y_close_mask]] = 1
        after_close_mask = (dist <= 0)
        y_close.loc[df.loc[mask].index[after_close_mask]] = pd.NA
        df[f"y_close_h{close_horizon}"] = y_close.astype("Int8")
    else:
        df[f"y_close_h{close_horizon}"] = pd.Series(np.zeros(len(df), dtype="Int8"), index=df.index)

    drop_cols = [f"y_drop_h{H}" for H in drop_horizons]
    comp = df[drop_cols + [f"y_close_h{close_horizon}"]].max(axis=1, skipna=True)
    df["y_risk_any"] = comp.astype("Int8")
    return df


# ----------------
# Core ETL stages
# ----------------

def data_extract(info_path: str, kpi_path: str, cust_path: str, sep: str = ",") -> Dict[str, pd.DataFrame]:
    df_info = pd.read_csv(info_path, sep=sep, dtype=str, encoding="utf-8")
    df_kpi  = pd.read_csv(kpi_path,  sep=sep, dtype=str, encoding="utf-8")
    df_cust = pd.read_csv(cust_path, sep=sep, dtype=str, encoding="utf-8")
    return {"info": df_info, "kpi": df_kpi, "cust": df_cust}


def data_transform(d: Dict[str, pd.DataFrame],
                   drop_horizons: List[int] = [1,2,3],
                   drop_thresh: float = -0.30,
                   close_horizon: int = 3) -> pd.DataFrame:
    df_info, df_kpi, df_cust = d["info"].copy(), d["kpi"].copy(), d["cust"].copy()

    if "TA_YM" in df_kpi.columns:
        df_kpi["TA_YM"] = to_period_month(df_kpi["TA_YM"])
    if "TA_YM" in df_cust.columns:
        df_cust["TA_YM"] = to_period_month(df_cust["TA_YM"])
    for col in ["ARE_D","MCT_ME_D"]:
        if col in df_info.columns:
            df_info[col] = pd.to_datetime(df_info[col], errors="coerce")

    df_kpi = add_bucket_features(df_kpi, BUCKET_COLS)

    def to_numeric_smart(df: pd.DataFrame) -> pd.DataFrame:
        for c in df.columns:
            if c in ["ENCODED_MCT","TA_YM"]:
                continue
            if df[c].dtype == object:
                df[c] = pd.to_numeric(df[c].str.replace(',',''), errors="ignore")
        return replace_special_missing(df)

    df_kpi  = to_numeric_smart(df_kpi)
    df_cust = to_numeric_smart(df_cust)

    df_kpi  = standardize_rates(df_kpi, RATE_COLS_0_100)
    df_cust = standardize_rates(df_cust, RATE_COLS_0_100)

    merged = pd.merge(df_kpi, df_cust, on=["ENCODED_MCT","TA_YM"], how="outer", suffixes=("_KPI","_CUST"))
    merged = pd.merge(merged, df_info, on=["ENCODED_MCT"], how="left")

    for col in ["MCT_SIGUNGU_NM","HPSN_MCT_BZN_CD_NM"]:
        if col not in merged.columns:
            merged[col] = np.nan
    num_cols = [c for c in merged.columns if pd.api.types.is_numeric_dtype(merged[c])]
    merged = build_peer_zscores(merged, value_cols=num_cols)
    merged = add_rolling_features(merged, id_col="ENCODED_MCT", time_col="TA_YM", windows=(3,6,12))

    merged = make_labels(merged,
                         id_col="ENCODED_MCT",
                         time_col="TA_YM",
                         drop_horizons=drop_horizons,
                         drop_thresh=drop_thresh,
                         close_horizon=close_horizon)
    return merged


def data_load(df: pd.DataFrame, outdir: str) -> Dict[str, str]:
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    p_parquet = out / "dataset_features_labels.parquet"
    p_csv     = out / "dataset_features_labels.csv"
    df.to_parquet(p_parquet, index=False)
    df.to_csv(p_csv, index=False, encoding="utf-8")

    label_cols = [c for c in df.columns if c.startswith("y_drop_h")] + \
                 [c for c in df.columns if c.startswith("y_close_h")] + ["y_risk_any"]
    summary = df[label_cols].astype("float").describe().T
    p_sum = out / "label_summary.csv"
    summary.to_csv(p_sum, encoding="utf-8")
    return {"parquet": str(p_parquet), "csv": str(p_csv), "summary": str(p_sum)}


# ----------------
# LightGBM track
# ----------------

def build_lgbm_model(cat_cols: List[str], num_cols: List[str]) -> Pipeline:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer(transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ])

    clf = LGBMClassifier(
        objective="binary",
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=127,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42
    )
    pipe = Pipeline(steps=[("preprocess", pre), ("clf", clf)])
    return pipe


def _time_split(df: pd.DataFrame, time_col: str, test_months: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    months = df[time_col].dropna().unique()
    months = np.sort(months)
    if len(months) < test_months + 3:
        raise ValueError("Not enough months to split. Need at least 3+test_months.")
    cutoff = months[-test_months]
    train = df[df[time_col] < cutoff].copy()
    test  = df[df[time_col] >= cutoff].copy()
    return train, test


def run_lgbm(merged: pd.DataFrame, out: Path, test_months: int = 2) -> None:
    print("\n[LightGBM] Train/Test on y_risk_any")
    train_df, test_df = _time_split(merged.dropna(subset=["y_risk_any"]), time_col="TA_YM", test_months=test_months)

    drop_cols = { "ENCODED_MCT", "TA_YM", "ARE_D","MCT_ME_D","__CLOSE_MONTH", "y_risk_any" }
    drop_cols |= {c for c in merged.columns if c.startswith("y_drop_h") or c.startswith("y_close_h")}
    feature_cols = [c for c in merged.columns if c not in drop_cols]

    cat_cols = [c for c in feature_cols if merged[c].dtype == "object" or pd.api.types.is_string_dtype(merged[c])]
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(merged[c])]

    model = build_lgbm_model(cat_cols, num_cols)
    model.fit(train_df[feature_cols], train_df["y_risk_any"].astype(int))

    p_test = model.predict_proba(test_df[feature_cols])[:,1]
    roc = roc_auc_score(test_df["y_risk_any"].astype(int), p_test)
    pr  = average_precision_score(test_df["y_risk_any"].astype(int), p_test)
    print(f"[LightGBM] ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}  (n_test={len(test_df)})")
    (out / "lgbm_info.txt").write_text(f"ROC-AUC={roc:.4f}\nPR-AUC={pr:.4f}\n", encoding="utf-8")


# -----------------------------
# Survival (time-varying) track
# -----------------------------

def build_survival_frame_timevarying(df: pd.DataFrame,
                                     id_col: str = "ENCODED_MCT",
                                     time_col: str = "TA_YM") -> pd.DataFrame:
    df = df.sort_values([id_col, time_col]).copy()
    df["__t_idx"] = df.groupby(id_col).cumcount()
    df["__t_idx_next"] = df["__t_idx"] + 1

    close_month = pd.to_datetime(df.get("MCT_ME_D", pd.NaT), errors="coerce").dt.to_period("M")
    df["__CLOSE_MONTH"] = close_month

    event = np.zeros(len(df), dtype=int)
    has_close = df["__CLOSE_MONTH"].notna()
    event_idx = has_close & (df[time_col] == df["__CLOSE_MONTH"])
    event[event_idx] = 1

    tv = df.copy()
    tv["start"] = tv["__t_idx"]
    tv["stop"]  = tv["__t_idx_next"]
    tv["event"] = event

    tv = tv.drop(columns=[c for c in ["__t_idx","__t_idx_next"] if c in tv.columns])
    return tv


def train_cox_timevarying(tv_df: pd.DataFrame,
                          id_col: str = "ENCODED_MCT",
                          time_cols: Tuple[str, str] = ("start","stop"),
                          event_col: str = "event",
                          exclude_cols: Optional[List[str]] = None) -> Tuple[CoxTimeVaryingFitter, List[str]]:
    exclude = set(exclude_cols or []) | {id_col, event_col, time_cols[0], time_cols[1]}
    exclude |= {c for c in tv_df.columns if c.startswith("y_drop_h") or c.startswith("y_close_h") or c=="y_risk_any"}
    exclude |= {"ARE_D","MCT_ME_D","__CLOSE_MONTH"}

    covariates = [c for c in tv_df.columns if c not in exclude and pd.api.types.is_numeric_dtype(tv_df[c])]
    tv_fit = tv_df[[id_col, *time_cols, event_col, *covariates]].copy()
    for c in covariates:
        tv_fit[c] = tv_fit[c].astype(float)
        tv_fit[c] = tv_fit[c].fillna(tv_fit[c].median())

    ctv = CoxTimeVaryingFitter(penalizer=0.1)
    ctv.fit(tv_fit, id_col=id_col, start_col=time_cols[0], stop_col=time_cols[1], event_col=event_col, show_progress=False)
    return ctv, covariates


def test_cox_timevarying(ctv: CoxTimeVaryingFitter, tv_df: pd.DataFrame,
                         id_col: str = "ENCODED_MCT",
                         time_cols: Tuple[str, str] = ("start","stop"),
                         event_col: str = "event",
                         covariates: Optional[List[str]] = None) -> Dict[str, any]:
    if covariates is None:
        covariates = [c for c in tv_df.columns if c not in {id_col, event_col, *time_cols} and pd.api.types.is_numeric_dtype(tv_df[c])]
    tv_eval = tv_df[[id_col, *time_cols, event_col, *covariates]].copy()
    for c in covariates:
        tv_eval[c] = tv_eval[c].astype(float).fillna(tv_eval[c].median())

    c_index = ctv.score_(tv_eval, scoring_method="concordance_index")
    pll = ctv._log_likelihood
    return {"concordance_index": float(c_index), "partial_log_likelihood": float(pll)}


# -----------------------------
# XGBoost AFT (survival) track
# -----------------------------

def build_aft_dmatrix(tv_df: pd.DataFrame,
                      id_col: str = "ENCODED_MCT",
                      time_cols: Tuple[str, str] = ("start", "stop"),
                      event_col: str = "event",
                      exclude_cols: Optional[List[str]] = None) -> Tuple[xgb.DMatrix, List[str]]:
    exclude = set(exclude_cols or []) | {id_col, event_col, time_cols[0], time_cols[1]}
    exclude |= {c for c in tv_df.columns if c.startswith("y_drop_h") or c.startswith("y_close_h") or c=="y_risk_any"}
    exclude |= {"ARE_D","MCT_ME_D","__CLOSE_MONTH"}

    covariates = [c for c in tv_df.columns if c not in exclude and pd.api.types.is_numeric_dtype(tv_df[c])]
    X = tv_df[covariates].astype(float).fillna(tv_df[covariates].median())

    y_lower = tv_df[time_cols[0]].astype(float).values
    y_upper = tv_df[time_cols[1]].astype(float).values
    mask_cens = tv_df[event_col] == 0
    y_upper[mask_cens] = np.inf

    dmat = xgb.DMatrix(X, label_lower=y_lower, label_upper=y_upper, feature_names=covariates)
    return dmat, covariates


def train_aft(dtrain: xgb.DMatrix, num_round: int = 300) -> xgb.Booster:
    params = {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "aft_loss_distribution": "normal",
        "aft_loss_distribution_scale": 1.0,
        "tree_method": "hist",
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
        "seed": 42,
    }
    model = xgb.train(params, dtrain, num_boost_round=num_round)
    return model


def test_aft(model: xgb.Booster, dtest: xgb.DMatrix) -> Dict[str, any]:
    pred = model.predict(dtest, output_margin=True)
    y_lower = dtest.get_float_info("label_lower_bound")
    mse = mean_squared_error(y_lower, pred)
    return {"aft_mse": float(mse)}


# -------------
# CLI Entrypoint
# -------------

def main():
    ap = argparse.ArgumentParser(description="Early Warning — choose method: lgbm / cox / aft / all")
    ap.add_argument("--method", choices=["lgbm", "cox", "aft", "all"], default="all",
                    help="모델 선택 (기본 all): lgbm / cox / aft / all")
    ap.add_argument("--info", required=True, help="dataset1 CSV path")
    ap.add_argument("--kpi",  required=True, help="dataset2 CSV path")
    ap.add_argument("--cust", required=True, help="dataset3 CSV path")
    ap.add_argument("--outdir", required=True, help="output folder")
    ap.add_argument("--sep", default=",", help="CSV separator")
    ap.add_argument("--drop_horizons", nargs="+", type=int, default=[1,2,3])
    ap.add_argument("--drop_thresh", type=float, default=-0.30)
    ap.add_argument("--close_horizon", type=int, default=3)
    ap.add_argument("--test_months", type=int, default=2)
    args = ap.parse_args()

    # ETL
    data = data_extract(args.info, args.kpi, args.cust, sep=args.sep)
    merged = data_transform(data,
                            drop_horizons=args.drop_horizons,
                            drop_thresh=args.drop_thresh,
                            close_horizon=args.close_horizon)
    paths = data_load(merged, args.outdir)

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    # --- LightGBM classification (y_risk_any)
    if args.method in ["lgbm", "all"]:
        run_lgbm(merged, out, test_months=args.test_months)

    # --- Survival: Cox Time-Varying
    if args.method in ["cox", "all"]:
        print("\n[Survival] Cox Time-Varying")
        tv = build_survival_frame_timevarying(merged, id_col="ENCODED_MCT", time_col="TA_YM")
        months = np.sort(merged["TA_YM"].dropna().unique())
        if len(months) < args.test_months + 3:
            raise RuntimeError("Not enough months for time-based split.")
        cutoff = months[-args.test_months]
        tv_train = tv[tv["TA_YM"] < cutoff].copy()
        tv_test  = tv[tv["TA_YM"] >= cutoff].copy()

        ctv, covs = train_cox_timevarying(tv_train, id_col="ENCODED_MCT",
                                          time_cols=("start","stop"), event_col="event")
        surv_metrics = test_cox_timevarying(ctv, tv_test, id_col="ENCODED_MCT",
                                            time_cols=("start","stop"), event_col="event", covariates=covs)
        print(f"[CoxTV] concordance_index={surv_metrics['concordance_index']:.4f}, "
              f"partial_log_likelihood={surv_metrics['partial_log_likelihood']:.2f}")
        (out / "cox_summary.txt").write_text(str(ctv.summary), encoding="utf-8")

    # --- Survival: XGBoost AFT
    if args.method in ["aft", "all"]:
        print("\n[Survival] XGBoost AFT")
        if 'tv' not in locals():
            tv = build_survival_frame_timevarying(merged, id_col="ENCODED_MCT", time_col="TA_YM")
            months = np.sort(merged["TA_YM"].dropna().unique())
            cutoff = months[-args.test_months]
            tv_train = tv[tv["TA_YM"] < cutoff].copy()
            tv_test  = tv[tv["TA_YM"] >= cutoff].copy()

        dtrain, _ = build_aft_dmatrix(tv_train, id_col="ENCODED_MCT")
        dtest, _  = build_aft_dmatrix(tv_test, id_col="ENCODED_MCT")
        aft_model = train_aft(dtrain, num_round=300)
        aft_metrics = test_aft(aft_model, dtest)
        print(f"[AFT] MSE (start vs pred log-time) = {aft_metrics['aft_mse']:.4f}")
        (out / "aft_metrics.txt").write_text(str(aft_metrics), encoding="utf-8")

    print("\n[SAVED] Data:", paths)
    print("[DONE] Method(s):", args.method)


if __name__ == "__main__":
    main()
