import os
import re
import json
import joblib
import pandas as pd

from pathlib import Path
from datetime import datetime, UTC
from typing import Dict, Any, Tuple

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "synthetic_emails_unificados.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts_email_ml_compare"
ARTIFACTS_DIR.mkdir(exist_ok=True)

TARGETS = [
    "tipo_base",
    "prioridad_label",
    "riesgo_label",
    "tono_label",
]

REQUIRED_COLUMNS = [
    "subject",
    "body",
    "sender_type",
    "tipo_base",
    "prioridad_label",
    "riesgo_label",
    "tono_label",
]

RANDOM_STATE = 42
TEST_SIZE = 0.15
VALID_SIZE = 0.1765  # deja aprox 70/15/15 final


# ============================================================
# UTILIDADES PICKLE-SAFE
# ============================================================

def to_text_list(x):
    """
    Convierte una columna seleccionada por ColumnTransformer
    en una lista de strings compatible con TfidfVectorizer.
    Debe estar a nivel global para que joblib pueda serializarla.
    """
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            s = x.iloc[:, 0]
        else:
            s = x.squeeze()
    elif isinstance(x, pd.Series):
        s = x
    else:
        s = pd.Series(x)

    return s.astype(str).fillna("").tolist()


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


# ============================================================
# LIMPIEZA / PREPROCESAMIENTO
# ============================================================

def normalize_text(text: str) -> str:
    if pd.isna(text):
        return ""

    text = str(text)

    # placeholders comunes del dataset
    text = re.sub(r"\[fecha\]", " TOKEN_FECHA ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[nombre del curso\]", " TOKEN_CURSO ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[nombre del trabajo\]", " TOKEN_TRABAJO ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[número de matrícula\]", " TOKEN_MATRICULA ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[motivo\]", " TOKEN_MOTIVO ", text, flags=re.IGNORECASE)

    # cualquier otro placeholder entre corchetes
    text = re.sub(r"\[[^\]]+\]", " TOKEN_PLACEHOLDER ", text)

    # normalizar espacios
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset(csv_file: Path) -> pd.DataFrame:
    if not csv_file.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {csv_file}")

    df = pd.read_csv(csv_file)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    df = df.dropna(subset=REQUIRED_COLUMNS).copy()

    df["subject"] = df["subject"].apply(normalize_text)
    df["body"] = df["body"].apply(normalize_text)
    df["sender_type"] = df["sender_type"].astype(str).str.strip().str.lower()

    df["text"] = "SUBJECT: " + df["subject"] + " BODY: " + df["body"]
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()

    df = df[df["text"].str.len() > 10].copy()

    if len(df) < 100:
        raise ValueError("Muy pocas filas para entrenar algo robusto.")

    return df


# ============================================================
# FEATURES
# ============================================================

def text_feature_pipeline_basic():
    return Pipeline([
        (
            "to_text_1d",
            FunctionTransformer(to_text_list, validate=False)
        ),
        (
            "features",
            FeatureUnion([
                (
                    "word_tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        strip_accents="unicode",
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.95,
                        sublinear_tf=True,
                        max_features=50000,
                    ),
                ),
                (
                    "char_tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        strip_accents="unicode",
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=2,
                        sublinear_tf=True,
                        max_features=30000,
                    ),
                ),
            ])
        ),
    ])


def text_feature_pipeline_plus_sender():
    return Pipeline([
        (
            "to_text_1d",
            FunctionTransformer(to_text_list, validate=False)
        ),
        (
            "features",
            FeatureUnion([
                (
                    "word_tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        strip_accents="unicode",
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.95,
                        sublinear_tf=True,
                        max_features=65000,
                    ),
                ),
                (
                    "char_tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        strip_accents="unicode",
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=2,
                        sublinear_tf=True,
                        max_features=40000,
                    ),
                ),
            ])
        ),
    ])


def build_experiment_pipeline(experiment_name: str) -> Pipeline:
    if experiment_name == "subject_body":
        preprocessor = ColumnTransformer(
            transformers=[
                ("text", text_feature_pipeline_basic(), "text"),
            ],
            remainder="drop"
        )

        clf = LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            solver="saga",
            C=2.0,
            random_state=RANDOM_STATE,
        )

    elif experiment_name == "subject_body_sender":
        preprocessor = ColumnTransformer(
            transformers=[
                ("text", text_feature_pipeline_plus_sender(), "text"),
                ("sender", OneHotEncoder(handle_unknown="ignore"), ["sender_type"]),
            ],
            remainder="drop"
        )

        clf = LogisticRegression(
            max_iter=4500,
            class_weight="balanced",
            solver="saga",
            C=1.5,
            random_state=RANDOM_STATE,
        )

    else:
        raise ValueError(f"Experimento no soportado: {experiment_name}")

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf),
    ])


# ============================================================
# ENTRENAMIENTO / EVALUACION
# ============================================================

def split_data(
    X: pd.DataFrame,
    y: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full,
        y_train_full,
        test_size=VALID_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train_full
    )

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def evaluate(model, X, y) -> Dict[str, Any]:
    preds = model.predict(X)

    return {
        "accuracy": float(accuracy_score(y, preds)),
        "f1_macro": float(f1_score(y, preds, average="macro")),
        "f1_weighted": float(f1_score(y, preds, average="weighted")),
        "classification_report": classification_report(y, preds, output_dict=True),
    }


def save_json(data: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def train_one_target(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"TARGET: {target}")
    print("=" * 80)

    X = df[["text", "sender_type"]].copy()
    y = df[target].astype(str)

    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(X, y)

    experiments = [
        "subject_body",
        "subject_body_sender",
    ]

    validation_results = {}
    best_name = None
    best_score = -1.0

    for exp_name in experiments:
        print(f"\nEntrenando experimento: {exp_name}")
        model = build_experiment_pipeline(exp_name)
        model.fit(X_train, y_train)

        metrics = evaluate(model, X_valid, y_valid)
        validation_results[exp_name] = metrics

        print(
            f"Valid -> accuracy={metrics['accuracy']:.4f} | "
            f"f1_macro={metrics['f1_macro']:.4f} | "
            f"f1_weighted={metrics['f1_weighted']:.4f}"
        )

        if metrics["f1_macro"] > best_score:
            best_score = metrics["f1_macro"]
            best_name = exp_name

    print(f"\nMejor experimento para {target}: {best_name}")

    final_model = build_experiment_pipeline(best_name)

    X_train_final = pd.concat([X_train, X_valid], axis=0)
    y_train_final = pd.concat([y_train, y_valid], axis=0)

    final_model.fit(X_train_final, y_train_final)
    test_metrics = evaluate(final_model, X_test, y_test)

    print(
        f"Test -> accuracy={test_metrics['accuracy']:.4f} | "
        f"f1_macro={test_metrics['f1_macro']:.4f} | "
        f"f1_weighted={test_metrics['f1_weighted']:.4f}"
    )

    artifact = {
        "target": target,
        "best_experiment": best_name,
        "pipeline": final_model,
        "labels": sorted(y.unique().tolist()),
        "created_at": utc_now_iso(),
        "dataset_rows_used": int(len(df)),
        "input_columns": ["subject", "body"] if best_name == "subject_body" else ["subject", "body", "sender_type"],
        "validation_results": validation_results,
        "test_metrics": test_metrics,
    }

    artifact_path = ARTIFACTS_DIR / f"{target}_best_model.joblib"
    joblib.dump(artifact, artifact_path)

    summary = {
        "target": target,
        "best_experiment": best_name,
        "artifact_path": str(artifact_path),
        "test_metrics": {
            "accuracy": test_metrics["accuracy"],
            "f1_macro": test_metrics["f1_macro"],
            "f1_weighted": test_metrics["f1_weighted"],
        },
    }

    return summary


def main():
    print("BASE_DIR:", BASE_DIR)
    print("CSV_FILE:", CSV_FILE)
    print("Existe CSV:", CSV_FILE.exists())

    df = load_dataset(CSV_FILE)
    print(f"Dataset cargado: {len(df)} filas")

    run_summary = {
        "csv_file": str(CSV_FILE),
        "created_at": utc_now_iso(),
        "targets": {},
    }

    for target in TARGETS:
        result = train_one_target(df, target)
        run_summary["targets"][target] = result

    summary_path = ARTIFACTS_DIR / "comparison_summary.json"
    save_json(run_summary, summary_path)

    print("\n" + "=" * 80)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 80)
    print(f"Resumen guardado en: {summary_path}")

    for target, info in run_summary["targets"].items():
        print(
            f"{target}: {info['best_experiment']} | "
            f"f1_macro={info['test_metrics']['f1_macro']:.4f}"
        )


if __name__ == "__main__":
    main()