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

TARGET = "prioridad_label"

REQUIRED_COLUMNS = [
    "subject",
    "body",
    "sender_type",
    "prioridad_label",
]

RANDOM_STATE = 42
TEST_SIZE = 0.15
VALID_SIZE = 0.1765  # deja aprox 70/15/15
SUMMARY_PATH = ARTIFACTS_DIR / "comparison_summary.json"


# ============================================================
# UTILIDADES PICKLE-SAFE
# ============================================================

def to_text_list(x):
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

    text = re.sub(r"\[fecha\]", " TOKEN_FECHA ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[nombre del curso\]", " TOKEN_CURSO ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[nombre del trabajo\]", " TOKEN_TRABAJO ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[número de matrícula\]", " TOKEN_MATRICULA ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[motivo\]", " TOKEN_MOTIVO ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[[^\]]+\]", " TOKEN_PLACEHOLDER ", text)

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

def text_features(max_word_features: int = 70000, max_char_features: int = 45000):
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
                        max_features=max_word_features,
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
                        max_features=max_char_features,
                    ),
                ),
            ])
        ),
    ])


def build_pipeline(experiment_name: str) -> Pipeline:
    if experiment_name == "subject_body":
        preprocessor = ColumnTransformer(
            transformers=[
                ("text", text_features(60000, 35000), "text"),
            ],
            remainder="drop"
        )

        clf = LogisticRegression(
            max_iter=10000,
            class_weight="balanced",
            solver="saga",
            C=1.5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    elif experiment_name == "subject_body_sender":
        preprocessor = ColumnTransformer(
            transformers=[
                ("text", text_features(70000, 45000), "text"),
                ("sender", OneHotEncoder(handle_unknown="ignore"), ["sender_type"]),
            ],
            remainder="drop"
        )

        clf = LogisticRegression(
            max_iter=12000,
            class_weight="balanced",
            solver="saga",
            C=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    elif experiment_name == "subject_body_sender_strong":
        preprocessor = ColumnTransformer(
            transformers=[
                ("text", text_features(80000, 50000), "text"),
                ("sender", OneHotEncoder(handle_unknown="ignore"), ["sender_type"]),
            ],
            remainder="drop"
        )

        clf = LogisticRegression(
            max_iter=15000,
            class_weight="balanced",
            solver="saga",
            C=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
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


def update_summary_file(summary_result: Dict[str, Any]) -> None:
    if SUMMARY_PATH.exists():
        with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
            summary = json.load(f)
    else:
        summary = {
            "csv_file": str(CSV_FILE),
            "created_at": utc_now_iso(),
            "targets": {}
        }

    summary["updated_at"] = utc_now_iso()
    if "targets" not in summary:
        summary["targets"] = {}

    summary["targets"][TARGET] = summary_result
    save_json(summary, SUMMARY_PATH)


def train_prioridad_only(df: pd.DataFrame) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"REENTRENANDO SOLO: {TARGET}")
    print("=" * 80)

    X = df[["text", "sender_type"]].copy()
    y = df[TARGET].astype(str)

    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(X, y)

    experiments = [
        "subject_body",
        "subject_body_sender",
        "subject_body_sender_strong",
    ]

    validation_results = {}
    best_name = None
    best_score = -1.0

    for exp_name in experiments:
        print(f"\nEntrenando experimento: {exp_name}")
        model = build_pipeline(exp_name)
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

    print(f"\nMejor experimento nuevo para {TARGET}: {best_name}")

    final_model = build_pipeline(best_name)

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
        "target": TARGET,
        "best_experiment": best_name,
        "pipeline": final_model,
        "labels": sorted(y.unique().tolist()),
        "created_at": utc_now_iso(),
        "dataset_rows_used": int(len(df)),
        "input_columns": ["subject", "body"] if best_name == "subject_body" else ["subject", "body", "sender_type"],
        "validation_results": validation_results,
        "test_metrics": test_metrics,
    }

    artifact_path = ARTIFACTS_DIR / f"{TARGET}_best_model.joblib"
    joblib.dump(artifact, artifact_path)

    result = {
        "target": TARGET,
        "best_experiment": best_name,
        "artifact_path": str(artifact_path),
        "test_metrics": {
            "accuracy": test_metrics["accuracy"],
            "f1_macro": test_metrics["f1_macro"],
            "f1_weighted": test_metrics["f1_weighted"],
        },
    }

    update_summary_file(result)
    return result


def main():
    print("BASE_DIR:", BASE_DIR)
    print("CSV_FILE:", CSV_FILE)
    print("Existe CSV:", CSV_FILE.exists())

    df = load_dataset(CSV_FILE)
    print(f"Dataset cargado: {len(df)} filas")

    result = train_prioridad_only(df)

    print("\n" + "=" * 80)
    print("REENTRENAMIENTO COMPLETADO")
    print("=" * 80)
    print(f"Modelo actualizado: {result['artifact_path']}")
    print(f"Mejor experimento: {result['best_experiment']}")
    print(
        f"Métricas test -> accuracy={result['test_metrics']['accuracy']:.4f} | "
        f"f1_macro={result['test_metrics']['f1_macro']:.4f} | "
        f"f1_weighted={result['test_metrics']['f1_weighted']:.4f}"
    )
    print(f"Resumen actualizado en: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()