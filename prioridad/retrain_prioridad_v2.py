import re
import json
import joblib
import pandas as pd

from pathlib import Path
from datetime import datetime, UTC
from typing import Dict, Any, Tuple

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, MaxAbsScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "synthetic_emails_unificados.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts_email_ml_compare"
VERSIONS_DIR = ARTIFACTS_DIR / "versions"
ARTIFACTS_DIR.mkdir(exist_ok=True)
VERSIONS_DIR.mkdir(exist_ok=True)

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
V2_SUMMARY_PATH = ARTIFACTS_DIR / "prioridad_label_v2_summary.json"


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


def version_tag() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


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
# FEATURES DE TEXTO
# ============================================================

def text_features(max_word_features: int = 70000, max_char_features: int = 45000):
    return Pipeline([
        ("to_text_1d", FunctionTransformer(to_text_list, validate=False)),
        ("features", FeatureUnion([
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
        ])),
    ])


# ============================================================
# FEATURES MANUALES PARA PRIORIDAD
# ============================================================

def count_any(text: str, patterns) -> int:
    return sum(len(re.findall(p, text, flags=re.IGNORECASE)) for p in patterns)


def extract_priority_signal_df(x) -> pd.DataFrame:
    """
    x llega como DataFrame con columnas: subject, body
    """
    if isinstance(x, pd.DataFrame):
        df = x.copy()
        df.columns = ["subject", "body"]
    else:
        df = pd.DataFrame(x, columns=["subject", "body"])

    subject = df["subject"].fillna("").astype(str)
    body = df["body"].fillna("").astype(str)
    text = (subject + " " + body).str.strip()

    lower_subject = subject.str.lower()
    lower_body = body.str.lower()
    lower_text = text.str.lower()

    urgency_patterns = [
        r"\burgente\b", r"\bprioridad\b", r"\binmediato\b", r"\bcuanto antes\b",
        r"\bhoy\b", r"\bahora\b", r"\blo antes posible\b", r"\bpronto\b"
    ]
    payment_patterns = [
        r"\bdeuda\b", r"\bpago\b", r"\bvencid", r"\bmora\b",
        r"\bregularizar\b", r"\bsuspensi", r"\bbloquead", r"\bfinanzas\b"
    ]
    request_patterns = [
        r"\bsolicito\b", r"\bagradeceria\b", r"\bnecesito\b", r"\bpor favor\b",
        r"\bconsult", r"\brevisi", r"\bprorroga\b", r"\bseguimiento\b"
    ]
    negative_patterns = [
        r"\bproblema\b", r"\berror\b", r"\bno puedo\b", r"\binconveniente\b",
        r"\bmolest", r"\bfrustr", r"\bdesesper", r"\bansios"
    ]
    deadline_patterns = [
        r"\bfecha limite\b", r"\bvence\b", r"\bplazo\b", r"\bhoy\b",
        r"\besta semana\b", r"\bmañana\b", r"\besta noche\b"
    ]

    def uppercase_ratio(s: str) -> float:
        letters = [c for c in s if c.isalpha()]
        if not letters:
            return 0.0
        uppers = [c for c in letters if c.isupper()]
        return len(uppers) / len(letters)

    features = pd.DataFrame({
        "subject_len": subject.str.len(),
        "body_len": body.str.len(),
        "text_len": text.str.len(),
        "subject_words": subject.str.split().str.len(),
        "body_words": body.str.split().str.len(),
        "text_words": text.str.split().str.len(),
        "exclamation_count": text.str.count(r"!"),
        "question_count": text.str.count(r"\?"),
        "digit_count": text.str.count(r"\d"),
        "caps_ratio": text.apply(uppercase_ratio),
        "subject_has_urgent": lower_subject.str.contains(r"\burgente\b").astype(int),
        "subject_has_pago": lower_subject.str.contains(r"\bpago\b").astype(int),
        "subject_has_revision": lower_subject.str.contains(r"\brevisi").astype(int),
        "subject_has_seguimiento": lower_subject.str.contains(r"\bseguimiento\b").astype(int),
        "body_has_bloqueo": lower_body.str.contains(r"\bbloquead|\bsuspend").astype(int),
        "body_has_hoy": lower_body.str.contains(r"\bhoy\b").astype(int),
        "body_has_deuda": lower_body.str.contains(r"\bdeuda\b|\bmora\b").astype(int),
        "body_has_plazo": lower_body.str.contains(r"\bplazo\b|\bvence\b|\bfecha limite\b").astype(int),
        "urgency_hits": lower_text.apply(lambda s: count_any(s, urgency_patterns)),
        "payment_hits": lower_text.apply(lambda s: count_any(s, payment_patterns)),
        "request_hits": lower_text.apply(lambda s: count_any(s, request_patterns)),
        "negative_hits": lower_text.apply(lambda s: count_any(s, negative_patterns)),
        "deadline_hits": lower_text.apply(lambda s: count_any(s, deadline_patterns)),
    })

    return features


def priority_signal_pipeline():
    return Pipeline([
        ("extract", FunctionTransformer(extract_priority_signal_df, validate=False)),
        ("scale", MaxAbsScaler()),
    ])


# ============================================================
# MODELOS
# ============================================================

def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("text", text_features(80000, 50000), "text"),
            ("sender", OneHotEncoder(handle_unknown="ignore"), ["sender_type"]),
            ("signals", priority_signal_pipeline(), ["subject", "body"]),
        ],
        remainder="drop"
    )


def build_candidates() -> Dict[str, Pipeline]:
    pre = build_preprocessor()

    candidates = {
        "logreg_v2": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression(
                max_iter=8000,
                class_weight="balanced",
                solver="saga",
                C=1.0,
                random_state=RANDOM_STATE,
            )),
        ]),
        "sgd_logloss_v2": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", SGDClassifier(
                loss="log_loss",
                alpha=1e-5,
                penalty="l2",
                class_weight="balanced",
                max_iter=6000,
                tol=1e-4,
                random_state=RANDOM_STATE,
            )),
        ]),
        "linear_svc_calibrated_v2": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", CalibratedClassifierCV(
                estimator=LinearSVC(
                    C=1.0,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
                method="sigmoid",
                cv=3,
            )),
        ]),
    }

    return candidates


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

    result = {
        "accuracy": float(accuracy_score(y, preds)),
        "f1_macro": float(f1_score(y, preds, average="macro")),
        "f1_weighted": float(f1_score(y, preds, average="weighted")),
        "classification_report": classification_report(y, preds, output_dict=True),
        "confusion_matrix": confusion_matrix(y, preds).tolist(),
    }

    try:
        probs = model.predict_proba(X)
        result["has_predict_proba"] = True
        result["avg_max_probability"] = float(probs.max(axis=1).mean())
    except Exception:
        result["has_predict_proba"] = False
        result["avg_max_probability"] = None

    return result


def save_json(data: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def train_prioridad_v2(df: pd.DataFrame) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"REENTRENANDO V2 SOLO: {TARGET}")
    print("=" * 80)

    X = df[["subject", "body", "text", "sender_type"]].copy()
    y = df[TARGET].astype(str)

    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(X, y)

    candidates = build_candidates()

    validation_results = {}
    best_name = None
    best_score = -1.0

    for name, model in candidates.items():
        print(f"\nEntrenando candidato: {name}")
        model.fit(X_train, y_train)

        metrics = evaluate(model, X_valid, y_valid)
        validation_results[name] = metrics

        print(
            f"Valid -> accuracy={metrics['accuracy']:.4f} | "
            f"f1_macro={metrics['f1_macro']:.4f} | "
            f"f1_weighted={metrics['f1_weighted']:.4f} | "
            f"avg_max_prob={metrics['avg_max_probability']}"
        )

        if metrics["f1_macro"] > best_score:
            best_score = metrics["f1_macro"]
            best_name = name

    print(f"\nMejor candidato V2 para {TARGET}: {best_name}")

    final_model = build_candidates()[best_name]
    X_train_final = pd.concat([X_train, X_valid], axis=0)
    y_train_final = pd.concat([y_train, y_valid], axis=0)

    final_model.fit(X_train_final, y_train_final)
    test_metrics = evaluate(final_model, X_test, y_test)

    print(
        f"Test -> accuracy={test_metrics['accuracy']:.4f} | "
        f"f1_macro={test_metrics['f1_macro']:.4f} | "
        f"f1_weighted={test_metrics['f1_weighted']:.4f} | "
        f"avg_max_prob={test_metrics['avg_max_probability']}"
    )

    tag = version_tag()
    versioned_path = VERSIONS_DIR / f"{TARGET}_v2_{best_name}_{tag}.joblib"
    latest_path = ARTIFACTS_DIR / f"{TARGET}_best_model_v2_latest.joblib"

    artifact = {
        "target": TARGET,
        "version": "v2",
        "best_experiment": best_name,
        "pipeline": final_model,
        "labels": sorted(y.unique().tolist()),
        "created_at": utc_now_iso(),
        "dataset_rows_used": int(len(df)),
        "input_columns": ["subject", "body", "sender_type"],
        "notes": "V2 con TF-IDF texto + sender_type + señales manuales de prioridad + comparación de 3 algoritmos",
        "validation_results": validation_results,
        "test_metrics": test_metrics,
        "versioned_path": str(versioned_path),
        "latest_path": str(latest_path),
    }

    joblib.dump(artifact, versioned_path)
    joblib.dump(artifact, latest_path)

    summary = {
        "target": TARGET,
        "version": "v2",
        "best_experiment": best_name,
        "versioned_path": str(versioned_path),
        "latest_path": str(latest_path),
        "test_metrics": {
            "accuracy": test_metrics["accuracy"],
            "f1_macro": test_metrics["f1_macro"],
            "f1_weighted": test_metrics["f1_weighted"],
            "avg_max_probability": test_metrics["avg_max_probability"],
        },
        "validation_results": {
            k: {
                "accuracy": v["accuracy"],
                "f1_macro": v["f1_macro"],
                "f1_weighted": v["f1_weighted"],
                "avg_max_probability": v["avg_max_probability"],
            }
            for k, v in validation_results.items()
        },
        "created_at": utc_now_iso(),
    }

    save_json(summary, V2_SUMMARY_PATH)
    return summary


def main():
    print("BASE_DIR:", BASE_DIR)
    print("CSV_FILE:", CSV_FILE)
    print("Existe CSV:", CSV_FILE.exists())

    df = load_dataset(CSV_FILE)
    print(f"Dataset cargado: {len(df)} filas")

    result = train_prioridad_v2(df)

    print("\n" + "=" * 80)
    print("REENTRENAMIENTO V2 COMPLETADO")
    print("=" * 80)
    print(f"Mejor candidato: {result['best_experiment']}")
    print(f"Modelo versionado: {result['versioned_path']}")
    print(f"Modelo latest: {result['latest_path']}")
    print(
        f"Test -> accuracy={result['test_metrics']['accuracy']:.4f} | "
        f"f1_macro={result['test_metrics']['f1_macro']:.4f} | "
        f"f1_weighted={result['test_metrics']['f1_weighted']:.4f}"
    )
    print(f"Resumen V2: {V2_SUMMARY_PATH}")


if __name__ == "__main__":
    main()