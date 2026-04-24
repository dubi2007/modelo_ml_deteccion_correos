"""
train_email_priority_v4.py
==========================
Mejoras sobre V3:
  - StratifiedKFold en lugar de un único split
  - Optuna para búsqueda de hiperparámetros
  - Calibración de probabilidades para TODOS los modelos
  - SelectKBest para reducción de features (evita sobreajuste)
  - SHAP para explicabilidad
  - Reporte de drift básico (distribución de predicciones)
  - Hash MD5 del dataset + skops/ONNX opcional para serialización robusta
  - Logging estructurado en lugar de print()
"""

import re
import json
import hashlib
import logging
import cloudpickle
import pandas as pd
import numpy as np

from pathlib import Path
from datetime import datetime, UTC
from typing import Dict, Any

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
)

# ============================================================
# CONFIG
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "../synthetic_emails_unificados.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts_email_ml_v4"
VERSIONS_DIR = ARTIFACTS_DIR / "versions"
ARTIFACTS_DIR.mkdir(exist_ok=True)
VERSIONS_DIR.mkdir(exist_ok=True)

TARGET = "prioridad_label"
RANDOM_STATE = 42
TEST_SIZE = 0.15
N_CV_FOLDS = 5          # StratifiedKFold
OPTUNA_TRIALS = 30      # aumentar para mejores resultados (costo: tiempo)
OPTUNA_TIMEOUT = 300    # segundos máximos por experimento

REQUIRED_COLUMNS = ["subject", "body", "sender_type", "prioridad_label"]

# ============================================================
# UTILIDADES
# ============================================================

def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()

def version_tag() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

def dataset_md5(df: pd.DataFrame) -> str:
    """Huella del dataset para detectar cambios entre versiones."""
    return hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()

def normalize_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    replacements = [
        (r"\[fecha\]",               " TOKEN_FECHA "),
        (r"\[nombre del curso\]",    " TOKEN_CURSO "),
        (r"\[nombre del trabajo\]",  " TOKEN_TRABAJO "),
        (r"\[número de matrícula\]", " TOKEN_MATRICULA "),
        (r"\[motivo\]",              " TOKEN_MOTIVO "),
        (r"\[[^\]]+\]",              " TOKEN_PLACEHOLDER "),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()

def count_any(text: str, patterns) -> int:
    return sum(
        len(re.findall(p, text, flags=re.IGNORECASE)) for p in patterns
    )

def save_json(data: Dict[str, Any], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ============================================================
# TRANSFORMERS
# ============================================================

class TextListExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            s = X.iloc[:, 0] if X.shape[1] == 1 else X.squeeze()
        elif isinstance(X, pd.Series):
            s = X
        else:
            s = pd.Series(X)
        return s.astype(str).fillna("").tolist()


class PrioritySignalExtractorV4(BaseEstimator, TransformerMixin):
    """
    Extractor de señales semánticas de prioridad.
    Igual a V3 pero con dos mejoras:
      - Nuevas features de longitud relativa (body_to_subject_ratio)
      - Normalización defensiva de NaN antes de retornar
    """

    _URGENCY   = [r"\burgente\b", r"\bprioridad\b", r"\binmediato\b",
                  r"\bcuanto antes\b", r"\bhoy\b", r"\bahora\b",
                  r"\blo antes posible\b", r"\bpronto\b",
                  r"\bde inmediato\b", r"\besta semana\b", r"\bmañana\b"]
    _PAYMENT   = [r"\bdeuda\b", r"\bpago\b", r"\bvencid", r"\bmora\b",
                  r"\bregularizar\b", r"\bsuspensi", r"\bbloquead",
                  r"\bfinanzas\b"]
    _REQUEST   = [r"\bsolicito\b", r"\bagradeceria\b", r"\bnecesito\b",
                  r"\bpor favor\b", r"\bconsult", r"\brevisi",
                  r"\bprorroga\b", r"\bseguimiento\b"]
    _NEGATIVE  = [r"\bproblema\b", r"\berror\b", r"\bno puedo\b",
                  r"\binconveniente\b", r"\bmolest", r"\bfrustr",
                  r"\bdesesper", r"\bansios"]
    _DEADLINE  = [r"\bfecha limite\b", r"\bvence\b", r"\bplazo\b",
                  r"\bhoy\b", r"\besta semana\b", r"\bmañana\b",
                  r"\besta noche\b"]
    _ABANDONO  = [r"\bretir", r"\bcongel", r"\babandon",
                  r"\bno se como seguir\b", r"\bdesmotivad",
                  r"\bsuperad", r"\bno puedo continuar\b",
                  r"\bme cuesta continuar\b", r"\bseguir adelante\b"]
    _APOYO     = [r"\bapoyo\b", r"\borientacion\b", r"\bacompañamiento\b",
                  r"\bayuda\b", r"\bfraccionamiento\b", r"\bbeca\b",
                  r"\bsoporte\b"]
    _E_ECON    = [r"\bproblemas econom", r"\bno puedo pagar\b",
                  r"\bdificultades econom", r"\bfraccionamiento\b",
                  r"\bapoyo economico\b"]
    _E_ACAD    = [r"\bcarga academica\b", r"\bentregas\b",
                  r"\bevaluaciones\b", r"\bno logro\b",
                  r"\bme cuesta\b", r"\britmo de clases\b"]

    def fit(self, X, y=None):
        return self

    @staticmethod
    def _uppercase_ratio(s: str) -> float:
        letters = [c for c in s if c.isalpha()]
        if not letters:
            return 0.0
        return sum(1 for c in letters if c.isupper()) / len(letters)

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            df.columns = ["subject", "body"]
        else:
            df = pd.DataFrame(X, columns=["subject", "body"])

        subject = df["subject"].fillna("").astype(str)
        body    = df["body"].fillna("").astype(str)
        text    = (subject + " " + body).str.strip()

        ls = subject.str.lower()
        lb = body.str.lower()
        lt = text.str.lower()

        urg  = lt.apply(lambda s: count_any(s, self._URGENCY))
        pay  = lt.apply(lambda s: count_any(s, self._PAYMENT))
        req  = lt.apply(lambda s: count_any(s, self._REQUEST))
        neg  = lt.apply(lambda s: count_any(s, self._NEGATIVE))
        dl   = lt.apply(lambda s: count_any(s, self._DEADLINE))
        abn  = lt.apply(lambda s: count_any(s, self._ABANDONO))
        apo  = lt.apply(lambda s: count_any(s, self._APOYO))
        eecon= lt.apply(lambda s: count_any(s, self._E_ECON))
        eacad= lt.apply(lambda s: count_any(s, self._E_ACAD))

        subj_len = subject.str.len()
        body_len = body.str.len()

        feat = pd.DataFrame({
            "subject_len":   subj_len,
            "body_len":      body_len,
            "text_len":      text.str.len(),
            # NUEVO: ratio cuerpo/asunto (emails con cuerpo largo y asunto
            # corto suelen ser soporte/consulta, no emergencia)
            "body_subj_ratio": (body_len / (subj_len + 1)).clip(upper=50),
            "subject_words": subject.str.split().str.len(),
            "body_words":    body.str.split().str.len(),
            "exclamation_count": text.str.count(r"!"),
            "question_count":    text.str.count(r"\?"),
            "digit_count":       text.str.count(r"\d"),
            "caps_ratio":        text.apply(self._uppercase_ratio),

            "subject_has_urgent":    ls.str.contains(r"\burgente\b").astype(int),
            "subject_has_pago":      ls.str.contains(r"\bpago\b").astype(int),
            "subject_has_revision":  ls.str.contains(r"\brevisi").astype(int),
            "subject_has_seguimiento": ls.str.contains(r"\bseguimiento\b").astype(int),
            "subject_has_retiro":    ls.str.contains(r"\bretir|\bcongel").astype(int),

            "body_has_bloqueo":  lb.str.contains(r"\bbloquead|\bsuspend").astype(int),
            "body_has_hoy":      lb.str.contains(r"\bhoy\b").astype(int),
            "body_has_deuda":    lb.str.contains(r"\bdeuda\b|\bmora\b").astype(int),
            "body_has_plazo":    lb.str.contains(r"\bplazo\b|\bvence\b|\bfecha limite\b").astype(int),

            "urgency_hits":         urg,
            "payment_hits":         pay,
            "request_hits":         req,
            "negative_hits":        neg,
            "deadline_hits":        dl,
            "abandono_hits":        abn,
            "apoyo_hits":           apo,
            "estres_economico_hits": eecon,
            "estres_academico_hits": eacad,
        })

        # Features compuestas (igual a V3)
        feat["abandono_sin_urgencia"] = (
            (abn >= 1).astype(int) * (urg == 0).astype(int)
            * (pay <= 1).astype(int) * (dl == 0).astype(int)
        )
        feat["abandono_con_apoyo"] = (
            (abn >= 1).astype(int) * (apo >= 1).astype(int)
        )
        feat["abandono_economico_no_critico"] = (
            (abn >= 1).astype(int) * (eecon >= 1).astype(int)
            * feat["body_has_plazo"].eq(0).astype(int)
            * feat["body_has_bloqueo"].eq(0).astype(int)
        )
        feat["urgencia_operativa_fuerte"] = (
            (urg >= 2).astype(int)
            + feat["body_has_bloqueo"]
            + feat["body_has_plazo"]
            + (pay >= 2).astype(int)
        )
        feat["abandono_vs_urgencia_balance"] = (
            abn + apo + eacad - urg - dl
        )
        feat["prioridad_media_hint"] = (
            feat["abandono_sin_urgencia"]
            + feat["abandono_con_apoyo"]
            + feat["abandono_economico_no_critico"]
        )
        feat["prioridad_alta_hint"] = (
            (feat["urgencia_operativa_fuerte"] >= 2).astype(int)
            + feat["body_has_bloqueo"]
            + feat["body_has_plazo"]
            + feat["subject_has_urgent"]
        )

        return feat.fillna(0)   # defensa contra NaN residuales


# ============================================================
# DATASET
# ============================================================

def load_dataset(csv_file: Path) -> pd.DataFrame:
    if not csv_file.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {csv_file}")

    df = pd.read_csv(csv_file)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas: {missing}")

    df = df.dropna(subset=REQUIRED_COLUMNS).copy()
    df["subject"]     = df["subject"].apply(normalize_text)
    df["body"]        = df["body"].apply(normalize_text)
    df["sender_type"] = df["sender_type"].astype(str).str.strip().str.lower()
    df["text"] = (
        "SUBJECT: " + df["subject"] + " BODY: " + df["body"]
    ).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["text"].str.len() > 10].copy()

    if len(df) < 100:
        raise ValueError("Muy pocas filas para entrenar.")

    log.info("Dataset cargado: %d filas | MD5: %s", len(df), dataset_md5(df))
    return df


# ============================================================
# PIPELINE BASE
# ============================================================

def text_features(max_word=80_000, max_char=50_000):
    return Pipeline([
        ("extract", TextListExtractor()),
        ("union", FeatureUnion([
            ("word_tfidf", TfidfVectorizer(
                lowercase=True, strip_accents="unicode",
                ngram_range=(1, 2), min_df=2, max_df=0.95,
                sublinear_tf=True, max_features=max_word,
            )),
            ("char_tfidf", TfidfVectorizer(
                lowercase=True, strip_accents="unicode",
                analyzer="char_wb", ngram_range=(3, 5),
                min_df=2, sublinear_tf=True, max_features=max_char,
            )),
        ])),
    ])


def build_preprocessor(max_word=80_000, max_char=50_000, k_best=None):
    """
    k_best: si se pasa un entero, agrega SelectKBest(chi2, k=k_best)
    para reducir la dimensionalidad del espacio TF-IDF + señales.
    """
    steps_text = [("text", text_features(max_word, max_char), "text")]
    ct = ColumnTransformer(
        transformers=steps_text + [
            ("sender", OneHotEncoder(handle_unknown="ignore"), ["sender_type"]),
            ("signals", Pipeline([
                ("extract", PrioritySignalExtractorV4()),
                ("scale",   MaxAbsScaler()),
            ]), ["subject", "body"]),
        ],
        remainder="drop",
    )
    if k_best is not None:
        return Pipeline([
            ("ct", ct),
            ("select", SelectKBest(f_classif, k=k_best)),
        ])
    return ct


def wrap_calibrated(estimator, cv=3):
    """Envuelve cualquier clasificador con calibración sigmoid."""
    return CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=cv)


# ============================================================
# OPTUNA: BÚSQUEDA DE HIPERPARÁMETROS
# ============================================================

def make_objective(X_train, y_train, cv: StratifiedKFold):
    """
    Función objetivo para Optuna. Prueba LogisticRegression y LinearSVC
    con distintos hiperparámetros, evaluados con CV estratificada.
    Retorna f1_macro promedio (a maximizar).
    """
    def objective(trial: optuna.Trial) -> float:
        model_type = trial.suggest_categorical("model", ["logreg", "svc"])
        k_best = trial.suggest_categorical("k_best", [None, 50_000, 80_000])

        preprocessor = build_preprocessor(k_best=k_best)

        if model_type == "logreg":
            C = trial.suggest_float("C", 0.1, 5.0, log=True)
            clf = wrap_calibrated(
                LogisticRegression(
                    max_iter=5000, class_weight="balanced",
                    solver="saga", C=C, random_state=RANDOM_STATE,
                )
            )
        else:
            C = trial.suggest_float("C_svc", 0.1, 5.0, log=True)
            clf = wrap_calibrated(
                LinearSVC(
                    C=C, class_weight="balanced",
                    random_state=RANDOM_STATE,
                )
            )

        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier",   clf),
        ])

        scores = cross_validate(
            pipe, X_train, y_train,
            cv=cv,
            scoring={"f1_macro": "f1_macro", "f1_weighted": "f1_weighted"},
            n_jobs=-1,
            error_score="raise",
        )
        return float(np.mean(scores["test_f1_macro"]))

    return objective


# ============================================================
# EVALUACIÓN
# ============================================================

def evaluate(model, X, y) -> Dict[str, Any]:
    preds = model.predict(X)
    result = {
        "accuracy":               float(accuracy_score(y, preds)),
        "f1_macro":               float(f1_score(y, preds, average="macro")),
        "f1_weighted":            float(f1_score(y, preds, average="weighted")),
        "classification_report":  classification_report(y, preds, output_dict=True),
        "confusion_matrix":       confusion_matrix(y, preds).tolist(),
    }
    try:
        probs = model.predict_proba(X)
        result["avg_max_probability"] = float(probs.max(axis=1).mean())
        # Distribución de predicciones (útil para detectar drift)
        unique, counts = np.unique(preds, return_counts=True)
        result["prediction_distribution"] = dict(zip(unique.tolist(), counts.tolist()))
    except Exception:
        result["avg_max_probability"] = None
        result["prediction_distribution"] = {}
    return result


# ============================================================
# SHAP (OPCIONAL — requiere pip install shap)
# ============================================================

def try_shap_summary(model, X_sample, labels, out_dir: Path):
    """
    Genera y guarda un resumen SHAP si el paquete está disponible.
    Solo funciona con modelos lineales (LogisticRegression, LinearSVC).
    """
    try:
        import shap
        log.info("Generando SHAP values (muestra de %d filas)...", len(X_sample))
        # Transformar los datos al espacio de features del pipeline
        preprocessor = model.named_steps["preprocessor"]
        clf = model.named_steps["classifier"]

        X_transformed = preprocessor.transform(X_sample)

        # shap.LinearExplainer funciona con LogisticRegression calibrada
        base_clf = getattr(clf, "estimator", clf)
        explainer = shap.LinearExplainer(
            base_clf, X_transformed, feature_perturbation="interventional"
        )
        shap_values = explainer.shap_values(X_transformed)

        # Guardar como JSON (top-20 features por clase)
        summary = {}
        if isinstance(shap_values, list):
            for i, class_vals in enumerate(shap_values):
                mean_abs = np.abs(class_vals).mean(axis=0)
                top_idx  = np.argsort(mean_abs)[::-1][:20]
                summary[labels[i]] = {
                    f"feature_{j}": float(mean_abs[j]) for j in top_idx
                }
        save_json(summary, out_dir / "shap_summary.json")
        log.info("SHAP summary guardado en %s", out_dir / "shap_summary.json")
    except ImportError:
        log.warning("shap no instalado — omitiendo análisis de explicabilidad.")
    except Exception as e:
        log.warning("SHAP falló: %s", e)


# ============================================================
# MAIN
# ============================================================

def main():
    log.info("CSV_FILE: %s | Existe: %s", CSV_FILE, CSV_FILE.exists())

    df = load_dataset(CSV_FILE)
    X  = df[["subject", "body", "text", "sender_type"]].copy()
    y  = df[TARGET].astype(str)

    # ── Split definitivo (test queda completamente apartado) ──────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    log.info("Train: %d | Test: %d", len(X_train), len(X_test))

    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # ── Búsqueda de hiperparámetros con Optuna ───────────────────────────
    log.info("Iniciando Optuna (%d trials, timeout=%ds)...", OPTUNA_TRIALS, OPTUNA_TIMEOUT)
    study = optuna.create_study(
        direction="maximize",
        study_name="email_priority_v4",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        make_objective(X_train, y_train, cv),
        n_trials=OPTUNA_TRIALS,
        timeout=OPTUNA_TIMEOUT,
        show_progress_bar=False,
    )

    best_params = study.best_params
    best_cv_score = study.best_value
    log.info("Mejor trial — f1_macro CV=%.4f | params=%s", best_cv_score, best_params)

    # ── Reconstruir el mejor pipeline con los hiperparámetros óptimos ────
    k_best = best_params.get("k_best", None)
    preprocessor = build_preprocessor(k_best=k_best)

    if best_params["model"] == "logreg":
        base_clf = LogisticRegression(
            max_iter=12000, class_weight="balanced",
            solver="saga", C=best_params["C"],
            random_state=RANDOM_STATE,
        )
    else:
        base_clf = LinearSVC(
            C=best_params["C_svc"], class_weight="balanced",
            random_state=RANDOM_STATE,
        )

    final_model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   wrap_calibrated(base_clf)),
    ])

    # ── Entrenamiento final sobre TODO el train ──────────────────────────
    log.info("Entrenando modelo final sobre %d muestras...", len(X_train))
    final_model.fit(X_train, y_train)

    # ── Evaluación en test ───────────────────────────────────────────────
    test_metrics = evaluate(final_model, X_test, y_test)
    log.info(
        "TEST — accuracy=%.4f | f1_macro=%.4f | f1_weighted=%.4f | avg_max_prob=%s",
        test_metrics["accuracy"],
        test_metrics["f1_macro"],
        test_metrics["f1_weighted"],
        test_metrics.get("avg_max_probability"),
    )
    log.info("Distribución predicciones test: %s", test_metrics["prediction_distribution"])

    # ── SHAP (muestra de hasta 500 filas del train) ──────────────────────
    sample_idx = X_train.sample(min(500, len(X_train)), random_state=RANDOM_STATE).index
    try_shap_summary(
        final_model,
        X_train.loc[sample_idx],
        sorted(y.unique().tolist()),
        ARTIFACTS_DIR,
    )

    # ── Serialización ────────────────────────────────────────────────────
    tag = version_tag()
    versioned_path = VERSIONS_DIR / f"prioridad_label_v4_{best_params['model']}_{tag}.pkl"
    latest_path    = ARTIFACTS_DIR / "prioridad_label_best_model_v4_latest.pkl"
    summary_path   = ARTIFACTS_DIR / "prioridad_label_v4_summary.json"

    artifact = {
        "target":          TARGET,
        "version":         "v4",
        "best_params":     best_params,
        "best_cv_f1_macro": best_cv_score,
        "pipeline":        final_model,
        "labels":          sorted(y.unique().tolist()),
        "created_at":      utc_now_iso(),
        "dataset_md5":     dataset_md5(df),
        "dataset_rows":    int(len(df)),
        "input_columns":   ["subject", "body", "sender_type"],
        "notes":           "V4: Optuna + StratifiedKFold + calibración universal + SHAP",
        "test_metrics":    test_metrics,
    }

    with open(versioned_path, "wb") as f:
        cloudpickle.dump(artifact, f)
    with open(latest_path, "wb") as f:
        cloudpickle.dump(artifact, f)

    save_json(
        {k: v for k, v in artifact.items() if k != "pipeline"},
        summary_path,
    )

    log.info("Artefactos guardados.")
    log.info("  Versionado: %s", versioned_path)
    log.info("  Latest:     %s", latest_path)
    log.info("  Resumen:    %s", summary_path)


if __name__ == "__main__":
    main()