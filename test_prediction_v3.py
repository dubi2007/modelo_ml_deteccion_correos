import cloudpickle
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts_email_ml_compare"

MODEL_FILE = ARTIFACTS_DIR / "prioridad_label_best_model_v3_latest.pkl"


def build_input(subject: str, body: str, sender_type: str = "estudiante") -> pd.DataFrame:
    text = f"SUBJECT: {subject.strip()} BODY: {body.strip()}"
    return pd.DataFrame([
        {
            "subject": subject.strip(),
            "body": body.strip(),
            "text": text,
            "sender_type": sender_type.strip().lower()
        }
    ])


def load_artifact(path: Path):
    with open(path, "rb") as f:
        return cloudpickle.load(f)


def predict_prioridad(subject: str, body: str, sender_type: str = "estudiante"):
    sample = build_input(subject, body, sender_type)

    artifact = load_artifact(MODEL_FILE)
    pipeline = artifact["pipeline"]

    pred = pipeline.predict(sample)[0]

    result = {
        "prediction": pred
    }

    try:
        probs = pipeline.predict_proba(sample)[0]
        classes = pipeline.classes_

        top_probs = sorted(
            zip(classes, probs),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        result["top_3_probabilities"] = [
            {"label": label, "probability": round(float(prob), 4)}
            for label, prob in top_probs
        ]
    except Exception:
        result["top_3_probabilities"] = "No disponible"

    return result


def main():
    subject = "Dificultades económicas y académicas para continuar este ciclo"
    body = (
        "Buenas noches, escribo este correo porque en las últimas semanas me he sentido "
        "bastante superado por la situación económica y académica que estoy atravesando. "
        "Actualmente estoy teniendo problemas para cumplir con algunos pagos y me está costando "
        "mantener el ritmo de las clases. No quisiera tomar una decisión apresurada, pero estoy "
        "considerando retirarme o congelar temporalmente mis estudios si no encuentro alguna "
        "alternativa. Quisiera saber si existe orientación, apoyo económico o acompañamiento."
    )
    sender_type = "estudiante"

    result = predict_prioridad(subject, body, sender_type)

    print("\n=== RESULTADO PRIORIDAD V3 AUTONOMO ===\n")
    print("Subject:", subject)
    print("Sender type:", sender_type)
    print("prioridad_label:", result["prediction"])
    print("Top 3:", result["top_3_probabilities"])


if __name__ == "__main__":
    main()