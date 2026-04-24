import re
import joblib
import pandas as pd
from pathlib import Path

# ============================================================
# FUNCIONES NECESARIAS PARA CARGAR LOS JOBLIB
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


def count_any(text: str, patterns) -> int:
    return sum(len(re.findall(p, text, flags=re.IGNORECASE)) for p in patterns)


def extract_priority_signal_df(x) -> pd.DataFrame:
    """
    Esta funcion debe existir porque el modelo V2 de prioridad
    la usa dentro del pipeline guardado.
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


# ============================================================
# RUTAS DE MODELOS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts_email_ml_compare"

MODEL_FILES = {
    "tipo_base": ARTIFACTS_DIR / "tipo_base_best_model.joblib",
    "prioridad_label": ARTIFACTS_DIR / "prioridad_label_best_model_v2_latest.joblib",
    "riesgo_label": ARTIFACTS_DIR / "riesgo_label_best_model.joblib",
    "tono_label": ARTIFACTS_DIR / "tono_label_best_model.joblib",
}


# ============================================================
# PREDICCION
# ============================================================

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


def predict_one(subject: str, body: str, sender_type: str = "estudiante"):
    sample = build_input(subject, body, sender_type)
    results = {}

    for target, model_path in MODEL_FILES.items():
        if not model_path.exists():
            print(f"No existe el modelo: {model_path}")
            continue

        artifact = joblib.load(model_path)
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
            result["top_3_probabilities"] = "No disponible para este modelo"

        results[target] = result

    return results


def main():
    test_cases = [
        {
            "name": "Caso 1 - Pago vencido urgente",
            "subject": "URGENTE pago vencido de matrícula y bloqueo de acceso",
            "body": (
                "Buenas tardes, me comunico porque al intentar ingresar hoy al sistema académico "
                "aparece un mensaje indicando que mi cuenta ha sido bloqueada por deuda pendiente. "
                "Tengo evaluaciones programadas esta semana y además debo descargar material del curso, "
                "por lo que esta situación me está afectando directamente. Hace unos días revisé mi estado "
                "de cuenta y vi un saldo vencido que no había podido regularizar por un inconveniente familiar, "
                "pero ahora necesito resolverlo con carácter inmediato. Les agradecería que me indiquen el monto "
                "exacto, los medios de pago habilitados y si existe alguna posibilidad de rehabilitar el acceso "
                "el mismo día una vez realizado el abono. Quedo atento a su pronta respuesta porque realmente "
                "necesito solucionar este problema hoy."
            ),
            "sender_type": "estudiante",
        },
        {
            "name": "Caso 2 - Consulta general de horarios",
            "subject": "Consulta detallada sobre horarios del próximo ciclo académico",
            "body": (
                "Buenos días, quisiera solicitar información sobre los horarios disponibles para el próximo ciclo "
                "académico, ya que me encuentro organizando mis actividades personales y laborales para poder "
                "llevar adecuadamente la carga de cursos. Me interesa especialmente conocer si ya se tiene publicado "
                "el cronograma tentativo de clases, si habrá turnos en la mañana y en la noche, y si existen cambios "
                "previstos respecto al semestre anterior. También quisiera saber si los cursos obligatorios de mi plan "
                "de estudios tendrán secciones adicionales y si es posible visualizar con anticipación los docentes "
                "asignados o al menos la distribución por sede. Les agradecería mucho que me orienten sobre dónde "
                "puedo consultar esta información o, en todo caso, si podrían compartirme el procedimiento correcto "
                "para acceder a esos datos. Muchas gracias por su atención."
            ),
            "sender_type": "externo",
        },
        {
            "name": "Caso 3 - Reclamo de nota",
            "subject": "Solicitud formal de revisión de nota del examen parcial",
            "body": (
                "Estimados, me dirijo a ustedes para solicitar de manera formal la revisión de la nota que figura "
                "registrada en el sistema respecto al examen parcial del curso. He revisado cuidadosamente los criterios "
                "de evaluación explicados en clase y, según mis cálculos y las observaciones que hice al momento de rendir "
                "la prueba, considero que la calificación publicada no refleja adecuadamente mi desempeño. Entiendo que "
                "pueden existir errores involuntarios al registrar o consolidar notas, por lo que me gustaría pedir "
                "respetuosamente una revisión del examen o una explicación más detallada sobre la forma en que fue evaluado. "
                "No busco generar inconvenientes, sino simplemente asegurarme de que el resultado final sea justo y correcto. "
                "Agradecería mucho su apoyo y quedo atento a cualquier indicación adicional que deba seguir para formalizar "
                "este pedido dentro del plazo correspondiente."
            ),
            "sender_type": "estudiante",
        },
        {
            "name": "Caso 4 - Riesgo de abandono",
            "subject": "Dificultades económicas y académicas para continuar este ciclo",
            "body": (
                "Buenas noches, escribo este correo porque en las últimas semanas me he sentido bastante superado por "
                "la situación económica y académica que estoy atravesando. Actualmente estoy teniendo problemas para "
                "cumplir con algunos pagos y, además, me está costando mantener el ritmo de las clases y las entregas. "
                "He intentado organizarme mejor, pero siento que cada vez se me hace más difícil continuar con normalidad "
                "y eso me está desmotivando bastante. No quisiera tomar una decisión apresurada, pero honestamente estoy "
                "considerando retirarme o congelar temporalmente mis estudios si no encuentro alguna alternativa. Por eso "
                "quisiera saber si existe algún tipo de orientación, apoyo económico, fraccionamiento o acompañamiento "
                "académico para estudiantes que están pasando por una situación complicada. Agradecería mucho cualquier "
                "información que puedan brindarme, porque en este momento realmente no sé cómo seguir adelante."
            ),
            "sender_type": "estudiante",
        },
        {
            "name": "Caso 5 - Seguimiento de trámite",
            "subject": "Seguimiento a solicitud de certificado presentada hace varios días",
            "body": (
                "Buenos días, escribo nuevamente para hacer seguimiento a la solicitud de certificado que presenté hace "
                "varios días mediante los canales indicados por la institución. Hasta la fecha no he recibido confirmación "
                "sobre el estado del trámite ni una fecha estimada de entrega, a pesar de que el documento me está siendo "
                "requerido con cierta urgencia para completar un proceso externo. Entiendo que puede haber carga administrativa "
                "o demoras operativas, pero agradecería mucho si pudieran indicarme en qué etapa se encuentra mi solicitud, "
                "si falta algún requisito adicional y cuánto tiempo más podría tomar la emisión del documento. En caso de que "
                "deba acercarme personalmente o reenviar algún comprobante, agradecería que me lo hagan saber para no seguir "
                "retrasando el trámite. Quedo atento a su respuesta y agradezco de antemano la ayuda brindada."
            ),
            "sender_type": "externo",
        },
    ]

    print("\n=== RESULTADOS DE PREDICCION POR LOTES ===\n")

    for i, case in enumerate(test_cases, start=1):
        results = predict_one(case["subject"], case["body"], case["sender_type"])

        print("=" * 90)
        print(f"{i}. {case['name']}")
        print("=" * 90)
        print("Subject:", case["subject"])
        print("Sender type:", case["sender_type"])
        print("Body:", case["body"])
        print()

        for target, info in results.items():
            print(f"{target}: {info['prediction']}")
            print(f"Top 3: {info['top_3_probabilities']}")
            print("-" * 60)

        print()
    subject = "URGENTE pago vencido de matrícula"
    body = (
        "Buenas tardes, intento ingresar al sistema y aparece bloqueado por deuda. "
        "Necesito regularizar el pago hoy porque tengo evaluaciones esta semana. "
        "Agradecería que me indiquen el procedimiento lo antes posible."
    )
    sender_type = "estudiante"

    results = predict_one(subject, body, sender_type)

    print("\n=== RESULTADOS DE PREDICCION ===\n")
    print("Subject:", subject)
    print("Sender type:", sender_type)
    print()

    for target, info in results.items():
        print(f"{target}: {info['prediction']}")
        print(f"Top 3: {info['top_3_probabilities']}")
        print("-" * 60)


if __name__ == "__main__":
    main()