from pathlib import Path
import os

import cv2
import numpy as np
import streamlit as st
from groq import Groq
from PIL import Image, ImageOps
from ultralytics import YOLO

from agent_parking import (
    analyser_resultat_yolo,
    generer_diagnostic,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Assistant de Parking IA",
    page_icon="🚗",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent

YOLO_WEIGHTS = (
    BASE_DIR
    / "yolov8s_50.pt"
)


st.title(
    "🚗 Tableau de Bord : Assistant de Parking"
)

st.caption(
    "Analyse de scènes avec YOLOv8 Small + "
    "moteur de risque déterministe + Agent LLM"
)

st.divider()


# ============================================================
# 2. SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Configuration"
)

api_key_env = os.getenv(
    "GROQ_API_KEY",
    "",
).strip()


if api_key_env:

    api_key = api_key_env

    st.sidebar.success(
        "Clé Groq chargée depuis GROQ_API_KEY"
    )

else:

    api_key = st.sidebar.text_input(
        "Clé API Groq",

        type="password",

        help=(
            "Sans clé, le diagnostic local "
            "continue de fonctionner."
        ),
    ).strip()


st.sidebar.markdown("---")

st.sidebar.subheader(
    "🎯 Réglages YOLO"
)


seuil_confiance = st.sidebar.slider(
    "Seuil de confiance",

    min_value=0.10,

    max_value=0.70,

    value=0.22,

    step=0.01,
)


taille_inference = st.sidebar.select_slider(
    "Résolution d'inférence",

    options=[
        640,
        768,
        960,
        1280,
    ],

    value=960,

    help=(
        "960 améliore généralement la "
        "détection des petits objets."
    ),
)


utiliser_tta = st.sidebar.checkbox(
    "Test-Time Augmentation",

    value=False,

    help=(
        "Peut améliorer certaines "
        "détections mais ralentit "
        "le traitement."
    ),
)


st.sidebar.caption(
    "Les distances sont approximatives. "
    "Une caméra RGB non calibrée ne donne "
    "pas une profondeur métrique exacte."
)


# ============================================================
# 3. YOLO
# ============================================================

@st.cache_resource
def charger_yolo():

    if not YOLO_WEIGHTS.exists():

        raise FileNotFoundError(
            "Le fichier yolov8s_50.pt "
            "est introuvable dans le dossier "
            "du projet."
        )

    return YOLO(
        str(YOLO_WEIGHTS)
    )


try:

    modele_yolo = charger_yolo()

except Exception as exc:

    st.error(
        f"Impossible de charger YOLO : {exc}"
    )

    st.stop()


# ============================================================
# 4. GROQ
# ============================================================

client_groq = None


if api_key:

    try:

        client_groq = Groq(
            api_key=api_key
        )

    except Exception as exc:

        st.sidebar.error(
            f"Erreur Groq : {exc}"
        )

else:

    st.sidebar.info(
        "Mode local actif : le moteur "
        "de sécurité fonctionne sans LLM."
    )


# ============================================================
# 5. COULEURS DES BOXES
# ============================================================

def couleur_detection(
    categorie,
    distance,
):

    if categorie == "pieton":

        if (
            distance is not None
            and distance < 2.0
        ):

            return (
                255,
                0,
                0,
            )

        return (
            255,
            80,
            80,
        )


    if categorie in {
        "voiture",
        "camion",
        "moto",
        "velo",
    }:

        if (
            distance is not None
            and distance < 1.2
        ):

            return (
                255,
                0,
                0,
            )

        return (
            255,
            120,
            0,
        )


    return (
        0,
        180,
        255,
    )


# ============================================================
# 6. DESSIN DES DÉTECTIONS
# ============================================================

def dessiner_detections(
    image_rgb,
    scene,
):

    image_annotee = image_rgb.copy()

    hauteur, largeur = (
        image_annotee.shape[:2]
    )

    epaisseur = max(
        2,
        round(
            min(
                largeur,
                hauteur,
            )
            / 350
        ),
    )

    taille_texte = max(
        0.45,
        min(
            0.8,
            min(
                largeur,
                hauteur,
            )
            / 900,
        ),
    )


    for detection in scene.get(
        "detections",
        [],
    ):

        x1, y1, x2, y2 = [
            int(round(v))
            for v
            in detection["bbox"]
        ]


        x1 = max(
            0,
            min(
                x1,
                largeur - 1,
            ),
        )

        y1 = max(
            0,
            min(
                y1,
                hauteur - 1,
            ),
        )

        x2 = max(
            0,
            min(
                x2,
                largeur - 1,
            ),
        )

        y2 = max(
            0,
            min(
                y2,
                hauteur - 1,
            ),
        )


        distance = detection[
            "distance_estimee_m"
        ]

        categorie = detection[
            "categorie"
        ]

        confiance = detection[
            "confiance"
        ]


        couleur = couleur_detection(
            categorie,
            distance,
        )


        cv2.rectangle(
            image_annotee,

            (
                x1,
                y1,
            ),

            (
                x2,
                y2,
            ),

            couleur,

            epaisseur,
        )


        if distance is not None:

            distance_texte = (
                f" | ~{distance:.1f} m"
            )

        else:

            distance_texte = ""


        etiquette = (
            f"{detection['classe']} "
            f"| {confiance:.2f}"
            f"{distance_texte}"
        )


        (
            largeur_texte,
            hauteur_texte,
        ), baseline = cv2.getTextSize(
            etiquette,

            cv2.FONT_HERSHEY_SIMPLEX,

            taille_texte,

            max(
                1,
                epaisseur - 1,
            ),
        )


        y_texte = max(
            hauteur_texte + 8,
            y1,
        )


        cv2.rectangle(
            image_annotee,

            (
                x1,
                y_texte
                - hauteur_texte
                - 8,
            ),

            (
                min(
                    largeur - 1,
                    x1
                    + largeur_texte
                    + 8,
                ),

                y_texte
                + baseline,
            ),

            couleur,

            thickness=-1,
        )


        cv2.putText(
            image_annotee,

            etiquette,

            (
                x1 + 4,
                y_texte - 4,
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            taille_texte,

            (
                255,
                255,
                255,
            ),

            max(
                1,
                epaisseur - 1,
            ),

            cv2.LINE_AA,
        )


    return image_annotee


# ============================================================
# 7. AFFICHAGE RISQUE
# ============================================================

def afficher_risque(
    risque,
):

    niveau = risque[
        "niveau"
    ]

    alerte = risque[
        "alerte"
    ]


    if niveau == "Critique":

        st.error(
            f"🛑 {alerte}"
        )


    elif niveau == "Élevé":

        st.warning(
            f"⚠️ {alerte}"
        )


    elif niveau == "Moyen":

        st.warning(
            f"🟠 {alerte}"
        )


    elif niveau == "Faible":

        st.success(
            f"✅ {alerte}"
        )


    else:

        st.info(
            f"ℹ️ {alerte}"
        )


def texte_distance(
    valeur,
):

    if valeur is None:

        return "Non détecté"

    return (
        f"~{valeur:.1f} m"
    )


# ============================================================
# 8. UPLOAD IMAGE
# ============================================================

st.subheader(
    "📷 Acquisition Visuelle"
)


fichier = st.file_uploader(
    "Choisissez une image de route ou de parking",

    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
    ],
)


if fichier is None:

    st.info(
        "Chargez une image pour lancer l'analyse."
    )

    st.stop()


try:

    image_pil = Image.open(
        fichier
    )

    image_pil = ImageOps.exif_transpose(
        image_pil
    )

    image_pil = image_pil.convert(
        "RGB"
    )

    image_np = np.array(
        image_pil
    )


except Exception as exc:

    st.error(
        f"Image illisible : {exc}"
    )

    st.stop()


# ============================================================
# 9. INFÉRENCE YOLO
# ============================================================

with st.spinner(
    "Analyse visuelle YOLOv8 Small en cours..."
):

    try:

        resultats = modele_yolo.predict(
            source=image_np,

            imgsz=taille_inference,

            conf=seuil_confiance,

            iou=0.45,

            max_det=300,

            augment=utiliser_tta,

            verbose=False,
        )

    except Exception as exc:

        st.error(
            f"Erreur pendant l'inférence YOLO : {exc}"
        )

        st.stop()


if not resultats:

    st.error(
        "YOLO n'a retourné aucun résultat."
    )

    st.stop()


# ============================================================
# 10. ANALYSE STRUCTURÉE
# ============================================================

scene = analyser_resultat_yolo(
    resultats[0],
    image_np.shape,
)


image_annotee = dessiner_detections(
    image_np,
    scene,
)


# ============================================================
# 11. AGENT IA
# ============================================================

with st.spinner(
    "Calcul du risque et génération "
    "du diagnostic..."
):

    diagnostic = generer_diagnostic(
        client_groq,
        scene,
    )


risque = diagnostic[
    "risque"
]


# ============================================================
# 12. DASHBOARD
# ============================================================

col_vision, col_agent = st.columns(
    [
        1.55,
        1.0,
    ],

    gap="large",
)


with col_vision:

    st.subheader(
        "📷 Flux Vidéo (Vision Module)"
    )

    st.image(
        image_annotee,

        use_container_width=True,
    )


    st.caption(
        f"{scene['nombre_detections']} détection(s)"
        f" | Confiance moyenne : "
        f"{scene['confiance_moyenne']:.2f}"
        f" | Piéton : "
        f"{texte_distance(scene['distance_pieton_min_m'])}"
        f" | Obstacle : "
        f"{texte_distance(scene['distance_obstacle_min_m'])}"
    )


with col_agent:

    st.subheader(
        "🧠 Diagnostic Agent IA"
    )


    st.caption(
        "Niveau de Risque Détecté"
    )


    st.markdown(
        f"# {risque['niveau']}"
    )


    afficher_risque(
        risque
    )


    with st.expander(
        "📍 Analyse Spatiale Détaillée",
        expanded=True,
    ):

        st.markdown(
            diagnostic[
                "rapport"
            ]
        )


        if not diagnostic[
            "llm_ok"
        ]:

            if diagnostic[
                "erreur_llm"
            ]:

                st.caption(
                    "Le LLM est indisponible. "
                    "Le résultat affiché est calculé "
                    "par le moteur local déterministe."
                )


                with st.expander(
                    "Détail technique de l'erreur LLM"
                ):

                    st.code(
                        diagnostic[
                            "erreur_llm"
                        ]
                    )


            else:

                st.caption(
                    "Mode local : aucune clé "
                    "Groq active."
                )


    with st.expander(
        "🛡️ Recommandations de Conduite",
        expanded=True,
    ):

        st.write(
            risque[
                "recommandation"
            ]
        )


# ============================================================
# 13. TABLEAU DES DÉTECTIONS
# ============================================================

st.divider()

st.subheader(
    "🔎 Détails des objets détectés"
)


if scene[
    "detections"
]:

    lignes = []


    for numero, detection in enumerate(
        scene["detections"],
        start=1,
    ):

        distance = detection[
            "distance_estimee_m"
        ]


        if distance is None:

            distance_texte = "N/A"

        else:

            distance_texte = (
                f"~{distance:.1f} m"
            )


        lignes.append(
            {
                "#":
                    numero,

                "Classe YOLO":
                    detection["classe"],

                "Catégorie":
                    detection["categorie"],

                "Confiance":
                    detection["confiance"],

                "Distance estimée":
                    distance_texte,

                "Occupation image":
                    (
                        f"{detection['occupation_image_pct']:.2f}%"
                    ),
            }
        )


    st.dataframe(
        lignes,

        use_container_width=True,

        hide_index=True,
    )


else:

    st.warning(
        "Aucun objet n'a dépassé le seuil "
        "de confiance. Le résultat est donc "
        "'Indéterminé' et jamais automatiquement "
        "'Faible'."
    )


st.caption(
    "⚠️ Démonstrateur académique : les distances "
    "monoculaires sont approximatives. "
    "Un système automobile réel nécessite une "
    "calibration caméra ou des capteurs de profondeur."
)
