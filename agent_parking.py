import json
import math
import os
import unicodedata
from collections import Counter

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


# ============================================================
# 1. NORMALISATION DES CLASSES YOLO
# ============================================================

ALIASES_CLASSES = {
    "person": "pieton",
    "pedestrian": "pieton",
    "pieton": "pieton",
    "pietons": "pieton",

    "car": "voiture",
    "cars": "voiture",
    "voiture": "voiture",
    "voitures": "voiture",

    "truck": "camion",
    "trucks": "camion",
    "camion": "camion",
    "camions": "camion",

    "motorcycle": "moto",
    "motorbike": "moto",
    "moto": "moto",
    "motos": "moto",

    "bicycle": "velo",
    "bike": "velo",
    "velo": "velo",
    "velos": "velo",

    "traffic sign": "panneau",
    "traffic signs": "panneau",
    "panneau": "panneau",
    "panneaux": "panneau",
    "panneau de circulation": "panneau",
    "panneaux de circulation": "panneau",

    "crosswalk": "passage_pieton",
    "passage pieton": "passage_pieton",
    "passages pietons": "passage_pieton",

    "cycle lane": "piste_cyclable",
    "bicycle lane": "piste_cyclable",
    "piste cyclable": "piste_cyclable",
    "pistes cyclables": "piste_cyclable",

    "arrow": "fleche",
    "arrows": "fleche",
    "fleche": "fleche",
    "fleches": "fleche",

    "parking": "parking",
}


HAUTEURS_REELLES_M = {
    "pieton": 1.70,
    "voiture": 1.50,
    "camion": 3.00,
    "moto": 1.15,
    "velo": 1.10,
}


def texte_normalise(texte):
    texte = str(texte or "")

    texte = unicodedata.normalize("NFKD", texte)

    texte = "".join(
        caractere
        for caractere in texte
        if not unicodedata.combining(caractere)
    )

    texte = texte.lower()
    texte = texte.replace("_", " ")
    texte = texte.replace("-", " ")

    return " ".join(texte.split())


def normaliser_categorie(nom_classe):
    cle = texte_normalise(nom_classe)

    return ALIASES_CLASSES.get(
        cle,
        cle.replace(" ", "_"),
    )


# ============================================================
# 2. ESTIMATION APPROXIMATIVE DE DISTANCE
# ============================================================

def estimer_distance_bbox(
    xyxy,
    largeur_image,
    hauteur_image,
    categorie,
    champ_horizontal_deg=70.0,
):
    """
    Estimation monoculaire approximative.

    Le calcul utilise la taille apparente de la bounding box.
    Il ne remplace pas un capteur de profondeur ou une caméra
    correctement calibrée.
    """

    hauteur_reelle = HAUTEURS_REELLES_M.get(categorie)

    if hauteur_reelle is None:
        return None

    x1, y1, x2, y2 = [
        float(v)
        for v in xyxy
    ]

    hauteur_bbox = max(
        y2 - y1,
        1.0,
    )

    focal_px = float(largeur_image) / (
        2.0
        * math.tan(
            math.radians(champ_horizontal_deg) / 2.0
        )
    )

    distance = (
        hauteur_reelle
        * focal_px
        / hauteur_bbox
    )

    distance = max(
        0.3,
        min(distance, 80.0),
    )

    return round(distance, 1)


# ============================================================
# 3. TRANSFORMATION DES RÉSULTATS YOLO
# ============================================================

def analyser_resultat_yolo(
    resultat,
    image_shape,
):
    hauteur_image, largeur_image = image_shape[:2]

    detections = []

    boxes = getattr(
        resultat,
        "boxes",
        None,
    )

    if boxes is not None:

        for box in boxes:

            classe_id = int(
                box.cls[0].item()
            )

            confiance = float(
                box.conf[0].item()
            )

            nom_classe = str(
                resultat.names[classe_id]
            )

            categorie = normaliser_categorie(
                nom_classe
            )

            xyxy = (
                box.xyxy[0]
                .detach()
                .cpu()
                .tolist()
            )

            x1, y1, x2, y2 = [
                float(v)
                for v in xyxy
            ]

            distance = estimer_distance_bbox(
                xyxy=xyxy,
                largeur_image=largeur_image,
                hauteur_image=hauteur_image,
                categorie=categorie,
            )

            surface_bbox = max(
                0.0,
                (x2 - x1) * (y2 - y1),
            )

            surface_image = max(
                1.0,
                float(
                    largeur_image
                    * hauteur_image
                ),
            )

            occupation = round(
                100.0
                * surface_bbox
                / surface_image,
                2,
            )

            detections.append(
                {
                    "classe": nom_classe,
                    "categorie": categorie,
                    "confiance": round(
                        confiance,
                        3,
                    ),
                    "bbox": [
                        round(x1, 1),
                        round(y1, 1),
                        round(x2, 1),
                        round(y2, 1),
                    ],
                    "distance_estimee_m": distance,
                    "occupation_image_pct": occupation,
                }
            )

    compte = Counter(
        detection["categorie"]
        for detection in detections
    )

    def distance_min(categories):

        valeurs = [
            detection["distance_estimee_m"]
            for detection in detections
            if detection["categorie"] in categories
            and detection["distance_estimee_m"] is not None
        ]

        if not valeurs:
            return None

        return min(valeurs)

    if detections:

        confiance_moyenne = round(
            sum(
                detection["confiance"]
                for detection in detections
            )
            / len(detections),
            3,
        )

    else:

        confiance_moyenne = 0.0

    return {
        "largeur_image": int(
            largeur_image
        ),

        "hauteur_image": int(
            hauteur_image
        ),

        "nombre_detections": len(
            detections
        ),

        "confiance_moyenne": confiance_moyenne,

        "comptage": dict(
            compte
        ),

        "distance_pieton_min_m": distance_min(
            {"pieton"}
        ),

        "distance_obstacle_min_m": distance_min(
            {
                "voiture",
                "camion",
                "moto",
                "velo",
            }
        ),

        "detections": detections,
    }


# ============================================================
# 4. MOTEUR DE RISQUE DÉTERMINISTE
# ============================================================

def evaluer_risque_scene(scene):

    nombre_detections = int(
        scene.get(
            "nombre_detections",
            0,
        )
    )

    distance_pieton = scene.get(
        "distance_pieton_min_m"
    )

    distance_obstacle = scene.get(
        "distance_obstacle_min_m"
    )

    confiance = float(
        scene.get(
            "confiance_moyenne",
            0.0,
        )
    )

    if nombre_detections == 0:

        return {
            "niveau": "Indéterminé",

            "alerte":
                "Aucun objet exploitable détecté.",

            "raisons": [
                "La vision n'a produit aucune "
                "détection suffisamment fiable."
            ],

            "recommandation":
                "Ne pas conclure que la zone est libre. "
                "Vérifier visuellement la scène avant "
                "toute manœuvre.",
        }

    # --------------------------------------------------------
    # PRIORITÉ AUX PIÉTONS
    # --------------------------------------------------------

    if distance_pieton is not None:

        if distance_pieton < 2.0:

            return {
                "niveau": "Critique",

                "alerte":
                    "Danger immédiat détecté.",

                "raisons": [
                    f"Piéton estimé à environ "
                    f"{distance_pieton:.1f} m."
                ],

                "recommandation":
                    "Arrêt immédiat de la manœuvre "
                    "et contrôle visuel.",
            }

        if distance_pieton < 5.0:

            return {
                "niveau": "Élevé",

                "alerte":
                    "Usager vulnérable à proximité.",

                "raisons": [
                    f"Piéton proche estimé à environ "
                    f"{distance_pieton:.1f} m."
                ],

                "recommandation":
                    "Suspendre ou ralentir fortement "
                    "la manœuvre et laisser une marge "
                    "de sécurité au piéton.",
            }

    # --------------------------------------------------------
    # VÉHICULE / CAMION / MOTO / VÉLO
    # --------------------------------------------------------

    if distance_obstacle is not None:

        if distance_obstacle < 0.6:

            return {
                "niveau": "Critique",

                "alerte":
                    "Collision potentiellement imminente.",

                "raisons": [
                    f"Obstacle estimé à environ "
                    f"{distance_obstacle:.1f} m."
                ],

                "recommandation":
                    "Arrêt immédiat et contrôle "
                    "de l'espace disponible.",
            }

        if distance_obstacle < 1.2:

            return {
                "niveau": "Élevé",

                "alerte":
                    "Marge de manœuvre insuffisante.",

                "raisons": [
                    f"Obstacle très proche estimé à "
                    f"{distance_obstacle:.1f} m."
                ],

                "recommandation":
                    "Avancer uniquement au pas et "
                    "contrôler l'environnement.",
            }

        if distance_obstacle < 2.5:

            return {
                "niveau": "Moyen",

                "alerte":
                    "Environnement encombré.",

                "raisons": [
                    f"Obstacle proche estimé à environ "
                    f"{distance_obstacle:.1f} m."
                ],

                "recommandation":
                    "Poursuivre très lentement avec "
                    "des contrôles visuels fréquents.",
            }

    # --------------------------------------------------------
    # CONFIANCE FAIBLE
    # --------------------------------------------------------

    if confiance < 0.35:

        return {
            "niveau": "Indéterminé",

            "alerte":
                "Perception insuffisamment fiable.",

            "raisons": [
                f"Confiance moyenne des détections "
                f"faible ({confiance:.2f})."
            ],

            "recommandation":
                "Ne pas automatiser la décision. "
                "Vérifier la scène manuellement.",
        }

    return {
        "niveau": "Faible",

        "alerte":
            "Aucun danger immédiat détecté.",

        "raisons": [
            "Aucun danger immédiat identifié "
            "par le moteur de sécurité."
        ],

        "recommandation":
            "La manœuvre peut être envisagée "
            "à faible vitesse avec surveillance "
            "continue de l'environnement.",
    }


# ============================================================
# 5. OUTIL POUR GROQ
# ============================================================

def evaluateur_global_parking(
    distance_pieton_m,
    distance_obstacle_m,
    nombre_detections,
    confiance_moyenne,
):

    return evaluer_risque_scene(
        {
            "nombre_detections":
                int(nombre_detections),

            "confiance_moyenne":
                float(confiance_moyenne),

            "distance_pieton_min_m":
                None
                if float(distance_pieton_m) >= 999
                else float(distance_pieton_m),

            "distance_obstacle_min_m":
                None
                if float(distance_obstacle_m) >= 999
                else float(distance_obstacle_m),
        }
    )


OUTILS_JSON = [
    {
        "type": "function",

        "function": {
            "name":
                "evaluateur_global_parking",

            "description":
                "Évalue de manière déterministe "
                "le risque d'une scène de parking.",

            "parameters": {
                "type": "object",

                "properties": {
                    "distance_pieton_m": {
                        "type": "number",
                        "description":
                            "Distance du piéton le plus "
                            "proche. 999 si absent.",
                    },

                    "distance_obstacle_m": {
                        "type": "number",
                        "description":
                            "Distance du véhicule, camion, "
                            "moto ou vélo le plus proche. "
                            "999 si absent.",
                    },

                    "nombre_detections": {
                        "type": "integer",
                    },

                    "confiance_moyenne": {
                        "type": "number",
                    },
                },

                "required": [
                    "distance_pieton_m",
                    "distance_obstacle_m",
                    "nombre_detections",
                    "confiance_moyenne",
                ],
            },
        },
    }
]


# ============================================================
# 6. CRÉATION DU RÉSUMÉ DE SCÈNE
# ============================================================

def resume_scene(scene):

    comptage = scene.get(
        "comptage",
        {},
    )

    if comptage:

        texte_comptage = ", ".join(
            f"{nom}: {nombre}"
            for nom, nombre
            in sorted(comptage.items())
        )

    else:

        texte_comptage = "aucun objet"

    distance_pieton = (
        f"~{scene['distance_pieton_min_m']:.1f} m"
        if scene["distance_pieton_min_m"] is not None
        else "non détecté"
    )

    distance_obstacle = (
        f"~{scene['distance_obstacle_min_m']:.1f} m"
        if scene["distance_obstacle_min_m"] is not None
        else "non détecté"
    )

    lignes = [
        (
            f"Résolution : "
            f"{scene['largeur_image']}x"
            f"{scene['hauteur_image']}"
        ),

        (
            f"Nombre de détections : "
            f"{scene['nombre_detections']}"
        ),

        (
            f"Confiance moyenne : "
            f"{scene['confiance_moyenne']:.2f}"
        ),

        f"Comptage : {texte_comptage}",

        (
            f"Piéton le plus proche : "
            f"{distance_pieton}"
        ),

        (
            f"Obstacle le plus proche : "
            f"{distance_obstacle}"
        ),

        "Objets principaux :",
    ]

    detections = sorted(
        scene.get(
            "detections",
            [],
        ),

        key=lambda detection: (
            detection["distance_estimee_m"]
            is None,

            detection["distance_estimee_m"]
            if detection["distance_estimee_m"]
            is not None
            else 999,

            -detection["confiance"],
        ),
    )

    for detection in detections[:20]:

        distance = detection[
            "distance_estimee_m"
        ]

        if distance is None:

            distance_texte = (
                "distance non estimée"
            )

        else:

            distance_texte = (
                f"~{distance:.1f} m"
            )

        lignes.append(
            "- "
            f"{detection['classe']} "
            f"(catégorie="
            f"{detection['categorie']}, "
            f"confiance="
            f"{detection['confiance']:.2f}, "
            f"{distance_texte})"
        )

    return "\n".join(
        lignes
    )


# ============================================================
# 7. RAPPORT LOCAL DE SECOURS
# ============================================================

def rapport_local(
    scene,
    risque,
    prefixe=None,
):

    comptage = scene.get(
        "comptage",
        {},
    )

    if comptage:

        objets = ", ".join(
            f"{nom}={nombre}"
            for nom, nombre
            in sorted(comptage.items())
        )

    else:

        objets = "aucun"

    raisons = "\n".join(
        f"- {raison}"
        for raison
        in risque.get(
            "raisons",
            [],
        )
    )

    if prefixe:

        introduction = (
            prefixe
            + "\n\n"
        )

    else:

        introduction = ""

    return (
        introduction

        + "### 👁️ Analyse de la scène\n"

        + (
            f"- Détections : "
            f"{scene.get('nombre_detections', 0)}\n"
        )

        + (
            f"- Objets : "
            f"{objets}\n"
        )

        + (
            f"- Confiance moyenne : "
            f"{scene.get('confiance_moyenne', 0.0):.2f}"
            f"\n\n"
        )

        + "### 📐 Évaluation technique\n"

        + (
            raisons
            if raisons
            else "- Aucun élément supplémentaire."
        )

        + "\n\n"

        + "### ⚠️ Niveau de risque\n"

        + (
            f"**{risque['niveau']}** — "
            f"{risque['alerte']}\n\n"
        )

        + "### 🚗 Recommandation\n"

        + risque["recommandation"]

        + "\n\n"

        + (
            "_Les distances sont des estimations "
            "monoculaires de démonstration et ne "
            "remplacent pas une caméra calibrée ou "
            "un capteur de profondeur._"
        )
    )


# ============================================================
# 8. AGENT GROQ ROBUSTE
# ============================================================

def generer_diagnostic(
    client_groq,
    scene,
):

    # Le risque est TOUJOURS calculé localement.
    risque_local = evaluer_risque_scene(
        scene
    )

    # --------------------------------------------------------
    # SANS API : L'APPLICATION CONTINUE
    # --------------------------------------------------------

    if client_groq is None:

        return {
            "risque":
                risque_local,

            "rapport":
                rapport_local(
                    scene,
                    risque_local,
                    prefixe=(
                        "ℹ️ Diagnostic local "
                        "déterministe."
                    ),
                ),

            "llm_ok":
                False,

            "erreur_llm":
                None,
        }

    scene_texte = resume_scene(
        scene
    )

    distance_pieton = (
        scene["distance_pieton_min_m"]
        if scene["distance_pieton_min_m"]
        is not None
        else 999
    )

    distance_obstacle = (
        scene["distance_obstacle_min_m"]
        if scene["distance_obstacle_min_m"]
        is not None
        else 999
    )

    try:

        # ----------------------------------------------------
        # PREMIER APPEL :
        # un seul Function Calling
        # ----------------------------------------------------

        messages_outil = [
            {
                "role":
                    "system",

                "content":
                    (
                        "Tu es un agent d'assistance "
                        "au parking. "
                        "Appelle exactement une fois "
                        "evaluateur_global_parking avec "
                        "les valeurs fournies. "
                        "N'invente aucune valeur."
                    ),
            },

            {
                "role":
                    "user",

                "content":
                    (
                        f"{scene_texte}\n\n"

                        "Valeurs exactes :\n"

                        f"distance_pieton_m="
                        f"{distance_pieton}\n"

                        f"distance_obstacle_m="
                        f"{distance_obstacle}\n"

                        f"nombre_detections="
                        f"{scene['nombre_detections']}\n"

                        f"confiance_moyenne="
                        f"{scene['confiance_moyenne']}"
                    ),
            },
        ]

        reponse_outil = (
            client_groq
            .chat
            .completions
            .create(
                model=GROQ_MODEL,

                messages=
                    messages_outil,

                tools=
                    OUTILS_JSON,

                tool_choice={
                    "type":
                        "function",

                    "function": {
                        "name":
                            "evaluateur_global_parking"
                    },
                },

                parallel_tool_calls=False,

                temperature=0,

                max_completion_tokens=500,
            )
        )

        message = (
            reponse_outil
            .choices[0]
            .message
        )

        # Lecture uniquement pour vérifier le JSON.
        if message.tool_calls:

            try:
                json.loads(
                    message
                    .tool_calls[0]
                    .function
                    .arguments
                    or "{}"
                )

            except Exception:
                pass

        # IMPORTANT :
        # on utilise les vraies valeurs Python,
        # pas les valeurs éventuellement modifiées
        # par le LLM.
        resultat_outil = risque_local

        # ----------------------------------------------------
        # SECOND APPEL :
        # NOUVELLE conversation SANS OUTILS
        #
        # Donc plus de :
        # "Tool choice is none, but model called a tool"
        # ----------------------------------------------------

        prompt_final = f"""
Tu rédiges le rapport final d'un démonstrateur
d'assistance au stationnement.

RÈGLES ABSOLUES :
- Réponds uniquement en français.
- N'invente aucune classe.
- N'invente aucune distance.
- N'invente aucune mesure.
- Le niveau de risque imposé est :
  {resultat_outil['niveau']}
- Ne modifie jamais ce niveau.
- Les distances précédées de ~ sont approximatives.
- N'écris aucun appel de fonction.
- Reste technique, clair et professionnel.

DONNÉES YOLO :

{scene_texte}

ÉVALUATION DÉTERMINISTE :

{json.dumps(
    resultat_outil,
    ensure_ascii=False
)}

FORMAT :

### 👁️ Analyse de la scène
2 à 5 phrases.

### 📐 Évaluation technique
Explique uniquement les éléments réellement détectés.

### ⚠️ Niveau de risque
**{resultat_outil['niveau']}**
avec une justification courte.

### 🚗 Recommandation
Une action concrète cohérente.

Termine obligatoirement par :

_Les distances sont des estimations monoculaires
de démonstration._
"""

        reponse_finale = (
            client_groq
            .chat
            .completions
            .create(
                model=GROQ_MODEL,

                messages=[
                    {
                        "role":
                            "system",

                        "content":
                            (
                                "Tu es un rédacteur "
                                "technique spécialisé "
                                "dans les systèmes "
                                "d'assistance au "
                                "stationnement."
                            ),
                    },

                    {
                        "role":
                            "user",

                        "content":
                            prompt_final,
                    },
                ],

                temperature=0.1,

                max_completion_tokens=900,
            )
        )

        rapport = (
            reponse_finale
            .choices[0]
            .message
            .content
            or ""
        ).strip()

        if not rapport:

            rapport = rapport_local(
                scene,
                risque_local,
            )

        return {
            "risque":
                risque_local,

            "rapport":
                rapport,

            "llm_ok":
                True,

            "erreur_llm":
                None,
        }

    # --------------------------------------------------------
    # SI GROQ PLANTE :
    # AUCUNE FAUSSE ALERTE CRITIQUE
    # --------------------------------------------------------

    except Exception as exc:

        return {
            "risque":
                risque_local,

            "rapport":
                rapport_local(
                    scene,
                    risque_local,
                    prefixe=(
                        "⚠️ Le LLM est temporairement "
                        "indisponible. "
                        "Le diagnostic ci-dessous "
                        "provient du moteur local "
                        "déterministe."
                    ),
                ),

            "llm_ok":
                False,

            "erreur_llm":
                str(exc),
        }
