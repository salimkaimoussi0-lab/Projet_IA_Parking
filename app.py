import streamlit as st
from ultralytics import YOLO
import cv2
import os
import json
from groq import Groq
import numpy as np
from PIL import Image

# 1. Configuration de la page streamlit
st.set_page_config(page_title ="IA Parking Autonome", page_icon="🚗", layout="wide")

# 2. Initialisation des modèles
@st.cache_resource
def load_yolo():
    return YOLO('yolov8s_50.pt')

model_yolo = load_yolo()  # Correction: ajout des parenthèses
client = Groq() 

# 3. L'outil d'evaluation (Module B)
def evaluateur_global_parking(largeur_place_cm, distance_vehicule_devant_m, distance_pieton_m):
    taille_voiture_minimale_cm = 250
    rapport = []
    if distance_pieton_m < 2.0 :
        rapport.append("🔴 ALERTE ROUGE : Piéton à moins de 2 mètres. Freinage d'urgence requis.")
    elif distance_pieton_m < 5.0 :
        rapport.append("🟠 ATTENTION : Piéton proche. Prudence imposée.")
    else:
        rapport.append("🟢 Sécurité piéton : Aucun danger immédiat.")

    if largeur_place_cm < taille_voiture_minimale_cm :
        rapport.append(f"🔴 STATIONNEMENT IMPOSSIBLE : Place trop petite ({largeur_place_cm}cm).")
    else:
        rapport.append("🟢 STATIONNEMENT POSSIBLE : La place est assez grande.")
        if distance_vehicule_devant_m < 0.5 :
            rapport.append("🟠 RISQUE MATÉRIEL : Le véhicule devant est très proche.")
        else:
            rapport.append("🟢 Espace de manoeuvre matériel : OK.")
    return " | ".join(rapport)

outils_json = [{
    "type": "function",
    "function": {
        "name": "evaluateur_global_parking",
        "description": "Évalue si une place de parking est assez grande ET évalue les risques de collision.",
        "parameters": {
            "type" : "object",
            "properties":{
                "largeur_place_cm": {"type": "integer"},
                "distance_vehicule_devant_m": {"type": "number"},
                "distance_pieton_m": {"type": "number"}
            },
            "required": ["largeur_place_cm", "distance_vehicule_devant_m", "distance_pieton_m"]
        }    
    }
}]

prompt_systeme = """ Tu es l'ordinateur de bord d'un véhicule autonome.
RÈGLE ABSOLUE : Utilise l'outil 'evaluateur_global_parking' pour valider la situation.

FORMAT DE RÉPONSE OBLIGATOIRE :
### 👁️ Analyse de la scène
...
### 🧮 Évaluation technique
...
### ⚠️ Niveau de Risque
[Faible / Moyen / Élevé / Critique]
### 🚗 Recommandations de conduite
..."""

def executer_agent(observation_yolo):
    messages = [
        {"role": "system", "content": prompt_systeme},
        {"role": "user", "content": f"Voici les données des caméras : {observation_yolo}"}
    ]
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        tools=outils_json,
        tool_choice="auto",
    )
    message_ia = response.choices[0].message
    if message_ia.tool_calls:
        for tool_call in message_ia.tool_calls:
            arguments = json.loads(tool_call.function.arguments)
            resultat_outil = evaluateur_global_parking(**arguments)
            messages.append(message_ia)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id, 
                "name": tool_call.function.name, 
                "content": resultat_outil
            })
            response_finale = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages
            )
            return response_finale.choices[0].message.content # Correction: choices avec un 's'
    return message_ia.content

# ==========================================
# 4. Interface Utilisateur (Le design) - DÉSINDENTÉ
# ==========================================
st.title("🚗 Système d'Assistance au Stationnement (IA)")
st.markdown("### Module C : Intégration Yolov8 + Agent LLM")
fichier_image = st.file_uploader("📷 Uploadez une image (Dashcam ou Caméra de recul)", type=["jpg", "jpeg", "png"])

if fichier_image is not None:
    # Lecture de l'image
    image = Image.open(fichier_image)
    image_np = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1️⃣ Analyse Visuelle (YOLO)")
        with st.spinner('Détectant les objets...'):
            resultats = model_yolo(image_np)
            image_annotee = resultats[0].plot()
            st.image(image_annotee, caption='Détection en temps réel', use_container_width=True)

            # Traduction pour l'Agent
            pieton_proche = 999
            voiture_proche = 999
            objets_detectes = set()

            for boite in resultats[0].boxes:
                nom_objet = resultats[0].names[int(boite.cls[0])]
                objets_detectes.add(nom_objet)
                hauteur = boite.xywh[0][3].item()
                distance_estimee = round(1000 / hauteur, 1)

                if nom_objet in ['person', 'pietons'] and distance_estimee < pieton_proche:
                    pieton_proche = distance_estimee
                elif nom_objet in ['car', 'Voitures', 'Camions'] and distance_estimee < voiture_proche:
                    voiture_proche = distance_estimee

            # Création de la phrase - SORTIE DE LA BOUCLE FOR
            phrase_yolo = f"Objets vus : {','.join(objets_detectes)}. Largeur estimée de la place: 280cm. "
            if voiture_proche != 999: phrase_yolo += f"Véhicule devant à {voiture_proche} mètres. "
            if pieton_proche != 999: phrase_yolo += f"Piéton détecté à {pieton_proche} mètres. "
            else: phrase_yolo += "Aucun piéton détecté."

    with col2:
        st.subheader("2️⃣ Rapport de l'Ordinateur de Bord (Agent)")
        with st.spinner('Le cerveau IA réfléchit...'):
            rapport = executer_agent(phrase_yolo)
            st.markdown(rapport)
