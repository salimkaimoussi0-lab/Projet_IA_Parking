import streamlit as st
from ultralytics import YOLO
import cv2
import os
import json
from groq import Groq
import numpy as np
from PIL import Image

# ==========================================
# 1. Configuration de la page
# ==========================================
st.set_page_config(page_title="IA Parking Autonome", page_icon="🚗", layout="wide")

# ==========================================
# 2. Design de l'Interface Principale
# ==========================================
st.title("🚗 Système d'Assistance Intelligent au Parking")
st.divider()

# ==========================================
# 3. Menu Latéral (Sidebar) pour la configuration
# ==========================================
st.sidebar.header("⚙️ Configuration Système")
api_key = st.sidebar.text_input(
    "Clé d'authentification API (Groq) :", 
    type="password", 
    help="Nécessaire pour initialiser le moteur de raisonnement LLM."
)
st.sidebar.markdown("---")

# ==========================================
# 4. Modèles et Fonctions Métier
# ==========================================
@st.cache_resource
def load_yolo():
    return YOLO('yolov8s_50.pt')

model_yolo = load_yolo() 

def evaluateur_global_parking(largeur_place_cm, distance_vehicule_devant_m, distance_pieton_m):
    taille_voiture_minimale_cm = 250
    rapport = []
    
    # Évaluation Piéton
    if distance_pieton_m < 2.0:
        rapport.append("🔴 ALERTE ROUGE : Piéton à moins de 2 mètres. Freinage d'urgence requis.")
    elif distance_pieton_m < 5.0:
        rapport.append("🟠 ATTENTION : Piéton proche. Prudence imposée.")
    else:
        rapport.append("🟢 Sécurité piéton : Aucun danger immédiat.")

    # Évaluation Place
    if largeur_place_cm < taille_voiture_minimale_cm:
        rapport.append(f"🔴 STATIONNEMENT IMPOSSIBLE : Place trop petite ({largeur_place_cm}cm).")
    else:
        rapport.append("🟢 STATIONNEMENT POSSIBLE : La place est assez grande.")
        if distance_vehicule_devant_m < 0.5:
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
RÈGLE ABSOLUE : Utilise l'outil 'evaluateur_global_parking' pour valider la situation mathématiquement.

FORMAT DE RÉPONSE OBLIGATOIRE (Utilise du Markdown propre) :
### 👁️ Analyse de la scène
[Description courte de ce que tu as compris]

### 🧮 Évaluation technique
[Résultat de ton outil mathématique]

### ⚠️ Niveau de Risque global
**[Faible / Moyen / Élevé / Critique]**

### 🚗 Recommandation d'action
[Action précise : Freiner, Avancer doucement, Manoeuvre autorisée...]"""

def executer_agent(client_ia, observation_yolo):
    messages = [
        {"role": "system", "content": prompt_systeme},
        {"role": "user", "content": f"Voici les données des caméras : {observation_yolo}"}
    ]
    response = client_ia.chat.completions.create(
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
            response_finale = client_ia.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages
            )
            return response_finale.choices[0].message.content
    return message_ia.content

# ==========================================
# 5. Zone d'Upload et Déclenchement de l'IA
# ==========================================
st.subheader("📷 Acquisition Visuelle")
fichier_image = st.file_uploader("Transmission de la capture caméra :", type=["jpg", "jpeg", "png"])

if fichier_image is not None:
    if not api_key:
        st.warning("⚠️ Image reçue. En attente de l'authentification API (menu de gauche) pour initialiser le système autonome.")
        st.stop() 

    try:
        client_groq = Groq(api_key=api_key)
    except Exception as e:
        st.error(f"Erreur de connexion au serveur d'IA : {e}")
        st.stop()

    st.success("✅ Connexion sécurisée établie. Lancement de la procédure d'analyse...")
    st.markdown("---")

    # Lancement du Pipeline
    image = Image.open(fichier_image)
    image_np = np.array(image)

    # ---------------------------------------------------------
    # PARTIE 1 : IMAGE TAILLE AJUSTÉE (66% de largeur)
    # ---------------------------------------------------------
    st.subheader("👁️ Perception (YOLOv8)")
    with st.spinner('Traitement visuel en cours...'):
        resultats = model_yolo(image_np)
        
        # labels=False pour garder uniquement les cadres
        image_annotee = resultats[0].plot(labels=False) 
        
        # Les proportions sont ici : 1 part de vide, 4 parts d'image, 1 part de vide
        col_vide1, col_image, col_vide2 = st.columns([1, 4, 1])
        with col_image:
            st.image(image_annotee, use_container_width=True)

        # Calculs en arrière-plan
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

        phrase_yolo = f"Objets détectés : {', '.join(objets_detectes)}. Largeur de la place visée estimée à 280cm. "
        if voiture_proche != 999: phrase_yolo += f"Véhicule devant à {voiture_proche} mètres. "
        if pieton_proche != 999: phrase_yolo += f"Piéton détecté à {pieton_proche} mètres. "
        else: phrase_yolo += "Aucun piéton détecté."

    st.markdown("---")

    # ---------------------------------------------------------
    # PARTIE 2 : ORDINATEUR DE BORD (En dessous)
    # ---------------------------------------------------------
    st.subheader("🧠 Ordinateur de Bord")
    with st.spinner('Analyse spatiale et calcul des trajectoires de sécurité...'):
        rapport = executer_agent(client_groq, phrase_yolo)
        st.info(rapport, icon="🤖")