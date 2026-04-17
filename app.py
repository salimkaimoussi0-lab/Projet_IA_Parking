import streamlit as st
from PIL import Image, ImageDraw
import time 
from ultralytics import YOLO 
from agent_parking import analyser_scene_parking

# 1. Configuration
st.set_page_config(page_title="Assistant Parking", page_icon="🚗", layout="wide")

# 2. Chargement du modèle YOLO
@st.cache_resource
def charger_modele():
    return YOLO("yolov8s_50.pt")

try:
    modele_vision = charger_modele()
    modele_charge = True
except Exception as e:
    modele_charge = False
    st.error("Erreur : Impossible de charger le modèle YOLO. Vérifiez que yolov8s_50.pt est bien dans le dossier.")

# 3. La Barre latérale
with st.sidebar:
    st.header("⚙️ Centre de Contrôle")
    
    st.subheader("🔑 Connexion Groq")
    api_key = st.text_input("Entrez la clé API Groq :", type="password")
    st.divider()
    
    uploaded_file = st.file_uploader("Importer une vue dashcam", type=["jpg", "jpeg", "png"])
    st.divider()
    
    if modele_charge:
        st.success("👁️ Vision : Prête (YOLOv8s)")
    else:
        st.error("👁️ Vision : Hors ligne")
        
    if api_key:
        st.success("🧠 Agent IA : Prêt (Llama 3)")
    else:
        st.warning("🧠 Agent IA : En attente de clé API")

# 4. En-tête principal
st.title("🚗 Tableau de Bord : Assistant de Parking")
st.markdown("Système d'analyse de scènes en temps réel propulsé par YOLOv8 & LLM")
st.divider()

if uploaded_file is not None and modele_charge:
    if not api_key:
        st.warning("👈 Veuillez renseigner une clé API Groq dans le panneau de gauche pour activer l'analyse intelligente.")
    else:
        col_gauche, col_droite = st.columns([0.6, 0.4])

        with col_gauche:
            st.subheader("📷 Flux Vidéo (Vision Module)")
            image = Image.open(uploaded_file)
            draw = ImageDraw.Draw(image) 
            
            with st.spinner("1/2 Détection YOLOv8 en cours..."):
                resultats = modele_vision(image)
                
            donnees_pour_agent = {"detections": []}
            
            for box in resultats[0].boxes:
                coords = box.xyxy[0].tolist() 
                confiance = box.conf[0].item() 
                nom_classe = modele_vision.names[int(box.cls[0].item())] 

                donnees_pour_agent["detections"].append({
                    "classe": nom_classe,
                    "confiance": round(confiance, 2),
                    "coordonnees": {"ymin": int(coords[1]), "ymax": int(coords[3])}
                })

                hauteur_boite = coords[3] - coords[1]
                distance_estimee = round(800 / hauteur_boite, 1) 
                
                draw.rectangle([coords[0], coords[1], coords[2], coords[3]], outline="#FF0000", width=4)
                texte_label = f"{nom_classe} - Dist: {distance_estimee}m"
                draw.text((coords[0], coords[1] - 15), texte_label, fill="#FF0000")

            st.image(image,width='stretch')

        with col_droite:
            st.subheader("🧠 Diagnostic Agent IA")
            
            # --- LA CORRECTION EST ICI ---
            if len(donnees_pour_agent["detections"]) == 0:
                st.metric(label="Niveau de Risque Détecté", value="Inconnu")
                st.info("👀 L'IA Visuelle n'a détecté aucun véhicule ou piéton sur cette image. Cela peut arriver si l'angle de vue n'est pas celui d'une caméra embarquée (dashcam).")
            # -----------------------------
            else:
                with st.spinner("2/2 Analyse cognitive par Llama 3 en cours..."):
                    reponse_agent = analyser_scene_parking(donnees_pour_agent, api_key)
                
                donnees_llm = reponse_agent.get("agent_llm", {})
                risque = donnees_llm.get("niveau_risque", "Inconnu")
                
                st.metric(label="Niveau de Risque Détecté", value=risque)
                
                if "Faible" in risque:
                    st.success("✅ Aucun danger immédiat.")
                elif "Moyen" in risque:
                    st.warning("⚠️ Attention requise.")
                else:
                    st.error("🛑 DANGER : Arrêt recommandé.")
                    
                with st.expander("📍 Analyse Spatiale Détaillée", expanded=True):
                    st.write(donnees_llm.get('analyse', 'Aucune analyse disponible.'))
                    
                with st.expander("🛡️ Recommandations de Conduite", expanded=True):
                    st.write(donnees_llm.get('recommandations', 'Aucune recommandation.'))

elif uploaded_file is None:
    st.info("👈 Veuillez charger une image depuis le panneau de gauche pour initier le pipeline d'analyse.")
