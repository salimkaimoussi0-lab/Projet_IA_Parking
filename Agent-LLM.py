!pip install groq -q 

import os
import json
from getpass import getpass
from groq import Groq

# 1- Connexion à Groq (Sécurisée)
print("🔑 Veuillez entrer votre clé API Groq :")
os.environ["GROQ_API_KEY"] = getpass()
client = Groq()

# 2- NOTRE OUTIL (Places Libres + Piétons)
def evaluateur_global_parking(largeur_place_cm, distance_vehicule_devant_m, distance_pieton_m):
    """Analyse si la place est assez grande ET s'il y a un danger."""
    taille_voiture_minimale_cm = 250 
    rapport = []

    # A. Analyse des dangers mortels
    if distance_pieton_m < 2.0:
        rapport.append("Alerte Rouge : Piéton à moins de 2 mètres. Freinage d'urgence requis.")
    elif distance_pieton_m < 5.0:
        rapport.append("ATTENTION : Piéton Proche (Zone des 5 mètres). Le code de la route impose la prudence.")
    else:
        rapport.append("Sécurité Piéton : Aucun danger immédiat.")
    
    # B. Analyse de la place de parking
    if largeur_place_cm < taille_voiture_minimale_cm:
        rapport.append(f"STATIONNEMENT IMPOSSIBLE : Place trop petite ({largeur_place_cm}cm détectés, 250cm requis).")
    else:
        rapport.append("STATIONNEMENT POSSIBLE : la place est assez grande.")

        # C. Analyse des collisions matérielles
        if distance_vehicule_devant_m < 0.5:
            rapport.append("RISQUE MATÉRIEL : Le véhicule devant très proche (moins de 50cm). Manœuvre difficile.")
        else: 
            rapport.append("Espace de manœuvre matériel : OK.")

    return " | ".join(rapport)

outils_disponibles = {
    "evaluateur_global_parking": evaluateur_global_parking,
}

# La définition Json pour Groq
outils_json = [
    {
        "type": "function",
        "function": {
            "name": "evaluateur_global_parking",
            "description": "Évalue si une place de parking est assez grande ET évalue les risques de collision avec les piétons ou véhicules.",
            "parameters": {
                "type" : "object",
                "properties":{
                    "largeur_place_cm": {"type": "integer", "description": "Largeur de la place vide en centimètres."},
                    "distance_vehicule_devant_m": {"type": "number", "description": "Distance avec le véhicule garé devant en mètres."},
                    "distance_pieton_m": {"type": "number", "description": "Distance avec le piéton le plus proche en mètres. Mettre 999 s'il n'y a aucun piéton."}
                },
                "required": ["largeur_place_cm", "distance_vehicule_devant_m", "distance_pieton_m"]
            }    
        }
    }
]

# 3- System Prompt
prompt_systeme = """Tu es l'ordinateur de bord d'un véhicule autonome.
Tu dois analyser les données des caméras (YOLO) pour aider au stationnement.

RÈGLE ABSOLUE : Tu DOIS utiliser l'outil 'evaluateur_global_parking' pour valider mathématiquement si la place est bonne ET si la situation est sûre.

ÉCHELLE DE RISQUE OBLIGATOIRE :
- Faible : Place assez grande, aucun obstacle, aucun piéton.
- Moyen : Place assez grande, mais manœuvre serrée ou piéton dans les environs (sans danger immédiat).
- Élevé : Place trop petite OU piéton très proche.
- Critique : FREINAGE D'URGENCE (Piéton à moins de 2 mètres ou collision imminente).

FORMAT DE RÉPONSE OBLIGATOIRE :
- **Analyse de la scène :**...
- **Résultats de l'outil d'évaluation :**...
- **Niveau de Risque :**[Faible / Moyen / Élevé / Critique]
- **Recommandations de conduite :**..."""

# 4- LA BOUCLE AGENT
def executer_agent(observation_yolo):
    print(f"👁️ YOLO voit : {observation_yolo}\n")
    print("🤖 L'Agent réfléchit...\n")

    messages = [
        {"role": "system", "content": prompt_systeme},
        {"role": "user", "content": f"Voici les données des caméras : {observation_yolo}"}
    ]

    # CHANGEMENT ICI : llama-3.3-70b-versatile
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        tools=outils_json,
        tool_choice="auto",
    )

    message_ia = response.choices[0].message

    if message_ia.tool_calls:
        for tool_call in message_ia.tool_calls:
            nom_fonction = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            print(f"🔧 Utilisation de l'outil [{nom_fonction}] avec les paramètres {arguments}")

            function_python = outils_disponibles[nom_fonction]
            resultat_outil = function_python(**arguments)

            messages.append(message_ia)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": nom_fonction,
                "content": resultat_outil
            })

            # CHANGEMENT ICI AUSSI : llama-3.3-70b-versatile
            response_finale = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages
            )

            print("\n" + " RAPPORT FINAL ".ljust(50, "="))
            print(response_finale.choices[0].message.content)
            print("="*50 + "\n")
    else:
        print(message_ia.content)

# 5- SIMULATION
print("--- SCÉNARIO 1 : Place parfaite ---")
executer_agent("Place de parking vide détectée. Largeur: 320 cm. Voiture devant à 1.5 mètres. Aucun piéton détecté.")

print("\n--- SCÉNARIO 2 : La place est bonne mais danger ! ---")
executer_agent("Place de parking vide détectée. Largeur: 280 cm. Voiture devant à 0.8 mètres. 1 piéton détecté à 1.2 mètres sur le côté droit.")

print("\n--- SCÉNARIO 3 : Place trop petite ---")
executer_agent("Espace entre deux voitures détectée. Largeur: 180 cm. Voiture devant à 0.2 mètres. Piéton éloigné à 15 mètres.")

# ==========================================
# LE PONT ENTRE LA VISION ET L'IA
# ==========================================

def traduire_yolo_pour_agent(resultats_yolo):
    """
    Cette fonction prend les boîtes de détection de YOLO, 
    calcule les distances, et crée la phrase pour l'Agent.
    """
    pieton_le_plus_proche = 999  # Valeur par défaut (aucun)
    voiture_la_plus_proche = 999
    largeur_place_estimee = 300  # Par défaut on suppose la place standard
    
    # On parcourt tout ce que YOLO a vu sur l'image
    for boite in resultats_yolo[0].boxes:
        classe_id = int(boite.cls[0])
        nom_objet = resultats_yolo[0].names[classe_id] # 'person', 'car', etc.
        
        # Calcul d'une distance fictive basée sur la taille de la boîte
        # (Plus la boîte est grande sur l'image, plus l'objet est proche)
        hauteur_boite = boite.xywh[0][3].item()
        distance_estimee = 1000 / hauteur_boite  # Formule mathématique simplifiée
        
        if nom_objet == 'person' or nom_objet == 'pietons':
            if distance_estimee < pieton_le_plus_proche:
                pieton_le_plus_proche = round(distance_estimee, 1)
                
        elif nom_objet == 'car' or nom_objet == 'Voitures':
             if distance_estimee < voiture_la_plus_proche:
                voiture_la_plus_proche = round(distance_estimee, 1)

    # On rédige la phrase automatiquement !
    observation_automatique = f"Tentative de stationnement. Largeur de place estimée: {largeur_place_estimee} cm. "
    
    if voiture_la_plus_proche != 999:
        observation_automatique += f"Véhicule devant à {voiture_la_plus_proche} mètres. "
    if pieton_le_plus_proche != 999:
        observation_automatique += f"Attention, piéton détecté à {pieton_le_plus_proche} mètres !"
    else:
        observation_automatique += "Aucun piéton détecté."

    return observation_automatique


# ---------------------------------------------------------
# EXEMPLE DE TON CODE FINAL :
# ---------------------------------------------------------
# 1. YOLO regarde l'image
# resultats = model_yolo('photo_du_parking.jpg')

# 2. On traduit l'image en texte
# phrase_pour_ia = traduire_yolo_pour_agent(resultats)

# 3. L'Agent Groq lit le texte et prend une décision !
# executer_agent(phrase_pour_ia)

