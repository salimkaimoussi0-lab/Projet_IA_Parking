import json
from groq import Groq

# OUTILS DE L'AGENT
def estimer_distance(ymax):
    """Calcule la distance approximative d'un obstacle."""
    if ymax >= 500:
        return "ALERTE : Objet très proche (Moins de 2 mètres) !"
    elif ymax >= 350:
        return "ATTENTION : Objet à moyenne distance (Environ 5 mètres)."
    else:
        return "SÉCURITÉ : Objet éloigné (Plus de 10 mètres)."

# FONCTION PRINCIPALE 
def analyser_scene_parking(donnees_vision, cle_api_groq):
    """
    Analyse les détections de la dashcam et retourne un diagnostic IA.
    
    Arguments:
    - donnees_vision (dict) : Le JSON provenant du Module A (Vision).
    - cle_api_groq (str) : La clé API pour se connecter à Llama 3.
    
    Retourne:
    - dict : Le diagnostic formaté avec le risque, l'analyse et les recommandations.
    """
    if "vision" not in donnees_vision:
        donnees_vision = {"vision": donnees_vision}
    try:
        # Initialisation du client
        client = Groq(api_key=cle_api_groq)
        
        # Définition des outils pour l'IA
        mes_outils = [{
            "type": "function",
            "function": {
                "name": "estimer_distance",
                "description": "Utilise cet outil pour connaître la distance réelle d'un obstacle. Passe-lui la valeur 'ymax' de la boîte de détection.",
                "parameters": {
                    "type": "object",
                    "properties": {"ymax": {"type": "integer"}},
                    "required": ["ymax"]
                }
            }
        }]

        system_prompt = """
        Tu es l'IA d'assistance au stationnement embarquée dans une dashcam. Ton rôle est d'analyser l'environnement et d'assurer la sécurité totale du véhicule et des usagers.

        INSTRUCTIONS D'ANALYSE :
        1. Contexte : Tu analyses des données de vision par ordinateur. Utilise toujours les outils à ta disposition pour obtenir des informations extérieures AVANT de rédiger ta conclusion.
        2. Échelle de risque stricte :
           - Faible : Manœuvre libre, obstacles lointains.
           - Moyen : Obstacle présent, manœuvre autorisée avec prudence.
           - Elevé : Obstacle très proche, manœuvre complexe, ralentissement exigé.
           - Critique : Danger immédiat, risque de collision, arrêt d'urgence.
        3. Concision : Le conducteur lit tes recommandations en temps réel. Sois direct et factuel.

        CONTRAINTES DE FORMAT (RÈGLE ABSOLUE) :
        Tu dois renvoyer UNIQUEMENT un objet JSON valide, sans AUCUN texte avant ou après, et SANS balises Markdown (n'utilise pas ```json).
        Ne fais aucune liste à puces. Utilise du texte simple.

        Format exact attendu :
        {
          "agent_llm": {
            "niveau_risque": "[Choix exact : Faible, Moyen, Elevé ou Critique]",
            "analyse": "[1 à 2 phrases courtes décrivant le danger principal]",
            "recommandations": "[1 seule phrase d'action directe pour le conducteur]"
          }
        }
        """

        messages_conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Voici les détections : {json.dumps(donnees_vision)}. Utilise tes outils si besoin."}
        ]

        # Appel 1 : L'IA réfléchit et décide si elle utilise un outil
        premiere_reponse = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages_conversation,
            tools=mes_outils,
            tool_choice="auto"
        )
        message_ia = premiere_reponse.choices[0].message

        # Si l'IA utilise un outil
        if message_ia.tool_calls:
            messages_conversation.append(message_ia)
            for tool_call in message_ia.tool_calls:
                if tool_call.function.name == "estimer_distance":
                    arguments = json.loads(tool_call.function.arguments)
                    resultat_outil = estimer_distance(arguments.get("ymax"))
                    
                    messages_conversation.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": resultat_outil
                    })
                    
            # Appel 2 : Réponse finale après avoir utilisé l'outil
            reponse_finale = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages_conversation,
                response_format={"type": "json_object"}
            )
            texte_final = reponse_finale.choices[0].message.content
            
        else:
            # Si l'IA n'a pas eu besoin d'outil, la première réponse est la bonne (en forçant le JSON)
            reponse_directe = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages_conversation,
                response_format={"type": "json_object"}
            )
            texte_final = reponse_directe.choices[0].message.content

        # On convertit le texte de l'IA en vrai dictionnaire Python pour le renvoyer proprement
        return json.loads(texte_final)

    except Exception as e:
        # Code de sécurité : si Groq plante ou n'a plus de réseau, on renvoie une erreur propre
        return {
            "agent_llm": {
                "niveau_risque": "Critique",
                "analyse": f"Erreur système IA : {str(e)}",
                "recommandations": "Arrêtez le véhicule et reprenez le contrôle manuel."
            }
        }
