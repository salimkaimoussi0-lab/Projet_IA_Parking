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


#  FONCTION PRINCIPALE (LE CERVEAU)
def analyser_scene_parking(donnees_vision, cle_api_groq):
    """
    Analyse les détections de la dashcam et retourne un diagnostic IA.
    
    Arguments:
    - donnees_vision (dict) : Le JSON provenant du Module A (Vision).
    - cle_api_groq (str) : La clé API pour se connecter à Llama 3.
    
    Retourne:
    - dict : Le diagnostic formaté avec le risque, l'analyse et les recommandations.
    """
    
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

        # Instructions strictes
        system_prompt = """
        Tu es le cerveau analytique d'un système d'aide au stationnement.
        Avant de répondre, vérifie toujours la distance des piétons en utilisant l'outil fourni.
        
        RÈGLE ABSOLUE : Tu dois répondre STRICTEMENT avec ce format JSON exact, sans faire de listes pour les recommandations (utilise une seule phrase) :
        {
          "agent_llm": {
            "niveau_risque": "Faible ou Moyen ou Elevé ou Critique",
            "analyse": "...",
            "recommandations": "..."
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