# 🚗 (Projet_IA_Parking) : Système Complet d'Assistance au Stationnement Autonome

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-yellow.svg)
![Groq](https://img.shields.io/badge/LLM-Groq_Llama_3-orange.svg)
![Streamlit](https://img.shields.io/badge/Interface-Streamlit-red.svg)

## 🎯 À quoi sert ce projet ?
Le stationnement en milieu urbain dense est l'une des principales causes d'accidents mineurs et de stress pour les conducteurs. Ce projet vise à créer un **copilote intelligent** (vision par ordinateur + IA générative) capable de :
1. **Sécuriser les manœuvres** : Détecter en temps réel les obstacles et les usagers vulnérables (piétons, cyclistes).
2. **Automatiser l'analyse spatiale** : Vérifier mathématiquement si un véhicule peut entrer dans une place détectée sans risque de collision.
3. **Réduire la charge cognitive** : Transformer des images complexes en recommandations textuelles claires (ex: "Freinage d'urgence" ou "Stationnement possible").

---

## 🚦 Architecture et Pipeline de Traitement

Ce projet intègre des technologies "State-of-the-Art" divisées en trois modules interconnectés formant un pipeline de bout en bout :

* **👁️ Module A : Perception (YOLOv8)** Extrait les données de l'image (Dashcam/Caméra de recul). Ce module détecte 13 classes (Voitures, Piétons, Panneaux, etc.) à haute fréquence, calcule leurs coordonnées, et agit comme les yeux du système.
* **🧠 Module B : Raisonnement (Agent Llama 3.3 via Groq)** Agit comme l'ordinateur de bord. Les données de YOLO lui sont envoyées. Grâce au **Function Calling**, l'Agent utilise un outil mathématique déterministe pour évaluer les distances d'arrêt et la largeur des places, puis génère une décision (Risque Faible à Critique).
* **🖥️ Module C : Interface Utilisateur (Streamlit)** Dashboard web interactif permettant de visualiser le retour caméra annoté et le rapport de l'IA de manière instantanée.

---

## 📊 Performances de la Vision (YOLOv8 Nano vs Small)

Pour le Module A, nous avons comparé deux architectures pour trouver l'équilibre idéal entre vitesse d'inférence et précision de sécurité (sur 50 epochs) :
- **YOLOv8 Nano** : Modèle ultra-léger, très rapide, mais peine sur les petits objets éloignés.
- **YOLOv8 Small** : Plus de paramètres, offrant une bien meilleure compréhension de la scène.

| Modèle | mAP@0.5 (Global) | Précision (P) | Rappel (R) | Détection Voitures |
| :--- | :---: | :---: | :---: | :---: |
| **YOLOv8n (Nano)** | 29.4 % | 78.9 % | 27.9 % | 93.8 % |
| **YOLOv8s (Small)** | **77.2 %** | **80.1 %** | **68.8 %** | **97.1 %** |

*Conclusion : Le modèle Nano a totalement échoué sur des classes critiques comme les panneaux (mAP 0%), tandis que le Small offre une fiabilité de 81% dessus, avec une précision de 76.5% sur les camions. Le **YOLOv8s** est donc l'architecture retenue en production.*

---

## 📁 Contenu du Dépôt
* `YOLOv8_Parking.ipynb` : Notebook de recherche et d'entraînement des modèles de vision.
* `Agent-LLM.py` : Script isolé de l'Agent IA (Cerveau) avec appels de fonctions JSON.
* `app.py` : Code source de l'interface Web interactive (Streamlit).
* `yolov8n_50.pt` / `yolov8s_50.pt` : Poids des réseaux de neurones entraînés.
* `PROJET--IA.pdf` : Rapport académique complet du projet.
* Dossiers d'images : Contient les graphiques d'entraînement et les images de validation.

---

## 💻 Installation & Usage

**1. Clonez le dépôt en local :**
```bash
git clone [https://github.com/salimkaimoussi0-lab/Projet_IA_Parking.git](https://github.com/salimkaimoussi0-lab/Projet_IA_Parking.git)
cd Projet_IA_Parking
## Installez les dépendances nécessaires:
pip install -r requirements.txt
## Lancez l'assistance au stationnement:
streamlit run app.py
