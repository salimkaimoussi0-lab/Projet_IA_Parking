# 🚗 Projet_IA_Parking : Système Complet d'Assistance au Stationnement Autonome

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-yellow.svg)
![Groq](https://img.shields.io/badge/LLM-Groq_Llama_3-orange.svg)
![Streamlit](https://img.shields.io/badge/Interface-Streamlit-red.svg)

Ce projet implémente un système de vision par ordinateur couplé à une Intelligence Artificielle générative pour créer un assistant de stationnement autonome. Le système détecte les objets dans un environnement routier et génère des recommandations de conduite sécuritaires en temps réel.

---

## 📁 Contenu du Dépôt
* `YOLOv8_Parking.ipynb` : Notebook d'entraînement des modèles de vision.
* `Agent-LLM.py` : Script de l'Agent IA (Cerveau) avec appels de fonctions.
* `app.py` : Interface Web interactive développée avec Streamlit.
* `yolov8n_50.pt` / `yolov8s_50.pt` : Poids des modèles entraînés.
* `PROJET--IA.pdf` : Rapport complet du projet.
* Dossiers d'images (Courbes et Validations).

---

## 🏗️ Architecture du Projet
1. **Module A (Vision - YOLOv8)** : Détection de 13 classes (Voitures, Piétons, Panneaux...) avec comparaison entre une architecture Nano et Small sur 50 epochs.
2. **Module B (Agent - Llama 3)** : Analyse géométrique et contextuelle. L'Agent évalue le risque (Faible à Critique) via un outil mathématique strict (distances d'arrêt, largeur de place).
3. **Module C (Interface - Streamlit)** : Pipeline de bout en bout permettant d'uploader une image Dashcam et d'obtenir le rapport de bord instantané.

---

## 📊 Résultats (YOLOv8 Nano vs Small)
Le modèle **YOLOv8s** a démontré une supériorité écrasante pour la compréhension de la scène.

| Modèle | mAP@0.5 (Global) | Précision (P) | Rappel (R) | Détection Voitures |
| :--- | :---: | :---: | :---: | :---: |
| **YOLOv8n (Nano)** | 29.4 % | 78.9 % | 27.9 % | 93.8 % |
| **YOLOv8s (Small)** | **77.2 %** | **80.1 %** | **68.8 %** | **97.1 %** |

*Note : Le Nano a totalement échoué sur les panneaux (mAP 0%), tandis que le Small offre une fiabilité de 81%. Le YOLOv8s est donc le modèle retenu pour l'application.*

---

## 🚀 Installation et Exécution
```bash
git clone [https://github.com/salimkaimoussi0-lab/Projet_IA_Parking.git](https://github.com/salimkaimoussi0-lab/Projet_IA_Parking.git)
cd Projet_IA_Parking
pip install -r requirements.txt