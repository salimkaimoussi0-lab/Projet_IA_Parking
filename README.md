# 🚗 Projet_IA_Parking : Système de Détection de Véhicules et Piétons

<p align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-111F68?style=for-the-badge&logo=yolo&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-007ACC?style=for-the-badge&logo=opencv&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)

</p>

Ce projet implémente un système de vision par ordinateur basé sur le Deep Learning pour détecter et classifier des objets dans un environnement routier et de parking. 

L'objectif principal est de comparer deux architectures d'intelligence artificielle de pointe (**YOLOv8 Nano** et **YOLOv8 Small**) sur une base de données complexe contenant 13 classes différentes (Voitures, Camions, Motos, Piétons, Panneaux de signalisation, etc.).

---

## 🔬 Modèles et Méthodologie

Deux modèles ont été entraînés et mis en compétition :
1. **YOLOv8n (Nano) :** Modèle ultra-léger (3 millions de paramètres), optimisé pour la vitesse.
2. **YOLOv8s (Small) :** Modèle plus profond (11.1 millions de paramètres), optimisé pour l'extraction de caractéristiques complexes.

### Hyperparamètres d'entraînement

Afin d'assurer une comparaison rigoureuse, les deux modèles ont partagé la configuration exacte suivante :

* **Taille des images (imgsz) :** 640x640 pixels
* **Époques (Epochs) :** 50 (avec mécanisme d'*Early Stopping*)
* **Taille du lot (Batch size) :** 16
* **Optimiseur :** AdamW (Learning Rate initial ~0.0005)

### Stratégie d'Augmentation de Données (Data Augmentation)

Pour éviter le sur-apprentissage (overfitting) et rendre le modèle robuste aux conditions réelles, nous avons appliqué :

* **Augmentations géométriques :** Mosaic (100%), Flip Horizontal (50%), Zoom et Translation aléatoires.
* **Augmentations photométriques :** Variations HSV (Luminosité/Teinte/Saturation), ajout de flou léger (Blur 1%), conversion en niveaux de gris (ToGray 1%) et ajustement de contraste adaptatif (CLAHE 1%) pour simuler des conditions météorologiques et d'éclairage variables (nuit, pluie, éblouissement).

---

## 📊 Résultats et Comparaison

Les entraînements ont révélé une supériorité écrasante du modèle **YOLOv8s**, capable de mieux appréhender la complexité des 13 classes.

| Modèle | mAP@0.5 (Global) | Précision (P) | Rappel (R) | Détection Voitures (mAP) |
| :--- | :---: | :---: | :---: | :---: |
| **YOLOv8n (Nano)** | 29.4 % | 78.9 % | 27.9 % | 93.8 % |
| **YOLOv8s (Small)** | **77.2 %** | **80.1 %** | **68.8 %** | **97.1 %** |

**Analyse des performances :**

* Le modèle **YOLOv8s** a obtenu un score mAP global de **77.2%**, surpassant largement le Nano (29.4%).
* Sur les classes complexes comme les "Camions", le Small atteint une précision de **76.5%** contre seulement 52.1% pour le Nano.
* Le Nano a totalement échoué à détecter les panneaux de circulation (mAP de 0%), tandis que le Small les détecte avec une excellente fiabilité (**81.0%**).

---

*Projet réalisé sur Google Colab avec le framework Ultralytics YOLOv8.*

---

# 🚀 Extension du Projet : Assistant Intelligent de Parking

Après la comparaison des modèles YOLOv8 Nano et YOLOv8 Small, le projet a été étendu afin de transformer le système de détection en une **application complète d'assistance intelligente au stationnement**.

Le modèle **YOLOv8 Small (`yolov8s_50.pt`)** a été retenu comme modèle principal grâce à ses performances supérieures.

Le système combine désormais :

- **YOLOv8 Small** pour la perception visuelle ;
- **OpenCV** pour le traitement et l'annotation des images ;
- **Python** pour les calculs et la logique de sécurité ;
- **Groq / LLM** pour l'analyse intelligente de la scène ;
- **Streamlit** pour l'interface utilisateur interactive.

---

## 🧠 Architecture du système final

Le fonctionnement général de l'application est le suivant :

```text
Image de route / parking
          │
          ▼
   YOLOv8 Small
          │
          ▼
Détection des objets
          │
          ▼
Classes + Bounding Boxes
+ Scores de confiance
          │
          ▼
Analyse géométrique Python
          │
          ▼
Estimation des distances
          │
          ▼
Moteur de risque déterministe
          │
          ▼
Agent IA via Groq
          │
          ▼
Diagnostic et recommandation
          │
          ▼
Interface Streamlit
```

---

## 🔎 Amélioration de la détection YOLOv8

L'application finale utilise le modèle :

```text
yolov8s_50.pt
```

L'inférence YOLOv8 a été adaptée afin d'améliorer l'analyse des scènes contenant plusieurs objets ou des éléments de petite taille.

La configuration permet notamment d'utiliser :

- une résolution d'inférence pouvant atteindre **960 pixels** ;
- un seuil de confiance réglable ;
- un seuil IoU adapté ;
- jusqu'à plusieurs centaines de détections ;
- le **Test-Time Augmentation (TTA)** en option.

L'utilisateur peut modifier certains de ces paramètres directement depuis l'interface Streamlit.

---

## 🏷️ Normalisation des classes détectées

Les noms des classes provenant du dataset sont normalisés afin de garantir une analyse cohérente.

Exemples :

```text
Voitures / voiture / car
→ voiture

Camions / camion / truck
→ camion

pietons / pieton / person
→ pieton

moto / motorcycle
→ moto

velo / bicycle
→ velo
```

Cette normalisation permet au moteur d'analyse d'utiliser la même logique quelle que soit la manière dont la classe est nommée par le modèle.

---

## 📐 Estimation approximative des distances

Le projet intègre également une estimation de la distance entre la caméra et certains objets détectés.

L'estimation repose notamment sur :

- la taille de la bounding box ;
- la résolution de l'image ;
- la catégorie de l'objet ;
- une taille physique approximative de l'objet.

Le principe général utilisé est :

```text
Distance ≈ Taille réelle × Focale / Taille apparente
```

Les distances affichées avec le symbole `~` sont donc des **estimations approximatives**.

Exemple :

```text
Voiture | Confiance : 0.91 | Distance : ~5.2 m
```

---

## 🛡️ Moteur de risque déterministe

Le niveau de risque n'est plus décidé uniquement par le LLM.

Une logique Python analyse d'abord les informations obtenues par YOLO afin de déterminer un niveau de risque cohérent.

Les niveaux possibles sont :

- 🟢 **Faible**
- 🟠 **Moyen**
- 🟠 **Élevé**
- 🔴 **Critique**
- ⚪ **Indéterminé**

Exemples de règles utilisées :

```text
Piéton très proche
→ Critique

Piéton à proximité
→ Élevé

Obstacle extrêmement proche
→ Critique

Obstacle proche
→ Moyen

Aucun danger immédiat détecté
→ Faible

Informations insuffisantes
→ Indéterminé
```

Cette architecture permet d'obtenir une décision plus stable et plus prévisible.

---

## 🤖 Agent IA et Groq

Un Agent LLM est intégré au système afin de transformer les résultats techniques en un diagnostic compréhensible.

Le modèle actuellement utilisé via l'API Groq est :

```text
openai/gpt-oss-20b
```

L'Agent IA reçoit les informations issues de YOLO et du moteur Python afin de générer :

- une analyse de la scène ;
- une évaluation technique ;
- un niveau de risque ;
- une recommandation de conduite.

---

## 🔧 Function Calling

Le projet utilise également le **Function Calling**.

L'Agent IA peut exploiter la fonction :

```python
evaluateur_global_parking
```

Le fonctionnement devient donc :

```text
YOLOv8
   │
   ▼
Analyse Python
   │
   ▼
Moteur de sécurité
   │
   ▼
Function Calling
   │
   ▼
Agent LLM
   │
   ▼
Rapport final
```

---

## 🧯 Gestion des erreurs de l'Agent IA

La gestion des erreurs de l'API Groq a également été améliorée.

Une erreur du LLM n'est plus interprétée automatiquement comme un risque critique.

Le système fonctionne désormais selon cette logique :

```text
YOLO fonctionne
      │
      ▼
Python analyse la scène
      │
      ▼
Calcul local du risque
      │
      ├── Groq disponible
      │       │
      │       ▼
      │   Rapport LLM
      │
      └── Groq indisponible
              │
              ▼
        Rapport local
```

Ainsi, même en cas de problème avec Groq, le module de vision et le moteur de risque Python continuent de fonctionner.

---

## 🖥️ Interface Streamlit

L'application possède désormais un tableau de bord interactif développé avec **Streamlit**.

L'interface affiche deux parties principales.

### 📷 Flux Vidéo — Vision Module

Cette partie permet d'afficher :

- l'image analysée ;
- les bounding boxes ;
- les classes détectées ;
- les scores de confiance ;
- les distances approximatives.

### 🧠 Diagnostic Agent IA

Cette partie affiche :

- le niveau de risque ;
- l'analyse spatiale détaillée ;
- l'évaluation technique ;
- la recommandation de conduite.

Un tableau supplémentaire permet également de consulter précisément les objets détectés.

---

## 🛠️ Technologies utilisées

<p align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-111F68?style=for-the-badge&logo=yolo&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-007ACC?style=for-the-badge&logo=opencv&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)

</p>

### Principales technologies

- 🔵 **Python** : logique principale du projet.
- 🔴 **Streamlit** : interface web interactive.
- 🔵 **OpenCV** : traitement et annotation des images.
- 🔵 **Ultralytics YOLOv8** : détection des objets.
- 🧠 **Groq** : exécution de l'Agent LLM.
- 🔵 **NumPy** : manipulation des données et images.
- 🟠 **Google Colab** : entraînement des modèles YOLO.

---

## 📁 Fichiers principaux

```text
Projet_IA_Parking/
│
├── app.py
├── agent_parking.py
├── requirements.txt
├── README.md
├── yolov8n_50.pt
├── yolov8s_50.pt
├── YOLOv8_Parking.ipynb
└── ...
```

---

# ▶️ Tester le projet

Cloner le dépôt :

```bash
git clone https://github.com/salimkaimoussi0-lab/Projet_IA_Parking.git
```

Entrer dans le dossier :

```bash
cd Projet_IA_Parking
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

Puis lancer l'application :

```bash
streamlit run app.py
```

Si la commande `streamlit` n'est pas reconnue :

```bash
python -m streamlit run app.py
```

L'application est ensuite accessible dans le navigateur à l'adresse :

```text
http://localhost:8501
```

Une clé API Groq peut être renseignée directement dans le menu latéral de l'application.

---

## 📦 Dépendances principales

```text
streamlit
ultralytics
groq
opencv-python-headless
pillow
numpy
```

---

## ⚠️ Limites

Ce projet constitue un **démonstrateur académique d'Intelligence Artificielle**.

Les distances calculées avec une seule image provenant d'une caméra RGB sont approximatives.

Une application automobile réelle nécessiterait notamment :

- une calibration précise de la caméra ;
- une caméra stéréo ou de profondeur ;
- un LiDAR ;
- un radar ;
- des capteurs ultrasons ;
- des mécanismes de sécurité redondants.

Le système présenté ici démontre principalement l'intégration entre :

```text
Vision par ordinateur
+
Analyse géométrique
+
Moteur de sécurité
+
Agent LLM
+
Interface Streamlit
```

---

## 🏁 Conclusion

Les expérimentations montrent que **YOLOv8 Small** est nettement plus adapté que YOLOv8 Nano pour l'analyse de scènes routières complexes.

Le projet a ensuite évolué d'un simple système de détection vers un démonstrateur complet combinant :

- la perception visuelle avec YOLOv8 ;
- l'analyse spatiale ;
- l'estimation approximative des distances ;
- un moteur de risque déterministe ;
- un Agent LLM ;
- le Function Calling ;
- une interface Streamlit interactive.

L'architecture finale permet ainsi de passer progressivement de la **perception de l'environnement** à une **analyse intelligente et explicable de la situation**.

---

### 👨‍💻 Auteur

**Salim Kaimoussi**

Projet académique réalisé autour de la **Vision par Ordinateur, du Deep Learning et des Agents IA**.
