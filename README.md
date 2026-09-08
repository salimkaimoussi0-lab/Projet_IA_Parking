# 🚗 Projet_IA_Parking : Système de Détection de Véhicules et Piétons

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

