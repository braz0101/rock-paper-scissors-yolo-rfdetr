# 🎮 Rock Paper Scissors — Fine-tuning YOLOv11n vs RF-DETR

**Auteurs :** Ibrahima Fall | Souaibou Dieng  
**Filière :** Master 2 SRT — ESP Dakar (UCAD)  
**Module :** Deep Learning / Traitement d'Images — Dr. Keita  
**Année universitaire :** 2025 – 2026

---

## Description

Ce projet compare deux architectures de détection d'objets — **YOLOv11n** (CNN) et **RF-DETR** (Transformer) — entraînées par fine-tuning sur le dataset **Rock Paper Scissors** (Pierre-Papier-Ciseaux) de Roboflow.

Le pipeline complet couvre : préparation du dataset → fine-tuning → évaluation → déploiement temps réel via Gradio et Hugging Face Spaces.

🚀 **Démo live :** [braz01-yolo-rps-detection.hf.space](https://braz01-yolo-rps-detection.hf.space)

---

## Structure du dépôt

```
rock-paper-scissors-yolo-rfdetr/
├── notebook.ipynb              # Pipeline complet (Kaggle, GPU Tesla T4)
├── app.py                      # Application Gradio (Hugging Face Spaces)
├── requirements.txt            # Dépendances Python
├── comparaison_finale.csv      # Résultats comparatifs des deux modèles
├── best.onnx                   # Modèle YOLOv11n exporté (ONNX)
├── yolo_best.pt                # Poids PyTorch du meilleur modèle YOLO
├── assets/
│   ├── image.png               # Résultats d'entraînement YOLO
│   ├── image1.png              # Résultats RF-DETR
│   └── labels.jpg              # Distribution des classes du dataset
└── .gitignore
```

---

## Résultats : YOLOv11n vs RF-DETR

| Critère | YOLOv11n ✅ | RF-DETR Base |
|---|---|---|
| **mAP50** | **0.892** | 0.602 |
| **mAP50-95** | 0.555 | 0.602 |
| **Précision** | **0.964** | 0.867 |
| **Rappel** | 0.799 | **0.964** |
| **Vitesse d'inférence** | **~14.5 ms** | ~80-120 ms |
| **Paramètres** | **2.58 M** | 31.9 M |
| **Taille du modèle** | **5.5 MB** | 120 MB |
| **Temps d'entraînement** | ~20 min | ~11 min (6 époques) |

**Conclusion :** YOLOv11n domine sur toutes les dimensions critiques pour ce dataset de petite taille (~237 images). RF-DETR nécessite beaucoup plus de données pour converger.

---

## Technologies utilisées

| Outil | Version | Rôle |
|---|---|---|
| Ultralytics YOLO | 8.x | Fine-tuning YOLOv11n |
| RF-DETR (Roboflow) | — | Fine-tuning RF-DETR Base |
| Kaggle (GPU Tesla T4) | — | Environnement d'entraînement |
| ONNX / TorchScript | — | Export pour déploiement |
| Gradio | — | Interface web temps réel |
| Hugging Face Spaces | — | Hébergement cloud |

---

## Lancer le notebook (Kaggle)

1. Importer `notebook.ipynb` dans [kaggle.com](https://www.kaggle.com)
2. Activer l'accélérateur GPU (Settings → Accelerator → **NVIDIA Tesla T4**)
3. Activer Internet (Settings → Internet → On)
4. Exécuter toutes les cellules dans l'ordre

Le notebook effectue automatiquement :
- L'installation des dépendances
- Le téléchargement du dataset via l'API Roboflow
- Le fine-tuning de YOLOv11n (25 époques) et RF-DETR (6 époques)
- La comparaison des métriques
- Le lancement de la démo Gradio temps réel

---

## Déploiement local

```bash
pip install ultralytics gradio onnxruntime opencv-python-headless

python app.py
```

---

## Dataset

- **Source :** [Roboflow Universe — Rock Paper Scissors](https://universe.roboflow.com/hihi159753/rock-paper-scissors-hugue)
- **Classes :** 3 (rock, paper, scissors)
- **Nombre d'images :** ~237
- **Format YOLO :** YOLOv8 (boîtes englobantes)
- **Format RF-DETR :** COCO JSON

---

## Pipeline MLOps

```
Dataset Roboflow → Formatage (YOLO/COCO) → Fine-Tuning YOLOv11n
                                          → Fine-Tuning RF-DETR
                                                    ↓
                                           Évaluation (mAP, FPS)
                                                    ↓
                                        Déploiement Gradio / HF Spaces
```
