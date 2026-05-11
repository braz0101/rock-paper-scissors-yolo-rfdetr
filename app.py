from ultralytics import YOLO
import gradio as gr

# Chargement du modèle ONNX (inférence CPU optimisée)
model = YOLO("best.onnx", task="detect")


def predict_rps(image):
    """
    Reçoit une image (webcam), applique YOLOv11n,
    retourne l'image annotée avec les détections.
    """
    if image is None:
        return None

    results = model.predict(
        image,
        conf=0.4,       # Seuil de confiance
        verbose=False,
        imgsz=480,      # Résolution réduite → inférence plus rapide sur CPU
        max_det=5       # Nombre maximum de détections par image
    )[0]

    return results.plot()


demo = gr.Interface(
    fn=predict_rps,
    inputs=gr.Image(
        sources=["webcam"],
        streaming=True,
        label="Montrez votre main (Pierre / Papier / Ciseaux)"
    ),
    outputs=gr.Image(label="Résultat Détection"),
    title="Rock Paper Scissors - Détection en Temps Réel",
    description="Modèle : YOLOv11n | Déployé sur Hugging Face Spaces",
    live=True,
    cache_examples=False
)

demo.launch()
