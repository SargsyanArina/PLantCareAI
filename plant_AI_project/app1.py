
from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

import numpy as np
import cv2
import base64
import torch.nn.functional as F

app = Flask(__name__)






# ======================
# MODEL
# ======================
class PlantCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*14*14,512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,num_classes)
        )

    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ======================
# LOAD MODEL
# ======================
checkpoint = torch.load("plant_disease_model1.pth", map_location="cpu")

classes = checkpoint["classes"]
num_classes = len(classes)

model = PlantCNN(num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# ======================
# TRANSFORM
# ======================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


# ======================
# GRADCAM
# ======================
class GradCAM:
    def __init__(self, model, target_layer, classes):
        self.model = model
        self.target_layer = target_layer
        self.classes = classes

        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate(self, x, class_idx=None):
        self.model.eval()

        output = self.model(x)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        grads = self.gradients[0]
        acts = self.activations[0]

        weights = grads.mean(dim=(1, 2))

        cam = torch.zeros(acts.shape[1:], device=acts.device)

        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = F.relu(cam)

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().detach().numpy(), self.classes[class_idx]


# ======================
# ROUTES
# ======================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]

    # ===== LOAD IMAGE =====
    pil_image = Image.open(file).convert("RGB")
    image = transform(pil_image).unsqueeze(0)

    # ===== MODEL =====
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)[0]

    conf, pred = torch.max(probs, 0)

    class_name = classes[pred.item()]
    confidence = round(conf.item() * 100, 2)

    # ===== INFECTION % =====
    healthy_indices = [
        i for i, cls in enumerate(classes)
        if "healthy" in cls.lower()
    ]

    healthy_prob = probs[healthy_indices].sum().item()
    infection_percent = (1 - healthy_prob) * 100

    # ===== GRADCAM =====
    image_for_cam = transform(pil_image).unsqueeze(0)

    target_layer = model.features[-2]
    cam_model = GradCAM(model, target_layer, classes)

    cam, _ = cam_model.generate(image_for_cam, pred.item())

    img_np = np.array(pil_image)

    cam = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam),
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = heatmap * 0.4 + img_np * 0.6
    overlay = np.uint8(overlay)

    # ===== DAMAGE % =====
    if np.any(cam > 0.5):
        damage_percent = float(np.mean(cam[cam > 0.5]) * 100)
    else:
        damage_percent = 0.0

    # ===== BASE64 =====
    def encode_image(img):
        _, buffer = cv2.imencode(".png", img)
        return base64.b64encode(buffer).decode("utf-8")

    heatmap_b64 = encode_image(heatmap)
    overlay_b64 = encode_image(overlay)

    # ===== RESPONSE =====
    return jsonify({
        "class": class_name,
        "confidence": confidence,
        "damage": round(damage_percent, 2),
        "heatmap": heatmap_b64,
        "overlay": overlay_b64
    })


# ======================
# RUN
# ======================
if __name__ == "__main__":
    print("🚀 Server starting...")
    app.run(host="127.0.0.1", port=5001, debug=True)

