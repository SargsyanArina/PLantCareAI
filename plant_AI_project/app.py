from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

app = Flask(__name__)

# ===== MODEL =====
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


# ===== LOAD MODEL =====
checkpoint = torch.load("plant_disease_model.pth", map_location="cpu")

classes = checkpoint["classes"]
num_classes = len(classes)

model = PlantCNN(num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# ===== TRANSFORM =====
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# ===== ADVICE (short demo) =====
disease_info = {
    "Tomato___Late_blight": "Fungicide օգտագործիր և նվազեցրու խոնավությունը",
    "Tomato___healthy": "Բույսը առողջ է 🌿"
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]

    image = Image.open(file).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    class_name = classes[pred.item()]
    confidence = round(conf.item()*100, 2)

    advice = disease_info.get(class_name, "Խորհուրդ")

    return jsonify({
        "class": class_name,
        "confidence": confidence,
        "advice": advice
    })

if __name__ == "__main__":
    print("🚀 Server starting...")
    app.run(debug=True)