import cv2
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 7)

model.load_state_dict(torch.load("emotion_cnn_final.pth", map_location=device))
model = model.to(device)
model.eval()

# Emotion dict (if folders are 1–7)
emotion_dict = {
    1: "Neutral",
    2: "Surprise",
    3: "Fear",
    4: "Disgust",
    5: "Happy",
    6: "Sad",
    7: "Angry",
}

# Transform
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = cv2.resize(img, (224, 224))
    img_pil = transforms.ToPILImage()(img_pil)

    input_img = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_img)
        _, predicted = torch.max(outputs, 1)
        emotion = emotion_dict[predicted.item()+1]

    cv2.putText(frame, f"Emotion: {emotion}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Live Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
