import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
import threading
import time

# ------------ MODEL DEFINITION ------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ------------ LOAD CLASSES ------------
train_dir = r"D:\Project\WildlifeMonitoring\animal-detection\train"
classes = os.listdir(train_dir)
num_classes = len(classes)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------ LOAD MODEL ------------
model = SimpleCNN(num_classes)
model.load_state_dict(torch.load("wildlife_cnn.pth", map_location=device))
model.to(device)
model.eval()

# ------------ TRANSFORM ------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ------------ TKINTER UI (LIGHT THEME) ------------
root = tk.Tk()
root.title("Wildlife Classifier (PyTorch)")
root.geometry("520x650")
root.configure(bg="#f2f2f2")   # Light background

# Title
title_label = tk.Label(root, text="🐾 Wildlife Species Classifier",
                       font=("Arial", 20, "bold"),
                       bg="#f2f2f2", fg="#333333")
title_label.pack(pady=15)

# Frame for image
frame = tk.Frame(root, bg="#ffffff", bd=2, relief="solid")
frame.pack(pady=10)

img_label = tk.Label(frame, bg="#ffffff")
img_label.pack()

# Prediction labels
status_label = tk.Label(root, text="Upload an image to start",
                        font=("Arial", 12),
                        bg="#f2f2f2", fg="#444444")
status_label.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 16, "bold"),
                        bg="#f2f2f2", fg="#007acc")
result_label.pack(pady=10)

confidence_label = tk.Label(root, text="", font=("Arial", 14),
                            bg="#f2f2f2", fg="#333333")
confidence_label.pack(pady=5)


# --- BUTTON STYLE (LIGHT THEME) ---
def create_button(text, command):
    return tk.Button(root,
                     text=text,
                     command=command,
                     font=("Arial", 14, "bold"),
                     fg="white",
                     bg="#007acc",
                     activebackground="#005f99",
                     relief="flat",
                     width=20,
                     height=1)


# ------------ IMAGE SELECTION FUNCTION ------------
def choose_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        return

    # Show status
    status_label.config(text="Processing image...", fg="#cc7a00")
    result_label.config(text="")
    confidence_label.config(text="")
    root.update_idletasks()

    # Load image
    img = Image.open(file_path).convert("RGB")
    img_resized = img.resize((300, 300))
    img_tk = ImageTk.PhotoImage(img_resized)

    img_label.config(image=img_tk)
    img_label.image = img_tk

    # Predict in background thread
    threading.Thread(target=predict_image, args=(img,)).start()


# ------------ PREDICTION FUNCTION ------------
def predict_image(img):

    time.sleep(0.4)  # small delay for UI smoothness

    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        conf, pred = torch.max(probabilities, 0)

    # Update UI
    status_label.config(text="Prediction completed!", fg="#2d7d46")
    result_label.config(text=f"Predicted Species: {classes[pred.item()]}")
    confidence_label.config(text=f"Confidence: {conf.item() * 100:.2f}%")



# Upload Button
upload_btn = create_button("📂 Select Image", choose_image)
upload_btn.pack(pady=20)

root.mainloop()
# Start the Tkinter event loop