import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

MODEL_PATH = "model_keras.keras"
model = load_model(MODEL_PATH)

def preprocess_image(image_path):
    image = load_img(image_path, color_mode="rgb", target_size=(150, 150))
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    print(f"Image shape after preprocessing: {image.shape}")
    return image

def predict_image():
    global img_label

    file_path = filedialog.askopenfilename(filetypes=[("PNG Files", "*.png")])
    if not file_path:
        return

    img = Image.open(file_path)
    img.thumbnail((200, 200))
    img_tk = ImageTk.PhotoImage(img)
    img_label.config(image=img_tk)
    img_label.image = img_tk

    preprocessed_image = preprocess_image(file_path)

    try:
        prediction = model.predict(preprocessed_image)
        result_text = f"Prediction: {'Pass' if prediction[0][0] >= 0.5 else 'Fail'}"
    except ValueError as e:
        result_text = f"Error: {str(e)}"

    result_label.config(text=result_text)


app = tk.Tk()
app.title("Image Classifier of BGA")
app.geometry("400x400")

img_label = Label(app)
img_label.pack(pady=10)

result_label = Label(app, text="No image", font=("Arial", 14))
result_label.pack(pady=10)

upload_button = Button(app, text="Upload Image", command=predict_image)
upload_button.pack(pady=20)

app.mainloop()
