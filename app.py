import tkinter as tk
import customtkinter as ctk
from PIL import ImageTk, Image
from authtoken import auth_token
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
import threading

# Create the app
app = tk.Tk()
app.geometry("532x632")
app.title("Stable Bud")
ctk.set_appearance_mode("dark")

# Create the entry widget
prompt = ctk.CTkEntry(app, height=40, width=512, text_color="black", fg_color="white")
prompt.place(x=10, y=10)

# Create a label widget with app as master
lmain = ctk.CTkLabel(app, height=512, width=512)
lmain.place(x=10, y=110)

modelid = "CompVis/stable-diffusion-v1-4"
device = "cpu"  # Use CPU

try:
    pipe = StableDiffusionPipeline.from_pretrained(modelid, use_auth_token=auth_token)
    pipe.to(device)
except Exception as e:
    print(f"Error occurred: {e}")

def generate_image():
    try:
        input_text = prompt.get()
        with autocast(device):
            image = pipe(input_text, guidance_scale=8.5)["sample"][0]

        # Convert PIL image to ImageTk format for displaying in Tkinter Label
        img = Image.fromarray(image)
        img = img.resize((512, 512), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)

        # Update the label with the generated image
        lmain.configure(image=img)
        lmain.image = img  # Keep a reference to avoid garbage collection

    except Exception as e:
        print(f"Error occurred: {e}")

def generate():
    threading.Thread(target=generate_image).start()

# Create the "Generate" button
trigger = ctk.CTkButton(app, height=40, width=120, text="Generate", text_color="white", fg_color="blue", command=generate)
trigger.place(x=206, y=60)

app.mainloop()
