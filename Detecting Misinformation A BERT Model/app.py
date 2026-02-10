import tkinter as tk
from tkinter import ttk, messagebox
from PIL import ImageTk, Image
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification

class ModernFakeNewsDetector:
    def __init__(self, master):
        self.master = master
        master.title("News Authenticity Analyzer")
        master.geometry("800x600")
        master.resizable(False, False)

        # Setup background image
        self.setup_background("background.jpg")

        # Main content frame - Fixed size
        self.main_frame = tk.Frame(master, bg='white', bd=0, relief='flat', width=600, height=400)
        self.main_frame.place(relx=0.5, rely=0.5, anchor='center')
        self.main_frame.pack_propagate(False)  # Prevent child widgets from resizing it

        # UI Elements
        self.create_header()
        self.create_input_section()
        self.create_result_section()

        # Load model
        self.device = torch.device("cpu")
        self.load_model()

    def setup_background(self, image_path):
        if not os.path.exists(image_path):
            messagebox.showwarning("Image Error", "Background image not found!")
            return
        try:
            self.bg_image = Image.open(image_path)
            self.bg_photo = ImageTk.PhotoImage(self.bg_image.resize((800, 600)))
            self.background_label = tk.Label(self.master, image=self.bg_photo)
            self.background_label.place(x=0, y=0, relwidth=1, relheight=1)
        except Exception as e:
            messagebox.showwarning("Image Error", f"Failed to load background: {str(e)}")

    def create_header(self):
        header_frame = tk.Frame(self.main_frame, bg='white')
        header_frame.pack(pady=10)
        
        self.title_label = ttk.Label(header_frame, text="News Authenticity Analyzer",
                                     font=('Arial', 20, 'bold'), foreground='#2c3e50', background='white')
        self.title_label.pack()

        self.subtitle_label = ttk.Label(header_frame, text="Detect Misinformation in News Headlines",
                                        font=('Arial', 14), foreground='#7f8c8d', background='white')
        self.subtitle_label.pack(pady=3)

    def create_input_section(self):
        input_frame = tk.Frame(self.main_frame, bg='white')
        input_frame.pack(pady=20)

        self.input_entry = ttk.Entry(input_frame, width=50, font=('Arial', 12))
        self.input_entry.pack(pady=5, ipady=6)
        self.input_entry.bind("<Return>", lambda event: self.detect_fake_news())

        self.detect_button = ttk.Button(input_frame, text="Analyze →",
                                        command=self.detect_fake_news)
        self.detect_button.pack(pady=10, anchor='center')

    def create_result_section(self):
        self.result_frame = tk.Frame(self.main_frame, bg='white')
        self.result_frame.pack()

        self.headline_label = ttk.Label(self.result_frame, text="",
                                        font=('Arial', 14, 'italic'),
                                        wraplength=550, background='white', foreground='#34495e')
        self.headline_label.pack(pady=(0, 3))

        self.result_label = ttk.Label(self.result_frame, text="",
                                      font=('Arial', 18, 'bold'),
                                      background='white')
        self.result_label.pack(pady=(3, 10))

    def load_model(self):
        model_path = "model/"  # Ensure this folder exists
        if not os.path.exists(model_path):
            messagebox.showerror("Model Error", "Model folder not found! Check path.")
            self.master.destroy()
            return

        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            self.model = BertForSequenceClassification.from_pretrained(
                model_path, num_labels=2, output_attentions=False, output_hidden_states=False
            )
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load model: {str(e)}")
            self.master.destroy()

    def detect_fake_news(self):
        title = self.input_entry.get().strip()
        
        if not self.is_valid_headline(title):
            messagebox.showwarning("Invalid Input", "Please enter a valid news headline.")
            return

        try:
            inputs = self.tokenizer.encode_plus(
                title, add_special_tokens=True, max_length=64,
                truncation=True, padding='max_length', return_tensors='pt'
            )

            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs['input_ids'].to(self.device),
                    attention_mask=inputs['attention_mask'].to(self.device)
                )

            prediction = torch.argmax(outputs.logits, dim=1).item()
            self.show_result(title, "Real News ✅" if prediction == 0 else "Fake News ❌")

        except Exception as e:
            messagebox.showerror("Analysis Error", f"Processing failed: {str(e)}")

    def show_result(self, title, result):
        color = '#2ecc71' if 'Real' in result else '#e74c3c'
        self.result_label.config(text=result, foreground=color)
        self.headline_label.config(text=f"Headline:\n{title}",justify='center', anchor='center')
        self.input_entry.delete(0, tk.END)
        
    def is_valid_headline(self, title):
        return bool(title) and len(title) < 200 and len(title.split()) >= 3 and not any(c in title for c in ['<', '>', '{', '}', ';'])

def main():
    root = tk.Tk()
    app = ModernFakeNewsDetector(root)
    root.mainloop()

if __name__ == "__main__":
    main()
