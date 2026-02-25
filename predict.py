import customtkinter as ctk
import joblib
from pathlib import Path

# --- Core Logic Setup ---
MODEL_DIR = Path("models")
DATA_DIR = Path("data")
FEEDBACK_FILE = DATA_DIR / "feedback.txt"

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)

# GUI Configuration
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class SentimentAnalyzerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI Sentiment Analyzer")
        self.geometry("650x650")
        self.minsize(550, 600)
        
        self.current_text = ""
        self.vectorizer = None
        self.model = None
        
        self.setup_ui()
        self.load_models()

    def load_models(self):
        """Loads the machine learning models with error handling."""
        try:
            self.vectorizer = joblib.load(MODEL_DIR / "vectorizer.joblib")
            self.model = joblib.load(MODEL_DIR / "sentiment_model.joblib")
            self.status_label.configure(text="✅ Models loaded successfully.", text_color="#28a745")
        except Exception as e:
            self.status_label.configure(text=f"❌ Error loading models: {e}", text_color="#dc3545")
            self.analyze_btn.configure(state="disabled")

    def setup_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # --- Header ---
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")
        
        header_label = ctk.CTkLabel(header_frame, text="Sentiment Analyzer", font=ctk.CTkFont(size=28, weight="bold"))
        header_label.pack(side="left")

        # --- SECTION 1: Input Card ---
        input_card = ctk.CTkFrame(self, corner_radius=15, fg_color=("gray90", "gray13"))
        input_card.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        input_card.grid_columnconfigure(0, weight=1)

        input_title = ctk.CTkLabel(input_card, text="Text Input", font=ctk.CTkFont(size=16, weight="bold"))
        input_title.grid(row=0, column=0, padx=20, pady=(15, 5), sticky="w")

        self.textbox = ctk.CTkTextbox(input_card, height=120, font=ctk.CTkFont(size=15), border_width=1, border_color=("gray70", "gray30"))
        self.textbox.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        self.textbox.insert("0.0", "Type or paste your sentence here...")
        self.textbox.bind("<FocusIn>", self.clear_placeholder)

        # Buttons
        btn_frame = ctk.CTkFrame(input_card, fg_color="transparent")
        btn_frame.grid(row=2, column=0, padx=20, pady=(5, 20), sticky="ew")
        
        self.analyze_btn = ctk.CTkButton(btn_frame, text="Analyze Sentiment", height=40, font=ctk.CTkFont(size=15, weight="bold"), command=self.analyze_text)
        self.analyze_btn.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        self.clear_btn = ctk.CTkButton(btn_frame, text="Clear", height=40, fg_color="gray", hover_color="#555555", command=self.clear_input)
        self.clear_btn.pack(side="right")

        # --- SECTION 2: Results & Feedback Card ---
        self.result_card = ctk.CTkFrame(self, corner_radius=15, fg_color=("gray85", "gray16"))
        self.result_card.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        self.result_card.grid_columnconfigure(0, weight=1)
        
        result_title = ctk.CTkLabel(self.result_card, text="Analysis Result", font=ctk.CTkFont(size=14), text_color="gray")
        result_title.grid(row=0, column=0, pady=(15, 0))

        self.result_label = ctk.CTkLabel(self.result_card, text="Waiting for input...", font=ctk.CTkFont(size=20))
        self.result_label.grid(row=1, column=0, pady=(10, 20))

        # Feedback Sub-section (Hidden initially)
        self.feedback_frame = ctk.CTkFrame(self.result_card, fg_color="transparent")
        self.feedback_frame.grid(row=2, column=0, padx=20, pady=(0, 15), sticky="ew")
        
        feedback_divider = ctk.CTkFrame(self.feedback_frame, height=2, fg_color=("gray75", "gray30"))
        feedback_divider.pack(fill="x", pady=(0, 15))

        feedback_title = ctk.CTkLabel(self.feedback_frame, text="Was this correct? (Help improve the model)", font=ctk.CTkFont(size=13, slant="italic"))
        feedback_title.pack(pady=(0, 10))

        fb_btn_frame = ctk.CTkFrame(self.feedback_frame, fg_color="transparent")
        fb_btn_frame.pack()

        ctk.CTkButton(fb_btn_frame, text="😊 Positive", width=100, fg_color="#28a745", hover_color="#218838", command=lambda: self.save_feedback("positive")).pack(side="left", padx=5)
        ctk.CTkButton(fb_btn_frame, text="😞 Negative", width=100, fg_color="#dc3545", hover_color="#c82333", command=lambda: self.save_feedback("negative")).pack(side="left", padx=5)
        ctk.CTkButton(fb_btn_frame, text="😐 Mixed", width=100, fg_color="#ffc107", text_color="black", hover_color="#e0a800", command=lambda: self.save_feedback("mixed")).pack(side="left", padx=5)
        ctk.CTkButton(fb_btn_frame, text="Skip", width=80, fg_color="transparent", border_width=1, text_color=("black", "white"), command=self.hide_feedback).pack(side="left", padx=(15, 5))

        self.feedback_frame.grid_remove()

        # --- Status Bar ---
        self.status_label = ctk.CTkLabel(self, text="Ready.", font=ctk.CTkFont(size=12))
        self.status_label.grid(row=3, column=0, padx=20, pady=(0, 15), sticky="w")

    def clear_placeholder(self, event):
        if self.textbox.get("0.0", "end-1c") == "Type or paste your sentence here...":
            self.textbox.delete("0.0", "end")

    def clear_input(self):
        self.textbox.delete("0.0", "end")
        self.result_label.configure(text="Waiting for input...", text_color=("black", "white"))
        self.hide_feedback()
        self.status_label.configure(text="Ready.", text_color=("black", "white"))

    def get_emoji_and_color(self, label):
        """Returns the emoji text and the specific color for that sentiment."""
        return {
            "positive": ("😊 Positive", "#28a745"),  # Green
            "negative": ("😞 Negative", "#dc3545"),  # Red
            "mixed": ("😐 Mixed", "#ffc107")         # Yellow
        }.get(label, ("❓ Unknown", ("black", "white")))

    def analyze_text(self):
        self.current_text = self.textbox.get("0.0", "end-1c").strip()
        
        if not self.current_text or self.current_text == "Type or paste your sentence here...":
            self.status_label.configure(text="⚠️ Please enter valid text.", text_color="#ffc107")
            return

        if self.model is None or self.vectorizer is None:
            self.status_label.configure(text="❌ Models not loaded. Cannot analyze.", text_color="#dc3545")
            return

        # --- Core Logic Execution ---
        try:
            vectorized_text = self.vectorizer.transform([self.current_text])
            pred = self.model.predict(vectorized_text)[0]
            
            # Update UI with colors based on prediction
            result_text, color = self.get_emoji_and_color(pred)
            self.result_label.configure(text=result_text, text_color=color, font=ctk.CTkFont(size=28, weight="bold"))
            
            self.feedback_frame.grid()
            self.status_label.configure(text="Analysis complete. Waiting for feedback...", text_color=("black", "white"))
            
        except Exception as e:
            self.status_label.configure(text=f"❌ Error during prediction: {e}", text_color="#dc3545")

    def save_feedback(self, label):
        if not self.current_text:
            return

        try:
            with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
                f.write(f"{label}|{self.current_text}\n")
            
            self.status_label.configure(text="✅ Feedback saved successfully.", text_color="#28a745")
            self.hide_feedback()
            
        except Exception as e:
            self.status_label.configure(text=f"❌ Failed to save feedback: {e}", text_color="#dc3545")

    def hide_feedback(self):
        self.feedback_frame.grid_remove()

if __name__ == "__main__":
    app = SentimentAnalyzerApp()
    app.mainloop()