# 🔐 Audio–Visual CAPTCHA Verification

## 📌 Overview
This project explores **multimodal CAPTCHA verification** by combining **automatic speech recognition (ASR)** and **optical character recognition (OCR)**.  

The idea:  
- **Audio CAPTCHA** → transcribed using **Wav2Vec2**  
- **Image CAPTCHA** → text extracted via **OCR (Tesseract / CRNN)**  
- **Similarity Check** → predictions compared against ground-truth text with Levenshtein ratio  

This shows how modern **speech + vision models** can be applied to security challenges and why CAPTCHA design needs to evolve against AI attacks.  

---

## ⚙️ Tech Stack
- **Python, PyTorch, Hugging Face Transformers**
- **Wav2Vec2** – pre-trained ASR model, fine-tuned for audio captchas  
- **Tesseract OCR** (or CRNN) – for image text extraction  
- **Librosa, SoundFile** – audio preprocessing  
- **OpenCV, Pillow** – image preprocessing  
- **Levenshtein** – accuracy metrics (character/word-level)  

---

## 📂 Dataset
- **Synthetic Generator**: Captcha images & audio generated with [`captcha`](https://pypi.org/project/captcha/) library → provides **paired audio+image captchas**  
- **Kaggle Audio Captchas**: [Text and Audio Captchas dataset](https://www.kaggle.com/datasets/mhassansaboor/text-and-audio-captchas) – used for fine-tuning Wav2Vec2 on real audio captcha data  

---

## 🚀 Features
- ✅ **Synthetic Data Generation** (custom pairs with ground truth)  
- ✅ **Audio Transcription** with **Wav2Vec2**  
- ✅ **Image OCR** with **Tesseract** (option to extend with CRNN)  
- ✅ **Fine-tuning on Kaggle Audio Captchas** for higher accuracy  
- ✅ **Character-level evaluation** with Levenshtein ratio  
- ✅ **Colab Notebook for Training + Demo**  

---

## 📊 Results (Current)
- Pre-trained Wav2Vec2 → ~6% accuracy on synthetic audio captchas  
- After fine-tuning (small synthetic set) → ~30%  
- With larger Kaggle dataset →  **78%+**  



---

## 🏃‍♂️ How to Run

### 1. Clone repo & install dependencies
```bash
git clone https://github.com/<your-username>/audio-visual-captcha.git
cd audio-visual-captcha
pip install -r requirements.txt
