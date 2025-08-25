import argparse, re
import cv2, pytesseract, torch, librosa
from Levenshtein import ratio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import numpy as np
from PIL import Image

def load_audio(file_path, target_sr=16000):
    audio, sr = librosa.load(file_path, sr=target_sr)
    audio = librosa.effects.preemphasis(audio)
    audio = librosa.util.normalize(audio)
    return audio

def transcribe_audio(file_path, processor, model, device):
    audio = load_audio(file_path)
    inputs = processor(audio, return_tensors="pt", sampling_rate=16000).input_values.to(device)
    with torch.no_grad():
        logits = model(inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)
    pred_text = processor.batch_decode(pred_ids)[0]
    
    return re.sub(r"[^A-Z0-9]", "", pred_text.upper())


def extract_text_from_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(binary, config="--psm 6")
    return re.sub(r"[^A-Z0-9]", "", text.upper())

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = Wav2Vec2Processor.from_pretrained(args.model_path)
    model = Wav2Vec2ForCTC.from_pretrained(args.model_path).to(device)

    pred_audio = transcribe_audio(args.audio, processor, model, device)
    pred_image = extract_text_from_image(args.image)

    acc = ratio(pred_audio, pred_image) * 100

    print("🎙️ Audio Prediction :", pred_audio)
    print("🖼️ Image Prediction :", pred_image)
    print(f"✅ Similarity Score : {acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, required=True, help="Path to captcha audio file (.wav)")
    parser.add_argument("--image", type=str, required=True, help="Path to captcha image file (.png)")
    parser.add_argument("--model_path", type=str, default="facebook/wav2vec2-large-960h", 
                        help="Path to pretrained/fine-tuned Wav2Vec2 model")
    args = parser.parse_args()
    main(args)
