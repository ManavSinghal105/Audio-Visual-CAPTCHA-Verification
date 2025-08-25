import os, random, string
from captcha.image import ImageCaptcha
from captcha.audio import AudioCaptcha

os.makedirs("data/images", exist_ok=True)
os.makedirs("data/audio", exist_ok=True)

image_gen = ImageCaptcha(width=200, height=80)
audio_gen = AudioCaptcha()

def random_text(length=5):
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choice(chars) for _ in range(length))

def generate(n=10):
    for i in range(1, n+1):
        text = random_text()
        image_file = f"data/images/captcha_{i:04d}.png"
        audio_file = f"data/audio/captcha_{i:04d}.wav"
        image_gen.write(text, image_file)
        audio_gen.write(text, audio_file)
        print(f"Generated {text} -> {image_file}, {audio_file}")

if __name__ == "__main__":
    generate(20)  
