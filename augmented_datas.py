
import os
import librosa
import soundfile as sf
import numpy as np
import random
from scipy.signal import convolve

# Chemin vers le dataset
DATASET_PATH = 'dataset/archive/Guitar Dataset/'

TARGET_TOTAL = 43000
AUGMENTATIONS_PER_FILE = 20

# Ajouter du bruit
def add_noise(y, noise_factor=None):
    if noise_factor is None:
        noise_factor = random.uniform(0.001, 0.01)
    noise = np.random.randn(len(y))
    augmented = y + noise_factor * noise
    return augmented / np.max(np.abs(augmented))

# Changer le pitch
def change_pitch(y, sr, pitch_factor=None):
    if pitch_factor is None:
        pitch_factor = random.uniform(-5, 5)
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=pitch_factor)

# Changer la vitesse
def change_speed(y, speed_factor=None):
    if speed_factor is None:
        speed_factor = random.uniform(0.8, 1.2)
    return librosa.effects.time_stretch(y=y, rate=speed_factor)

# Ajouter de la réverbération
def add_reverb(y, sr, reverb_factor=None):
    if reverb_factor is None:
        reverb_factor = random.uniform(0.1, 0.3)
    impulse_response = np.exp(-np.linspace(0, 1, int(sr * reverb_factor)))
    reverb = convolve(y, impulse_response, mode='full')[:len(y)]
    return reverb / np.max(np.abs(reverb))

# Inverser la phase
def invert_phase(y):
    return -y

# Appliquer plusieurs augmentations
def augment_audio(y, sr):
    augmented = y.copy()
    
    if random.random() < 0.5:
        augmented = add_noise(augmented)
    
    if random.random() < 0.5:
        pitch_factor = random.uniform(-5, 5)
        augmented = change_pitch(y=augmented, sr=sr, pitch_factor=pitch_factor)
    
    if random.random() < 0.5:
        speed_factor = random.uniform(0.8, 1.2)
        augmented = change_speed(y=augmented, speed_factor=speed_factor)
    
    if random.random() < 0.5:
        augmented = add_reverb(augmented, sr)
    
    if random.random() < 0.5:
        augmented = invert_phase(augmented)
    
    return augmented



# Script principal pour augmentation des données
def main():
    total_samples = sum([len(os.listdir(os.path.join(DATASET_PATH, note))) for note in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, note))])
    
    if total_samples >= TARGET_TOTAL:
        print("Le nombre de samples est déjà suffisant.")
        return
    
    samples_needed = TARGET_TOTAL - total_samples
    
    for note in os.listdir(DATASET_PATH):
        note_path = os.path.join(DATASET_PATH, note)
        if not os.path.isdir(note_path):
            continue
        
        for audio_file in os.listdir(note_path):
            audio_path = os.path.join(note_path, audio_file)
            #y, sr = librosa.load(audio_path, sr=None)

            try:
                y, sr = librosa.load(audio_path, sr=None)
            except Exception as e:
                print(f"Erreur lors du chargement de {audio_path}: {e}")
                continue
        
            
            for _ in range(AUGMENTATIONS_PER_FILE):
                augmented_y = augment_audio(y=y, sr=sr)
                new_file_name = f"{os.path.splitext(audio_file)[0]}_aug_{random.randint(1000, 9999)}.wav"
                
                new_file_path = os.path.join(note_path, f"new/{new_file_name}")

                sf.write(new_file_path, augmented_y, sr)
                samples_needed -= 1
                if samples_needed <= 0:
                    break
            if samples_needed <= 0:
                break
        if samples_needed <= 0:
            break

    print(f"Nombre total de samples : {total_samples}")

if __name__ == "__main__":
    main()
