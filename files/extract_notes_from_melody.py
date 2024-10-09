import os
import librosa
import soundfile as sf

# Créer le dossier "notes" s'il n'existe pas
output_dir = 'notes'
os.makedirs(output_dir, exist_ok=True)

# Charger la mélodie
audio_path = 'random_melodies/sample.wav'
y, sr = librosa.load(audio_path)

# Détecter les onsets
onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames')

# Découper et enregistrer les segments
for i, frame in enumerate(onset_frames):
    # Convertir l'index de frame en échantillon
    onset_sample = librosa.frames_to_samples(frame)

    # Déterminer la durée jusqu'au prochain onset
    if i < len(onset_frames) - 1:
        next_onset_frame = onset_frames[i + 1]
        duration_samples = librosa.frames_to_samples(next_onset_frame) - onset_sample
    else:
        duration_samples = sr // 10  # Par exemple, 100 ms pour le dernier onset

    # Extraire le segment
    segment = y[onset_sample:onset_sample + duration_samples]

    # Enregistrer le segment en tant que fichier WAV
    note_filename = os.path.join(output_dir, f'note_{i + 1}.wav')
    sf.write(note_filename, segment, sr)

print(f"Notes enregistrées dans le dossier '{output_dir}'.")
