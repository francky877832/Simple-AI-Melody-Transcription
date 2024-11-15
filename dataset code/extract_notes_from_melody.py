import sys
import os
import librosa
import soundfile as sf
import shutil


sys.stdout.reconfigure(encoding='utf-8')



def clear_directory(path):
    # Supprimer tous les fichiers et sous-dossiers dans le dossier spécifié
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Supprime le fichier
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Supprime le dossier
        except Exception as e:
            print(f'Erreur lors de la suppression de {file_path}: {e}')



def extract_notes(audio_path):
    # Créer le dossier "notes" s'il n'existe pas
    output_dir = 'notes'

    clear_directory(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Charger la mélodie
    print(f"Chargement du fichier audio : {audio_path}")
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

if __name__ == "__main__":
    # Vérification des arguments en ligne de commande
    if len(sys.argv) < 2:
        print("Usage : python extract_note_from_melody.py <chemin_du_fichier_audio>")
        sys.exit(1)

    # Récupérer le chemin du fichier audio depuis les arguments
    audio_path = sys.argv[1]
    print(sys.argv)

    # Vérifier si le fichier existe
    if not os.path.exists(audio_path):
        print(f"Erreur : Le fichier '{audio_path}' n'existe pas.")
        sys.exit(1)

    # Lancer l'extraction des notes
    extract_notes(audio_path)
