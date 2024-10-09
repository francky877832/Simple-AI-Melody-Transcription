import librosa
import numpy as np
import joblib
import tensorflow as tf
import os
import re

def extract_features(file_path, n_mfcc=13, sr=22050, max_duration=3.0):
    """
    Extrait les MFCC d'un fichier audio et retourne la moyenne et l'écart-type des coefficients.

    Parameters:
    - file_path (str): Chemin vers le fichier audio.
    - n_mfcc (int): Nombre de coefficients MFCC à extraire.
    - sr (int): Taux d'échantillonnage.
    - max_duration (float): Durée maximale en secondes à charger.

    Returns:
    - np.ndarray: Tableau des caractéristiques concaténées (moyenne et écart-type).
    """
    try:
        # Charger l'audio
        signal, sr = librosa.load(file_path, sr=sr, duration=max_duration)
        # Normaliser le signal
        signal = librosa.util.normalize(signal)
        # Extraire les MFCC
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
        # Calculer la moyenne et l'écart-type
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        # Concaténer les caractéristiques
        features = np.concatenate((mfcc_mean, mfcc_std))
        return features
    except Exception as e:
        print(f"Erreur lors de l'extraction des caractéristiques de {file_path}: {e}")
        return None

# 1. Charger le scaler et le LabelEncoder
scaler = joblib.load('datas/scaler.joblib')
label_encoder = joblib.load('datas/label_encoder.joblib')

# 2. Charger le modèle entraîné
model = tf.keras.models.load_model('datas/note_classification_model.keras')


input_dir = 'notes/'  # Remplacez par le chemin de votre dossier source
output_dir = 'results/'  # Dossier pour enregistrer les notes prédites

# Créer le dossier "notes" s'il n'existe pas
#os.makedirs(output_dir, exist_ok=True)

# Nom du fichier de sortie
output_file_path = os.path.join(output_dir, 'predicted_notes.txt')


def extract_number(filename):
    """
    Extrait le nombre du nom de fichier (ex: note_10 -> 10).
    """
    match = re.search(r'_(\d+)', filename)
    return int(match.group(1)) if match else float('inf')  # Retourne un grand nombre si aucune correspondance

# Liste des fichiers audio avec leurs chemins
files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
# Trier les fichiers par numéro
sorted_files = sorted(files, key=extract_number)

# Ouvrir le fichier de sortie en écriture
with open(output_file_path, 'w') as output_file:
    # Boucle pour traiter chaque fichier audio dans le dossier
    for filename in sorted_files:
        if filename.endswith('.wav'):  # Vérifiez que le fichier est un WAV
            file_path = os.path.join(input_dir, filename)
            
            # Charger l'audio
            y, sr = librosa.load(file_path)

            # Extraire les caractéristiques MFCC du fichier audio
            mfcc_features = extract_features(file_path)  # Appellez votre fonction d'extraction ici
            duration = librosa.get_duration(y=y, sr=sr)

            if mfcc_features is not None:
                # Normaliser les caractéristiques
                mfcc_features_scaled = scaler.transform([mfcc_features])

                # Faire la prédiction
                prediction = model.predict(mfcc_features_scaled)
                predicted_label_index = np.argmax(prediction)

                # Décoder la classe prédite en note de musique
                predicted_note = label_encoder.inverse_transform([predicted_label_index])[0]

                # Écrire la note prédite dans le fichier de sortie
                output_file.write(f"{filename} : {predicted_note} - {duration}\n")
                print(f"Fichier: {filename}, La note prédite est : {predicted_note}")
            else:
                print(f"Impossible d'extraire les caractéristiques pour {filename}.")

print(f"Toutes les notes prédites ont été enregistrées dans {output_file_path}.")


