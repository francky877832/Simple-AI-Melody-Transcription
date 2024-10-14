import os
import pandas as pd
import glob
import librosa
import numpy as np
import wave

# Définir le chemin de la racine de votre dataset
# Remplacez ce chemin par celui de votre dataset de piano
dataset_root = 'dataset/archive/Guitar Dataset/'  # Exemple : 'dataset/archive/Piano Dataset/'

# Paramètres pour l'extraction des MFCC
N_MFCC = 13            # Nombre de coefficients MFCC à extraire
MAX_DURATION = 5.0     # Durée maximale en secondes pour les fichiers audio (ajustez selon vos données)
SAMPLING_RATE = 22050  # Taux d'échantillonnage (standard pour librosa)

# Fonction pour vérifier si un fichier WAV est valide
def is_valid_wav(file_path):
    try:
        with wave.open(file_path, 'rb') as wf:
            # Si le fichier peut être ouvert et lu, il est considéré comme valide
            return True
    except wave.Error:
        return False
    except Exception as e:
        print(f"Erreur inattendue lors de la vérification de {file_path}: {e}")
        return False

# Fonction pour extraire les caractéristiques MFCC d'un fichier audio
def extract_mfcc_features(file_path, n_mfcc=13, sr=22050, max_duration=5.0):
    try:
        # Charger le fichier audio
        signal, sr = librosa.load(file_path, sr=sr, duration=max_duration)
        
        # Normaliser le signal audio
        signal = librosa.util.normalize(signal)
        
        # Extraire les MFCC
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
        
        # Calculer la moyenne et l'écart-type pour chaque coefficient MFCC
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        
        # Concatenation des moyennes et écarts-types
        features = np.concatenate((mfccs_mean, mfccs_std))
        
        return features
    except Exception as e:
        print(f"Erreur lors de l'extraction des MFCC de {file_path}: {e}")
        return None

# Initialiser une liste pour stocker les données
data = []

# Parcourir chaque dossier de note dans le dataset
note_folders = [folder for folder in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, folder))]

for note_folder in note_folders:
    note_path = os.path.join(dataset_root, note_folder)
    
    # Utiliser glob pour trouver tous les fichiers WAV dans le dossier
    wav_files = glob.glob(os.path.join(note_path, '*.wav'))
    
    for wav_file in wav_files:
        # Vérifier si le fichier WAV est valide
        if not is_valid_wav(wav_file):
            print(f"Fichier WAV invalide ou corrompu : {wav_file}. Ignoré.")
            continue
        
        # Extraire les caractéristiques MFCC
        features = extract_mfcc_features(wav_file, n_mfcc=N_MFCC, sr=SAMPLING_RATE, max_duration=MAX_DURATION)
        
        if features is not None:
            # Utiliser le nom complet du dossier comme label (par exemple, 'A2', 'A#3')
            label = note_folder.upper() if len(note_folder) > 0 else 'Unknown'
            
            # Créer un dictionnaire pour les données
            data_entry = {'label': label}
            
            # Ajouter les MFCC moyens
            for i in range(N_MFCC):
                data_entry[f'mfcc_{i+1}_mean'] = features[i]
            
            # Ajouter les MFCC écarts-types
            for i in range(N_MFCC):
                data_entry[f'mfcc_{i+1}_std'] = features[N_MFCC + i]
            
            # Ajouter l'entrée à la liste des données
            data.append(data_entry)

# Vérifier si des données ont été collectées
if not data:
    print("Aucune donnée n'a été collectée. Vérifiez la structure de votre dataset et les fichiers WAV.")
else:
    # Créer un DataFrame pandas à partir des données collectées
    df = pd.DataFrame(data)
    
    # Afficher les premières lignes pour vérifier
    print("Aperçu des données collectées :")
    print(df.head())
    
    # Définir le chemin de sortie pour le CSV
    csv_output_path = 'dataset_mfcc_features.csv'
    
    # Sauvegarder le DataFrame en CSV
    try:
        df.to_csv(csv_output_path, index=False)
        print(f"CSV avec les caractéristiques MFCC créé avec succès à l'emplacement : {csv_output_path}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du CSV : {e}")
