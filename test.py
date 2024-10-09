import os
import librosa
import soundfile as sf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import load_model
import joblib

# Chemins vers les fichiers sauvegardés et le fichier audio à transcrire
MODEL_PATH = 'datas/music_transcription.keras'        # Remplacez par le chemin réel
ENCODER_PATH = 'datas/label_encoder.pkl'              # Remplacez par le chemin réel
SCALER_PATH = 'datas/scaler.pkl'                      # Remplacez par le chemin réel
MELODY_PATH = 'melody/Slow Acoustic Guitar Instrumental - Quiet Place (Original).wav'         # Remplacez par le chemin réel
OUTPUT_TXT = 'transcribed_notes.txt'
OUTPUT_CSV = 'transcribed_notes.csv'



# Paramètres d'extraction des MFCC
SAMPLE_RATE = 22050       # Taux d'échantillonnage utilisé lors de l'entraînement
DURATION = 2.0             # Durée des clips audio en secondes (peut être ajusté)
SAMPLES_PER_TRACK = int(SAMPLE_RATE * DURATION)
N_MFCC = 13                # Nombre de coefficients MFCC

# Fonction pour extraire les MFCC d'un segment de note
def extract_mfcc_for_note(note_signal, sr, n_mfcc=N_MFCC):
    try:
        # Convertir en mono si nécessaire
        if len(note_signal.shape) == 2:
            note_signal = np.mean(note_signal, axis=1)
        
        # Resampler si nécessaire
        if sr != SAMPLE_RATE:
            note_signal = librosa.resample(note_signal, orig_sr=sr, target_sr=SAMPLE_RATE)
        
        # Troncature ou padding
        if len(note_signal) < SAMPLES_PER_TRACK:
            padding = SAMPLES_PER_TRACK - len(note_signal)
            note_signal = np.pad(note_signal, (0, padding), 'constant')
        else:
            note_signal = note_signal[:SAMPLES_PER_TRACK]
        
        # Extraction des MFCC
        mfcc = librosa.feature.mfcc(y=note_signal, sr=SAMPLE_RATE, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        
        return mfcc_mean
    except Exception as e:
        print(f"Erreur lors de l'extraction des MFCC : {e}")
        return None

def main():
    # 1. Charger le modèle et les objets encodés
    print("Chargement du modèle et des objets encodés...")
    model = load_model(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Chargement terminé.\n")
    
    # 2. Charger le fichier mélodie
    print(f"Chargement du fichier audio : {MELODY_PATH}")
    signal, sr = librosa.load(MELODY_PATH, sr=None)
    print(f"Taux d'échantillonnage : {sr} Hz")
    print(f"Durée de l'audio : {len(signal)/sr:.2f} secondes\n")
    
    # 3. Détecter les onsets
    print("Détection des onsets (début des notes)...")
    onset_frames = librosa.onset.onset_detect(y=signal, sr=sr, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    print(f"Temps des onsets (secondes) : {onset_times}\n")
    
    # 4. Segmenter l'audio en notes
    print("Segmentation de l'audio en notes individuelles...")
    onset_times = np.append(onset_times, len(signal)/sr)  # Ajouter la fin de l'audio comme dernier onset
    notes_segments = []
    for i in range(len(onset_times)-1):
        start_time = onset_times[i]
        end_time = onset_times[i+1]
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        note_signal = signal[start_sample:end_sample]
        notes_segments.append(note_signal)
    print(f"Nombre de notes détectées : {len(notes_segments)}\n")
    
    # 5. Extraire les MFCC pour chaque note
    print("Extraction des MFCC pour chaque note...")
    mfcc_features = []
    valid_notes = []
    
    for i, note in enumerate(notes_segments):
        mfcc = extract_mfcc_for_note(note, sr)
        if mfcc is not None:
            mfcc_features.append(mfcc)
            valid_notes.append(i+1)  # Numéro de la note (optionnel)
        else:
            print(f"Note {i+1} : Extraction des MFCC a échoué.")
    
    if not mfcc_features:
        print("Aucune caractéristique MFCC n'a été extraite. Terminaison du programme.")
        return
    
    # Convertir les MFCC en DataFrame
    mfcc_df = pd.DataFrame(mfcc_features, columns=[f'MFCC_{i+1}' for i in range(N_MFCC)])
    print("Extraction des MFCC terminée.\n")
    

    expected_columns = scaler.feature_names_in_  # Colonnes utilisées lors de l'entraînement du scaler
    mfcc_df = mfcc_df.reindex(columns=expected_columns, fill_value=0)  

    # 6. Normaliser les caractéristiques
    print("Normalisation des caractéristiques...")
    mfcc_scaled = scaler.transform(mfcc_df)
    print("Normalisation terminée.\n")
    

    print(mfcc_scaled)

    # 7. Classer chaque note avec le modèle entraîné
    print("Classification des notes...")
    y_pred_prob = model.predict(mfcc_scaled)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_pred_labels = encoder.inverse_transform(y_pred)
    print("Classification terminée.\n")
    
    # 8. Afficher et sauvegarder les résultats
    print("Affichage des notes prédites :")
    #for i, note in enumerate(y_pred_labels, 1):
    #   print(f"Note {i}: {note}")
    
    # Sauvegarder dans un fichier texte
    print(f"\nSauvegarde des notes prédites dans {OUTPUT_TXT}...")
    with open(OUTPUT_TXT, 'w') as f:
        for i, note in enumerate(y_pred_labels, 1):
            f.write(f"Note {i}: {note}\n")
    print(f"Notes transcrites sauvegardées dans {OUTPUT_TXT}\n")
    
    # Sauvegarder dans un fichier CSV
    print(f"Sauvegarde des notes prédites dans {OUTPUT_CSV}...")
    transcribed_df = pd.DataFrame({
        'Note_Number': range(1, len(y_pred_labels)+1),
        'Predicted_Note': y_pred_labels
    })
    transcribed_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Notes transcrites sauvegardées dans {OUTPUT_CSV}\n")
    
    print("Transcription terminée avec succès.")

if __name__ == "__main__":
    main()
