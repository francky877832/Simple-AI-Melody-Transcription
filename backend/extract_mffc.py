import sys
import numpy as np
import librosa
import json
import io

def main():
    # Lire le buffer audio depuis stdin
    audio_bytes = sys.stdin.buffer.read()
    audio_stream = io.BytesIO(audio_bytes)
    y, sr = librosa.load(audio_stream, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1).tolist()
    # Envoyer les MFCC au stdout en format JSON
    print(json.dumps(mfcc_mean))

if __name__ == "__main__":
    main()
