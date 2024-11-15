
import librosa
import numpy as np
import os
import soundfile as sf

NOTE_FREQUENCIES = {'A': 440.0, 'B': 493.88, 'C': 261.63, 'D': 293.66, 'E': 329.63, 'F': 349.23, 'G': 392.0,}

TOLERANCE = 20
NOTE_RANGES = {note: (freq - TOLERANCE, freq + TOLERANCE) for note, freq in NOTE_FREQUENCIES.items()}

def classify_note(frequency):
    for note, (low, high) in NOTE_RANGES.items():
        if low <= frequency <= high:
            return note
    return None

def segment_and_save_notes(input_path, output_dir):
    y, sr = librosa.load(input_path)
    hop_length = 512
    frame_length = 2048

    stft_result = librosa.stft(y, n_fft=frame_length, hop_length=hop_length)
    magnitudes = np.abs(stft_result)

    for i in range(magnitudes.shape[1]):
        frequencies = librosa.fft_frequencies(sr=sr, n_fft=frame_length)
        dominant_index = magnitudes[:, i].argmax()
        dominant_frequency = frequencies[dominant_index]

        note = classify_note(dominant_frequency)
        if note:
            start_sample = i * hop_length
            end_sample = start_sample + frame_length
            segment = y[start_sample:end_sample]

            note_folder = os.path.join(output_dir, note)
            os.makedirs(note_folder, exist_ok=True)
            file_path = os.path.join(note_folder, f"{note}-{i+1}.wav")
            sf.write(file_path, segment, sr)
            print(f"Saved segment to {file_path}")

input_path = 'guitar melodies/guitar_melody-1.wav'
output_dir = 'guitar dataset'
segment_and_save_notes(input_path, output_dir)

