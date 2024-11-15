import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def visualize_dataset(dataset_path):
 
    fig, ax = plt.subplots(3, 1, figsize=(15, 15))

    all_waveforms = []
    all_spectrograms = []
    all_frequency_distributions = []

    for note_folder in os.listdir(dataset_path):
        note_folder_path = os.path.join(dataset_path, note_folder)
        if os.path.isdir(note_folder_path):
            for file in os.listdir(note_folder_path):
                if file.endswith('.wav') or file.endswith('.mp3'):
                    file_path = os.path.join(note_folder_path, file)

                    # Load audio file
                    y, sr = librosa.load(file_path)

                    # Waveform: Append all waveforms for plotting later
                    all_waveforms.append(y)

                    # Spectrogram: Compute and append the spectrogram
                    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
                    S_dB = librosa.power_to_db(S, ref=np.max)
                    all_spectrograms.append(S_dB)

                    # Frequency Distribution: Compute and append the frequency distribution
                    stft_result = np.abs(librosa.stft(y))
                    frequencies = librosa.fft_frequencies(sr=sr)
                    magnitude_sums = stft_result.sum(axis=1)
                    all_frequency_distributions.append((frequencies, magnitude_sums))

    # Combine all waveforms into one plot
    ax[0].set_title('Waveform of All Notes')
    for waveform in all_waveforms:
        ax[0].plot(np.linspace(0, len(waveform) / sr, num=len(waveform)), waveform, alpha=0.5)
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Amplitude')

    # Combine all spectrograms into one plot
    ax[1].set_title('Spectrogram of All Notes')
    for S_dB in all_spectrograms:
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000, ax=ax[1], cmap='viridis', alpha=0.5)
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Mel Frequency')

    # Combine all frequency distributions into one plot
    ax[2].set_title('Frequency Distribution of All Notes')
    for frequencies, magnitude_sums in all_frequency_distributions:
        ax[2].plot(frequencies, magnitude_sums, alpha=0.5)
    ax[2].set_xlabel('Frequency (Hz)')
    ax[2].set_ylabel('Magnitude')
    ax[2].set_xlim(0, 5000) 

    plt.tight_layout()
    plt.show()


dataset_path = './dataset/guitar dataset'
visualize_dataset(dataset_path)
