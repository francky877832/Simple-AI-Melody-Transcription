from pydub import AudioSegment
from pydub.playback import play
import numpy as np

# Define the sample rate
sample_rate = 44100

# Load guitar samples
E2 = AudioSegment.from_file("dataset/archive/Guitar Dataset/E2/E2-1-spn.wav")
DSHARP3 = AudioSegment.from_file("dataset/archive/Guitar Dataset/Dsharp3/Dsharp3-1-spn.wav")
E5 = AudioSegment.from_file("dataset/archive/Guitar Dataset/E5/E5-1-spn.wav")
F3 = AudioSegment.from_file("dataset/archive/Guitar Dataset/F3/F3-1-spn.wav")
FSHARP3 = AudioSegment.from_file("dataset/archive/Guitar Dataset/Fsharp3/Fsharp3-1-spn.wav")

# Define the melody with notes and durations
melody = [
    (E2, 1.6718367346938776),
    (DSHARP3, 0.046439909297052155),
    (E2, 0.37151927437641724),
    (E2, 0.3947392290249433),
    (E2, 0.3947392290249433),
    (E5, 0.3947392290249433),
    (E2, 0.3947392290249433),
    (DSHARP3, 0.3947392290249433),
    (DSHARP3, 0.3947392290249433),
    (F3, 0.3947392290249433),
    (E2, 0.3947392290249433),
    (E2, 0.3947392290249433),
    (F3, 0.4179591836734694),
    (FSHARP3, 0.37151927437641724),
    (E2, 0.3947392290249433),
    (E2, 0.3947392290249433),
    (DSHARP3, 0.3947392290249433),
    (F3, 0.1)
]

# Generate the melody
melody_audio = AudioSegment.silent(duration=0)
for note, duration in melody:
    note_audio = note[:int(duration * 1000)]  # Convert duration to milliseconds
    melody_audio += note_audio

# Export the melody to a WAV file
melody_audio.export("results/melody_guitar.wav", format="wav")

# Play the melody
#play(melody_audio)

print("The melody has been generated and saved as 'melody_guitar.wav'.")
