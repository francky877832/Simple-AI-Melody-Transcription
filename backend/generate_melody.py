from pydub import AudioSegment
from pydub.playback import play
import numpy as np
import sys
import json
import os



sys.stdout.reconfigure(encoding='utf-8')
current_dir = os.path.dirname(os.path.abspath(__file__))  


# Define the sample rate
sample_rate = 44100

# Load guitar samples
E2 = AudioSegment.from_file(os.path.join(current_dir,"samples_notes/Guitar Dataset/E2/E2-1-spn.wav"))
DSHARP3 = AudioSegment.from_file(os.path.join(current_dir, "samples_notes/Guitar Dataset/Dsharp3/Dsharp3-1-spn.wav"))
E5 = AudioSegment.from_file(os.path.join(current_dir, "samples_notes/Guitar Dataset/E5/E5-1-spn.wav"))
F3 = AudioSegment.from_file(os.path.join(current_dir, "samples_notes/Guitar Dataset/F3/F3-1-spn.wav"))
F2 = AudioSegment.from_file(os.path.join(current_dir, "samples_notes/Guitar Dataset/F2/F2-1-spn.wav"))
D2 = AudioSegment.from_file(os.path.join(current_dir, "samples_notes/Guitar Dataset/D2/D2-1-spn.wav"))
FSHARP3 = AudioSegment.from_file(os.path.join(current_dir, "samples_notes/Guitar Dataset/Fsharp3/Fsharp3-1-spn.wav"))
FSHARP2 = AudioSegment.from_file(os.path.join(current_dir, "samples_notes/Guitar Dataset/Fsharp2/Fsharp2-1-spn.wav"))
ASHARP2 = AudioSegment.from_file(os.path.join(current_dir, "samples_notes/Guitar Dataset/Asharp2/Asharp2-1-spn.wav"))
GSHARP3 = AudioSegment.from_file(os.path.join(current_dir, "samples_notes/Guitar Dataset/Gsharp3/Gsharp3-1-spn.wav"))


# Define the melody with notes and durations
melody = [

]


# Chemin vers le fichier JSON
json_file_path = os.path.join(current_dir, 'results/predicted_notes.json')

# Charger le fichier JSON
with open(json_file_path, 'r') as file:
    data = json.load(file) 

for el in data:
    note = el.get('note')     
    duration = el.get('duration') 
    melody.append((globals()[note], duration)) 

#print(melody)



# Generate the melody
melody_audio = AudioSegment.silent(duration=0)
for note, duration in melody:
    note_audio = note[:int(duration * 1000)]  # Convert duration to milliseconds
    melody_audio += note_audio

# Export the melody to a WAV file
melody_audio.export(os.path.join(current_dir, "results/melody_guitar.wav"), format="wav")

# Play the melody
#play(melody_audio)

print("The melody has been generated and saved as 'melody_guitar.wav'.")
