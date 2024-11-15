import os
import librosa
import soundfile as sf

def clean_audio(input_path, output_path):

    try:
        y, sr = librosa.load(input_path)
        
        y = librosa.util.normalize(y)
        
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        
        # Save cleaned audio
        sf.write(output_path, y_trimmed, sr)
        print(f"Cleaned and saved: {output_path}")
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def clean_dataset(dataset_path, cleaned_dataset_path):

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.wav') or file.endswith('.mp3'):
                input_file = os.path.join(root, file)
                
                relative_path = os.path.relpath(root, dataset_path)
                output_dir = os.path.join(cleaned_dataset_path, relative_path)
                os.makedirs(output_dir, exist_ok=True)
                
                # Output file path
                output_file = os.path.join(output_dir, file)
                
                # Clean and save the audio file
                clean_audio(input_file, output_file)

dataset_path = './guitar dataset'  # Path to your original dataset
cleaned_dataset_path = './cleaned_note_segments'  # Path to save cleaned dataset
clean_dataset(dataset_path, cleaned_dataset_path)
