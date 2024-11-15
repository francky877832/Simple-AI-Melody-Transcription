# Simple-AI-Melody-Transcription


This project aims to detect musical notes (A, B, C, D, E, F, G) from guitar melodies using machine learning and signal processing techniques. The dataset consists of guitar audio files (MP3/WAV) scraped from YouTube. The project includes data scraping, audio segmentation, and visualization of musical notes in the form of waveforms, spectrograms, and frequency distributions.

## Features
- **Data Scraping**: Scrapes guitar melody videos from YouTube and downloads the audio files (MP3/WAV).
- **Audio Segmentation**: Segments audio into individual musical notes based on their frequencies and stores them in folders for each note.
- **Data Cleaning**: Removes unnecessary noise and cleans the audio files for accurate note detection.
- **Exploratory Data Analysis (EDA)**: Visualizes audio data to analyze the characteristics of different musical notes.
- **Visualization**: Plots waveforms, spectrograms, and frequency distributions for all notes.

## Installation


### Prerequisites


### Steps to Install



## Project Structure

Here’s a brief overview of the project structure:

```
.
├── dataset/
│   └── guitar dataset/    # Folder containing segmented notes (A, B, C, etc.)
├── scripts/
│   ├── scraping.py          # Script for scraping YouTube for guitar audio
│   ├── extract_notes_from_melodu.py          # Script for segmenting audio into individual notes
│   └── visualisation_data.py         # Script for visualizing audio data (waveform, spectrogram)
    └── cleaning.py  
    
```

## Usage

### 1. **Data Scraping**
   To scrape guitar melody videos from YouTube and download their audio files (MP3/WAV), use the following script:
   
   ```bash
   python scripts/scraping.py
   ```

   This script will download the audio files and store them in the `dataset/` folder.

### 2. **Audio Segmentation**
   After scraping the data, you can segment each melody into individual musical notes and store them in separate folders. Run the following script:

   ```bash
   python scripts/extract_notes_from_melody.py
   ```

   This will analyze the audio and split the melodies into 7 subdirectories corresponding to the musical notes (A, B, C, D, E, F, G).

### 3. **Data Visualization**
   To visualize the waveform, spectrogram, and frequency distribution of the audio files, run:

   ```bash
   python scripts/visualisation.py
   ```

   This will display a series of plots for the audio files in the dataset, helping you explore and understand the audio characteristics.

### 4. **Customizing**
   You can modify the following parameters to suit your needs:
   - **Dataset Path**: In the `scripts/visualisation.py` and `scripts/extract_notes_from_melody.py` files, you can change the `dataset_path` to the directory where your dataset is stored.
   - **Scraping Settings**: Modify the scraping script to filter specific YouTube channels, video types, or audio formats.

## Dependencies

The following Python packages are required to run this project:

- `librosa` – for audio analysis
- `yt-dlp` – for YouTube video downloading
- `numpy` – for numerical computations
- `matplotlib` – for plotting
- `scipy` – for signal processing
- `pydub` – for audio file conversion
- `tqdm` – for progress bars during scraping

Install them using:

```bash
pip install -r requirements.txt
```

## Data Preprocessing

Before analyzing the audio files, you should clean and preprocess the dataset:

1. **Remove Silence**: Audio segments containing only silence are filtered out.
2. **Noise Reduction**: Any background noise in the audio is removed to improve the accuracy of note detection.
3. **Normalization**: Audio files are normalized to ensure consistent volume levels across the dataset.

## Data Analysis & Visualization

After cleaning the dataset, you can analyze the audio data through visualization. The project generates the following types of plots:

- **Waveform**: Displays the amplitude of the audio signal over time.
- **Spectrogram**: Displays the frequency content of the audio over time, helping to distinguish between different musical notes.
- **Frequency Distribution**: Displays the frequency spectrum of the audio signal, helping to identify the dominant frequencies corresponding to musical notes.

### Example Output

- **Waveform**: The plot shows the shape of the audio signal over time.
- **Spectrogram**: The plot shows how the frequency content of the audio evolves over time.
- **Frequency Distribution**: The plot shows the magnitudes of the frequencies present in the audio.

## Contributing

Contributions are welcome! If you find a bug or want to add new features, feel free to open an issue or submit a pull request.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -am 'Add feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Create a new Pull Request.

## License

```

### Key Sections:
1. **Project Overview**: Describes the purpose and features of the project.
2. **Installation Instructions**: Guides users on how to set up the project on their local machine.
3. **Usage**: Explains how to run the different parts of the project, including data scraping, segmentation, and visualization.
4. **Dependencies**: Lists the required Python packages.![graphs](https://github.com/user-attachments/assets/bc528f07-2405-4e85-8646-56393d2656ca)

5. **Data Preprocessing**: Details the steps for cleaning and preparing the dataset.
6. **Data Analysis & Visualization**: Provides information on how the data is analyzed and visualized.
7. **Contributing**: Explains how to contribute to the project.
8. **License**: Includes the licensing information.
![graphs](https://github.com/user-attachments/assets/5f0ddeea-de6b-4305-9564-b6f312a5c2de)


