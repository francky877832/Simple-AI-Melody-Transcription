const express = require('express');
const multer = require('multer');
const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const path = require('path');
const { extractMFCC } = require('./audioProcessing'); // Implémentez une fonction pour extraire les MFCC
const { spawn } = require('child_process');


const app = express();
const port = 8001;

// Configuration de Multer pour le téléchargement des fichiers
// Configuration de multer pour gérer l'upload de fichiers
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, 'medias/uploads'); // Dossier où les fichiers seront stockés
    },
    filename: (req, file, cb) => {
        cb(null, Date.now() + path.extname(file.originalname)); // Renommer le fichier
    },
});

const upload = multer({ storage });

// Activer CORS pour permettre les requêtes depuis le frontend
app.use((req, res, next) => {
    res.header("Access-Control-Allow-Origin", "*"); // Ajustez selon vos besoins de sécurité
    res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
    next();
});

// Charger le modèle TensorFlow.js
let model;
/*
(async () => {
    model = await tf.loadLayersModel('file://path/to/tfjs_model/model.json');
    console.log("Modèle chargé avec succès.");
})();*/

// Définir les étiquettes de classe
const CLASS_LABELS = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#'];

app.post('/transcribe', upload.single('file'), async (req, res) => {
    try {
        console.log("Fichier reçu");

        const filePath = path.resolve(req.file.path);
        
        const runPythonScript = (script) => {
            return new Promise((resolve, reject) => {
                const pythonProcess = spawn('python', [script, filePath]);

                pythonProcess.stdout.on('data', (data) => {
                    console.log(`Sortie : ${data}`);
                });

                pythonProcess.stderr.on('data', (data) => {
                    console.error(`Erreur : ${data}`);
                });

                pythonProcess.on('close', (code) => {
                    if (code === 0) {
                        resolve();
                    } else {
                        reject(new Error(`Le script ${script} a échoué avec le code ${code}`));
                    }
                });
            });
        };

        // Exécuter les scripts Python en séquence
        await runPythonScript('extract_notes_from_melody.py');
        await runPythonScript('predict_notes_from_melody.py');
        await runPythonScript('generate_melody.py');

        console.log("Tous les scripts Python exécutés avec succès");

        // Chemins des fichiers résultants
        const jsonFilePath = path.join(__dirname, 'results', 'predicted_notes.json');
        const wavFilePath = path.join(__dirname, 'results', 'melody_guitar.wav');

        res.download(jsonFilePath, 'predicted_notes.json', (err) => {
            if (err) {
                console.error('Erreur lors de l\'envoi du fichier JSON:', err);
                return res.status(500).json({ error: 'Erreur lors de l\'envoi du fichier JSON' });
            }

            /*// Si l'envoi du JSON a réussi, ensuite envoyer le fichier WAV
            res.download(wavFilePath, 'melody_guitar.wav', (err) => {
                if (err) {
                    console.error('Erreur lors de l\'envoi du fichier wav:', err);
                    return res.status(500).json({ error: 'Erreur lors de l\'envoi du fichier wav' });
                }
            });*/
        });
    } catch (error) {
        console.error(error);
        return res.status(500).json({ error: 'Une erreur est survenue lors de la transcription.' });
    }
});


app.get('/download', (req, res) => {
    const filePath = path.join(__dirname, 'results', 'melody_guitar.wav');
    res.download(filePath, 'melody_guitar.wav', (err) => {
        if (err) {
            console.error('Erreur lors du téléchargement:', err);
            res.status(500).send('Erreur lors du téléchargement du fichier.');
        }
    });
});


// Démarrer le serveur
app.listen(port, () => {
    console.log(`Serveur en écoute sur le port ${port}`);
});
