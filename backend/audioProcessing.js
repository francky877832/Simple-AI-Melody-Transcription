const { spawn } = require('child_process');
const path = require('path');

const extractMFCC = (audioBuffer) => {
    return new Promise((resolve, reject) => {
        const pythonProcess = spawn('python', [path.join(__dirname, 'extract_mfcc.py')]);

        let mfccData = '';
        let errorData = '';

        pythonProcess.stdout.on('data', (data) => {
            mfccData += data.toString();
        });

        pythonProcess.stderr.on('data', (data) => {
            errorData += data.toString();
        });

        pythonProcess.on('close', (code) => {
            if (code !== 0) {
                return reject(new Error(errorData));
            }
            try {
                const mfcc = JSON.parse(mfccData);
                resolve(mfcc);
            } catch (e) {
                reject(e);
            }
        });

        // Envoyer le buffer audio au script Python via stdin
        pythonProcess.stdin.write(audioBuffer);
        pythonProcess.stdin.end();
    });
};

module.exports = { extractMFCC };
