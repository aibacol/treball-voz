from flask import Flask, render_template
import numpy as np
import librosa
import librosa.display

app = Flask(__name__)

# Funciones para el cálculo del espectrograma
def calcular_espectrograma(x, fs, hop_length=256, n_fft=512):
    S = librosa.stft(x.astype(float), hop_length=hop_length, n_fft=n_fft)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    return S_db

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/espectrograma')
def espectrograma():
    # Aquí cargas tu archivo de audio
    # fs, x = cargar_audio(file_path)
    # Luego, calculas el espectrograma
    # S_db = calcular_espectrograma(x, fs)
    # Por simplicidad, crearemos un espectrograma de ejemplo
    S_db = np.random.rand(10, 10) * 100  # Esto es solo un espectrograma de ejemplo
    return render_template('espectrograma.html', espectrograma=S_db.tolist())

@app.route('/espectrograma_log')
def espectrograma_log():
    # Aquí calculas el espectrograma en escala logarítmica
    return "Espectrograma en escala logarítmica"

@app.route('/espectrograma_mel')
def espectrograma_mel():
    # Aquí calculas el espectrograma en escala MEL
    return "Espectrograma en escala MEL"

if __name__ == '__main__':
    app.run(debug=True)
