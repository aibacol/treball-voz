from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import vozyaudio as vz
import numpy as np
from scipy.io.wavfile import write, read
from scipy import signal
import os

app = Flask(__name__)
CORS(app)

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/aplicar-efecto/<efecto>', methods=['POST'])
def aplicar_efecto(efecto):
    archivo_audio = request.files['audio']

    # Guardar el archivo de audio temporalmente
    archivo_temporal = 'temp_audio.wav'
    archivo_audio.save(archivo_temporal)

    try:
        # Aplicar el efecto correspondiente
        if efecto == 'tremolo':
            resultado = tremolo(archivo_temporal)
        elif efecto == 'vibrato':  # Corregir el nombre del efecto
            resultado = vibrato(archivo_temporal)
        elif efecto == 'cambio-pitch':
            resultado = cambio_pitch(archivo_temporal)
        else:
            raise ValueError('Efecto no válido')
        
        # Devolver el audio resultante
        return send_file(resultado, as_attachment=True, mimetype='audio/wav')
    except Exception as e:
        # Enviar mensaje de error si ocurre algún problema
        return jsonify({'error': str(e)})

def tremolo(parametro):
    fs, x = vz.lee_audio(parametro)
    fc = 20  # Frecuencia del tremolo
    n = np.arange(len(x))
    carrier = 0.5 * (1 + np.cos(2 * np.pi * fc * n / fs))  # Modulación de amplitud entre 0 y 1
    
    y = x * carrier
    y = np.int16(y / np.max(np.abs(y)) * 32767)  # Normalizar y convertir a 16 bits
    resultado_path = "trem.wav"
    write(resultado_path, fs, y)
    return resultado_path

def vibrato(parametro):
    fs, x = vz.lee_audio(parametro)
    vibrato_depth = 0.002 * fs  # Profundidad del vibrato (en muestras)
    vibrato_rate = 3  # Frecuencia del vibrato (en Hz)
    
    modulation = vibrato_depth * np.sin(2 * np.pi * vibrato_rate * np.arange(len(x)) / fs)
    y = np.zeros_like(x)
    
    for i in range(len(x)):
        mod_index = int(i + modulation[i])
        if mod_index < len(x):
            y[i] = x[mod_index]
        else:
            y[i] = 0
    
    y = np.int16(y / np.max(np.abs(y)) * 32767)  # Normalizar y convertir a 16 bits
    resultado_path = "vibrato.wav"
    write(resultado_path, fs, y)
    return resultado_path

def cambio_pitch(parametro):
    fs, x = vz.lee_audio(parametro)
    
    H = 256  # Tamaño de salto
    B = 1024  # Tamaño del bloque/FFT
    fmax = 1600  # Frecuencia máxima a considerar
    noverlap = B - H  # Solape
    f, t, Zxx = signal.stft(x, fs=fs, nperseg=B, noverlap=noverlap)  # Calculamos STFT
    kmax = int(np.ceil(fmax * B / fs))  # Índice de la DFT correspondiente a la frecuencia máxima
    picos, fi = vz.encuentra_fi_espectrograma(Zxx[0:kmax, :], fs, B, H, umbral=30, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=0.01, wait=2)  # Buscamos picos en el espectro -> Candidatos a pitch

    # Seleccionamos el pitch como el candidato de frecuencia inferior
    nk, nn = np.shape(fi)
    fpitch = np.zeros(nn)
    for n in range(nn):
        if np.sum(fi[:, n]) > 0:
            k = np.argwhere(fi[:, n] > 0)
            fpitch[n] = np.min(fi[k, n])
    f0 = vz.segmentaf0(fpitch)  # Segmentamos/Post-procesamos el pitch -> obtenemos trayectoria del pitch

    # Buscamos la amplitud de los valores de la STFT asociados a la trayectoria del pitch
    n = np.argwhere(f0 > 0)  # Bloques con pitch válido
    a = np.zeros(len(f0))
    indices_k = np.int16(np.rint(f0[n] * B / fs))
    a[n] = np.abs(Zxx[indices_k, n])  # Amplitud asociada a cada frecuencia fundamental

    # Calculamos la frecuencia de la nota musical más cercana a la trayectoria del pitch
    pfref = np.zeros(len(f0))
    pfref[n] = np.rint(12 * np.log2(f0[n] / 440) + 69)
    f0midi = np.zeros(len(f0))
    f0midi[n] = 440 * 2**((pfref[n] - 69) / 12)  # Calculamos las frecuencias nominales de las notas musicales más cercanas al pitch

    # Sintetizamos el audio
    y = vz.sintetizaf0(f0midi, a, fs, H)
    
    y = np.int16(y / np.max(np.abs(y)) * 32767)  # Normalizar y convertir a 16 bits
    resultado_path = "pitch.wav"
    write(resultado_path, fs, y)
    return resultado_path

if __name__ == '__main__':
    app.run(debug=True)
