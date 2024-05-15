from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import vozyaudio as vz
import numpy as np
import os
from scipy.io.wavfile import write

app = Flask(__name__)
CORS(app)


# Ruta para servir archivos estáticos (HTML, CSS, JavaScript, etc.)
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
        elif efecto == 'vibratto':
            resultado = vibratto(archivo_temporal)
        elif efecto == 'cambio-pitch':
            resultado = cambio_pitch(archivo_temporal)
        else:
            raise ValueError('Efecto no válido')
        
        # Devolver el audio resultante
        return send_from_directory('', resultado, as_attachment=True)
    except Exception as e:
        # Enviar mensaje de error si ocurre algún problema
        return jsonify({'error': str(e)})



def tremolo(parametro):
    fs, x = vz.lee_audio(parametro)
    fc = 20
    n = np.arange(len(x))
    carrier = np.cos(2 * np.pi * fc * n / fs)
    
    y = x * carrier
    y= np.int16(y)
    write("trem.wav",fs,y)
    return "trem.wav"

def cambio_pitch(parametro):
    fs, x = vz.lee_audio(parametro)
    # Efecto de cambio de tono
    semitonos = -2  # Número de semitonos para cambiar el tono (negativo para bajar)
    pitch_shift = 2 ** (semitonos / 12)  # Factor de cambio de tono
    y = vz.cambia_tono(x, fs, pitch_shift)
    # Guardar el audio resultante en formato WAV
    write("pitch.wav", fs, y)
    return "pitch.wav"

def vibratto(parametro):
    fs, x = vz.lee_audio(parametro)
    # Parámetros para el vibrato
    vibrato_depth = 0.002 * fs  # Profundidad del vibrato (en muestras), reducido
    vibrato_rate = 3 / fs  # Frecuencia del vibrato (en Hz), reducido
    # Generar la señal moduladora para el vibrato
    modulation = vibrato_depth * np.sin(2 * np.pi * vibrato_rate * np.arange(len(x)))
    # Aplicar el efecto de vibrato
    y = x * np.cos(2 * np.pi * modulation)
    # Convertir a tipo de datos de 16 bits
    y = np.int16(y)
    # Guardar el audio resultante
    write("vibrato.wav", fs, y)
    return "vibrato.wav"

if __name__ == '__main__':
    app.run(debug=True)
