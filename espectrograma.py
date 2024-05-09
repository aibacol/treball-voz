import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import vozyaudio as vz

fs,x = vz.lee_audio('trompeta.wav')

NFFT=np.ceil(0.02*fs) # Tamaño del bloque (y número de puntos de la FFT) 
hop=np.ceil(0.005*fs) # Tamaño entre bloques
S_l = librosa.stft(x.astype(float), hop_length=int(hop), n_fft=int(NFFT))

S_db = librosa.amplitude_to_db(np.abs(S_l), ref=np.max)

fig, ax = plt.subplots()
img = librosa.display.specshow(S_db, sr=fs, hop_length=int(hop),x_axis='time', y_axis='linear', ax=ax)
ax.set(title='Espectrograma')
fig.colorbar(img, ax=ax, format="%+2.f dB");


fig, ax = plt.subplots()
img = librosa.display.specshow(S_db, sr=fs, hop_length=int(hop),x_axis='time', y_axis='log', ax=ax)
ax.set(title='Espectrograma con rango frecuencual logarítmico')
fig.colorbar(img, ax=ax, format="%+2.f dB");


S_mel = librosa.feature.melspectrogram(y=x.astype(float), hop_length=int(hop), sr=fs)
S_db = librosa.amplitude_to_db(np.abs(S_mel), ref=np.max)

fig, ax = plt.subplots()
img = librosa.display.specshow(S_db, sr=fs, hop_length=int(hop),x_axis='time', y_axis='mel', ax=ax)
ax.set(title='Espectrograma en escala MEL')
fig.colorbar(img, ax=ax, format="%+2.f dB");

plt.savefig('wachi.png')