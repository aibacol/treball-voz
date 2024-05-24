! Para el uso de esta página web es necesario tener Python instalado !

Debe tener instaladas las siguientes librerias:
- [flask](https://flask.palletsprojects.com/en/3.0.x/)
- [flask_cors](https://pypi.org/project/Flask-Cors/1.10.3/)
- [numpy](https://numpy.org/)
- [scipy.io.wavfile](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html)
- [scipy](https://scipy.org/)
- os

Para que funcione tienes que activar el servidor flask. Para ello es necesario que abras 2 terminales.
En ambas terminales debera ir hasta el directorio donde esta contenido tanto el py como html.

`python aplicar-funcion.py`
`python -m http.server`

Una vez esté en el directorio ejecute los codigos de arriba en el orden indicado. 
Una vez realizado esto podra ejecutar de nuevo el html y poder usar los filtros.
---
## Explicación de las funciones python empleadas:
- El efecto trémolo hace que el volumen fluctúe a una frecuencia determinada. Usa una frecuencia de trémolo de 20, oscilando el volumen del audio 20 veces por segundo. Modulamos de amplitud varia entre 0 y 1. Lo normalizamos y lo convertimos a 16 bits.
- El vibratto hace lo mismo pero trabaja sobre el tono, no sobre el volumen. En este caso, fijamos la frecuencia a 3 Hz, fluctuando el tono 3 veces por segundo. Empleamos un parámetro que representa la cantidad máxima de desviación que se aplica a la señal original. Para acabar, normalizamos 
- El cambio de pitch es un cambio de tono de todo el audio. Primero realizamos una transformada de Fourier de 4º tiempo STFT para obtener el espectro de frecuencias para identificar los picos de frecuencia que corresponden al pitch. De esos picos de frdecuencia seleccionamos el pitch más bajo en cada bloque de tiempo, ajustamos las frecuencias al tono de la nota más cercana y normalizamos.