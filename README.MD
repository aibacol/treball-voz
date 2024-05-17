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

```
python aplicar-funcion.py 
```
```
python -m http.server
```

Una vez este en el directorio ejecute los codigos de arriba en el orden indicado. 
Una vez realizado esto podra ejecutar de nuevo el html y poder usar los filtros.