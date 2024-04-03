
function verEspectrograma() {
    fetch('/espectrograma')
        .then(response => response.json())
        .then(data => {
            document.getElementById('espectrograma').innerText = JSON.stringify(data);
        });
}

function verEspectrogramaLog() {
    fetch('/espectrograma_log')
        .then(response => response.text())
        .then(data => {
            document.getElementById('espectrograma').innerText = data;
        });
}

function verEspectrogramaMel() {
     fetch('/espectrograma_mel')
        .then(response => response.text())
        .then(data => {
            document.getElementById('espectrograma').innerText = data;
        });
}

// Para grabar
var mediaRecorder; // Variable global para almacenar el objeto MediaRecorder
        var audioChunks = []; // Array para almacenar los fragmentos de audio grabados. Usar AudioChunks para los datos del espectrograma

        function iniciarGrabacion() {
            // Solicitar acceso al micrófono del usuario
            navigator.mediaDevices.getUserMedia({ audio: true })
                .then(function(stream) {
                    // Crear un objeto MediaRecorder para grabar audio
                    mediaRecorder = new MediaRecorder(stream);

                    // Evento que se activa cuando se graba un nuevo fragmento de audio
                    mediaRecorder.addEventListener('dataavailable', function(event) {
                        audioChunks.push(event.data);
                    });

                    // Comenzar a grabar audio
                    mediaRecorder.start();
                })
                .catch(function(error) {
                    console.error('Error al acceder al micrófono:', error);
                });
        }

        function detenerGrabacion() {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop(); // Detener la grabación si está en curso
            }
        }

        function descargarGrabacion() {
            // Comprobar si hay fragmentos de audio grabados
            if (audioChunks.length === 0) {
                console.warn('No hay audio grabado.');
                return;
            }

            // Convertir los fragmentos de audio a un blob
            var audioBlob = new Blob(audioChunks, { type: 'audio/wav' });

            // Crear un objeto URL para el blob
            var audioURL = URL.createObjectURL(audioBlob);

            // Crear un enlace invisible para descargar el archivo WAV
            var descargarEnlace = document.createElement('a');
            descargarEnlace.href = audioURL;
            descargarEnlace.download = 'grabacion.wav';
            document.body.appendChild(descargarEnlace);

            // Hacer clic en el enlace para iniciar la descarga del archivo WAV
            descargarEnlace.click();

            // Limpiar el array de fragmentos de audio
            audioChunks = [];
        }