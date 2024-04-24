
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

// Create a new audio context
const audioCtx = new AudioContext();

// Load the audio file
fetch('trompeta.wav')
  .then(response => response.arrayBuffer())
  .then(arrayBuffer => audioCtx.decodeAudioData(arrayBuffer))
  .then(audioBuffer => {
    // Store the audio buffer
    const x = audioBuffer;

    // Create an analyser node
    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 2048;
    analyser.smoothingTimeConstant = 0.3;

    // Connect the audio buffer source node to the analyser node
    const source = audioCtx.createBufferSource();
    source.buffer = x;
    source.connect(analyser);

    // Start the audio
    source.start(0);

    // Update the spectrogram data in the requestAnimationFrame loop
    function updateSpectrogram() {
      // Get the time domain data
      analyser.getByteTimeDomainData(timeByteData);

      // Get the frequency domain data
      analyser.getByteFrequencyData(freqByteData);

      // Create the canvas and context
      const canvas = document.getElementById('espectrograma');
      const ctx = canvas.getContext('2d');

      // Clear the canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw the time domain data
      ctx.fillStyle = 'rgb(0, 0, 0)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.lineWidth = 2;
      ctx.strokeStyle = 'rgb(255, 255, 255)';
      ctx.beginPath();
      let x = 0;
      for (let i = 0; i < timeByteData.length; i++) {
        const v = timeByteData[i] / 128.0;
        const y = (1 - v) * canvas.height;
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
        x++;
      }
      ctx.stroke();

      // Draw the frequency domain data
      ctx.fillStyle = 'rgb(0, 0, 0)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.lineWidth = 2;
      ctx.strokeStyle = 'rgb(255, 255, 255)';
      ctx.beginPath();
      x = 0;
      for (let i = 0; i < freqByteData.length; i++) {
        const v = freqByteData[i] / 128.0;
        const y = (1 - v) * canvas.height;
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
        x += canvas.width / freqByteData.length;
      }
      ctx.stroke();

      // Request the next frame
      requestAnimationFrame(updateSpectrogram);
    }

    // Start the update loop
    updateSpectrogram();
  });