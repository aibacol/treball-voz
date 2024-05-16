let mediaRecorder;
let recordedChunks = [];
let audioStream;

navigator.mediaDevices.getUserMedia({ audio: true })
  .then(stream => {
    // Guarda el stream de audio para usarlo más tarde
    audioStream = stream;
  })
  .catch(error => {
    console.error('Error al acceder al micrófono:', error);
  });

function convertirTextoAVoz() {
    // Obtener el texto del textarea
    var texto = document.getElementById("texto").value;

    // Obtener el idioma seleccionado
    var idioma = document.getElementById("idioma").value;

    // Verificar si el navegador soporta la API SpeechSynthesis
    if ('speechSynthesis' in window) {
        // Crear un nuevo objeto SpeechSynthesisUtterance
        var mensaje = new SpeechSynthesisUtterance();

        // Establecer el texto que se va a convertir a voz
        mensaje.text = texto;

        // Establecer el idioma de la voz
        mensaje.lang = idioma;

        // Hablar el texto
        window.speechSynthesis.speak(mensaje);

    } else {
        alert("Tu navegador no soporta la API SpeechSynthesis.");
    }
}

function iniciarGrabacion() {
    // Detener grabación si ya está en curso
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        detenerGrabacion();
    }

    // Crear un nuevo objeto MediaRecorder utilizando el stream de audio
    mediaRecorder = new MediaRecorder(audioStream);

    // Manejar los eventos de grabación
    mediaRecorder.ondataavailable = event => {
        recordedChunks.push(event.data);
    };

    mediaRecorder.onstop = () => {
        // Crear un archivo de audio a partir de los datos grabados
        const blob = new Blob(recordedChunks, { type: 'audio/wav' });
        const audioUrl = URL.createObjectURL(blob);

        // Crear un enlace de descarga para el archivo de audio
        const downloadLink = document.createElement('a');
        downloadLink.href = audioUrl;
        downloadLink.download = 'grabacion.wav';
        document.body.appendChild(downloadLink);
        downloadLink.click();
        document.body.removeChild(downloadLink);

        // Limpiar los datos grabados
        recordedChunks = [];
    };

    // Iniciar la grabación
    mediaRecorder.start();
}

function detenerGrabacion() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
}

function aplicarEfecto(efecto) {
    // Obtener el archivo de audio
    var archivoAudio = document.getElementById("archivoAudio").files[0];
    var formData = new FormData();
    formData.append("audio", archivoAudio);

    // Realizar la solicitud POST al servidor Flask
    fetch('http://localhost:5000/aplicar-efecto/' + efecto, {
        method: 'POST',
        body: formData
    })
    .then(response => response.blob())
    .then(blob => {
        // Reproducir el audio resultante
        var audio = new Audio(URL.createObjectURL(blob));
        audio.play();
    })
    .catch(error => console.error('Error:', error));
}
