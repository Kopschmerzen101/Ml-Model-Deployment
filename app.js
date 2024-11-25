const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const clearButton = document.getElementById('clear');
const predictButton = document.getElementById('predict');
const resultText = document.getElementById('result');

// Variables for drawing
let isDrawing = false;

// Event listeners for drawing
canvas.addEventListener('mousedown', () => (isDrawing = true));
canvas.addEventListener('mouseup', () => (isDrawing = false));
canvas.addEventListener('mousemove', draw);

// Clear canvas
clearButton.addEventListener('click', () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    resultText.innerHTML = 'Prediction: <strong>N/A</strong>';
});

// Draw on canvas
function draw(event) {
    if (!isDrawing) return;
    ctx.fillStyle = '#000';
    ctx.lineWidth = 20;
    ctx.lineCap = 'round';

    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    ctx.beginPath();
    ctx.arc(x, y, ctx.lineWidth / 2, 0, Math.PI * 2);
    ctx.fill();
}

// Predict button click
predictButton.addEventListener('click', async () => {
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const resizedData = resizeImage(imageData, 28, 28);
    const normalizedData = preprocessData(resizedData);

    // Load the model
    const modelUrl = 'https://storage.googleapis.com/tfjs-models/tfjs/mnist/model.json';
    const model = await tf.loadLayersModel(modelUrl);

    // Predict
    const inputTensor = tf.tensor(normalizedData).reshape([1, 28, 28, 1]);
    const prediction = model.predict(inputTensor);
    const predictedClass = prediction.argMax(1).dataSync()[0];

    resultText.innerHTML = `Prediction: <strong>${predictedClass}</strong>`;
});

// Resize the canvas data to 28x28
function resizeImage(imageData, width, height) {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = width;
    tempCanvas.height = height;
    const tempCtx = tempCanvas.getContext('2d');

    tempCtx.drawImage(canvas, 0, 0, width, height);

    const resizedData = tempCtx.getImageData(0, 0, width, height).data;
    const grayscaleData = [];
    for (let i = 0; i < resizedData.length; i += 4) {
        // Convert to grayscale (using only red channel as grayscale equivalent)
        grayscaleData.push(resizedData[i]);
    }
    return grayscaleData;
}

// Normalize pixel values to [0, 1]
function preprocessData(data) {
    return data.map(pixel => pixel / 255);
}
