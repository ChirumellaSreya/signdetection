document.addEventListener('DOMContentLoaded', () => {
    const videoElement = document.getElementById('input_video');
    const canvasElement = document.getElementById('output_canvas');
    const canvasCtx = canvasElement.getContext('2d');
    
    const predictionText = document.getElementById('predictionText');
    const predictionBox = document.getElementById('predictionBox');
    const toggleAudioBtn = document.getElementById('toggleAudioBtn');
    const toggleCameraBtn = document.getElementById('toggleCameraBtn');
    
    let isAudioEnabled = true;
    let isCameraEnabled = true;
    let lastSpokenWord = "";
    let lastSpokenTime = 0;
    
    // Temporal smoothing
    const BUFFER_SIZE = 4; // Require 4 consecutive identical predictions
    let predictionBuffer = [];
    
    // Resize canvas to match video
    function resizeCanvas() {
        if (videoElement.videoWidth) {
            canvasElement.width = videoElement.videoWidth;
            canvasElement.height = videoElement.videoHeight;
        }
    }
    videoElement.addEventListener('loadedmetadata', resizeCanvas);

    // Toggle Camera
    let camera = null;
    toggleCameraBtn.addEventListener('click', () => {
        isCameraEnabled = !isCameraEnabled;
        if (isCameraEnabled) {
            toggleCameraBtn.classList.add('active');
            toggleCameraBtn.innerHTML = `
                <svg viewBox="0 0 24 24" width="24" height="24" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" class="icon"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path><circle cx="12" cy="13" r="4"></circle></svg>
                Webcam: ON
            `;
            if (camera) camera.start();
            canvasElement.style.opacity = '1';
        } else {
            toggleCameraBtn.classList.remove('active');
            toggleCameraBtn.innerHTML = `
                <svg viewBox="0 0 24 24" width="24" height="24" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" class="icon"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path><line x1="2" y1="2" x2="22" y2="22"></line></svg>
                Webcam: OFF
            `;
            if (camera) camera.stop();
            canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
            canvasElement.style.opacity = '0.3';
            updateUI("Camera Off", false);
        }
    });
    
    // Toggle Audio
    toggleAudioBtn.addEventListener('click', () => {
        isAudioEnabled = !isAudioEnabled;
        if (isAudioEnabled) {
            toggleAudioBtn.classList.add('active');
            toggleAudioBtn.innerHTML = `
                <svg viewBox="0 0 24 24" width="24" height="24" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" class="icon"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path></svg>
                Voice: ON
            `;
        } else {
            toggleAudioBtn.classList.remove('active');
            toggleAudioBtn.innerHTML = `
                <svg viewBox="0 0 24 24" width="24" height="24" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" class="icon"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><line x1="23" y1="9" x2="17" y2="15"></line><line x1="17" y1="9" x2="23" y2="15"></line></svg>
                Voice: OFF
            `;
            window.speechSynthesis.cancel();
        }
    });

    // Speak text
    function speak(text) {
        if (!isAudioEnabled) return;
        
        const now = Date.now();
        if (text !== lastSpokenWord || (now - lastSpokenTime) > 3000) {
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 1.0;
            window.speechSynthesis.speak(utterance);
            lastSpokenWord = text;
            lastSpokenTime = now;
        }
    }

    // Update UI
    function updateUI(text, isConfident) {
        predictionText.innerText = text;
        if (isConfident) {
            predictionBox.classList.add('active');
            speak(text); // Speak the full label
        } else {
            predictionBox.classList.remove('active');
        }
    }

    // Initialize MediaPipe Hands
    const hands = new Hands({locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
    }});
    
    hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 1, // Keep at 1 for high accuracy
        minDetectionConfidence: 0.75, // Slightly higher for better precision
        minTrackingConfidence: 0.5
    });

    let lastApiCallTime = 0;
    const API_THROTTLE_MS = 500; // Call AI every 500ms (balanced for speed/accuracy)
    let isApiPending = false; // Prevent multiple overlapping API calls

    // Process Hand Landmarks
    hands.onResults(async (results) => {
        resizeCanvas();
        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        
        canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
        
        if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
            const handLandmarks = results.multiHandLandmarks[0];
            
            // Draw landmarks on frontend (Smooth 30fps)
            drawConnectors(canvasCtx, handLandmarks, HAND_CONNECTIONS, {color: '#00FF00', lineWidth: 3});
            drawLandmarks(canvasCtx, handLandmarks, {color: '#FF0000', lineWidth: 1, radius: 3});
            
            const now = Date.now();
            // Only call API if throttle time passed AND no other call is pending
            if (now - lastApiCallTime > API_THROTTLE_MS && !isApiPending) {
                lastApiCallTime = now;
                isApiPending = true;

                const wrist = handLandmarks[0];
                let x_coords = [], y_coords = [], z_coords = [];
                
                for (let i = 0; i < handLandmarks.length; i++) {
                    x_coords.push(-(handLandmarks[i].x - wrist.x)); 
                    y_coords.push(handLandmarks[i].y - wrist.y);
                    z_coords.push(handLandmarks[i].z - wrist.z);
                }
                
                let normalizedLandmarks = x_coords.concat(y_coords, z_coords);
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({landmarks: normalizedLandmarks})
                    });
                    
                    const data = await response.json();
                    
                    if (data.prediction !== "Error" && data.confidence >= 0.6) {
                        predictionBuffer.push(data.prediction);
                        if (predictionBuffer.length > 2) predictionBuffer.shift();
                        
                        const allSame = predictionBuffer.length >= 2 && predictionBuffer.every(val => val === predictionBuffer[0]);
                        if (allSame) {
                            updateUI(`${data.prediction}`, true);
                        }
                    } else {
                        predictionBuffer = [];
                        updateUI("Detecting...", false);
                    }
                } catch (err) {
                    console.error("API Error:", err);
                } finally {
                    isApiPending = false;
                }
            }
        } else {
            predictionBuffer = [];
            const now = Date.now();
            if (now - lastSpokenTime > 2000) {
                updateUI("Position Hand", false);
            }
        }
        canvasCtx.restore();
    });

    // Start Camera
    camera = new Camera(videoElement, {
        onFrame: async () => {
            if (isCameraEnabled) {
                await hands.send({image: videoElement});
            }
        },
        width: 480,  // Optimized resolution for speed
        height: 360, // Optimized resolution for speed
        facingMode: 'user'
    });
    
    // Start automatically
    predictionText.innerText = "Initializing AI...";
    camera.start();
});
