const canvas = document.getElementById('paintCanvas');
//console.log(canvas);

const ctx = canvas.getContext('2d');

let mode = 'draw'; // Modes: 'draw', 'select', 'move', 'resize'
let lines = []; // Stores all lines
let selectedLine = null;
let selectedPoint = null; // Tracks which point (start/end) is selected for resizing
let isDrawing = false;
let startX, startY;
let canvasStates = [];

let currentStateIndex = -1;
let currentType = null; // Stores the current line type (Long, Short, or Verte)
let isLineTypeEnabled = false;
// Define a color mapping for line types
const LINE_COLORS = {
    Long: '#288e0a',  // 
    Short: '#87128b', 
    Verte: '#198754',

};
const STROKE_COLOR = '#1513a0';

// Load and set a background image
let backgroundImage = new Image();
let originalImageWidth, originalImageHeight;
let scaleX = 1;
let scaleY = 1;

let defaultImageWidth = 800;
let defaultImageHeight = 600;


// Initialize dotenv
// window.dotenv.config();

//const apiUrl = process.env.API_URL;


function showToast(message, isSuccess = true) {
    const toastElement = document.getElementById('dynamicToast');
    const toastBody = document.getElementById('toastBody');

    // Change text and background color based on success/failure
    toastBody.textContent = message;
    toastElement.classList.remove('bg-success', 'bg-danger');
    toastElement.classList.add(isSuccess ? 'bg-success' : 'bg-danger');

    // Show the toast
    const toast = new bootstrap.Toast(toastElement);
    toast.show();

    // Auto-dismiss after 3 seconds
    setTimeout(() => {
        toast.hide();
    }, 3000);
}

// Add event listeners for buttons
document.getElementById('longBtn').addEventListener('click', () => {
    currentType = 'Long';
    isLineTypeEnabled = !lines.some(line => line.type === 'Long');
    updateButtonStates();
});

document.getElementById('shortBtn').addEventListener('click', () => {
    currentType = 'Short';
    isLineTypeEnabled = !lines.some(line => line.type === 'Short');
    updateButtonStates();
});

document.getElementById('verteBtn').addEventListener('click', () => {
    currentType = 'Verte';
    isLineTypeEnabled = !lines.some(line => line.type === 'Verte');
    updateButtonStates();
});

function clearCanvas() {
    lines = [];
    selectedLine = null;
    document.getElementById('longBtn').disabled = 0
    document.getElementById('shortBtn').disabled = 0
    document.getElementById('verteBtn').disabled = 0
    drawCanvas();
}

document.getElementById('clearCanvas').addEventListener('click', () => {
    clearCanvas();
});

document.getElementById('calculateVHS').addEventListener('click', printCoordinates);

canvas.addEventListener('mousedown', (e) => {
    const { offsetX, offsetY } = e;
    //console.log(offsetX, offsetY);
    //console.log(lines);
    //console.log(currentType, "mode", mode);

    const existingLine = lines.find(line => line.type === currentType);
    //console.log("existing lines", existingLine);

    if (existingLine) {
        selectedLine = existingLine;
        mode = 'select';
        canvas.style.cursor = 'default';
        handleInteraction(offsetX, offsetY);
        return;
    }

    if (mode === 'draw' && lines.length < 3 && currentType) {
        isDrawing = true;
        startX = offsetX;
        startY = offsetY;

    } else {
        handleInteraction(offsetX, offsetY);
    }
});


canvas.addEventListener('mousemove', (e) => {
    const { offsetX, offsetY } = e;
    if (mode === 'draw' && isDrawing) {
        drawCanvas();
        drawTempLine(startX, startY, offsetX, offsetY);
    } else if (isDrawing && selectedLine) {
        if (selectedPoint) {
            // Resize
            if (selectedPoint === 'start') {
                selectedLine.x1 = offsetX;
                selectedLine.y1 = offsetY;
            } else if (selectedPoint === 'end') {
                selectedLine.x2 = offsetX;
                selectedLine.y2 = offsetY;
            }
        } else {
            // Move
            const dx = offsetX - startX;
            const dy = offsetY - startY;
            selectedLine.x1 += dx;
            selectedLine.y1 += dy;
            selectedLine.x2 += dx;
            selectedLine.y2 += dy;
            startX = offsetX;
            startY = offsetY;
        }
        drawCanvas();
    }
});

canvas.addEventListener('mouseup', (e) => {
    if (mode === 'draw' && isDrawing && lines.length < 3 && currentType) {
        const { offsetX, offsetY } = e;
        const newLine = { x1: startX, y1: startY, x2: offsetX, y2: offsetY, type: currentType };
        lines.push(newLine);
        selectedLine = newLine;
        mode = 'select';
        canvas.style.cursor = 'default';
    }
    isDrawing = false;
    selectedPoint = null;
    drawCanvas();
    saveState();
});

function adjustShortLongLines() {
    const shortLine = lines.find(line => line.type === 'Short');
    const longLine = lines.find(line => line.type === 'Long');

    if (shortLine && longLine) {
        // Point A is the start of the long line
        const Ax = longLine.x1;
        const Ay = longLine.y1;

        // Point B is the end of the long line
        const Bx = longLine.x2;
        const By = longLine.y2;

        // Point C is the start of the short line
        const Cx = shortLine.x1;
        const Cy = shortLine.y1;

        // Calculate the vector AB
        const ABx = Bx - Ax;
        const ABy = By - Ay;

        // Normalize AB vector
        const ABLength = Math.sqrt(ABx * ABx + ABy * ABy);
        const ABNormalizedX = ABx / ABLength;
        const ABNormalizedY = ABy / ABLength;

        // Calculate the perpendicular vector to AB (rotated 90 degrees clockwise)
        const perpX = ABNormalizedY;
        const perpY = -ABNormalizedX;

        // Calculate the length of CD
        const CDLength = Math.sqrt(Math.pow(shortLine.x2 - Cx, 2) + Math.pow(shortLine.y2 - Cy, 2));

        // Adjust point D to be perpendicular to AB and to the right of C
        shortLine.x2 = Cx + perpX * CDLength;
        shortLine.y2 = Cy + perpY * CDLength;

        drawCanvas();

    }
}

function handleInteraction(x, y) {
    const clickedLine = getLineAtPosition(x, y);
    if (clickedLine) {
        selectedLine = clickedLine;
        selectedPoint = getPointAtPosition(x, y, selectedLine);
        if (selectedPoint) {
            mode = 'resize';
            canvas.style.cursor = 'pointer';
        } else {
            mode = 'move';
            canvas.style.cursor = 'move';
        }
        isDrawing = true;
        startX = x;
        startY = y;
    } else {
        selectedLine = null;
        mode = 'draw';
        canvas.style.cursor = 'crosshair';
        // if (currentType) {
        //     mode = 'draw';
        //     canvas.style.cursor = 'crosshair';
        // }
    }
    adjustShortLongLines();
    updateButtonStates();
    drawCanvas();
    printCoordinates();

    //Save automatically
    document.getElementById("saveCoordinates").click();
}

function drawCanvas() {
    // console.log("drawCanvas called....")
    // console.log(canvas.width, canvas.height);

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(backgroundImage, 0, 0, defaultImageWidth, defaultImageHeight);

    // Calculate scaling factor to fit image within canvas
    // const scale = Math.min(canvas.width / backgroundImage.width, canvas.height / backgroundImage.height);

    // // Calculate centered position
    // const x = (canvas.width / 2) - (backgroundImage.width / 2) * scale;
    // const y = (canvas.height / 2) - (backgroundImage.height / 2) * scale;

    // // Draw the image centered and scaled
    // //ctx.drawImage(backgroundImage, x, y, backgroundImage.width * scale, backgroundImage.height * scale);

    // ctx.drawImage(backgroundImage, x, y);

    lines.forEach(line => {
        drawLine(line, line === selectedLine);
    });

    //drawAxes()
}


// function drawCanvas() {
//     //console.log("drawCanvas called....")

//     canvas.width = backgroundImage.width;
//     canvas.height = backgroundImage.height;

//     ctx.clearRect(0, 0, canvas.width, canvas.height);

//     // Draw the image as the background
//     ctx.drawImage(backgroundImage, 0, 0, canvas.width, canvas.height);

//     const scaleX = canvas.width / originalImageWidth;
//     const scaleY = canvas.height / originalImageHeight;

//     // Draw each line with scaling
//     lines.forEach(line => {
//         drawLine({
//             x1: line.x1 * scaleX,
//             y1: line.y1 * scaleY,
//             x2: line.x2 * scaleX,
//             y2: line.y2 * scaleY,
//             type:line.type
//         }, line.type);
//     });

//     //drawAxes()
// }

const coordinatesDisplay = document.getElementById('coordinatesDisplay');

canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = Math.round(e.clientX - rect.left);
    const y = Math.round(e.clientY - rect.top);
    coordinatesDisplay.textContent = `X: ${x}, Y: ${y}`;
});

canvas.addEventListener('mouseout', () => {
    coordinatesDisplay.textContent = '';
});


function drawLine(line, isSelected = false) {
    const strokeColor = isSelected ? STROKE_COLOR : LINE_COLORS[line.type];
    
    ctx.beginPath();

    x1 = line.x1;
    y1 = line.y1;

    x2 = line.x2;
    y2 = line.y2;

    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 4;
    ctx.stroke();

    drawPoint(x1, y1, "#0529f0");
    drawPoint(x2, y2, "#0529f0");

    // ctx.fillStyle = strokeColor;
    // ctx.font = '12px Arial';
    // ctx.fillText(line.type, (line.x1 + line.x2) / 2, (line.y1 + line.y2) / 2);
}


function drawPoint(x, y, color) {
    ctx.beginPath();
    ctx.arc(x, y, 0, 0, 2 * Math.PI); // 5 is a good visible size
    ctx.fillStyle = color;
    ctx.fill();
    ctx.stroke();
}

function drawTempLine(x1, y1, x2, y2) {
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.strokeStyle = LINE_COLORS[currentType] || '#1b77b0';
    ctx.lineWidth = 2;
    ctx.stroke();
}

function getLineAtPosition(x, y) {
    return lines.find(line => pointToLineDistance(x, y, line) < 5);
}

function getPointAtPosition(x, y, line) {
    const startDistance = Math.hypot(x - line.x1, y - line.y1);
    const endDistance = Math.hypot(x - line.x2, y - line.y2);
    if (startDistance < 8) return 'start';
    if (endDistance < 8) return 'end';
    return null;
}

function pointToLineDistance(x, y, line) {
    const { x1, y1, x2, y2 } = line;
    const A = x - x1;
    const B = y - y1;
    const C = x2 - x1;
    const D = y2 - y1;

    const dot = A * C + B * D;
    const lenSq = C * C + D * D;
    const param = lenSq !== 0 ? dot / lenSq : -1;

    let xx, yy;

    if (param < 0) {
        xx = x1;
        yy = y1;
    } else if (param > 1) {
        xx = x2;
        yy = y2;
    } else {
        xx = x1 + param * C;
        yy = y1 + param * D;
    }

    const dx = x - xx;
    const dy = y - yy;
    return Math.sqrt(dx * dx + dy * dy);
}

function saveState() {
    currentStateIndex++;
    if (currentStateIndex < canvasStates.length) {
        canvasStates.length = currentStateIndex;
    }
    canvasStates.push(canvas.toDataURL());
}

function undo() {
    if (currentStateIndex > 0) {
        currentStateIndex--;
        restoreState(canvasStates[currentStateIndex]);
    }
}

function redo() {
    if (currentStateIndex < canvasStates.length - 1) {
        currentStateIndex++;
        restoreState(canvasStates[currentStateIndex]);
    }
}

function restoreState(state) {
    let img = new Image();
    img.src = state;
    img.onload = function () {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0);
    }
}

let seconds = 0;
let interval = null;

let sec=0
interval = setInterval(() => {
    sec++;
  
    const minutes = Math.floor(sec / 60);
    const seconds = sec % 60;

    // Format with leading zero if needed (e.g., 2:05 instead of 2:5)
    const formatted = `${minutes}:${seconds.toString().padStart(2, '0')}`;

    document.getElementById("timer1").textContent = formatted ;
    var imageName = document.getElementById('imageNameDisplay').innerText.trim();
    var totalImageText = imageName + "<br/>" + (currentImageIndex + 1);
    document.getElementById("totalImages").innerHTML = totalImageText;

    
  }, 1000);

 // Start when page loads
 function startTimer() {
    interval = setInterval(() => {
      seconds++;
      document.getElementById("timer").textContent = seconds;
    }, 1000);
  }

  function resetTimer() {
    clearInterval(interval);
    seconds = 0;
    document.getElementById("timer").textContent = seconds;
    startTimer();
  }
  startTimer();
  image_no = 0

document.addEventListener('keydown', function (e) {
    if (e.key === 'Delete' && selectedLine) {
        const index = lines.indexOf(selectedLine);
        if (index > -1) {
            lines.splice(index, 1);
            selectedLine = null;
            drawCanvas();
        }
    }
    if (e.ctrlKey && e.key === 'z') {
        e.preventDefault();
        undo();
    }
    if (e.ctrlKey && e.key === 'y') {
        e.preventDefault();
        redo();
    }
    if (e.ctrlKey && e.key === 's') {
        e.preventDefault();
        document.getElementById("saveCoordinates").click();
    }

    if (e.key === 'ArrowLeft') {
        e.preventDefault();
        
        if (currentImageIndex > 0) {
            loadImage(currentImageIndex - 1);
            //resetPredictions();
        }
    };

    

    if (e.key === 'ArrowRight') {
        console.log("Right Arrow")
        e.preventDefault();
        if (currentImageIndex < totalImages - 1) {
            loadImage(currentImageIndex + 1);
            resetTimer();
            image_no++
            document.getElementById("totalImage").innerText = ` / ${image_no}`; // Display total images
            document.getElementById("vhslabel").innerText = '';
            document.getElementById("vhslabel1").innerText = '';
            //resetPredictions();
        }
    }


});

canvas.addEventListener('mouseup', function () {
    saveState();
    //console.log(mode, currentType)

});

window.addEventListener('message', function (event) {
    //console.log("listener")
    if (event.data.type === 'FROM_CHILD') {
        window.parent.postMessage({
            type: 'streamlit:setComponentValue',
            value: event.data.value
        }, '*');
    }
});


document.getElementById("saveCoordinates").addEventListener("click", () => {
    if (!lines || lines.length !== 3) {
        showToast("Please ensure exactly 3 lines are drawn before saving.",false);
        return;
    }

    // Extract six points from the drawn lines
    let sixPoints = [];
    lines.forEach(line => {
        sixPoints.push([line.x1 / scaleX, line.y1 / scaleY]);
        sixPoints.push([line.x2 / scaleX, line.y2 / scaleY]);
    });

    // Get the image name (Modify as per how you store images)
    const imageName = document.getElementById("imageNameDisplay").innerText.trim() || "default_image.png";

    // Prepare the payload
    const coordinatesData = {
        six_points: sixPoints
    };

    // Send data to Flask API
    fetch(`/save_coordinates/${imageName}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(coordinatesData)
    })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showToast(`Error: ${data.error}`,false);
            } else {
                showToast(`Coordinates saved successfully!`);

                printCoordinates();
            }
        })
        .catch(error => {
            showToast('Error saving coordinates. Please try again.',false);
        });
});

//#region "VHS Calculation"

function roundPoint([x, y]) {
    return [+(x.toFixed(4)), +(y.toFixed(4))];
}
async function calculateVHS(lines) {
    if (lines.length !== 3) {
        console.error('Invalid number of lines. Expected 3, got', lines.length);
        return null;
    }

    // const points = [
    //     roundPoint([lines[0].x1 / scaleX, lines[0].y1 / scaleY]),
    //     roundPoint([lines[0].x2 / scaleX, lines[0].y2 / scaleY]),
    //     roundPoint([lines[1].x1 / scaleX, lines[1].y1 / scaleY]),
    //     roundPoint([lines[1].x2 / scaleX, lines[1].y2 / scaleY]),
    //     roundPoint([lines[2].x1 / scaleX, lines[2].y1 / scaleY]),
    //     roundPoint([lines[2].x2 / scaleX, lines[2].y2 / scaleY]),
    // ];

    const points = [
        ([lines[0].x1 / scaleX, lines[0].y1 / scaleY]),
        ([lines[0].x2 / scaleX, lines[0].y2 / scaleY]),
        ([lines[1].x1 / scaleX, lines[1].y1 / scaleY]),
        ([lines[1].x2 / scaleX, lines[1].y2 / scaleY]),
        ([lines[2].x1 / scaleX, lines[2].y1 / scaleY]),
        ([lines[2].x2 / scaleX, lines[2].y2 / scaleY]),
    ];
    
    const imageName = document.getElementById('imageNameDisplay').innerText.trim();
 

    // const points = [
    //     [lines[0].x1, lines[0].y1],
    //     [lines[0].x2, lines[0].y2],
    //     [lines[1].x1, lines[1].y1],
    //     [lines[1].x2, lines[1].y2],
    //     [lines[2].x1, lines[2].y1],
    //     [lines[2].x2, lines[2].y2]
    // ];

    try {
        const response = await fetch('/calculate_vhs', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ imageName,points }),
            credentials: 'include'  // Add this line
        });
        return await response.json();
    } catch (error) {
        console.error('Error sending points to API:', error);
        return null;
    }
}

function formatNumber(num) {
    return parseFloat(num.toFixed(8)).toString();
}

function printCoordinates() {
    // let coordinatesText = "Line Coordinates:<br>";
    // lines.forEach((line, index) => {
    //     coordinatesText += `${line.type} Line ${index + 1}: (${Math.round(line.x1)}, ${Math.round(line.y1)}) to (${Math.round(line.x2)}, ${Math.round(line.y2)})<br>`;
    // });


    //let tableHTML = '<table border="0" style="width:300px;"><tr><th>X1, Y1</th><th>X2, Y2</th></tr>';

    // lines.forEach((line, index) => {
    //     tableHTML += `
    //         <tr>
    //             <td>(${formatNumber(line.x1)}, ${formatNumber(line.y1)})</td>
    //             <td>(${formatNumber(line.x2)}, ${formatNumber(line.y2)})</td>
    //         </tr>
    //     `;
    // });

    // tableHTML += '</table>';

    let tableHTML = ''


    const longLine = lines.find(line => line.type === 'Long');
    const shortLine = lines.find(line => line.type === 'Short');
    const verteLine = lines.find(line => line.type === 'Verte');

    if (!longLine || !shortLine || !verteLine) return null;

    calculateVHS([longLine, shortLine, verteLine]).then(result => {
        if (result) {
            if (result !== null) {
                tableHTML += `<b>VHS:</b> ${result.VHS}`;
                document.getElementById('vhslabel').innerText = "\n"+ result.VHS;
            } else {
                tableHTML += `<b>VHS:</b> N/A (Not enough data or invalid lines)`;
            }

            const coordinatesLabel = document.getElementById('coordinatesLabel');
            coordinatesLabel.style.display = "block";

            if (coordinatesLabel) {
                coordinatesLabel.innerHTML = tableHTML;
            } else {
                const label = document.createElement('div');
                label.id = 'coordinatesLabel';
                label.innerHTML = tableHTML;
                document.body.appendChild(label);
            }

           
        } else {
            showToast("error",false)
        }
    });

    // calculateVHS(lines).then(result => {
    //     if (result !== null) {
    //         coordinatesText += `<br>AB: ${result.AB}<br>CD: ${result.CD}<br>EF: ${result.EF}<br>VHS: ${result.VHS}`;
    //     } else {
    //         coordinatesText += `<br>AB: N/A<br>CD: N/A<br>EF: N/A<br>VHS: N/A (Not enough data or invalid lines)`;
    //     }

    //     const coordinatesLabel = document.getElementById('coordinatesLabel');
    //     if (coordinatesLabel) {
    //         coordinatesLabel.innerHTML = coordinatesText;
    //     } else {
    //         const label = document.createElement('div');
    //         label.id = 'coordinatesLabel';
    //         label.innerHTML = coordinatesText;
    //         document.body.appendChild(label);
    //     }
    // });
}

//#endregion


function updateButtonStates() {
    document.getElementById('longBtn').disabled = lines.some(line => line.type === 'Long');
    document.getElementById('shortBtn').disabled = lines.some(line => line.type === 'Short');
    document.getElementById('verteBtn').disabled = lines.some(line => line.type === 'Verte');
}

// canvas.addEventListener('mousedown', (e) => {
//     const { offsetX, offsetY } = e;
//     if (mode === 'draw' && lines.length < 3 && currentType && isLineTypeEnabled) {
//         isDrawing = true;
//         startX = offsetX;
//         startY = offsetY;
//     } else {
//         handleInteraction(offsetX, offsetY);
//     }
// });

///*************** Upload Images  *****************/
document.addEventListener('DOMContentLoaded', function () {

    //console.log("DOM loaded", document.getElementById('fileInput'))

    document.getElementById('fileInput').addEventListener('change', function () {
        const files = Array.from(this.files);
        if (files.length > 0) {
            //console.log('Files selected:', files.map(file => file.name));
        } else {
            //console.log('No files selected.');
        }
    });


    document.getElementById('modeSelect').addEventListener('change', function () {
        const fileInput = document.getElementById('fileInput');
        if (this.value === 'truth') {
            fileInput.accept = '.mat'; // Only allow .mat files for Ground Truth
        } else {
            fileInput.accept = 'image/*'; // Allow images for Images mode
        }
    });

    document.getElementById('uploadForm').addEventListener('submit', function (e) {
        e.preventDefault();

        const fileInput = document.getElementById('fileInput');
        const modeSelect = document.getElementById('modeSelect');

        if (!fileInput || !fileInput.files.length) {
            showModal('Error', 'Please select files before uploading.');
            return;
        }

        const formData = new FormData(this);

        // Add the selected mode to the form data
        formData.append('mode', modeSelect.value);

        fetch('/upload_images', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    showModal('Error', data.error);
                } else {
                    showModal('Success', 'Files uploaded successfully!');
                    setTimeout(() => {
                        window.location.reload();
                    }, 2000);
                }
            })
            .catch(error => {
                console.error('Error uploading files:', error);
                showModal('Error', 'Error uploading files. Please try again.');
            });
    });

    function showModal(type, message) {
        const modalBody = document.getElementById('uploadModalBody');

        // Set text and styling based on message type
        if (type === 'Error') {
            modalBody.style.color = 'red'; // Error messages in red
        } else if (type === 'Success') {
            modalBody.style.color = 'green'; // Success messages in green
        }

        document.getElementById('uploadModalLabel').innerText = type;
        modalBody.innerHTML = message;

        var modal = new bootstrap.Modal(document.getElementById('uploadModal'));
        modal.show();
    }

});

// document.getElementById('imageInput').addEventListener('change', function (e) {
//     const preview = document.getElementById('imagePreview');
//     preview.innerHTML = '';

//     for (const file of this.files) {
//         const reader = new FileReader();
//         reader.onload = function (e) {
//             const img = document.createElement('img');
//             img.src = e.target.result;
//             img.style.maxWidth = '200px';
//             img.style.margin = '5px';
//             preview.appendChild(img);
//         }
//         reader.readAsDataURL(file);
//     }
// });


/********** Image Slider *************/
let currentImageIndex = 1;
let totalImages = 0;

function updateImageNavigation() {
    document.getElementById('firstImage').disabled = currentImageIndex === 0;
    document.getElementById('prevImage').disabled = currentImageIndex === 0;
    document.getElementById('nextImage').disabled = currentImageIndex === totalImages - 1;
    document.getElementById('lastImage').disabled = currentImageIndex === totalImages - 1;
    document.getElementById('imageIndex').value = currentImageIndex + 1;
}

function loadImage(index) {
    clearCanvas();

    fetch(`/get_image/${index}`)
        .then(response => response.json())
        .then(data => {
            if (data.url) {
                currentImageIndex = data.index;
                updateImageNavigation();

                // // Store original image dimensions for scaling points
                // originalImageWidth = data.width;
                // originalImageHeight = data.height;

                // canvas.height = 600;
                // canvas.width = 800;

                // // // Scale factor (if canvas size is different from original image)
                // scaleX = canvas.width / data.width;
                // scaleY = canvas.height / data.height;

                // Load image onto canvas
                loadImageOnCanvas(data.url, data.image_name);

                document.getElementById('imageNameDisplay').innerText = data.image_name;
                modeSelectionChanged();

            } else {
                //console.log('Image not found');
            }
        })
        .catch(error => console.error('Error loading image:', error));
}

function loadImageOnCanvas(imageUrl, imageName) {
    let img = new Image();
    img.src = imageUrl;

    backgroundImage = img;
    //console.log("bg", img);


    backgroundImage.onload = function () {
        // ✅ Update the displayed image name
        document.getElementById('imageNameDisplay').innerText = imageName;

        drawCanvas();
        //here if the 6 points present, then just draw the lines. 
    };
}

function resetPredictions() {
    document.getElementById("groundTruthRadio").checked = true;
}

document.addEventListener("DOMContentLoaded", function () {

    // Load the first image when the page loads
    document.getElementById('firstImage').addEventListener('click', () => {
        loadImage(0);
        //resetPredictions();
    });

    document.getElementById('prevImage').addEventListener('click', () => {
        if (currentImageIndex > 0) {
            loadImage(currentImageIndex - 1);
            //resetPredictions();
        }
    });

    
    document.getElementById('nextImage').addEventListener('click', () => {
        if (currentImageIndex < totalImages - 1) {
            loadImage(currentImageIndex + 1);
            
            //resetPredictions();
        }
    });

    document.getElementById('lastImage').addEventListener('click', () => {
        loadImage(totalImages - 1);
        //resetPredictions();
    });

    document.getElementById('imageIndex').addEventListener('change', (e) => {
        const newIndex = parseInt(e.target.value) - 1;
        if (newIndex >= 0 && newIndex < totalImages) {
            loadImage(newIndex);
            //resetPredictions();
        }
    });

    // Load total number of images and first image on page load
    fetch('/get_total_images')
        .then(response => response.json())
        .then(data => {
            totalImages = data.total; // Store total images count
            document.getElementById("totalImagesSpan").innerText = ` / ${totalImages}`; // Display total images

            updateImageNavigation(); // Update navigation controls
            loadImage(0); // Load the first image
        })
        .catch(error => console.error('Error getting total images:', error));
});


/** PREDICT BUTTON  */
// document.getElementById('predictBtn').addEventListener('click', function () {
//     const imageName = document.getElementById('imageNameDisplay').innerText.trim(); // Get current image name

//     if (!imageName) {
//         alert("No image selected for prediction.");
//         return;
//     }

//     fetch(`/predict_points/${imageName}`)
//         .then(response => response.json())
//         .then(data => {
//             if (data.error) {
//                 alert(data.error);
//                 return;
//             }

//             //console.log("Predicted Data:", data);

//             // Update predictedLabels div with values
//             updatePredictedLabels(data.labeled_points, data.VHS);

//             // Draw labeled points and connecting lines
//             drawPredictedPointsAndLines(data.labeled_points);
//         })
//         .catch(error => console.error('Error fetching predicted points:', error));
// });

// // Function to reset canvas and draw new points and lines

// function resetAndDraw(responseData) {
//     // Load background image
//     const img = new Image();
//     img.src = responseData.overlayed_image;
//     img.onload = function () {
//         // Resize canvas to match image dimensions
//         canvas.width = img.width;
//         canvas.height = img.height;

//         // Clear canvas
//         ctx.clearRect(0, 0, canvas.width, canvas.height);

//         // Draw the image as the background
//         ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

//         // Extract six points from response
//         const sixPoints = responseData.six_points;

//         // Scale factor (if canvas size is different from original image)
//         const scaleX = canvas.width / responseData.width;
//         const scaleY = canvas.height / responseData.height;

//         // Define line pairs with corresponding types
//         const lines = [
//             { x1: sixPoints.PA[0], y1: sixPoints.PA[1], x2: sixPoints.PB[0], y2: sixPoints.PB[1], type: "Long" },
//             { x1: sixPoints.PC[0], y1: sixPoints.PC[1], x2: sixPoints.PD[0], y2: sixPoints.PD[1], type: "Short" },
//             { x1: sixPoints.PE[0], y1: sixPoints.PE[1], x2: sixPoints.PF[0], y2: sixPoints.PF[1], type: "Verte" }
//         ];

//         // Draw each line with scaling
//         lines.forEach(line => {
//             drawLine({
//                 x1: line.x1 * scaleX,
//                 y1: line.y1 * scaleY,
//                 x2: line.x2 * scaleX,
//                 y2: line.y2 * scaleY
//             }, line.type);
//         });
//     };
// }


function showPredictions(mode) {
    //console.log("show predictions called....");

    predictions_data = {}
    const imageName = document.getElementById('imageNameDisplay').innerText.trim(); // Get current image name
    // ✅ Fetch prediction when turned ON
    fetch(`/predict_direct/${imageName}?mode=${mode}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                //console.log(data.error);
                //Clear all existing datapoints.
                clearCanvas();

                //predictToggle.checked = false; // Reset switch if error
                return;
            }

            /********** START : FOR MIN WIDTH /HEIGHT ****************/

            if (data.width > 800) {
                console.clear();
                //console.log("Width More than 800");

                defaultImageWidth = data.width;
                defaultImageHeight = data.height;

                // Calculate aspect ratio
                const aspectRatio = defaultImageWidth / defaultImageHeight;

                // Set maximum dimension
                const maxDimension = 800;

                if (defaultImageWidth > defaultImageHeight) {
                    // Width is greater, so set it to 800px and adjust height
                    canvas.width = maxDimension;
                    canvas.height = Math.round(maxDimension / aspectRatio);
                } else {
                    // Height is greater or equal, so set it to 800px and adjust width
                    canvas.height = maxDimension;
                    canvas.width = Math.round(maxDimension * aspectRatio);
                }

                // Update defaultImageWidth and defaultImageHeight to the new canvas dimensions
                defaultImageWidth = canvas.width;
                defaultImageHeight = canvas.height;

                /********** END: FOR MIN WIDTH /HEIGHT ****************/
            }

            else {
                defaultImageWidth = data.width;
                defaultImageHeight = data.height;

                // Store original image dimensions for scaling points
                canvas.width = defaultImageWidth;
                canvas.height = defaultImageHeight;
            }


            //Show VHS

            const coordinatesLabel = document.getElementById('coordinatesLabel');
            coordinatesLabel.style.display = "block";
            coordinatesLabel.innerHTML = `<b>VHS: ${data.VHS}</b>`;
            document.getElementById('vhslabel1').innerText = data.VHS;
            


            //Load the Six Points into AB,CD,EF
            // Extract six points from response
            const sixPoints = data.six_points;

            // Scale factor (if canvas size is different from original image)
            scaleX = canvas.width / data.width;
            scaleY = canvas.height / data.height;

            // Define line pairs with corresponding types
            lines = [
                { x1: sixPoints.PA[0] * scaleX, y1: sixPoints.PA[1] * scaleY, x2: sixPoints.PB[0] * scaleX, y2: sixPoints.PB[1] * scaleY, type: "Long" },
                { x1: sixPoints.PC[0] * scaleX, y1: sixPoints.PC[1] * scaleY, x2: sixPoints.PD[0] * scaleX, y2: sixPoints.PD[1] * scaleY, type: "Short" },
                { x1: sixPoints.PE[0] * scaleX, y1: sixPoints.PE[1] * scaleY, x2: sixPoints.PF[0] * scaleX, y2: sixPoints.PF[1] * scaleY, type: "Verte" }
            ];

            selectedLine = lines[0];
            currentType = "Long";

            backgroundImage.src = data.overlayed_image;
            //originalImageHeight = responseData.height;
            //originalImageWidth = responseData.width;
            drawCanvas();

            //console.log(scaleX, scaleY);


            //loadImageOnCanvas(data.overlayed_image, imageName)
        })
        // .then(data => {
        //     if (data.error) {
        //         alert(data.error);
        //         predictToggle.checked = false; // Reset switch if error
        //         return;
        //     }

        //     //console.log("Predicted Data:", data);
        //     predictions_data = data;

        //     document.getElementById("predictedLabels").innerHTML = "<span>Predicted VHS: " + data.VHS + "</span>";

        //     loadImageOnCanvas(data.overlayed_image, imageName)

        // })
        .catch(error => {
            console.error('Error fetching predicted points:', error);
        });
}

function getSelectedViewMode() {
    const selectedRadio = document.querySelector('input[name="viewMode"]:checked');
    return selectedRadio ? selectedRadio.value : '';
}

function modeSelectionChanged() {
    const imageName = document.getElementById('imageNameDisplay').innerText.trim(); // Get current image name

    if (!imageName) {
        return;
    }

    var mode = getSelectedViewMode();
    //console.log(mode);

    if (mode == "both") {
        canvas.style.pointerEvents = 'none';
    }
    else {
        canvas.style.pointerEvents = 'auto';
    }

    showPredictions(mode);
}

document.addEventListener("DOMContentLoaded", function () {



    // Example usage
    document.querySelector('.radio-container').addEventListener('change', () => {
        modeSelectionChanged();
    });

    // const predictRadio = document.getElementById('predictRadio');
    // const showBothRadio = document.getElementById('showBothRadio');
    // const groundTruthRadio = document.getElementById('groundTruthRadio');


    // // ✅ Handle "Show Both" mode
    // showBothRadio.addEventListener("change", function () {
    //     modeSelectionChanged();
    //     // const imageName = document.getElementById('imageNameDisplay').innerText.trim(); // Get current image name

    //     // if (!imageName) {
    //     //     return;
    //     // }
    //     // if (this.checked) {
    //     //     showPredictions("both");

    //     //     // in this case , disable the canvas.
    //     // }
    // });

    // // ✅ Handle "Ground Truth" mode
    // groundTruthRadio.addEventListener("change", function () {
    //     modeSelectionChanged();
    //     // //console.log("Truth")
    //     // const imageName = document.getElementById('imageNameDisplay').innerText.trim(); // Get current image name

    //     // if (!imageName) {
    //     //     return;
    //     // }
    //     // if (this.checked) {
    //     //     showPredictions("truth");

    //     //     canvas.style.pointerEvents = 'auto';

    //     // }
    // });

    // // ✅ Handle "Prediction" mode
    // predictRadio.addEventListener("change", function () {
    //     modeSelectionChanged();

    //     // //console.log("Predict checked");
    //     // const imageName = document.getElementById('imageNameDisplay').innerText.trim(); // Get current image name

    //     // if (!imageName) {
    //     //     return;
    //     // }
    //     // if (this.checked) {
    //     //     showPredictions("predict");
    //     //     canvas.style.pointerEvents = 'auto';

    //     // }
    // });
});

// document.addEventListener("DOMContentLoaded", function () {
//     const predictToggle = document.getElementById('predictToggle');

//     if (!predictToggle) {
//         console.error("Predict toggle switch not found in the DOM.");
//         return;
//     }

//     predictToggle.onchange = function () {
//         const imageName = document.getElementById('imageNameDisplay').innerText.trim(); // Get current image name

//         if (!imageName) {
//             alert("No image selected for prediction.");
//             predictToggle.checked = false; // Reset switch if no image
//             return;
//         }

//         if (predictToggle.checked) {
//             showPredictions();
//         } else {
//             //Reset the background image.
//             var imgSrc = backgroundImage.src;
//             imgSrc = imgSrc.replace("overlayed_images", "images");

//             backgroundImage.src = imgSrc;
//             //console.log(imgSrc);

//             handleInteraction();
//             //loadImage(currentImageIndex);
//         }
//     };

//     document.getElementById("showBothButton").addEventListener('click', function () {
//         showPredictions(true);
//     });
// });

document.getElementById('startPredictionBtn').addEventListener('click', function () {
    const progressContainer = document.getElementById('progress-container');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');

    // Show progress bar
    progressContainer.style.display = 'block';
    progressBar.value = 0;
    progressText.innerText = "0%";

    // Create EventSource to receive progress updates
    const eventSource = new EventSource('/generate_predictions');

    eventSource.onmessage = function (event) {
        const data = JSON.parse(event.data);

        if (data.progress) {
            progressBar.value = data.progress;
            progressText.innerText = `${data.progress}%`;

            if (data.progress >= 100) {
                eventSource.close();
                showToast("✅ Predictions generated successfully!");
                progressContainer.style.display = 'none'; // Hide after completion
            }
        }
    };

    eventSource.onerror = function (event) {
        console.error("EventSource failed.", event);
       // alert("SSE Error: " + event.type);
        progressContainer.style.display = 'none'; // Hide after completion
    };
    
});




