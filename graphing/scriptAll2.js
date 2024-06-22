document.getElementById('upload-csv1').addEventListener('change', function (event) {
    handleFileUpload(event, 0);
});

document.getElementById('upload-csv2').addEventListener('change', function (event) {
    handleFileUpload(event, 1);
});

const filesData = [null, null];

function handleFileUpload(event, index) {
    const file = event.target.files[0];
    if (file) {
        Papa.parse(file, {
            header: true,
            dynamicTyping: true,
            complete: function (results) {
                filesData[index] = results.data;
                if (filesData[0] && filesData[1]) {
                    conversionOfData();
                    plotAllGraphs();
                }
            },
            error: function (error) {
                console.error('Error parsing the CSV file:', error);
            }
        });
    }
}
function conversionOfData() {
    // calculate the conversion ratio for pixels to mm
    file1rulerDeltaX = [0];
    file1rulerDeltaY = [0];
    file1XLabels = [0];
    file1YLabels = [0];
    filesData[0].forEach((row, index) => {
        file1Labels.push(index);
        file1rulerDeltaX.push(row['ruler 0 mm x'] - row['ruler 20 mm x']);
        file1YLabels.push(index);
        file1rulerDeltaY.push(row['ruler 0 mm y'] - row['ruler 20 mm y']);
    });

    file2rulerDeltaX = [0];
    file2rulerDeltaY = [0];
    file2XLabels = [0];
    file2YLabels = [0];
    filesData[1].forEach((row, index) => {
        file2XLabels.push(index);
        file2rulerDeltaX.push(row['ruler 0 mm x'] - row['ruler 20 mm x']);
        file2YLabels.push(index);
        file2rulerDeltaY.push(row['ruler 0 mm y'] - row['ruler 20 mm y']);
    });

    let file1TotalDeltaX = 0;
    for (let i = 1; i < file1Labels.length; i++) {
        file1TotalDeltaX = file1rulerDeltaX[i] + file1TotalDeltaX;
    }
    const file1DeltaXAvg = file1TotalDeltaX / file1Labels.length;

    let file2TotalDeltaX = 0;
    for (let i = 1; i < file2Labels.length; i++) {
        file2TotalDeltaX = file2rulerDeltaX[i] + file2TotalDeltaX;
    }
    const file2DeltaXAvg = file2TotalDeltaX / file2Labels.length;

    let file1TotalDeltaY = 0;
    for (let i = 1; i < file1Labels.length; i++) {
        file1TotalDeltaY = file1rulerDeltaY[i] + file1TotalDeltaY;
    }
    const file1DeltaYAvg = file1TotalDeltaY / file1Labels.length;

    let file2TotalDeltaY = 0;
    for (let i = 1; i < file2Labels.length; i++) {
        file2TotalDeltaY = file2rulerDeltaY[i] + file2TotalDeltaY;
    }
    const file2DeltaYAvg = file2TotalDeltaX / file2Labels.length;

    const file1Pixels20 = Math.sqrt(file1DeltaXAvg * file1DeltaXAvg + file1DeltaYAvg * file1DeltaYAvg);
    const FILE1MMCONVERSION = 20 / file1Pixels20;

    const file2Pixels20 = Math.sqrt(file2DeltaXAvg * file2DeltaXAvg + file2DeltaYAvg * file2DeltaYAvg);
    const FILE2MMCONVERSION = 20 / file2Pixels20;
}

function plotAllGraphs() {
    clearPreviousGraphs();
    plotTimeVsApexX();
    plotTimeVsApexY();
    plotApexXvsApexY(0, 'Apex X vs Apex Y (Dataset 1)', 'apexXvsApexY1');
    plotApexXvsApexY(1, 'Apex X vs Apex Y (Dataset 2)', 'apexXvsApexY2');
    plotTimeVsDisplacement();
}

function clearPreviousGraphs() {
    const chartArea = document.getElementById('chartArea');
    chartArea.innerHTML = '';
}

function createChartContainer(title, canvasId) {
    const chartArea = document.getElementById('chartArea');
    const container = document.createElement('div');
    container.className = 'chart-container';
    const titleElement = document.createElement('h3');
    titleElement.textContent = title;
    const canvas = document.createElement('canvas');
    canvas.id = canvasId;
    container.appendChild(titleElement);
    container.appendChild(canvas);
    chartArea.appendChild(container);
}

function plotTimeVsApexX() {
    createChartContainer('Time vs Apex X', 'timeVsApexX');
    const FPH = 100;
    const framesToHoursX = [0];
    const apexXValues = [0];

    filesData[0].forEach((row, index) => {
        framesToHoursX.push(index * FPH);
        apexXValues.push(row['apex x'] * FILE1MMCONVERSION); // Convert to mm
    });

    const ctx = document.getElementById('timeVsApexX').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: framesToHoursX,
            datasets: [{
                label: 'Apex X (mm)',
                data: apexXValues,
                borderColor: 'rgba(75, 192, 192, 1)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Time (Minutes)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Apex X (mm)'
                    }
                }
            }
        }
    });
}

function plotTimeVsApexY() {
    createChartContainer('Time vs Apex Y', 'timeVsApexY');
    const framesToHoursY = [0];
    const apexYValues = [0];

    filesData[0].forEach((row, index) => {
        labels.push(index);
        apexYValues.push(row['apex y'] * FILE1MMCONVERSION); // Convert to mm
    });

    const ctx = document.getElementById('timeVsApexY').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: framesToHoursY,
            datasets: [{
                label: 'Apex Y (mm)',
                data: apexYValues,
                borderColor: 'rgba(75, 192, 192, 1)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Time (frames)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Apex Y (mm)'
                    }
                }
            }
        }
    });
}

function plotApexXvsApexY(index, title, canvasId) {
    createChartContainer(title, canvasId);
    const apexXValues = [];
    const apexYValues = [];

    filesData[index].forEach(row => {
        apexXValues.push(row['apex x'] * FILE1MMCONVERSION); // Convert to mm
        apexYValues.push(row['apex y'] * FILE1MMCONVERSION); // Convert to mm
    });

    const ctx = document.getElementById(canvasId).getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: apexXValues,
            datasets: [{
                label: title,
                data: apexYValues,
                borderColor: 'rgba(75, 192, 192, 1)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderWidth: 1,
                pointRadius: 0, // Hide points
                fill: false
            }]
        },
        options: {
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Apex X (mm)'
                    },
                    type: 'linear',
                    position: 'bottom'
                },
                y: {
                    title: {
                        display: true,
                        text: 'Apex Y (mm)'
                    }
                }
            }
        }
    });
}

function plotTimeVsDisplacement() {
    createChartContainer('Time vs Displacement', 'timeVsDisplacement');
    const frames = [0];
    const displacements = [0];
    let cumulativeDisplacement = 0;

    for (let i = 1; i < filesData[0].length; i++) {
        const prev = filesData[0][i - 1];
        const curr = filesData[0][i];

        const deltaX = (curr['apex x'] - prev['apex x']) * FILE1MMCONVERSION;
        const deltaY = (curr['apex y'] - prev['apex y']) * FILE1MMCONVERSION;
        const displacement = Math.sqrt(deltaX * deltaX + deltaY * deltaY);
        cumulativeDisplacement += displacement;

        labels.push(i);
        displacements.push(cumulativeDisplacement);
    }

    const ctx = document.getElementById('timeVsDisplacement').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: frames,
            datasets: [{
                label: 'Displacement (mm)',
                data: displacements,
                borderColor: 'rgba(75, 192, 192, 1)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Time (hours)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Displacement (mm)'
                    }
                }
            }
        }
    });
}
