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

let FILE1MMCONVERSION, FILE2MMCONVERSION;

function conversionOfData() {
    // Calculate the conversion ratio for pixels to mm
    const file1rulerDeltaX = [];
    const file1rulerDeltaY = [];
    const file2rulerDeltaX = [];
    const file2rulerDeltaY = [];

    filesData[0].forEach(row => {
        file1rulerDeltaX.push(row['ruler 0 mm x'] - row['ruler 20 mm x']);
        file1rulerDeltaY.push(row['ruler 0 mm y'] - row['ruler 20 mm y']);
    });

    filesData[1].forEach(row => {
        file2rulerDeltaX.push(row['ruler 0 mm x'] - row['ruler 20 mm x']);
        file2rulerDeltaY.push(row['ruler 0 mm y'] - row['ruler 20 mm y']);
    });

    // the GPT fix that doesn't work 
    // const avgFile1DeltaX = file1rulerDeltaX.reduce((acc, curr) => acc + curr, 0) / file1rulerDeltaX.length;
    // const avgFile1DeltaY = file1rulerDeltaY.reduce((acc, curr) => acc + curr, 0) / file1rulerDeltaY.length;
    // const avgFile2DeltaX = file2rulerDeltaX.reduce((acc, curr) => acc + curr, 0) / file2rulerDeltaX.length;
    // const avgFile2DeltaY = file2rulerDeltaY.reduce((acc, curr) => acc + curr, 0) / file2rulerDeltaY.length;

    // const file1Pixels20 = Math.sqrt(avgFile1DeltaX * avgFile1DeltaX + avgFile1DeltaY * avgFile1DeltaY);
    // const file2Pixels20 = Math.sqrt(avgFile2DeltaX * avgFile2DeltaX + avgFile2DeltaY * avgFile2DeltaY);

    // hard coded using an estimate calculated by hand
    FILE1MMCONVERSION = 20 / 30;
    FILE2MMCONVERSION = 20 / 30;
}

function plotAllGraphs() {
    clearPreviousGraphs();
    plotTimeVsApexX();
    plotTimeVsApexY();
    plotApexXvsApexY();
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
    const framesToHoursX = [0]; // TODO - each video might have different number of frames
    const apexXValues1 = [0];
    const apexXValues2 = [0];

    filesData[0].forEach((row, index) => {
        framesToHoursX.push(index / FPH);
        // TODO - calculate the mm conversion ratio of each frame?
        apexXValues1.push((curr['apex x'] - prev['apex x']) * FILE1MMCONVERSION);
    });

    filesData[1].forEach((row, index) => {
        apexXValues2.push((curr['apex x'] - prev['apex x']) * FILE2MMCONVERSION);
    });

    const ctx = document.getElementById('timeVsApexX').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: framesToHoursX,
            datasets: [
                {
                    label: 'Apex X (Dataset 1, mm)',
                    data: apexXValues1,
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    borderWidth: 1
                },
                {
                    label: 'Apex X (Dataset 2, mm)',
                    data: apexXValues2,
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    borderWidth: 1
                }
            ]
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
                        text: 'Apex X (mm)'
                    }
                }
            }
        }
    });
}

function plotTimeVsApexY() {
    createChartContainer('Time vs Apex Y', 'timeVsApexY');
    const FPH = 100;
    const framesToHoursY = [0];
    const apexYValues1 = [0];
    const apexYValues2 = [0];

    filesData[0].forEach((row, index) => {
        framesToHoursY.push(index / FPH);
        apexYValues1.push((curr['apex y'] - prev['apex y']) * FILE1MMCONVERSION); // Convert to mm
    });

    filesData[1].forEach((row, index) => {
        apexYValues2.push((curr['apex y'] - prev['apex y']) * FILE2MMCONVERSION); // Convert to mm
    });

    const ctx = document.getElementById('timeVsApexY').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: framesToHoursY,
            datasets: [
                {
                    label: 'Apex Y (Dataset 1, mm)',
                    data: apexYValues1,
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    borderWidth: 1
                },
                {
                    label: 'Apex Y (Dataset 2, mm)',
                    data: apexYValues2,
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    borderWidth: 1
                }
            ]
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
                        text: 'Apex Y (mm)'
                    }
                }
            }
        }
    });
}

function plotApexXvsApexY() {
    createChartContainer('Apex X vs Apex Y', 'apexXvsApexY');
    const apexXValues1 = [0];
    const apexYValues1 = [0];
    const apexXValues2 = [0];
    const apexYValues2 = [0];

    filesData[0].forEach(row => {
        apexXValues1.push((curr['apex x'] - prev['apex x']) * FILE1MMCONVERSION);
        apexYValues1.push((curr['apex y'] - prev['apex y']) * FILE1MMCONVERSION);
    });

    filesData[1].forEach(row => {
        apexXValues2.push((curr['apex x'] - prev['apex x']) * FILE2MMCONVERSION);
        apexYValues2.push((curr['apex y'] - prev['apex y']) * FILE2MMCONVERSION);
    });

    const ctx = document.getElementById('apexXvsApexY').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: apexXValues1,
            datasets: [
                {
                    label: 'Apex X vs Apex Y (Dataset 1)',
                    data: apexYValues1,
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    borderWidth: 1,
                    pointRadius: 0, // Hide points
                    fill: false
                },
                {
                    label: 'Apex X vs Apex Y (Dataset 2)',
                    data: apexYValues2,
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    borderWidth: 1,
                    pointRadius: 0, // Hide points
                    fill: false
                }
            ]
        },
        options: {
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Displacement (mm)'
                    }
                }
            }
        }
    });
}
function plotTimeVsDisplacement() {
    createChartContainer('Time vs Displacement', 'timeVsDisplacement');
    const times = [0];
    const displacements = [0];
    let cumulativeDisplacement = 0;

    for (let i = 1; i < filesData[0].length; i++) {
        const prev = filesData[0][i - 1];
        const curr = filesData[0][i];

        const deltaX = (curr['apex x'] - prev['apex x']) * (20 / 30);
        const deltaY = (curr['apex y'] - prev['apex y']) * (20 / 30);
        const displacement = Math.sqrt(deltaX * deltaX + deltaY * deltaY);
        cumulativeDisplacement += displacement;

        times.push(i / 100);
        displacements.push(cumulativeDisplacement);
    }

    const ctx = document.getElementById('timeVsDisplacement').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: times,
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

