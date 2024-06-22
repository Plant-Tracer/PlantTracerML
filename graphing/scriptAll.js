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
                    plotAllGraphs();
                }
            },
            error: function (error) {
                console.error('Error parsing the CSV file:', error);
            }
        });
    }
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
    const labels = [];
    const apexXValues = [];

    filesData[0].forEach((row, index) => {
        labels.push(index);
        apexXValues.push(row['apex x'] * (20 / 30)); // Convert to mm
    });

    const ctx = document.getElementById('timeVsApexX').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
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
                        text: 'Time (frames)'
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
    const labels = [];
    const apexYValues = [];

    filesData[0].forEach((row, index) => {
        labels.push(index);
        apexYValues.push(row['apex y'] * (20 / 30)); // Convert to mm
    });

    const ctx = document.getElementById('timeVsApexY').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
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
        apexXValues.push(row['apex x'] * (20 / 30)); // Convert to mm
        apexYValues.push(row['apex y'] * (20 / 30)); // Convert to mm
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
    const labels = [];
    const displacements = [];
    let cumulativeDisplacement = 0;

    for (let i = 1; i < filesData[0].length; i++) {
        const prev = filesData[0][i - 1];
        const curr = filesData[0][i];

        const deltaX = (curr['apex x'] - prev['apex x']) * (20 / 30);
        const deltaY = (curr['apex y'] - prev['apex y']) * (20 / 30);
        const displacement = Math.sqrt(deltaX * deltaX + deltaY * deltaY);
        cumulativeDisplacement += displacement;

        labels.push(i);
        displacements.push(cumulativeDisplacement);
    }

    const ctx = document.getElementById('timeVsDisplacement').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
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
                        text: 'Time (frames)'
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
