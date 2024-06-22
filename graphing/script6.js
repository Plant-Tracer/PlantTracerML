const filesData = [null, null];

document.getElementById('upload-csv1').addEventListener('change', function (event) {
    handleFileUpload(event, 0);
});

document.getElementById('upload-csv2').addEventListener('change', function (event) {
    handleFileUpload(event, 1);
});

function handleFileUpload(event, index) {
    const file = event.target.files[0];
    if (file) {
        Papa.parse(file, {
            header: true,
            dynamicTyping: true,
            complete: function (results) {
                filesData[index] = results.data;
                if (filesData[0] && filesData[1]) {
                    plotData();
                }
            },
            error: function (error) {
                console.error('Error parsing the CSV file:', error);
            }
        });
    }
}

function plotData() {
    const maxFrames = 300;
    const conversionFactor = 20 / 30; // 20 mm / 30 pixels
    const data1 = [];
    const data2 = [];
    let cumulativeDisplacement1 = { x: 0, y: 0 };
    let cumulativeDisplacement2 = { x: 0, y: 0 };

    for (let i = 1; i < Math.min(maxFrames, filesData[0].length, filesData[1].length); i++) {
        const point1Prev = filesData[0][i - 1];
        const point1 = filesData[0][i];
        const point2Prev = filesData[1][i - 1];
        const point2 = filesData[1][i];

        // Calculate displacement for the first CSV
        const deltaX1 = (point1['apex x'] - point1Prev['apex x']) * conversionFactor;
        const deltaY1 = (point1['apex y'] - point1Prev['apex y']) * conversionFactor;
        cumulativeDisplacement1.x += deltaX1;
        cumulativeDisplacement1.y += deltaY1;
        data1.push({ x: cumulativeDisplacement1.x, y: cumulativeDisplacement1.y });

        // Calculate displacement for the second CSV
        const deltaX2 = (point2['apex x'] - point2Prev['apex x']) * conversionFactor;
        const deltaY2 = (point2['apex y'] - point2Prev['apex y']) * conversionFactor;
        cumulativeDisplacement2.x += deltaX2;
        cumulativeDisplacement2.y += deltaY2;
        data2.push({ x: cumulativeDisplacement2.x, y: cumulativeDisplacement2.y });
    }

    const ctx = document.getElementById('apexLinePlot').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [
                {
                    label: 'Dataset 1',
                    data: data1,
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    borderWidth: 1,
                    pointRadius: 0, // Hide points
                    fill: false
                },
                {
                    label: 'Dataset 2',
                    data: data2,
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
                        text: 'Cumulative Displacement X (mm)'
                    },
                    type: 'linear',
                    position: 'bottom'
                },
                y: {
                    title: {
                        display: true,
                        text: 'Cumulative Displacement Y (mm)'
                    }
                }
            }
        }
    });
}

