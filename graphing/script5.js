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
    const data1 = [];
    const data2 = [];

    for (let i = 0; i < Math.min(maxFrames, filesData[0].length, filesData[1].length); i++) {
        const point1 = filesData[0][i];
        const point2 = filesData[1][i];

        data1.push({ x: point1['apex x'], y: point1['apex y'] });
        data2.push({ x: point2['apex x'], y: point2['apex y'] });
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
                        text: 'Apex X'
                    },
                    type: 'linear',
                    position: 'bottom'
                },
                y: {
                    title: {
                        display: true,
                        text: 'Apex Y'
                    }
                }
            }
        }
    });
}
