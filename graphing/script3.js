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
    const frameRate = 48 / 3600; // 48 frames per hour = 48/3600 frames per second
    const maxFrames = 300;

    const labels = [];
    const displacements1 = [];
    const displacements2 = [];

    for (let i = 1; i < Math.min(maxFrames, filesData[0].length, filesData[1].length); i++) {
        const frame1A = filesData[0][i - 1];
        const frame2A = filesData[0][i];
        const frame1B = filesData[1][i - 1];
        const frame2B = filesData[1][i];

        // Calculate displacement for the first CSV
        const deltaXA = frame2A['apex x'] - frame1A['apex x'];
        const deltaYA = frame2A['apex y'] - frame1A['apex y'];
        const displacementA = Math.sqrt(deltaXA * deltaXA + deltaYA * deltaYA);

        // Calculate displacement for the second CSV
        const deltaXB = frame2B['apex x'] - frame1B['apex x'];
        const deltaYB = frame2B['apex y'] - frame1B['apex y'];
        const displacementB = Math.sqrt(deltaXB * deltaXB + deltaYB * deltaYB);

        // Calculate time in seconds since the start
        const time = frame2A.frame_number * frameRate;

        labels.push(time.toFixed(2));
        displacements1.push(displacementA);
        displacements2.push(displacementB);
    }

    const ctxDisplacement = document.getElementById('displacementChart').getContext('2d');
    new Chart(ctxDisplacement, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Displacement CSV 1',
                    data: displacements1,
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    borderWidth: 1
                },
                {
                    label: 'Displacement CSV 2',
                    data: displacements2,
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
                        text: 'Time (seconds)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Displacement'
                    }
                }
            }
        }
    });
}
