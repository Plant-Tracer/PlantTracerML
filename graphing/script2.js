document.getElementById('upload-csv').addEventListener('change', function (event) {
    const file = event.target.files[0];
    if (file) {
        Papa.parse(file, {
            header: true,
            dynamicTyping: true,
            complete: function (results) {
                const data = results.data;
                const frameRate = 48 / 3600; // 48 frames per hour = 48/3600 frames per second

                const labels = [];
                const displacements = [];
                const rates = [];

                for (let i = 1; i < data.length; i++) {
                    const frame1 = data[i - 1];
                    const frame2 = data[i];

                    // Calculate displacement using Pythagorean theorem
                    const deltaX = frame2['apex x'] - frame1['apex x'];
                    const deltaY = frame2['apex y'] - frame1['apex y'];
                    const displacement = Math.sqrt(deltaX * deltaX + deltaY * deltaY);

                    // Calculate time in seconds since the start
                    const time = frame2.frame_number * frameRate;

                    // Calculate rate of movement
                    const rate = displacement * frameRate; // displacement per second

                    labels.push(time.toFixed(2));
                    displacements.push(displacement);
                    rates.push(rate);
                }

                // Create displacement chart
                const ctxDisplacement = document.getElementById('displacementChart').getContext('2d');
                new Chart(ctxDisplacement, {
                    type: 'line',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: 'Displacement',
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

                // Create rate of movement chart
                const ctxRate = document.getElementById('rateChart').getContext('2d');
                new Chart(ctxRate, {
                    type: 'line',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: 'Rate of Movement',
                            data: rates,
                            borderColor: 'rgba(255, 99, 132, 1)',
                            backgroundColor: 'rgba(255, 99, 132, 0.2)',
                            borderWidth: 1
                        }]
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
                                    text: 'Rate of Movement'
                                }
                            }
                        }
                    }
                });
            },
            error: function (error) {
                console.error('Error parsing the CSV file:', error);
            }
        });
    }
});
