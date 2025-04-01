document.getElementById('recommendForm').addEventListener('submit', function(event) {
    event.preventDefault();

    // Get the selected contentId from the radio buttons
    const selectedRadio = document.querySelector('input[name="content_id"]:checked');
    if (!selectedRadio) {
        alert('Please select a content ID.');
        return;
    }

    const contentId = selectedRadio.value;

    fetch('/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: `content_id=${contentId}`
    })
    .then(response => response.json())
    .then(data => {
        let recommendations = document.getElementById('recommendations');
        recommendations.innerHTML = '<h2>Recommendations:</h2>';
        
        if (data.length === 0) {
            recommendations.innerHTML += '<p>No recommendations found for this content ID.</p>';
        } else {
            data.forEach(item => {
                recommendations.innerHTML += `<p>Content ID: ${item.contentId}, Title: ${item.title}</p>`;
            });
        }
    })
    .catch(error => console.error('Error:', error));
});
