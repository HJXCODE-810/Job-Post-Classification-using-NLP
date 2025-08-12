document.addEventListener('DOMContentLoaded', function () {
    const form = document.querySelector('form');
    const resultDiv = document.getElementById('result');
    const fileInput = form.querySelector('input[type="file"]');
    const submitButton = form.querySelector('button[type="submit"]');

    // Add animations on form submission
    form.addEventListener('submit', function (event) {
        event.preventDefault(); // Prevent the default form submission

        const formData = new FormData(form);

        // Add animation to the submit button
        submitButton.classList.add('btn-loading');
        submitButton.innerHTML = "Uploading...";

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
            .then(response => response.text())
            .then(data => {
                resultDiv.innerHTML = data; // Display the result in the result div

                // Success animation
                resultDiv.classList.add('animate-success');
                setTimeout(() => {
                    resultDiv.classList.remove('animate-success');
                }, 2000);

                resetButton(submitButton, "Upload");
            })
            .catch(error => {
                console.error('Error:', error);
                resultDiv.innerHTML = "An error occurred while processing the file.";

                // Error animation
                resultDiv.classList.add('animate-error');
                setTimeout(() => {
                    resultDiv.classList.remove('animate-error');
                }, 2000);

                resetButton(submitButton, "Upload");
            });
    });

    // Add hover animation for file input
    fileInput.addEventListener('change', function () {
        if (fileInput.files.length > 0) {
            resultDiv.innerHTML = `<span class="file-ready">File ready to upload!</span>`;
        }
    });

    // Helper function to reset button
    function resetButton(button, text) {
        button.classList.remove('btn-loading');
        button.innerHTML = text;
    }
});
