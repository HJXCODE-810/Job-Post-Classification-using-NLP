// scripts.js
document.getElementById("jobForm").onsubmit = async function(event) {
    event.preventDefault(); // Prevent default form submission

    const formData = new FormData(this); // Gather form data

    const response = await fetch("/predict", {
        method: "POST",
        body: formData // Send form data in the request body
    });
    
    if (response.ok) { // Check if the response is successful
        const result = await response.json(); // Parse JSON response
        document.getElementById("result").innerText = "Prediction: " + result.prediction; // Display prediction
    } else {
        document.getElementById("result").innerText = "Error: Unable to get prediction."; // Handle error
    }
};
