
document.getElementById("registerForm").addEventListener("submit", async function(event) {
    event.preventDefault(); // Prevent default form submission

    let username = document.getElementById("username").value;
    let password = document.getElementById("password").value;

    let response = await fetch("/register", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ username: username, password: password })
    });

    let result = await response.json();
    
    if (result.success) {
        document.getElementById("successMessage").innerText = result.message;
        let successModal = new bootstrap.Modal(document.getElementById("successModal"));
        successModal.show();

        setTimeout(() => {
            window.location.href = "/dashboard"; // Redirect after success
        }, 2000); // 2-second delay before redirection
    } else {
        alert(result.message); // Show error as alert
    }
});
