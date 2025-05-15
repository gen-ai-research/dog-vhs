document.getElementById("loginForm").addEventListener("submit", async function(event) {
    event.preventDefault(); // Prevent default form submission

    let username = document.getElementById("username").value;
    let password = document.getElementById("password").value;

    try {
        let response = await fetch("/login", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ username: username, password: password })
        });

        let modalHeader = document.getElementById("modalHeader");
        let modalTitle = document.getElementById("statusModalLabel");
        let modalMessage = document.getElementById("statusMessage");

        // ✅ Ensure response is JSON before parsing
        let result;
        try {
            result = await response.json();
        } catch (error) {
            throw new Error("Server returned non-JSON response. Check Flask output.");
        }

        if (response.ok) {
            modalHeader.classList.remove("bg-danger");
            modalHeader.classList.add("bg-success");
            modalTitle.innerText = "Success";
            modalMessage.innerText = result.message;

            let successModal = new bootstrap.Modal(document.getElementById("statusModal"));
            successModal.show();

            setTimeout(() => {
                window.location.href = "/dashboard"; // Redirect after success
            }, 2000);
        } else {
            modalHeader.classList.remove("bg-success");
            modalHeader.classList.add("bg-danger");
            modalTitle.innerText = "Error";
            modalMessage.innerText = result.message || "Login failed! Please check your username and password.";

            let errorModal = new bootstrap.Modal(document.getElementById("statusModal"));
            errorModal.show();
        }
    } catch (error) {
        console.error("Error logging in:", error);

        modalHeader.classList.remove("bg-success");
        modalHeader.classList.add("bg-danger");
        modalTitle.innerText = "Error";
        modalMessage.innerText = "An unexpected error occurred. Please try again later.";

        let errorModal = new bootstrap.Modal(document.getElementById("statusModal"));
        errorModal.show();
    }
});
