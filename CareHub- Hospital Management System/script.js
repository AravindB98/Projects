document.addEventListener("DOMContentLoaded", function () {
    // Appointment Booking Form Handling
    const appointmentForm = document.getElementById("appointmentForm");
    if (appointmentForm) {
        appointmentForm.addEventListener("submit", function (event) {
            event.preventDefault();

            let doctor = document.getElementById("doctor").value;
            let patientName = document.getElementById("patientName").value.trim();
            let date = document.getElementById("date").value;
            let time = document.getElementById("time").value;

            if (doctor && patientName && date && time) {
                let appointments = JSON.parse(localStorage.getItem("appointments")) || [];
                appointments.push({ doctor, patientName, date, time });
                localStorage.setItem("appointments", JSON.stringify(appointments));

                document.getElementById("confirmationMessage").style.display = "block";
                setTimeout(() => {
                    document.getElementById("confirmationMessage").style.display = "none";
                }, 3000);
                appointmentForm.reset();
            } else {
                alert("Please fill in all fields before booking an appointment.");
            }
        });
    }

    // Login Form Validation
    const loginForm = document.getElementById("loginForm");
    if (loginForm) {
        loginForm.addEventListener("submit", function (event) {
            event.preventDefault();

            let email = document.getElementById("email");
            let password = document.getElementById("password");
            let isValid = true;

            const emailPattern = /^[a-zA-Z0-9._%+-]+@northeastern\.edu$/;
            if (!emailPattern.test(email.value.trim())) {
                email.classList.add("is-invalid");
                isValid = false;
            } else {
                email.classList.remove("is-invalid");
            }

            if (password.value.trim() === "") {
                password.classList.add("is-invalid");
                isValid = false;
            } else {
                password.classList.remove("is-invalid");
            }

            if (isValid) {
                alert("Login successful! Redirecting to Home...");
                setTimeout(() => {
                    window.location.href = "index.html";
                }, 1000);
            }
        });
    }
});