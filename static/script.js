document.getElementById("predictionForm").addEventListener("submit", async function(e) {
    e.preventDefault();

    const data = {
        age: parseInt(document.getElementById("age").value),
        gender: document.getElementById("gender").value,
        employment_status: document.getElementById("employment_status").value,
        work_environment: document.getElementById("work_environment").value,
        mental_health_history: document.getElementById("mental_health_history").value,
        seeks_treatment: document.getElementById("seeks_treatment").value,
        stress_level: parseInt(document.getElementById("stress_level").value),
        sleep_hours: parseInt(document.getElementById("sleep_hours").value),
        physical_activity_days: parseInt(document.getElementById("physical_activity_days").value),
        depression_score: parseInt(document.getElementById("depression_score").value),
        anxiety_score: parseInt(document.getElementById("anxiety_score").value),
        social_support_score: parseInt(document.getElementById("social_support_score").value),
        productivity_score: parseInt(document.getElementById("productivity_score").value),
    };

    const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
    });

    const result = await response.json();
    document.getElementById("result").innerText =
        "Predicted Mental Health Risk: " + result.prediction;
});
