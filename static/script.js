const preview = document.getElementById("preview");
const imageInput = document.getElementById("image");
const form = document.getElementById("ocr-form");
const resultContainer = document.getElementById("result-container");
const resultText = document.getElementById("result");
const copyBtn = document.getElementById("copy-btn");

imageInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) {
        preview.style.display = "none";
        preview.src = "";
        return;
    }
    const reader = new FileReader();
    reader.onload = (event) => {
        preview.src = event.target.result;
        preview.style.display = "block";
    };
    reader.readAsDataURL(file);
});

copyBtn.addEventListener("click", () => {
    navigator.clipboard.writeText(resultText.textContent)
        .then(() => alert("Texto copiado al portapapeles"))
        .catch(() => alert("Error al copiar el texto"));
});

form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (imageInput.files.length === 0) {
        alert("Selecciona una imagen primero");
        return;
    }

    const formData = new FormData();
    formData.append("image", imageInput.files[0]);

    try {
        const response = await fetch("/ocr", { method: "POST", body: formData });
        const data = await response.json();
        if (data.ok) {
            resultText.textContent = data.text;
            resultContainer.style.display = "block";
        } else {
            alert("Error al procesar la imagen: " + (data.error || "desconocido"));
        }
    } catch (err) {
        alert("Error al conectar con el servidor: " + err.message);
    }
});
