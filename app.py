from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import pytesseract
import cv2
import numpy as np
import re

# En Render la ruta es la siguiente:
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================
# Preprocesamiento avanzado
# ============================
def preprocess_image_bytes_advanced(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Imagen no válida")

    # Escalado moderado
    scale_percent = 160
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

    # Convertir a gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Mejorar contraste suavemente
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # Suavizado ligero
    gray = cv2.GaussianBlur(gray, (3,3), 0)

    # No binarizamos agresivamente; usamos Otsu si es necesario
    _, clean = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Deskew solo si el ángulo es significativo
    coords = np.column_stack(np.where(clean < 128))  # texto negro sobre fondo blanco
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) > 0.5:  # rotar solo si es relevante
            (h, w) = clean.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            clean = cv2.warpAffine(clean, M, (w, h),
                                   flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REPLICATE)

    return Image.fromarray(clean)

# ============================
# Postprocesamiento de texto
# ============================
def postprocess_text(text):
    # Limpiar espacios innecesarios
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'\s+\.', '.', text)
    text = re.sub(r'\s+,', ',', text)
    text = text.strip()

    # Reconstruir viñetas y numeración
    lines = text.split('\n')
    processed_lines = []
    for line in lines:
        line = line.strip()
        # Detectar letras seguidas de punto o paréntesis: a. b) etc
        if re.match(r'^[a-zA-Z]\s*[.)]\s*', line):
            processed_lines.append(line)
        # Detectar números de lista
        elif re.match(r'^\d+\s*[.)]\s*', line):
            processed_lines.append(line)
        else:
            processed_lines.append(line)
    return '\n'.join(processed_lines)

# ============================
# Rutas
# ============================
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ocr")
async def ocr_upload(image: UploadFile = File(...)):
    contents = await image.read()
    try:
        pil_img = preprocess_image_bytes_advanced(contents)
    except Exception as e:
        return JSONResponse({"ok": False, "error": "Error en preprocesamiento: " + str(e)}, status_code=400)

    try:
        # OCR optimizado para documentos complejos
        custom_config = r'--oem 3 --psm 1'
        text = pytesseract.image_to_string(pil_img, lang="spa", config=custom_config)
        text = postprocess_text(text)
        return {"ok": True, "text": text}
    except Exception as e:
        return JSONResponse({"ok": False, "error": "Error al extraer texto: " + str(e)}, status_code=500)
