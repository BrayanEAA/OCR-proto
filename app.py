from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import numpy as np
import cv2
import easyocr
import io

# Inicializar FastAPI
app = FastAPI()

# Montar carpeta de archivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar el lector de EasyOCR para español
reader = easyocr.Reader(['es'], gpu=False)

def preprocess_image_bytes(image_bytes):
    # Convertir bytes a imagen OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Imagen no válida")

    # Escalar la imagen para texto pequeño
    scale_percent = 150
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

    # Escala de grises y contraste
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ocr")
async def ocr_upload(image: UploadFile = File(...)):
    contents = await image.read()
    try:
        img = preprocess_image_bytes(contents)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

    try:
        # EasyOCR requiere imagen en formato RGB
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
        np_img = np.array(pil_img)
        results = reader.readtext(np_img)

        # Concatenar texto detectado
        text = "\n".join([res[1] for res in results])
        return {"ok": True, "text": text.strip()}
    except Exception as e:
        return JSONResponse({"ok": False, "error": "Error al extraer texto: " + str(e)}, status_code=500)
