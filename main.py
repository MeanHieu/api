from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from app.face_recognition import process_image  
from io import BytesIO
from PIL import Image
import numpy as np

app = FastAPI()

@app.get("/", response_class=RedirectResponse)
async def root():
    return "/docs"

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    img = Image.open(BytesIO(await file.read()))  
    result_image, info = process_image(np.array(img))  

    result_image_pil = Image.fromarray(result_image)  
    result_image_pil.save("app/static/result_image.jpg")  

    # Trả về HTML hiển thị thông tin và hình ảnh
    return HTMLResponse(content=f"""
    <html>
        <body>
            <h2>Information:</h2>
            <p>{info}</p>
            <img src="/static/result_image.jpg" />
        </body>
    </html>
    """)
