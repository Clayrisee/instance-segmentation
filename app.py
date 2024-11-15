import io
import base64
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import numpy as np
from pydantic import BaseModel
import uvicorn
from src.model import UNET


# Function to convert PIL image to base64
def pil_to_base64(image: Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

class InferenceModule(object):
    def __init__(self, model_path: str = "weights/best.pt", input_size=(256, 256)) -> None:
        self.model = UNET(in_channels=3, out_channels=13)
        self.model.load_state_dict(torch.load(model_path))  # Load your model weights
        self.model.eval()
        self.input_size = input_size
    
    def preprocess_image(self, img: Image) -> torch.Tensor:
        img = img.resize(self.input_size)
        img = np.array(img) / 255.0  # Normalization example, modify according to your model needs
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()  # Convert to torch.Tensor
        return img
    
    def postprocess(self, mask: torch.Tensor) -> Image:
        mask = mask.squeeze().detach().numpy()  # Convert to NumPy array and remove singleton dimensions
        mask = (mask * 255).astype(np.uint8)  # Rescale the mask to 0-255 range and cast to uint8
        if len(mask.shape) == 2:  # Ensure the mask is 2D (grayscale)
            return Image.fromarray(mask)
        else:
            # If the mask has an additional dimension (e.g., if it's a multi-channel mask), handle it accordingly
            # For example, using the first channel or converting it to grayscale
            return Image.fromarray(mask[0])  # If the mask has multiple channels, use the first one
    
    @torch.no_grad()
    def predict(self, img: Image) -> Image:
        preprocessed_img = self.preprocess_image(img)
        out_mask = self.model(preprocessed_img)
        return self.postprocess(out_mask)
    
    
# Load the trained model (modify this according to your model initialization process)
model = InferenceModule()

app = FastAPI()

# Define a model input/output schema
class ImageInput(BaseModel):
    image_base64: str

class ImageOutput(BaseModel):
    segmented_image_base64: str

# API endpoint for image segmentation
@app.post("/segment-image/", response_model=ImageOutput)
async def segment_image(file: UploadFile = File(...)):
    # Open the image file
    image = Image.open(io.BytesIO(await file.read()))
    
    mask_image = model.predict(img=image)
    
    # Convert mask to base64 format
    mask_base64 = pil_to_base64(mask_image)
    
    # Return the segmented mask as base64
    return {"segmented_image_base64": mask_base64}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)