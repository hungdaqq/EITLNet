from fastapi import FastAPI, File, UploadFile, HTTPException, APIRouter, Form
from fastapi.responses import Response
from io import BytesIO
import numpy as np
import cv2
from PIL import Image
from utils.utils import decompose, merge, rm_and_make_dir
import os
from tqdm import tqdm
import shutil
from utils.test_utils_res import SegFormer_Segmentation
from typing import Optional

router = APIRouter(prefix="/api/v1")

# Load the pre-trained SegFormer model for segmentation
used_weight = "./weights/weights_EITL_new.pth"
segformer = SegFormer_Segmentation("b2", used_weight)


def read_imagefile(file) -> np.ndarray:
    """
    Read uploaded image file and convert it to OpenCV format.

    Args:
        file (UploadFile): Uploaded image file.

    Returns:
        np.ndarray: Image in OpenCV format (BGR).
    """
    # Open image using PIL and convert to numpy array
    image = Image.open(BytesIO(file))
    return image


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    show_boxes: Optional[bool] = Form(False),
    show_contours: Optional[bool] = Form(True),
):
    """
    Handle image segmentation prediction.

    Args:
        file (UploadFile): Image file to be processed.
        show_boxes (Optional[bool]): Whether to show bounding boxes around segmented objects.
        show_contours (Optional[bool]): Whether to show contours of segmented objects.

    Returns:
        Response: Segmented image with optional bounding boxes and contours.
    """
    try:
        # Read and save the uploaded image
        image = read_imagefile(await file.read())

        # Define the path for saving uploaded file
        test_path = f"./uploads/{file.filename}/"
        os.makedirs(test_path, exist_ok=True)
        image = image.convert("RGB")
        image.save(test_path + file.filename)
        image_np = np.array(image)
        test_size = "512"

        # Decompose the image for processing
        _, path_out = decompose(f"./uploads/{file.filename}/", test_size)

        dir_pre_path = "test_out/temp/input_decompose_" + test_size + "_pred/"
        rm_and_make_dir(dir_pre_path)
        img_names = os.listdir(path_out)

        # Apply segmentation on each decomposed image
        for img_name in tqdm(img_names):
            print(img_name)
            if img_name.lower().endswith(
                (
                    ".bmp",
                    ".dib",
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".pbm",
                    ".pgm",
                    ".ppm",
                    ".tif",
                    ".tiff",
                )
            ):
                image_path = os.path.join(path_out, img_name)
                image = Image.open(image_path)
                _, seg_pred = segformer.detect_image_resize(image)
                save_name = img_name[:-4] + ".png"
                if not os.path.exists(dir_pre_path):
                    os.makedirs(dir_pre_path)
                seg_pred.save(os.path.join(dir_pre_path, save_name))

        # Clean up temporary directories
        if os.path.exists("test_out/temp/input_decompose_" + test_size + "/"):
            shutil.rmtree("test_out/temp/input_decompose_" + test_size + "/")

        # Merge segmented parts into the final output
        dir_save_path = "./test_out/samples_predict/"
        merge(test_path, dir_pre_path, dir_save_path, test_size)
        mask_dir = dir_save_path + file.filename.split(".")[0] + ".png"

        # Load the mask image and create a binary mask
        mask = Image.open(mask_dir)
        mask_np = np.array(mask)
        mask_np = np.where(mask_np > 50, 255, 0).astype(np.uint8)
        mask_np = mask_np[:, :, 0]

        # Create a red mask and blend with the original image
        red_mask = np.zeros_like(image_np)
        red_mask[mask_np == 255] = [255, 0, 0]
        masked_image = cv2.addWeighted(image_np, 1.0, red_mask, 0.5, 0)

        # Find and draw contours if requested
        contours, _ = cv2.findContours(
            mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if show_boxes:
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(masked_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if show_contours:
            cv2.drawContours(masked_image, contours, -1, (255, 0, 0), 2)

        # Convert image to RGB and encode to PNG format
        masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
        _, img_encoded = cv2.imencode(".png", masked_image)
        img_bytes = img_encoded.tobytes()

        # Return the processed image as a response
        return Response(content=img_bytes, media_type="image/png")

    except Exception as e:
        # Handle exceptions and return a server error response
        raise HTTPException(status_code=500, detail=str(e))


app = FastAPI()
app.include_router(router)

if __name__ == "__main__":
    import uvicorn

    # Run the FastAPI application
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
