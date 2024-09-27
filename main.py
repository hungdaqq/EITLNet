from fastapi import FastAPI, File, UploadFile, HTTPException, APIRouter, Form
from fastapi.responses import Response, JSONResponse
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
import tensorflow as tf
from PIL import Image, ImageChops, ImageEnhance

router = APIRouter(prefix="/api/v1")

# Load the pre-trained SegFormer model for segmentation
used_weigth = "./weights/weights_EITL_new.pth"
segformer = SegFormer_Segmentation("b2", used_weigth)
ela_model = tf.keras.models.load_model("model_data/new_model_casia.h5")


def convert_to_ela_image(image_path, quality):

    temp_filename = "temp.jpg"

    image = Image.open(image_path).convert("RGB")
    image.save(temp_filename, "JPEG", quality=quality)
    temp_image = Image.open(temp_filename)

    ela_image = ImageChops.difference(image, temp_image)

    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff

    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    return ela_image


def read_imagefile(file):
    # Open image using PIL and convert to numpy array
    image = Image.open(BytesIO(file))
    return image


def prepare_image(image_path, image_size=(128, 128)):
    return (
        np.array(convert_to_ela_image(image_path, 91).resize(image_size)).flatten()
        / 255.0
    )


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    show_boxes: Optional[bool] = Form(False),
    show_contours: Optional[bool] = Form(True),
    use_ela_auth: Optional[bool] = Form(False),
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
        image_np = np.array(image)
        test_path = f"./uploads/{file.filename}/"
        os.makedirs(test_path, exist_ok=True)
        save_path = test_path + file.filename

        # Check the format and adjust save options accordingly
        if image.format == "JPEG":
            image.save(save_path, "JPEG", quality=100)
        elif image.format == "PNG":
            image.save(save_path, "PNG", compress_level=0)
        elif format == "tiff":
            image.save(save_path, "TIFF", compression="none")
        else:
            image.save(save_path)
        test_size = "512"

        if use_ela_auth:
            # Check if the image is authentic
            ela_image = prepare_image(test_path + file.filename, image_size=(128, 128))
            preds = ela_model.predict(ela_image.reshape(-1, 128, 128, 3))
            y_pred_class = np.argmax(preds, axis=1)[0]
            print(preds)
            print(
                f"Predict: {y_pred_class}, Confidence: {preds[0][y_pred_class] * 100:.2f}"
            )
            if y_pred_class == 0:
                return JSONResponse(
                    status_code=200,
                    content={
                        "detail": "The uploaded image is authentic",
                        "data": {"confidence": float(preds[0][y_pred_class])},
                    },
                )

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
                print(seg_pred)
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
