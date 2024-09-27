from fastapi import FastAPI, File, UploadFile, HTTPException, APIRouter, Form
from fastapi.responses import Response, JSONResponse
from io import BytesIO
import numpy as np
import cv2
from PIL import Image, ImageChops, ImageEnhance
from utils.utils import decompose, merge, rm_and_make_dir
import os
from tqdm import tqdm
import shutil
from utils.test_utils_res import SegFormer_Segmentation
from typing import Optional
import tensorflow as tf

# Khởi tạo router cho các API
router = APIRouter(prefix="/api/v1")

# Tải mô hình SegFormer
used_weight = "./weights/weights_EITL_new.pth"
segformer = SegFormer_Segmentation("b2", used_weight)

# Tải mô hình ELA
ela_model = tf.keras.models.load_model("model_data/new_model_casia.h5")


def ELA(img_path):
    """Performs Error Level Analysis over a directory of images"""

    TEMP = "ela_" + "temp.jpg"
    SCALE = 10
    original = Image.open(img_path)
    try:
        original.save(TEMP, quality=90)
        temporary = Image.open(TEMP)
        diff = ImageChops.difference(original, temporary)

    except:

        original.convert("RGB").save(TEMP, quality=90)
        temporary = Image.open(TEMP)
        diff = ImageChops.difference(original.convert("RGB"), temporary)

    d = diff.load()

    WIDTH, HEIGHT = diff.size
    for x in range(WIDTH):
        for y in range(HEIGHT):
            # print(d[x,y])
            d[x, y] = tuple(k * SCALE for k in d[x, y])

    return diff


def read_imagefile(file):
    image = Image.open(BytesIO(file))
    return image


def prepare_image(image_path, image_size=(128, 128)):
    return np.array(ELA(image_path).resize(image_size)).flatten() / 255.0


### API 1: Phát hiện ảnh giả mạo bằng ELA ###
@router.post("/predict")
async def detect_ela(file: UploadFile = File(...)):
    """
    API để phát hiện ảnh giả mạo bằng phương pháp ELA (Error Level Analysis).

    Args:
        file (UploadFile): File ảnh được tải lên.

    Returns:
        JSONResponse: Kết quả dự đoán liệu ảnh có bị giả mạo hay không và độ tự tin.
    """
    try:
        # Đọc ảnh từ file tải lên
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

        # image.save(save_path)

        # Xử lý ảnh với ELA
        ela_image = ELA(save_path)
        ela_image_np = np.array(ela_image)

        # Dự đoán bằng mô hình ELA
        ela_image_resized = prepare_image(save_path, image_size=(128, 128))
        preds = ela_model.predict(ela_image_resized.reshape(-1, 128, 128, 3))
        y_pred_class = np.argmax(preds, axis=1)[0]
        confidence = float(preds[0][y_pred_class])

        # Xác định kết quả thật hay giả
        if y_pred_class == 0:
            result = "authentic"
        else:
            result = "fake"

        # if y_pred_class == 1:
        #         return JSONResponse(
        #             status_code=200,
        #             content={
        #                 "detail": "The uploaded image is authentic",
        #                 "data": {"confidence": float(preds[0][y_pred_class])},
        #             },
        #         )
        # else:
        #         return JSONResponse(
        #                 status_code=200,
        #                 content={
        #                     "detail": "The uploaded image is tampered",
        #                     "data": {"confidence": float(preds[0][y_pred_class])},
        #                 },
        #             )
        print(result, confidence)
        # Chuyển đổi ảnh ELA thành byte để trả về
        img_byte_arr = BytesIO()
        ela_image.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()

        # Trả về ảnh ELA kèm thông tin dự đoán
        return Response(
            content=img_byte_arr,
            media_type="image/png",
            headers={"X-Result": result, "X-Confidence": str(confidence)},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


### API 2: Phân đoạn ảnh giả mạo bằng SegFormer ###
@router.post("/segment-image")
async def segment_image(
    file: UploadFile = File(...),
    show_boxes: Optional[bool] = Form(False),
    show_contours: Optional[bool] = Form(True),
):
    """
    API phân đoạn ảnh bằng mô hình SegFormer.

    Args:
        file (UploadFile): File ảnh cần phân đoạn.
        show_boxes (Optional[bool]): Có hiển thị khung hình quanh vùng phát hiện hay không.
        show_contours (Optional[bool]): Có hiển thị đường bao quanh vùng phát hiện hay không.

    Returns:
        Response: Trả về ảnh phân đoạn với khung và đường bao quanh vùng phát hiện.
    """
    try:
        # Đọc và lưu ảnh tải lên
        image = read_imagefile(await file.read())
        image_np = np.array(image)
        test_path = f"./uploads/{file.filename}/"
        os.makedirs(test_path, exist_ok=True)
        save_path = test_path + file.filename
        image.save(save_path)

        # Check the format and adjust save options accordingly
        if image.format == "JPEG":
            image.save(save_path, "JPEG", quality=100)
        elif image.format == "PNG":
            image.save(save_path, "PNG", compress_level=0)
        elif format == "tiff":
            image.save(save_path, "TIFF", compression="none")
        else:
            image.save(save_path)

        # Tách nhỏ ảnh để xử lý
        test_size = "512"
        _, path_out = decompose(f"./uploads/{file.filename}/", test_size)
        dir_pre_path = "test_out/temp/input_decompose_" + test_size + "_pred/"
        rm_and_make_dir(dir_pre_path)
        img_names = os.listdir(path_out)

        # Phân đoạn từng ảnh nhỏ
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
            ):
                image_path = os.path.join(path_out, img_name)
                image = Image.open(image_path)
                _, seg_pred = segformer.detect_image_resize(image)
                save_name = img_name[:-4] + ".png"
                if not os.path.exists(dir_pre_path):
                    os.makedirs(dir_pre_path)
                seg_pred.save(os.path.join(dir_pre_path, save_name))

        # Ghép lại các phân đoạn
        dir_save_path = "./test_out/samples_predict/"
        merge(test_path, dir_pre_path, dir_save_path, test_size)
        mask_dir = dir_save_path + file.filename.split(".")[0] + ".png"

        # Load mask ảnh và tạo mặt nạ nhị phân
        mask = Image.open(mask_dir)
        mask_np = np.array(mask)
        mask_np = np.where(mask_np > 50, 255, 0).astype(np.uint8)
        mask_np = mask_np[:, :, 0]

        # Tạo mặt nạ đỏ và kết hợp với ảnh gốc
        red_mask = np.zeros_like(image_np)
        red_mask[mask_np == 255] = [255, 0, 0]
        masked_image = cv2.addWeighted(image_np, 1.0, red_mask, 0.5, 0)

        # Vẽ đường bao và khung nếu yêu cầu
        contours, _ = cv2.findContours(
            mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if show_boxes:
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(masked_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if show_contours:
            cv2.drawContours(masked_image, contours, -1, (255, 0, 0), 2)

        # Chuyển đổi ảnh và trả về
        masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
        _, img_encoded = cv2.imencode(".png", masked_image)
        img_bytes = img_encoded.tobytes()

        return Response(content=img_bytes, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app = FastAPI()
app.include_router(router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("test_main:app", host="0.0.0.0", port=8000, reload=True)
