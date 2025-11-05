import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

IMAGE_DIR = 'src/templates/images'

SAMPLE_IMAGE_NAME = 'src/templates/images/images.jpg'


def read_prepare_images():
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.bmp', '.jpeg'))]

    if not image_files:
        print(f"❌ Lỗi: Không tìm thấy ảnh trong thư mục '{IMAGE_DIR}'. Vui lòng kiểm tra lại.")
        return None, None
    images = {}
    for filename in image_files:
        path = os.path.join(IMAGE_DIR, filename)
        img = cv2.imread(path)
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images[filename] = img_rgb
        else:
            print(f"⚠️ Cảnh báo: Không thể đọc file ảnh: {filename}")

    if not images:
        print("❌ Lỗi: Không có ảnh nào được đọc thành công.")
        return None, None
    
    sample_img_rgb = images.get(SAMPLE_IMAGE_NAME)
    if sample_img_rgb is None:
        sample_img_rgb = next(iter(images.values()))
        print(f"⚠️ Cảnh báo: Không tìm thấy '{SAMPLE_IMAGE_NAME}'. Sử dụng ảnh đầu tiên cho yêu cầu 2.4.")
    return images, sample_img_rgb


def show_each_image_in_window(images):
    print("\n[HW2.1] Hiển thị từng ảnh trên từng cửa sổ...")
    for filename, img_rgb in images.items():
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow(f"Ảnh: {filename}", img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_rgb_layers_in_windows(images):
    print("\n[HW2.2] Hiển thị từng lớp màu RGB trên từng cửa sổ...")
    for filename, img_rgb in images.items():
        for i, channel in enumerate(['R', 'G', 'B']):
            blank = np.zeros_like(img_rgb[:, :, 0])
            layer = [blank, blank, blank]
            layer[i] = img_rgb[:, :, i]
            channel_img = np.stack(layer, axis=-1)
            channel_bgr = cv2.cvtColor(channel_img, cv2.COLOR_RGB2BGR)
            cv2.imshow(f"{filename} - Kênh {channel}", channel_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_grayscale_images(images):
    print("\n[HW2.3] Hiển thị ảnh xám trên từng cửa sổ...")
    for filename, img_rgb in images.items():
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        cv2.imshow(f"{filename} - Ảnh Xám", gray_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def rotate_image_100_times(sample_img_rgb):
    print("\n[HW2.4] Xoay ảnh 100 lần, mỗi lần 5 độ...")
    img_bgr = cv2.cvtColor(sample_img_rgb, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    cX, cY = w // 2, h // 2

    for i in range(100):
        angle = i * 5
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        rotated = cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        cv2.imshow("Xoay ảnh", rotated)
        key = cv2.waitKey(100)
        if key == 27:  # ESC
            break
    cv2.destroyAllWindows()

def show_cropped_center_images(images):
    print("\n[HW2.5] Hiển thị ảnh đã cắt 1/4 từ tâm...")
    for filename, img_rgb in images.items():
        h, w = img_rgb.shape[:2]
        crop_h, crop_w = h // 4, w // 4
        cX, cY = w // 2, h // 2
        startX = cX - crop_w // 2
        startY = cY - crop_h // 2
        cropped = img_rgb[startY:startY+crop_h, startX:startX+crop_w]
        cropped_bgr = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
        cv2.imshow(f"{filename} - Cắt 1/4", cropped_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    images, sample_img_rgb = read_prepare_images()
    if images is not None:
        show_each_image_in_window(images)
        show_rgb_layers_in_windows(images)
        show_grayscale_images(images)
        rotate_image_100_times(sample_img_rgb)
        show_cropped_center_images(images)
