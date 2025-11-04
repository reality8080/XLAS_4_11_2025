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

def display_all_on_one_figure(images, title):
    num_images = len(images)

    cols = min(5, num_images)
    rows = int(np.ceil(num_images/cols))

    plt.figure(figsize=(15, 5 * rows))
    plt.suptitle(title, fontsize=16)
    
    for i, (filename, img) in enumerate(images.items()):
        plt.subplot(rows, cols, i + 1)

        plt.imshow(img) 
        plt.title(f"{i+1}. {filename}", fontsize=10)
        plt.axis('off')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 

def display_color_channels_grid(images):
    print("\n[Yêu cầu 2.2] Tách lớp màu RGB và hiển thị trên 3 figure riêng biệt...")
    
    # Lọc chỉ những ảnh màu (có 3 kênh)
    color_images = {k: v for k, v in images.items() if len(v.shape) == 3}
    if not color_images:
        print("❌ Lỗi: Không có ảnh màu để thực hiện tách kênh.")
        return
        
    num_images = len(color_images)
    cols = min(5, num_images) # Tối đa 5 cột
    rows = int(np.ceil(num_images / cols))
    
    channels = {'R': 0, 'G': 1, 'B': 2} # Kênh 0=R, 1=G, 2=B sau khi chuyển sang RGB
    color_map  = {'R': 'red', 'G': 'green', 'B': 'blue'}

    for channel_name, channel_index in channels.items():
        plt.figure(figsize=(15, 5 * rows))

        plt.suptitle(f"Kênh Màu {channel_name} (Tất Cả Ảnh)", fontsize=16, color=color_map.get(channel_name, 'black'))
        
        for i, (filename, img_rgb) in enumerate(color_images.items()):
            # Lấy kênh màu mong muốn (ví dụ: R = img_rgb[:,:,0])
            blank = np.zeros_like(img_rgb[:,:,0])
            channel_img  =np.stack([ 
                img_rgb[:,:,0] if channel_index == 0  else blank,
                img_rgb[:,:,1] if channel_index == 1  else blank,
                img_rgb[:,:,2] if channel_index == 2  else blank,

            ], axis=-1)

            
            plt.subplot(rows, cols, i + 1)
            # Dùng cmap='gray' hoặc colormap tương ứng để hiển thị
            plt.imshow(channel_img) 
            plt.title(f"{filename} ({channel_name})", fontsize=10)
            plt.axis('off')
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

def display_grayscale_with_colorbar(images):
    """
    Yêu cầu 2.3: Hiển thị ảnh màu và ảnh xám tương ứng, kèm thanh màu.
    """
    print("\n[Yêu cầu 2.3] Hiển thị Ảnh Màu, Ảnh Xám, và Thanh Màu...")
    
    color_images = {k: v for k, v in images.items() if len(v.shape) == 3}
    if not color_images:
        print("❌ Lỗi: Không có ảnh màu để chuyển đổi.")
        return
        
    num_images = len(color_images)
    # Hiển thị 2 cột (Ảnh Màu | Ảnh Xám)
    cols = 2
    rows = num_images
    
    plt.figure(figsize=(10, 5 * rows))
    plt.suptitle("So Sánh Ảnh Màu và Ảnh Xám (Kèm Thanh Màu)", fontsize=16)

    for i, (filename, img_rgb) in enumerate(color_images.items()):
        # 1. Chuyển BGR sang Xám (CVT_BGR2GRAY) - dùng ảnh gốc BGR của OpenCV
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Cột 1: Ảnh Màu
        plt.subplot(rows, cols, 2 * i + 1)
        plt.imshow(img_rgb)
        plt.title(f"{filename} (Màu)")
        plt.axis('off')

        # Cột 2: Ảnh Xám (Sử dụng `cmap='gray'` và `colorbar`)
        ax = plt.subplot(rows, cols, 2 * i + 2)
        # Hiển thị ảnh xám
        im = ax.imshow(gray_img, cmap='gray', vmin=0, vmax=255)
        plt.title(f"{filename} (Xám)")
        plt.axis('off')
        
        # Thêm thanh màu (colorbar)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Mức Độ Xám (0-255)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# def display_rotation_animation(sample_img_rgb):
#     """
#     Yêu cầu 2.4: Xoay và Thu phóng 50 lần, hiển thị các bước chính trên 1 figure.
#     """
#     print("\n[Yêu cầu 2.4] Xoay và Thu phóng (50 lần, 15 độ, 90% size) - Hiển thị 9 bước chính...")

#     # Chuyển lại sang BGR cho các phép biến đổi hình học của OpenCV
#     sample_img_bgr = cv2.cvtColor(sample_img_rgb, cv2.COLOR_RGB2BGR)

#     (h, w) = sample_img_bgr.shape[:2]
#     (cX, cY) = (w // 2, h // 2)
    
#     current_rotation = 0.0
#     current_scale = 1.0
    
#     # Chỉ hiển thị 9 bước chính (bước 1, 5, 10, 15, 20, 25, 30, 40, 50)
#     steps_to_display = [1, 5, 10, 15, 20, 25, 30, 40, 50]
#     results = {}
    
#     for i in range(1, 51):
#         current_rotation += 15.0
#         current_scale *= 0.9
        
#         M = cv2.getRotationMatrix2D((cX, cY), current_rotation, current_scale)
#         rotated_scaled_bgr = cv2.warpAffine(sample_img_bgr, M, (w, h))
        
#         if i in steps_to_display:
#              # Chuyển kết quả về RGB để lưu và hiển thị bằng Matplotlib
#             rotated_scaled_rgb = cv2.cvtColor(rotated_scaled_bgr, cv2.COLOR_BGR2RGB)
#             results[f"Bước {i} (Góc: {int(current_rotation % 360)}°, Tỷ lệ: {current_scale:.2f})"] = rotated_scaled_rgb

#     # Hiển thị 9 kết quả chính trên 1 figure
#     plt.figure(figsize=(18, 12))
#     plt.suptitle(f"Biến đổi Xoay và Thu phóng (Ảnh Gốc: {SAMPLE_IMAGE_NAME})", fontsize=16)
    
#     cols = 3
#     rows = int(np.ceil(len(results) / cols))
    
#     for i, (title, img) in enumerate(results.items()):
#         plt.subplot(rows, cols, i + 1)
#         plt.imshow(img)
#         plt.title(title, fontsize=10)
#         plt.axis('off')
        
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.show()

from matplotlib.animation import FuncAnimation

def animate_rotation_only(sample_img_rgb):
    print("\n🎞️ Đang tạo mô hình động xoay tròn ảnh...")

    # Chuyển sang BGR để xử lý bằng OpenCV
    sample_img_bgr = cv2.cvtColor(sample_img_rgb, cv2.COLOR_RGB2BGR)

    h, w = sample_img_bgr.shape[:2]
    cX, cY = w // 2, h // 2

    fig, ax = plt.subplots(figsize=(6, 6))
    img_display = ax.imshow(sample_img_rgb)
    ax.axis('off')

    def update(frame):
        angle = frame * 5  # Xoay mỗi bước 5 độ
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)  # scale = 1.0 (giữ nguyên kích thước)
        rotated = cv2.warpAffine(sample_img_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        rotated_rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
        img_display.set_data(rotated_rgb)
        return [img_display]

    anim = FuncAnimation(fig, update, frames=72, interval=100, blit=True)
    plt.suptitle("Mô hình động: Xoay tròn ảnh quanh tâm", fontsize=16)
    plt.show()



def display_cropped_images(images):
    """
    Yêu cầu 2.5: Hiển thị toàn bộ ảnh gốc và ảnh đã cắt 1/4 từ tâm.
    """
    print("\n[Yêu cầu 2.5] Hiển thị Ảnh Gốc và Ảnh Đã Cắt 1/4 từ Tâm...")

    num_images = len(images)
    # Hiển thị 2 cột (Ảnh Gốc | Ảnh Cắt)
    cols = 2
    rows = num_images
    
    plt.figure(figsize=(12, 5 * rows))
    plt.suptitle("So Sánh Ảnh Gốc và Ảnh Đã Cắt 1/4", fontsize=16)

    for i, (filename, img_rgb) in enumerate(images.items()):
        h, w = img_rgb.shape[:2]
        
        # 1. Tính toán vùng cắt (giống như code gốc)
        crop_h, crop_w = h // 4, w // 4
        cX, cY = w // 2, h // 2
        
        startX = cX - (crop_w // 2)
        endX = cX + (crop_w - (crop_w // 2) if crop_w % 2 != 0 else crop_w // 2)
        startY = cY - (crop_h // 2)
        endY = cY + (crop_h - (crop_h // 2) if crop_h % 2 != 0 else crop_h // 2)
        
        # 2. Cắt ảnh (dùng NumPy slicing)
        cropped_img = img_rgb[startY:endY, startX:endX]

        # Cột 1: Ảnh Gốc
        plt.subplot(rows, cols, 2 * i + 1)
        plt.imshow(img_rgb)
        plt.title(f"{filename} (Gốc)")
        plt.axis('off')

        # Cột 2: Ảnh Đã Cắt
        plt.subplot(rows, cols, 2 * i + 2)
        plt.imshow(cropped_img)
        plt.title(f"{filename} (Cắt 1/4)")
        plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def main():
    print("🚀 Bắt đầu thực thi các yêu cầu lập trình OpenCV kết hợp Matplotlib...")
    
    images, sample_img_rgb = read_prepare_images()
    
    if images is None:
        print("Chương trình kết thúc do không đọc được ảnh.")
        return

    # --- Thực thi các yêu cầu hiển thị mới ---

    # Yêu cầu 1: Hiển thị toàn bộ ảnh trên cùng figure
    display_all_on_one_figure(images, "Yêu cầu 2.1: Toàn Bộ Ảnh Gốc")
    
    # Yêu cầu 2: Hiển thị 3 figure cho 3 kênh màu R, G, B
    display_color_channels_grid(images)
    
    # Yêu cầu 3: Hiển thị ảnh màu, ảnh xám kèm thanh màu
    display_grayscale_with_colorbar(images)
    
    # Yêu cầu 4: Hiển thị các bước xoay/thu phóng (chỉ 9 bước quan trọng)
    animate_rotation_only(sample_img_rgb)
    
    # Yêu cầu 5: Hiển thị ảnh gốc và ảnh cắt
    display_cropped_images(images)

    print("\n✅ Chương trình đã hoàn tất tất cả các yêu cầu hiển thị sử dụng Matplotlib.")


if __name__ == "__main__":
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
        print(f"Đã tạo thư mục '{IMAGE_DIR}'. Vui lòng đặt ít nhất 10 ảnh vào đó với các định dạng khác nhau (png, jpg, bmp) và đặt tên file mẫu là '{SAMPLE_IMAGE_NAME}'.")
    else:
        main()