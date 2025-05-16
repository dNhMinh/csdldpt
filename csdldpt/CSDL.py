import mysql.connector
import numpy as np
from scipy.spatial import KDTree
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import dlib
import os
# from sklearn.preprocessing import StandardScaler
# import faiss

BASE_IMAGE_DIR = "D:/Minh/ky2nam4/he co so du lieu da phuong tien/project2/csdldpt/"


# 🔹 Khởi tạo detector & predictor của dlib
detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Đường dẫn đến file model

predictor = dlib.shape_predictor("D:/Minh/ky2nam4/he co so du lieu da phuong tien/project2/csdldpt/shape_predictor_68_face_landmarks.dat")


# 🔹 Hàm trích xuất đặc trưng từ ảnh
def extract_facial_features(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    if len(faces) == 0:
        return None

    for face in faces:
        landmarks = predictor(gray, face)

        points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])

        # Tính toán các đặc trưng
        features = [
            np.linalg.norm(points[42] - points[39]),   # Khoảng cách hai mắt
            np.linalg.norm(points[17] - points[21]),   # Chiều dài cung mày trái
            np.linalg.norm(points[22] - points[26]),   # Chiều dài cung mày phải
            np.linalg.norm(points[21] - points[22]),   # Khoảng cách giữa hai lông mày
            np.linalg.norm(points[48] - points[54]),   # Chiều rộng miệng
            np.linalg.norm(points[51] - points[57]),   # Chiều cao miệng
            np.linalg.norm(points[37] - points[41]),   # Độ mở mắt trái
            np.linalg.norm(points[43] - points[47]),   # Độ mở mắt phải
            np.linalg.norm(points[27] - points[33]),   # Chiều dài sống mũi
            np.linalg.norm(points[31] - points[35]),   # Chiều rộng mũi
            np.linalg.norm(points[0] - points[16]) / np.linalg.norm(points[8] - points[27]),  # Tỉ lệ mặt
            np.arctan2(points[8][1] - points[0][1], points[8][0] - points[0][0]),  # Độ dốc hàm
            #đặc trưng thêm người già
            np.linalg.norm(points[43] - points[47]) - np.linalg.norm(points[37] - points[41]), #độ sụp mí
            # np.mean(gray[points[37][1]:points[41][1], points[37][0]:points[41][0]]), #nếp nhăn quanh mắt
            np.linalg.norm(points[1] - points[14]), #gò má, má hóp
            np.linalg.norm(points[8] - points[10]),  # Khoảng cách từ cằm đến một điểm trên cằm (ví dụ điểm 10)
            np.linalg.norm(points[8] - points[17])  # Khoảng cách giữa cằm và điểm trên cổ (ví dụ điểm 17)
        ]
        return np.array(features)

# 🔹 Hàm tải dữ liệu từ MySQL
def load_data():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="123456",
        database="face_img_db"
    )
    cursor = conn.cursor()

    cursor.execute("SELECT path, eye_distance, left_eyebrow_length, right_eyebrow_length, brow_distance, "
                   "mouth_width, mouth_height, left_eye_opening, right_eye_opening, "
                   "nose_length, nose_width, face_ratio, jaw_slope, eyelid_droop, cheek_bone, chin_distance, neck_distance FROM face_features")
    
    rows = cursor.fetchall()
    paths = [row[0] for row in rows]
    data = np.array([row[1:] for row in rows])

    cursor.close()
    conn.close()
    return paths, data

# 🔹 Hàm tìm ảnh gần nhất
# def find_nearest_image(query_point, k):
#     ## distances, indices  = kd_tree.query([query_point], k = k)
#     ## nearest_paths = [paths[i] for i in indices[0]]

#     distances, indices  = kd_tree.query([query_point], k = k + 1)  # tìm k+1 ảnh để loại bỏ ảnh gốc
#     result_idx = []
#     for i, d in enumerate(distances[0]):
#         if d > 1e-3:  # bỏ qua ảnh chính nó (khoảng cách gần bằng 0)
#             result_idx.append(indices[0][i])
#         if len(result_idx) == k:
#             break

#     nearest_paths = [paths[i] for i in result_idx]

#     # Hiển thị ảnh gần nhất
#     for i in range(0,3):
#         # img = Image.open(nearest_paths[i]).resize((200, 200))
#         img_path = os.path.join(BASE_IMAGE_DIR, nearest_paths[i])
#         img = Image.open(img_path).resize((200, 200))
#         img = ImageTk.PhotoImage(img)
#         result_image_labels[i].config(image=img)
#         result_image_labels[i].image = img
#         result_labels[i].config(text=f"Ảnh gần thứ {i+1}: {nearest_paths[i]}\nKhoảng cách: {distances[0][i]:.2f}")


# def find_nearest_image(query_point, k):
#     distances, indices = kd_tree.query([query_point], k + 5)  # lấy dư để lọc ảnh gốc

#     result_idx = []
#     result_dist = []

#     for idx, dist in zip(indices[0], distances[0]):
#         if dist < 1e-6:  # ảnh giống chính nó → bỏ qua
#             continue
#         result_idx.append(idx)
#         result_dist.append(dist)
#         if len(result_idx) == k:
#             break

#     for i in range(k):
#         img_path = os.path.join(BASE_IMAGE_DIR, paths[result_idx[i]])
#         img = Image.open(img_path).resize((200, 200))
#         img = ImageTk.PhotoImage(img)
#         result_image_labels[i].config(image=img)
#         result_image_labels[i].image = img
#         result_labels[i].config(text=f"Ảnh gần thứ {i+1}: {paths[result_idx[i]]}\nKhoảng cách: {result_dist[i]:.2f}")


def find_nearest_image(query_point, k, query_path):
    distances, indices = kd_tree.query([query_point], k + 10)  # lấy dư để lọc ảnh gốc

    result_idx = []
    result_dist = []

    for idx, dist in zip(indices[0], distances[0]):
        candidate_path = os.path.join(BASE_IMAGE_DIR, paths[idx])
        if os.path.normpath(candidate_path) == os.path.normpath(query_path):
            continue  # loại ảnh gốc
        result_idx.append(idx)
        result_dist.append(dist)
        if len(result_idx) == k:
            break

    for i in range(k):
        img_path = os.path.join(BASE_IMAGE_DIR, paths[result_idx[i]])
        img = Image.open(img_path).resize((200, 200))
        img = ImageTk.PhotoImage(img)
        result_image_labels[i].config(image=img)
        result_image_labels[i].image = img
        # result_labels[i].config(text=f"Ảnh gần thứ {i+1}: {paths[result_idx[i]]}\nKhoảng cách: {result_dist[i]:.2f}")
        result_labels[i].config(text=f"Ảnh giống thứ {i+1} \nKhoảng cách: {result_dist[i]:.2f}")


# 🔹 Hàm chọn ảnh từ máy tính
def choose_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if not file_path:
        return

    # Hiển thị ảnh đầu vào
    img = Image.open(file_path).resize((200, 200))
    img = ImageTk.PhotoImage(img)
    input_image_label.config(image=img)
    input_image_label.image = img

    # Trích xuất đặc trưng & tìm ảnh gần nhất
    query_features = extract_facial_features(file_path)
    if query_features is not None:
        # find_nearest_image(query_features, 3)
        find_nearest_image(query_features, 3, file_path)
    else:
        messagebox.showerror("Lỗi", "Không phát hiện khuôn mặt trong ảnh!")

# 🔹 Tải dữ liệu MySQL và tạo KD-tree
paths, data = load_data()
kd_tree = KDTree(data)

# 🔹 Giao diện Tkinter
root = tk.Tk()
root.title("Tìm kiếm ảnh giống nhất")

# btn_choose = tk.Button(root, text="Chọn ảnh", command=choose_image)
# btn_choose.pack(pady=10)

# input_image_label = tk.Label(root)
# input_image_label.pack(pady=5)

# result_labels = [tk.Label(root, text="", fg="blue"),tk.Label(root, text="", fg="blue"),tk.Label(root, text="", fg="blue")]

# result_image_labels = [tk.Label(root),tk.Label(root),tk.Label(root)]
# for i in range(0,3):
#     result_labels[i].pack(pady=i*6+5)
#     result_image_labels[i].pack(pady=i*6+5)

# root.mainloop()



#v2
# # Nút chọn ảnh
# btn_choose = tk.Button(root, text="Chọn ảnh", command=choose_image)
# btn_choose.pack(pady=10)

# # Ảnh đầu vào
# input_image_label = tk.Label(root)
# input_image_label.pack(pady=5)

# # Canvas + Scrollbar
# scroll_frame = tk.Frame(root)
# scroll_frame.pack(fill=tk.BOTH, expand=True)

# canvas = tk.Canvas(scroll_frame, width=600, height=600)
# scrollbar = tk.Scrollbar(scroll_frame, orient="vertical", command=canvas.yview)
# scrollable_frame = tk.Frame(canvas)

# scrollable_frame.bind(
#     "<Configure>",
#     lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
# )

# canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
# canvas.configure(yscrollcommand=scrollbar.set)

# canvas.pack(side="left", fill="both", expand=True)
# scrollbar.pack(side="right", fill="y")

# # Kết quả ảnh và label (có thể mở rộng số lượng)
# result_labels = []
# result_image_labels = []
# NUM_RESULTS = 3  # có thể tăng lên 10, 20 nếu muốn

# for i in range(NUM_RESULTS):
#     lbl = tk.Label(scrollable_frame, text="", fg="blue")
#     lbl.pack(pady=5)
#     img_lbl = tk.Label(scrollable_frame)
#     img_lbl.pack(pady=5)

#     result_labels.append(lbl)
#     result_image_labels.append(img_lbl)

# root.mainloop()

#v3
# Nút chọn ảnh
btn_choose = tk.Button(root, text="Chọn ảnh", command=choose_image)
btn_choose.pack(pady=10)

# Ảnh đầu vào
input_image_label = tk.Label(root)
input_image_label.pack(pady=5)

# Frame chứa canvas và scrollbar
scroll_frame = tk.Frame(root)
scroll_frame.pack(fill=tk.BOTH, expand=True)

canvas = tk.Canvas(scroll_frame, width=700, height=300)  # có thể chỉnh chiều cao cho phù hợp
scrollbar = tk.Scrollbar(scroll_frame, orient="horizontal", command=canvas.xview)

scrollable_frame = tk.Frame(canvas)

# Khi frame bên trong thay đổi kích thước, cập nhật vùng scroll của canvas
def on_frame_configure(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

scrollable_frame.bind("<Configure>", on_frame_configure)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(xscrollcommand=scrollbar.set)

canvas.pack(side="top", fill="both", expand=True)
scrollbar.pack(side="bottom", fill="x")

# Kết quả ảnh và label (hiển thị theo hàng ngang)
result_labels = []
result_image_labels = []
NUM_RESULTS = 3  # có thể tăng lên 10, 20 nếu muốn

for i in range(NUM_RESULTS):
    lbl = tk.Label(scrollable_frame, text="", fg="blue")
    img_lbl = tk.Label(scrollable_frame)

    # Dùng grid để sắp xếp theo hàng ngang
    lbl.grid(row=0, column=i, padx=10, pady=5)
    img_lbl.grid(row=1, column=i, padx=10, pady=5)

    result_labels.append(lbl)
    result_image_labels.append(img_lbl)

root.mainloop()

