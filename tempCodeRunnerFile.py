import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

class FaceDetector:
    def __init__(self):
        # Khởi tạo các module của MediaPipe
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils  # Dùng để vẽ khung hình xung quanh khuôn mặt
        self.cap = cv2.VideoCapture(0)  # Mở webcam

        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()

        self.mode = None  # Khởi tạo chế độ là None

        # Tạo giao diện Tkinter
        self.root = tk.Tk()
        self.root.title("Face Detection")

        # Khung để hiển thị video
        self.video_frame = tk.Label(self.root)
        self.video_frame.pack()

        # Khung cho các nút
        button_frame = tk.Frame(self.root)
        button_frame.pack(side=tk.BOTTOM)

        # Tạo nút bấm cho các chế độ
        self.btn_blur = tk.Button(button_frame, text="Làm mờ khuôn mặt", command=self.toggle_blur_mode)
        self.btn_blur.grid(row=0, column=0, padx=5, pady=5)

        self.btn_pixelize = tk.Button(button_frame, text="Pixel hóa khuôn mặt", command=self.toggle_pixelize_mode)
        self.btn_pixelize.grid(row=0, column=1, padx=5, pady=5)

        self.btn_square = tk.Button(button_frame, text="Thay thế bằng hình vuông", command=self.toggle_square_mode)
        self.btn_square.grid(row=0, column=2, padx=5, pady=5)

        self.btn_smiley = tk.Button(button_frame, text="Thay thế bằng biểu tượng cảm xúc", command=self.toggle_smiley_mode)
        self.btn_smiley.grid(row=0, column=3, padx=5, pady=5)

        # Bắt đầu vòng lặp video
        self.update_video()

        self.root.mainloop()  # Bắt đầu giao diện

    def toggle_blur_mode(self):
        if self.mode == 'blur':
            self.mode = None  # Tắt chế độ làm mờ
            self.btn_blur.config(relief=tk.RAISED)  # Đổi kiểu nút về bình thường
        else:
            self.mode = 'blur'  # Bật chế độ làm mờ
            self.btn_blur.config(relief=tk.SUNKEN)  # Đổi kiểu nút về nhấn xuống

    def toggle_pixelize_mode(self):
        if self.mode == 'pixelize':
            self.mode = None  # Tắt chế độ pixel hóa
            self.btn_pixelize.config(relief=tk.RAISED)  # Đổi kiểu nút về bình thường
        else:
            self.mode = 'pixelize'  # Bật chế độ pixel hóa
            self.btn_pixelize.config(relief=tk.SUNKEN)  # Đổi kiểu nút về nhấn xuống

    def toggle_square_mode(self):
        if self.mode == 'square':
            self.mode = None  # Tắt chế độ hình vuông
            self.btn_square.config(relief=tk.RAISED)  # Đổi kiểu nút về bình thường
        else:
            self.mode = 'square'  # Bật chế độ hình vuông
            self.btn_square.config(relief=tk.SUNKEN)  # Đổi kiểu nút về nhấn xuống

    def toggle_smiley_mode(self):
        if self.mode == 'smiley':
            self.mode = None  # Tắt chế độ biểu tượng cảm xúc
            self.btn_smiley.config(relief=tk.RAISED)  # Đổi kiểu nút về bình thường
        else:
            self.mode = 'smiley'  # Bật chế độ biểu tượng cảm xúc
            self.btn_smiley.config(relief=tk.SUNKEN)  # Đổi kiểu nút về nhấn xuống

    def detect_faces(self, frame):
        """Phát hiện khuôn mặt trong khung hình."""
        # Chuyển đổi khung hình từ BGR sang RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)  # Xử lý hình ảnh để phát hiện khuôn mặt
        return results

    def blur_faces(self, frame, faces):
        """Làm mờ khuôn mặt trong khung hình."""
        for detection in faces.detections:
            # Lấy tọa độ của khuôn mặt
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

            # Làm mờ khuôn mặt
            face_region = frame[y:y + height, x:x + width]
            blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
            frame[y:y + height, x:x + width] = blurred_face  # Thay thế khuôn mặt bằng khuôn mặt đã làm mờ

        return frame

    def pixelize_faces(self, frame, faces):
        """Ẩn danh khuôn mặt bằng cách pixel hóa."""
        for detection in faces.detections:
            # Lấy tọa độ của khuôn mặt
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

            # Pixelize khuôn mặt
            face_region = frame[y:y + height, x:x + width]
            pixelized_face = cv2.resize(face_region, (16, 16))  # Giảm kích thước
            pixelized_face = cv2.resize(pixelized_face, (width, height), interpolation=cv2.INTER_NEAREST)  # Tăng kích thước trở lại
            frame[y:y + height, x:x + width] = pixelized_face  # Thay thế khuôn mặt bằng khuôn mặt đã pixel hóa

        return frame

    def replace_with_square(self, frame, faces):
        """Thay thế khuôn mặt bằng hình vuông."""
        for detection in faces.detections:
            # Lấy tọa độ của khuôn mặt
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

            # Vẽ hình vuông
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), -1)  # Hình vuông màu xanh lá cây

        return frame

    def replace_with_smiley(self, frame, faces):
        """Thay thế khuôn mặt bằng biểu tượng cảm xúc."""
        smiley = np.zeros((100, 100, 3), dtype=np.uint8)  # Tạo ảnh biểu tượng cảm xúc
        cv2.circle(smiley, (50, 50), 40, (0, 255, 255), -1)  # Màu vàng
        cv2.circle(smiley, (35, 35), 8, (0, 0, 0), -1)  # Mắt trái
        cv2.circle(smiley, (65, 35), 8, (0, 0, 0), -1)  # Mắt phải
        cv2.ellipse(smiley, (50, 65), (20, 10), 0, 0, 180, (0, 0, 0), -1)  # Miệng

        for detection in faces.detections:
            # Lấy tọa độ của khuôn mặt
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

            # Thay thế khuôn mặt bằng biểu tượng cảm xúc
            smiley_resized = cv2.resize(smiley, (width, height))  # Thay đổi kích thước biểu tượng cảm xúc
            frame[y:y + height, x:x + width] = smiley_resized  # Thay thế khuôn mặt bằng biểu tượng cảm xúc

        return frame

    def update_video(self):
        """Cập nhật khung hình video."""
        ret, frame = self.cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            return

        # Ph
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

class FaceDetector:
    def __init__(self):
        # Khởi tạo các module của MediaPipe
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils  # Dùng để vẽ khung hình xung quanh khuôn mặt
        self.cap = cv2.VideoCapture(0)  # Mở webcam

        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()

        self.mode = None  # Khởi tạo chế độ là None

        # Tạo giao diện Tkinter
        self.root = tk.Tk()
        self.root.title("Face Detection")

        # Khung để hiển thị video
        self.video_frame = tk.Label(self.root)
        self.video_frame.pack()

        # Khung cho các nút
        button_frame = tk.Frame(self.root)
        button_frame.pack(side=tk.BOTTOM)

        # Tạo nút bấm cho các chế độ
        self.btn_blur = tk.Button(button_frame, text="Làm mờ khuôn mặt", command=self.toggle_blur_mode)
        self.btn_blur.grid(row=0, column=0, padx=5, pady=5)

        self.btn_pixelize = tk.Button(button_frame, text="Pixel hóa khuôn mặt", command=self.toggle_pixelize_mode)
        self.btn_pixelize.grid(row=0, column=1, padx=5, pady=5)

        self.btn_square = tk.Button(button_frame, text="Thay thế bằng hình vuông", command=self.toggle_square_mode)
        self.btn_square.grid(row=0, column=2, padx=5, pady=5)

        self.btn_smiley = tk.Button(button_frame, text="Thay thế bằng biểu tượng cảm xúc", command=self.toggle_smiley_mode)
        self.btn_smiley.grid(row=0, column=3, padx=5, pady=5)

        # Bắt đầu vòng lặp video
        self.update_video()

        self.root.mainloop()  # Bắt đầu giao diện

    def toggle_blur_mode(self):
        if self.mode == 'blur':
            self.mode = None  # Tắt chế độ làm mờ
            self.btn_blur.config(relief=tk.RAISED)  # Đổi kiểu nút về bình thường
        else:
            self.mode = 'blur'  # Bật chế độ làm mờ
            self.btn_blur.config(relief=tk.SUNKEN)  # Đổi kiểu nút về nhấn xuống

    def toggle_pixelize_mode(self):
        if self.mode == 'pixelize':
            self.mode = None  # Tắt chế độ pixel hóa
            self.btn_pixelize.config(relief=tk.RAISED)  # Đổi kiểu nút về bình thường
        else:
            self.mode = 'pixelize'  # Bật chế độ pixel hóa
            self.btn_pixelize.config(relief=tk.SUNKEN)  # Đổi kiểu nút về nhấn xuống

    def toggle_square_mode(self):
        if self.mode == 'square':
            self.mode = None  # Tắt chế độ hình vuông
            self.btn_square.config(relief=tk.RAISED)  # Đổi kiểu nút về bình thường
        else:
            self.mode = 'square'  # Bật chế độ hình vuông
            self.btn_square.config(relief=tk.SUNKEN)  # Đổi kiểu nút về nhấn xuống

    def toggle_smiley_mode(self):
        if self.mode == 'smiley':
            self.mode = None  # Tắt chế độ biểu tượng cảm xúc
            self.btn_smiley.config(relief=tk.RAISED)  # Đổi kiểu nút về bình thường
        else:
            self.mode = 'smiley'  # Bật chế độ biểu tượng cảm xúc
            self.btn_smiley.config(relief=tk.SUNKEN)  # Đổi kiểu nút về nhấn xuống

    def detect_faces(self, frame):
        """Phát hiện khuôn mặt trong khung hình."""
        # Chuyển đổi khung hình từ BGR sang RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)  # Xử lý hình ảnh để phát hiện khuôn mặt
        return results

    def blur_faces(self, frame, faces):
        """Làm mờ khuôn mặt trong khung hình."""
        for detection in faces.detections:
            # Lấy tọa độ của khuôn mặt
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

            # Làm mờ khuôn mặt
            face_region = frame[y:y + height, x:x + width]
            blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
            frame[y:y + height, x:x + width] = blurred_face  # Thay thế khuôn mặt bằng khuôn mặt đã làm mờ

        return frame

    def pixelize_faces(self, frame, faces):
        """Ẩn danh khuôn mặt bằng cách pixel hóa."""
        for detection in faces.detections:
            # Lấy tọa độ của khuôn mặt
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

            # Pixelize khuôn mặt
            face_region = frame[y:y + height, x:x + width]
            pixelized_face = cv2.resize(face_region, (16, 16))  # Giảm kích thước
            pixelized_face = cv2.resize(pixelized_face, (width, height), interpolation=cv2.INTER_NEAREST)  # Tăng kích thước trở lại
            frame[y:y + height, x:x + width] = pixelized_face  # Thay thế khuôn mặt bằng khuôn mặt đã pixel hóa

        return frame

    def replace_with_square(self, frame, faces):
        """Thay thế khuôn mặt bằng hình vuông."""
        for detection in faces.detections:
            # Lấy tọa độ của khuôn mặt
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

            # Vẽ hình vuông
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), -1)  # Hình vuông màu xanh lá cây

        return frame

    def replace_with_smiley(self, frame, faces):
        """Thay thế khuôn mặt bằng biểu tượng cảm xúc."""
        smiley = np.zeros((100, 100, 3), dtype=np.uint8)  # Tạo ảnh biểu tượng cảm xúc
        cv2.circle(smiley, (50, 50), 40, (0, 255, 255), -1)  # Màu vàng
        cv2.circle(smiley, (35, 35), 8, (0, 0, 0), -1)  # Mắt trái
        cv2.circle(smiley, (65, 35), 8, (0, 0, 0), -1)  # Mắt phải
        cv2.ellipse(smiley, (50, 65), (20, 10), 0, 0, 180, (0, 0, 0), -1)  # Miệng

        for detection in faces.detections:
            # Lấy tọa độ của khuôn mặt
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

            # Thay thế khuôn mặt bằng biểu tượng cảm xúc
            smiley_resized = cv2.resize(smiley, (width, height))  # Thay đổi kích thước biểu tượng cảm xúc
            frame[y:y + height, x:x + width] = smiley_resized  # Thay thế khuôn mặt bằng biểu tượng cảm xúc

        return frame

    def update_video(self):
        """Cập nhật khung hình video."""
        ret, frame = self.cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            return

        # Phát hiện khuôn mặt
        results = self.detect_faces(frame)

        # Chọn chế độ (làm mờ, pixel hóa, hình vuông, biểu tượng cảm xúc)
        if results.detections and self.mode:
            if self.mode == 'blur':
                frame = self.blur_faces(frame, results)
            elif self.mode == 'pixelize':
                frame = self.pixelize_faces(frame, results)
            elif self.mode == 'square':
                frame = self.replace_with_square(frame, results)
            elif self.mode == 'smiley':
                frame = self.replace_with_smiley(frame, results)

        # Chuyển đổi khung hình BGR sang RGB để hiển thị trên Tkinter
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_frame.imgtk = imgtk
        self.video_frame.configure(image=imgtk)
        self.video_frame.after(10, self.update_video)  # Gọi lại hàm sau 10ms

# Chạy ứng dụng
if __name__ == "__main__":
    detector = FaceDetector()
