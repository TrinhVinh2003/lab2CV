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
        self.blur_intensity = 30  # Độ mờ mặc định

        # Tạo giao diện Tkinter
        self.root = tk.Tk()
        self.root.title("Face Detection")

        # Khung để hiển thị video
        self.video_frame = tk.Label(self.root)
        self.video_frame.pack()

        # Khung cho các nút và thanh điều chỉnh
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

        # Thanh điều chỉnh độ mờ
        self.blur_scale = tk.Scale(button_frame, from_=1, to=99, orient=tk.HORIZONTAL, label="Độ mờ",
                                   command=self.update_blur_intensity)
        self.blur_scale.set(self.blur_intensity)  # Đặt giá trị mặc định là 30
        self.blur_scale.grid(row=1, column=0, columnspan=4, pady=5)

        # Thêm nút radio để chuyển đổi các phương pháp ẩn danh
        self.anonymize_method = tk.StringVar(value="blur")
        modes = [("Làm mờ", "blur"), ("Pixel hóa", "pixelize"), ("Hình vuông", "square"), ("Biểu tượng cảm xúc", "smiley")]
        for i, (label, value) in enumerate(modes):
            tk.Radiobutton(button_frame, text=label, variable=self.anonymize_method, value=value).grid(row=2, column=i, padx=5)

        # Bắt đầu vòng lặp video
        self.update_video()

        self.root.mainloop()  # Bắt đầu giao diện

    def update_blur_intensity(self, value):
        """Cập nhật giá trị độ mờ khi người dùng thay đổi thanh trượt."""
        self.blur_intensity = int(value)

    def detect_faces(self, frame):
        """Phát hiện khuôn mặt trong khung hình."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        return results

    def blur_faces(self, frame, faces):
        """Làm mờ khuôn mặt trong khung hình với độ mờ tùy chỉnh."""
        for detection in faces.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

            # Làm mờ khuôn mặt với giá trị điều chỉnh
            face_region = frame[y:y + height, x:x + width]
            blurred_face = cv2.GaussianBlur(face_region, (self.blur_intensity, self.blur_intensity), 30)
            frame[y:y + height, x:x + width] = blurred_face

        return frame

    def pixelize_faces(self, frame, faces):
        """Ẩn danh khuôn mặt bằng cách pixel hóa."""
        for detection in faces.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

            face_region = frame[y:y + height, x:x + width]
            pixelized_face = cv2.resize(face_region, (16, 16))  # Giảm kích thước
            pixelized_face = cv2.resize(pixelized_face, (width, height), interpolation=cv2.INTER_NEAREST)  # Tăng kích thước trở lại
            frame[y:y + height, x:x + width] = pixelized_face

        return frame

    def replace_with_square(self, frame, faces):
        """Thay thế khuôn mặt bằng hình vuông."""
        for detection in faces.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

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
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

            smiley_resized = cv2.resize(smiley, (width, height))
            frame[y:y + height, x:x + width] = smiley_resized

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
        mode = self.anonymize_method.get()
        if results.detections and mode:
            if mode == 'blur':
                frame = self.blur_faces(frame, results)
            elif mode == 'pixelize':
                frame = self.pixelize_faces(frame, results)
            elif mode == 'square':
                frame = self.replace_with_square(frame, results)
            elif mode == 'smiley':
                frame = self.replace_with_smiley(frame, results)

        # Chuyển đổi khung hình BGR sang RGB để hiển thị trên Tkinter
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_frame.imgtk = imgtk
        self.video_frame.configure(image=imgtk)

        self.video_frame.after(10, self.update_video)

    def toggle_blur_mode(self):
        self.mode = 'blur'

    def toggle_pixelize_mode(self):
        self.mode = 'pixelize'

    def toggle_square_mode(self):
        self.mode = 'square'

    def toggle_smiley_mode(self):
        self.mode = 'smiley'


if __name__ == "__main__":
    FaceDetector()
