import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

class FaceDetector:
    def __init__(self):
      
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils  
        self.cap = cv2.VideoCapture(0) 

        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()

        self.mode = None  
        self.blur_intensity = 30 

        # Tạo giao diện Tkinter
        self.root = tk.Tk()
        self.root.title("Face Detection")


        self.video_frame = tk.Label(self.root)
        self.video_frame.pack()

        button_frame = tk.Frame(self.root)
        button_frame.pack(side=tk.BOTTOM)

      
        self.btn_blur = tk.Button(button_frame, text="Làm mờ khuôn mặt", command=self.toggle_blur_mode)
        self.btn_blur.grid(row=0, column=0, padx=5, pady=5)

        self.btn_pixelize = tk.Button(button_frame, text="Pixel hóa khuôn mặt", command=self.toggle_pixelize_mode)
        self.btn_pixelize.grid(row=0, column=1, padx=5, pady=5)

        self.btn_square = tk.Button(button_frame, text="Thay thế bằng hình vuông", command=self.toggle_square_mode)
        self.btn_square.grid(row=0, column=2, padx=5, pady=5)

        self.btn_smiley = tk.Button(button_frame, text="Thay thế bằng biểu tượng cảm xúc", command=self.toggle_smiley_mode)
        self.btn_smiley.grid(row=0, column=3, padx=5, pady=5)

        self.blur_scale = tk.Scale(button_frame, from_=1, to=99, orient=tk.HORIZONTAL, label="Độ mờ",
                                   command=self.update_blur_intensity)
        self.blur_scale.set(self.blur_intensity) 
        self.blur_scale.grid(row=1, column=0, columnspan=4, pady=5)

        self.anonymize_method = tk.StringVar(value="blur")
        modes = [("Làm mờ", "blur"), ("Pixel hóa", "pixelize"), ("Hình vuông", "square"), ("Biểu tượng cảm xúc", "smiley")]
        for i, (label, value) in enumerate(modes):
            tk.Radiobutton(button_frame, text=label, variable=self.anonymize_method, value=value).grid(row=2, column=i, padx=5)

    
        self.update_video()

        self.root.mainloop()  

    def update_blur_intensity(self, value):
     
        self.blur_intensity = int(value)

    def detect_faces(self, frame):

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        return results

    def blur_faces(self, frame, faces):
      
        for detection in faces.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

      
            face_region = frame[y:y + height, x:x + width]
            blurred_face = cv2.GaussianBlur(face_region, (self.blur_intensity, self.blur_intensity), 30)
            frame[y:y + height, x:x + width] = blurred_face

        return frame

    def pixelize_faces(self, frame, faces):

        for detection in faces.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

            face_region = frame[y:y + height, x:x + width]
            pixelized_face = cv2.resize(face_region, (16, 16))  
            pixelized_face = cv2.resize(pixelized_face, (width, height), interpolation=cv2.INTER_NEAREST)  
            frame[y:y + height, x:x + width] = pixelized_face

        return frame

    def replace_with_square(self, frame, faces):
     
        for detection in faces.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), -1)  # Hình vuông màu xanh lá cây

        return frame

    def replace_with_smiley(self, frame, faces):
    
        smiley = np.zeros((100, 100, 3), dtype=np.uint8)  
        cv2.circle(smiley, (50, 50), 40, (0, 255, 255), -1)  
        cv2.circle(smiley, (35, 35), 8, (0, 0, 0), -1)  
        cv2.circle(smiley, (65, 35), 8, (0, 0, 0), -1)  
        cv2.ellipse(smiley, (50, 65), (20, 10), 0, 0, 180, (0, 0, 0), -1)

        for detection in faces.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

            smiley_resized = cv2.resize(smiley, (width, height))
            frame[y:y + height, x:x + width] = smiley_resized

        return frame

    def update_video(self):
        
        ret, frame = self.cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            return

     
        results = self.detect_faces(frame)


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
