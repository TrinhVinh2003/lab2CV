# Ứng dụng Phát hiện và Ẩn danh Khuôn mặt

Ứng dụng này sử dụng **MediaPipe** và **OpenCV** để phát hiện khuôn mặt theo thời gian thực qua webcam và áp dụng các phương pháp ẩn danh khác nhau. Giao diện người dùng được xây dựng bằng **Tkinter**, cung cấp các tùy chọn để áp dụng các phương pháp ẩn danh khuôn mặt như làm mờ, pixel hóa, thay thế bằng hình vuông hoặc bằng biểu tượng cảm xúc.

## Tính năng
- **Phát hiện khuôn mặt**: Ứng dụng phát hiện khuôn mặt theo thời gian thực sử dụng mô-đun phát hiện khuôn mặt của MediaPipe.
- **Làm mờ khuôn mặt**: Cho phép làm mờ khuôn mặt được phát hiện với cường độ điều chỉnh thông qua thanh trượt.
- **Pixel hóa khuôn mặt**: Áp dụng pixel hóa cho khuôn mặt được phát hiện.
- **Thay thế bằng hình vuông**: Thay thế khuôn mặt được phát hiện bằng một hình vuông màu.
- **Thay thế bằng biểu tượng cảm xúc**: Thay thế khuôn mặt được phát hiện bằng biểu tượng cảm xúc hình mặt cười.

## Hướng dẫn sử dụng

### Yêu cầu
- Python 3.x
- OpenCV
- MediaPipe
- Pillow
