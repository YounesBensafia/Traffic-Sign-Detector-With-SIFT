import os
import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QStackedWidget, QProgressBar)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
import cv2
import numpy as np

class SIFTProcessor(QThread):
    finished = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    
    def __init__(self, image):
        super().__init__()
        self.image = image 
        
    def run(self):
        try:
            if not isinstance(self.image, np.ndarray):
                raise ValueError("Input image is not a valid NumPy array")

            img1 = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT_create(nfeatures=5000)
            bf = cv2.BFMatcher()

            best_match = None
            max_good_matches = 0
            output = None

            kp1, des1 = sift.detectAndCompute(img1, None)
            if des1 is None:
                raise ValueError("No descriptors found in the query image")

            images_dir = 'images'
            if not os.path.exists(images_dir):
                raise ValueError(f"Images directory not found: {images_dir}")

            image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            total_files = len(image_files)

            for index, filename in enumerate(image_files):
                img2_path = os.path.join(images_dir, filename)
                img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
                
                if img2 is None:
                    print(f"Failed to load image: {filename}")
                    continue

                kp2, des2 = sift.detectAndCompute(img2, None)
                if des2 is None:
                    print(f"No descriptors found in image {filename}")
                    continue

                try:
                    matches = bf.knnMatch(des1, des2, k=2)
                    good = []
                    for m,n in matches:
                        if m.distance < 0.75 * n.distance:
                            good.append(m)

                    if len(good) > max_good_matches:
                        max_good_matches = len(good)
                        best_match = filename

                except Exception as e:
                    print(f"Error matching with {filename}: {str(e)}")
                    continue

                self.progress.emit(int((index + 1) / total_files * 100))

            if best_match:
                print(f"Best match: {best_match} with {max_good_matches} matches")
                img2 = cv2.imread(os.path.join(images_dir, best_match), cv2.IMREAD_GRAYSCALE)
                kp2, des2 = sift.detectAndCompute(img2, None)
                
                matches = bf.knnMatch(des1, des2, k=2)
                good = []
                for match_group in matches:
                    if len(match_group) == 2:
                        m, n = match_group
                        if m.distance < 0.75 * n.distance:
                            good.append(m)
                
                output = cv2.drawMatches(img1, kp1, img2, kp2, good, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            else:
                output = cv2.drawKeypoints(img1, kp1, None, 
                                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                print("No significant match found")

            if output is None:
                raise ValueError("Failed to generate output image")
                
            self.finished.emit(output)
        
        except Exception as e:
            self.error.emit(str(e))

class WelcomePage(QWidget):
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        welcome_label = QLabel("Traffic Sign Detection")
        welcome_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
                margin: 20px;
            }
        """)
        
        start_button = QPushButton("Start Detection")
        start_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 10px;
                border-radius: 5px;
                font-size: 16px;
                min-width: 200px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        start_button.clicked.connect(lambda: self.parent().parent().switch_to_detection())
        
        layout.addStretch()
        layout.addWidget(welcome_label)
        layout.addWidget(start_button, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addStretch()
        
        self.setLayout(layout)

class DetectionPage(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.current_image = None
        self.sift_thread = None
        
    def init_ui(self):
        layout = QHBoxLayout()
        
        # Left panel
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        
        upload_button = QPushButton("Upload Image")
        upload_button.clicked.connect(self.upload_image)
        
        self.image_preview = QLabel()
        self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview.setStyleSheet("border: 1px solid #bdc3c7;")
        self.image_preview.setMinimumSize(400, 300)
        
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self.clear_upload)
        
        left_layout.addWidget(upload_button)
        left_layout.addWidget(self.image_preview)
        left_layout.addWidget(clear_button)
        left_panel.setLayout(left_layout)
        
        # Right panel
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        
        self.output_preview = QLabel()
        self.output_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.output_preview.setStyleSheet("border: 1px solid #bdc3c7;")
        self.output_preview.setMinimumSize(400, 300)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        right_layout.addWidget(self.output_preview)
        right_layout.addWidget(self.progress_bar)
        right_layout.addWidget(self.status_label)
        right_panel.setLayout(right_layout)
        
        layout.addWidget(left_panel)
        layout.addWidget(right_panel)
        self.setLayout(layout)
        
    def upload_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "C:\\Users\\chems\\OneDrive\\Bureau\\TRAITEMENT_IMAGES\\SIFT-Sign-Detector\\inputs",
            "Images (*.png *.jpg *.jpeg)"
        )
        
        if file_name:
            try:
                self.current_image = cv2.imread(file_name)
                if self.current_image is None:
                    raise ValueError("Failed to load image")
                
                height, width = self.current_image.shape[:2]
                bytes_per_line = 3 * width
                qt_image = QImage(
                    cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB).data,
                    width,
                    height,
                    bytes_per_line,
                    QImage.Format.Format_RGB888
                )
                
                pixmap = QPixmap.fromImage(qt_image)
                scaled_pixmap = pixmap.scaled(
                    self.image_preview.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                
                self.image_preview.setPixmap(scaled_pixmap)
                self.process_image()
                
            except Exception as e:
                self.status_label.setText(f"Error: {str(e)}")
                self.status_label.setStyleSheet("color: red;")
    
    def process_image(self):
        if self.current_image is not None:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.status_label.setText("Processing...")
            self.status_label.setStyleSheet("color: #2c3e50;")
            
            # Create and start SIFT processing thread
            self.sift_thread = SIFTProcessor(self.current_image.copy())
            self.sift_thread.finished.connect(self.handle_processed_image)
            self.sift_thread.error.connect(self.handle_processing_error)
            self.sift_thread.progress.connect(self.update_progress_bar)
            self.sift_thread.start()
    
    def update_progress_bar(self, value):
        """Update the progress bar value"""
        self.progress_bar.setValue(value)
    
    def handle_processed_image(self, output_image):
        """Handle the processed image result"""
        height, width = output_image.shape[:2]
        bytes_per_line = 3 * width
        qt_image = QImage(
            cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB).data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888
        )
        
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.output_preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.output_preview.setPixmap(scaled_pixmap)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Processing complete")
        self.status_label.setStyleSheet("color: green;")
    
    def handle_processing_error(self, error_message):
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Error: {error_message}")
        self.status_label.setStyleSheet("color: red;")
    
    def clear_upload(self):
        self.current_image = None
        self.image_preview.clear()
        self.output_preview.clear()
        self.status_label.clear()
        self.progress_bar.setVisible(False)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Traffic Sign Detector")
        self.setMinimumSize(900, 600)
        
        self.stacked_widget = QStackedWidget()
        self.welcome_page = WelcomePage()
        self.detection_page = DetectionPage()
        
        self.stacked_widget.addWidget(self.welcome_page)
        self.stacked_widget.addWidget(self.detection_page)
        
        self.setCentralWidget(self.stacked_widget)
    
    def switch_to_detection(self):
        self.stacked_widget.setCurrentWidget(self.detection_page)
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
