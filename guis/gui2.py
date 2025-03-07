import sys
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QScrollArea,
    QStatusBar,
)
from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QImage
import cv2
import numpy as np
import os
from sift2 import find_best_match


class ImageViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None
        self.pixmap = None
        self.zoom_factor = 1.0
        self.pan_start = None
        self.offset = QPointF(0, 0)
        self.crop_rect = None
        self.is_cropping = False
        self.crop_start = None

        # Enable mouse tracking for hover events
        self.setMouseTracking(True)

    def loadImage(self, image_path):
        self.image = QImage(image_path)
        self.pixmap = QPixmap.fromImage(self.image)
        self.resetView()
        self.update()

    def resetView(self):
        self.zoom_factor = 1.0
        self.offset = QPointF(0, 0)
        self.crop_rect = None
        self.update()

    def paintEvent(self, event):
        if not self.pixmap:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Calculate scaled dimensions
        scaled_width = self.pixmap.width() * self.zoom_factor
        scaled_height = self.pixmap.height() * self.zoom_factor

        # Calculate centered position
        x = (self.width() - scaled_width) / 2 + self.offset.x()
        y = (self.height() - scaled_height) / 2 + self.offset.y()

        # Draw image
        painter.drawPixmap(
            QRectF(x, y, scaled_width, scaled_height),
            self.pixmap,
            QRectF(0, 0, self.pixmap.width(), self.pixmap.height()),
        )

        # Draw crop rectangle if active
        if self.crop_rect:
            painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.PenStyle.DashLine))
            painter.drawRect(self.crop_rect)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if not self.is_cropping:
                self.pan_start = event.pos()
            else:
                # Convert QPoint to QPointF
                pos = QPointF(event.pos())
                self.crop_start = pos
                # Create initial rectangle with zero size at the start point
                self.crop_rect = QRectF(pos, pos)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            if not self.is_cropping and self.pan_start:
                delta = event.pos() - self.pan_start
                self.offset += QPointF(delta.x(), delta.y())
                self.pan_start = event.pos()
            elif self.is_cropping and self.crop_start:
                # Convert current position to QPointF
                current_pos = QPointF(event.pos())
                self.crop_rect = QRectF(self.crop_start, current_pos).normalized()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.pan_start = None

    def wheelEvent(self, event):
        zoom_in_factor = 1.1
        zoom_out_factor = 1 / zoom_in_factor

        if event.angleDelta().y() > 0:
            self.zoom_factor *= zoom_in_factor
        else:
            self.zoom_factor *= zoom_out_factor

        self.zoom_factor = max(0.1, min(5.0, self.zoom_factor))
        self.update()

    def setCroppingMode(self, enabled):
        self.is_cropping = enabled
        if not enabled:
            self.crop_rect = None
        self.update()

    def getCroppedImage(self):
        if not self.crop_rect or not self.image:
            return None

        # Convert QRectF to image coordinates
        scale = 1 / self.zoom_factor
        x = (
            self.crop_rect.x()
            - self.offset.x()
            - (self.width() - self.pixmap.width() * self.zoom_factor) / 2
        ) * scale
        y = (
            self.crop_rect.y()
            - self.offset.y()
            - (self.height() - self.pixmap.height() * self.zoom_factor) / 2
        ) * scale
        width = self.crop_rect.width() * scale
        height = self.crop_rect.height() * scale

        # Ensure coordinates are within image bounds
        x = max(0, min(x, self.image.width()))
        y = max(0, min(y, self.image.height()))
        width = min(width, self.image.width() - x)
        height = min(height, self.image.height() - y)

        # Crop the image
        cropped = self.image.copy(int(x), int(y), int(width), int(height))
        return cropped


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Comparison Tool")
        self.setMinimumSize(1000, 700)

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # Create left panel for source image
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Add source image viewer
        self.source_viewer = ImageViewer()
        source_scroll = QScrollArea()
        source_scroll.setWidget(self.source_viewer)
        source_scroll.setWidgetResizable(True)
        left_layout.addWidget(QLabel("Source Image"))
        left_layout.addWidget(source_scroll)

        # Add buttons for source image
        button_layout = QHBoxLayout()
        self.open_btn = QPushButton("Open Image")
        self.crop_btn = QPushButton("Start Crop")
        self.reset_btn = QPushButton("Reset View")
        self.scan_btn = QPushButton("Run Scan")
        self.select_repo_btn = QPushButton("Select Repository")

        self.open_btn.clicked.connect(self.openImage)
        self.crop_btn.clicked.connect(self.toggleCrop)
        self.reset_btn.clicked.connect(self.resetView)
        self.scan_btn.clicked.connect(self.runScan)
        self.select_repo_btn.clicked.connect(self.selectRepository)

        button_layout.addWidget(self.open_btn)
        button_layout.addWidget(self.crop_btn)
        button_layout.addWidget(self.reset_btn)
        button_layout.addWidget(self.scan_btn)
        button_layout.addWidget(self.select_repo_btn)
        left_layout.addLayout(button_layout)

        # Create right panel for result image
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Add result image viewer
        self.result_viewer = ImageViewer()
        result_scroll = QScrollArea()
        result_scroll.setWidget(self.result_viewer)
        result_scroll.setWidgetResizable(True)
        right_layout.addWidget(QLabel("Most Similar Image"))
        right_layout.addWidget(result_scroll)

        # Add panels to main layout
        layout.addWidget(left_panel)
        layout.addWidget(right_panel)

        # Add status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Initialize state
        self.current_image_path = None
        self.is_cropping = False
        self.scan_btn.setEnabled(False)
        self.repo_path = None

        # Set the status
        self.updateStatus("Ready to load image")

    def openImage(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)"
        )

        if file_path:
            self.current_image_path = file_path
            self.source_viewer.loadImage(file_path)
            self.scan_btn.setEnabled(True)
            self.updateStatus("Image loaded successfully")

    def toggleCrop(self):
        self.is_cropping = not self.is_cropping
        self.source_viewer.setCroppingMode(self.is_cropping)
        self.crop_btn.setText("Finish Crop" if self.is_cropping else "Start Crop")
        self.updateStatus(
            "Cropping mode " + ("enabled" if self.is_cropping else "disabled")
        )

    def resetView(self):
        self.source_viewer.resetView()
        self.updateStatus("View reset")

    def selectRepository(self):
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Image Repository", "", QFileDialog.Option.ShowDirsOnly
        )
        if folder_path:
            self.repo_path = folder_path
            self.updateStatus(f"Repository selected: {folder_path}")

    def runScan(self):
        if not self.current_image_path:
            self.updateStatus("Please load an image first")
            return

        if not self.repo_path:
            self.updateStatus("Please select a repository folder first")
            return

        cropped_image = self.source_viewer.getCroppedImage()
        if not cropped_image:
            self.updateStatus("Please crop the image first")
            return

        # Save the cropped image temporarily
        temp_path = "temp_cropped.png"
        cropped_image.save(temp_path)

        try:
            self.updateStatus("Scanning... Please wait")

            # Run SIFT matching
            (
                best_match,
                max_matches,
                kp_input,
                kp_best,
                good_matches,
                img_input,
                img_best,
            ) = find_best_match(
                temp_path, self.repo_path, min_matches=10, ratio_thresh=0.7
            )

            if best_match:
                # Draw matches and keypoints
                img_matches = cv2.drawMatchesKnn(
                    img_input,
                    kp_input,
                    img_best,
                    kp_best,
                    good_matches,
                    None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                )

                # Convert to RGB for Qt display
                img_matches_rgb = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)
                height, width, channel = img_matches_rgb.shape
                bytes_per_line = 3 * width

                # Convert to QImage and QPixmap
                q_img = QImage(
                    img_matches_rgb.data,
                    width,
                    height,
                    bytes_per_line,
                    QImage.Format.Format_RGB888,
                )
                pixmap = QPixmap.fromImage(q_img)

                # Display in result viewer
                self.result_viewer.pixmap = pixmap
                self.result_viewer.image = q_img
                self.result_viewer.resetView()
                self.result_viewer.update()

                self.updateStatus(
                    f"Best match found: {best_match} with {max_matches} matches"
                )
            else:
                self.updateStatus("No matches found")

        except Exception as e:
            self.updateStatus(f"Error during scan: {str(e)}")
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def updateStatus(self, message):
        self.status_bar.showMessage(message)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
