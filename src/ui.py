import numpy as np
import cv2
import copy

from PyQt5 import QtCore, QtGui, QtWidgets, Qt, uic

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi("./src/main.ui", self)

        # original image box
        self.lb_original    = self.findChild(QtWidgets.QLabel, "lb_original")
        self.lb_original.mousePressEvent = self.onOriginalClicked

        # transformed image box
        self.lb_transformed = self.findChild(QtWidgets.QLabel, "lb_transformed")

        # connect click event to button Load
        self.bt_load = self.findChild(QtWidgets.QPushButton, "bt_load")
        self.bt_load.clicked.connect(self.buttonLoadClicked)

        # connect click event to button Clear
        self.bt_clear = self.findChild(QtWidgets.QPushButton, "bt_clear")
        self.bt_clear.clicked.connect(self.buttonClearClicked)

        # connect click event to button Transform
        self.bt_transform = self.findChild(QtWidgets.QPushButton, "bt_transform")
        self.bt_transform.clicked.connect(self.buttonTransformClicked)
        
        self.img_original    = None
        self.img_transformed = None

        # image vertices vector
        self.vertices = []

        # counter for image vertices vector
        self.vertices_added = 0

        # flag for image loading
        self.image_loaded = False

    def buttonLoadClicked(self):
        fileName = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open image",
            QtCore.QDir.homePath(),
            "Image Files (*.png *.bmp *.jpg)"
        )[0]

        if fileName != "":
            print(fileName)
            self.img_original = cv2.imread(fileName)
            
            # set image colors to RGB
            self.img_original = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2RGB)
            
            # resize it to window size
            self.img_original = cv2.resize(self.img_original, (256, 256))
            
            height, width, channel = self.img_original.shape
            bytesPerLine = 3 * width

            self.lb_original.setPixmap(
                QtGui.QPixmap.fromImage(QtGui.QImage(self.img_original.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888))
            )

            self.image_loaded = True

            # init transformed image
            self.img_transformed = copy.deepcopy(self.img_original)

            # init vertices count, in case of loading new image
            self.vertices_added = 0

    # reset vertices selection
    def buttonClearClicked(self):
        self.vertices_added = 0
        
        height, width, channel = self.img_original.shape
        bytesPerLine = 3 * width
        
        self.lb_original.setPixmap(
            QtGui.QPixmap.fromImage(QtGui.QImage(self.img_original.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888))
        )

        self.img_transformed = copy.deepcopy(self.img_original)
        self.vertices = []

    def buttonTransformClicked(self):
        if self.image_loaded:
            if self.vertices_added == 4:
                vertices = np.array(self.vertices, np.float32)
                vertices_image = np.array([[60, 90], [185, 90], [60, 185], [185, 185]], np.float32)
                
                matrix = cv2.getPerspectiveTransform(vertices, vertices_image)

                result = cv2.warpPerspective(copy.deepcopy(self.img_original), matrix, (256, 256))

                height, width, channel = result.shape
                bytesPerLine = 3 * width

                # set image
                self.lb_transformed.setPixmap(
                    QtGui.QPixmap.fromImage(QtGui.QImage(result.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888))
                )
            else:
                QtWidgets.QMessageBox.warning(self, "Vertices selection", "Please select 4 vertices for transformation.")

    def onOriginalClicked(self, event):
        if self.image_loaded == True:
            if self.vertices_added != 4:
                self.vertices_added += 1
                
                # add circle on image
                self.img_transformed = cv2.circle(self.img_transformed, (event.x(), event.y()), 4, (255, 50, 50), -1)
                
                height, width, channel = self.img_original.shape
                bytesPerLine = 3 * width
                
                self.lb_original.setPixmap(
                    QtGui.QPixmap.fromImage(QtGui.QImage(self.img_transformed.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888))
                )

                self.vertices.append([event.x(), event.y()])
