# -*- coding: utf-8 -*-

# This file is used for creating the user interface to allow the user to choose the input image
# is the main file of this project

from PyQt5 import QtCore, QtWidgets
import sys
from PyQt5.QtWidgets import *
import depth


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("Select Image")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(120, 90, 171, 150))
        self.groupBox.setObjectName("groupBox")
        self.radioButton = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton.setGeometry(QtCore.QRect(10, 20, 89, 16))
        self.radioButton.setObjectName("radioButton")
        self.radioButton_2 = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton_2.setGeometry(QtCore.QRect(10, 50, 89, 16))
        self.radioButton_2.setObjectName("radioButton_2")
        self.radioButton_3 = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton_3.setGeometry(QtCore.QRect(10, 80, 89, 16))
        self.radioButton_3.setObjectName("radioButton_3")

        self.pushButton = QtWidgets.QPushButton("ok", self.groupBox)
        self.pushButton.setGeometry(QtCore.QRect(10, 110, 89, 30))
        self.pushButton.setObjectName("pushButton")

        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(110, 290, 400, 267))
        # self.graphicsView.setStyleSheet("image: url(./images/output_correction/left1.jpg);\n"
        #                                 "border-image: url(./images/output_correction/left1.jpg);")
        self.graphicsView.setObjectName("graphicsView")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.radioButton.clicked.connect(self.slot1)
        self.radioButton_2.clicked.connect(self.slot2)
        self.radioButton_3.clicked.connect(self.slot3)
        self.pushButton.clicked.connect(self.run)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def slot1(self):
        self.graphicsView.setStyleSheet("image: url(./images/output_correction/left1.jpg);\n"
                                        "border-image: url(./images/output_correction/left1.jpg);")
        global flag
        flag = 1
        print(flag)

    def slot2(self):
        self.graphicsView.setStyleSheet("image: url(./images/output_correction/left2.jpg);\n"
                                        "border-image: url(./images/output_correction/left2.jpg);")
        global flag
        flag = 2
        print(flag)

    def slot3(self):
        self.graphicsView.setStyleSheet("image: url(./images/output_correction/left3.jpg);\n"
                                        "border-image: url(./images/output_correction/left3.jpg);")
        global flag
        flag = 3
        print(flag)

    def run(self):
        print("flag: ", flag)
        depth.depth_calculation(flag)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("Select Image", "Select Image"))
        self.groupBox.setTitle(_translate("Select Image", ""))
        self.radioButton.setText(_translate("Select Image", "image1"))
        self.radioButton_2.setText(_translate("Select Image", "image2"))
        self.radioButton_3.setText(_translate("Select Image", "image3"))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainwindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(mainwindow)
    mainwindow.show()
    sys.exit(app.exec_())
