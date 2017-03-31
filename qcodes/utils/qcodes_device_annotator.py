# A device image plotter

import sys
import os
import json
import glob
import PyQt5.QtWidgets as qt
import PyQt5.QtGui as gui
import PyQt5.QtCore as core

from shutil import copyfile

class MakeDeviceImage(qt.QWidget):
    """
    Class for clicking and adding labels
    """

    def __init__(self, folder, station):

        super().__init__()

        self.folder = folder
        self.station = station

        # BACKEND
        self._data = {}
        self.filename = None

        # FRONTEND

        grid = qt.QGridLayout()
        self.setLayout(grid)

        self.imageCanvas = qt.QLabel()
        self.loadButton = qt.QPushButton('Load image')
        self.labelButton = qt.QRadioButton("Insert Label")
        self.labelButton.setChecked(True)
        self.annotButton = qt.QRadioButton('Place annotation')

        self.okButton = qt.QPushButton('Save and close')

        self.loadButton.clicked.connect(self.loadimage)
        self.imageCanvas.mousePressEvent = self.set_label_or_annotation
        self.imageCanvas.setStyleSheet('background-color: white')
        self.okButton.clicked.connect(self.saveAndClose)

        self.treeView = qt.QTreeView()
        self.model = gui.QStandardItemModel()
        self.model.setHorizontalHeaderLabels([self.tr("Instruments")])
        self.addStation(self.model, station)
        self.treeView.setModel(self.model)
        self.treeView.sortByColumn(0, core.Qt.AscendingOrder)
        self.treeView.setSortingEnabled(True)
        grid.addWidget(self.imageCanvas, 0, 0, 4, 6)
        grid.addWidget(self.loadButton, 4, 0)
        grid.addWidget(self.labelButton, 4, 1)
        grid.addWidget(self.annotButton, 4, 2)
        grid.addWidget(self.okButton, 4, 9)
        grid.addWidget(self.treeView, 0, 6, 4, 4)

        self.resize(600, 400)
        self.move(100, 100)
        self.setWindowTitle('Generate annotated device image')
        self.show()

    def addStation(self, parent, station):

        for inst in station.components:
            item = gui.QStandardItem(inst)
            item.setEditable(False)
            item.setSelectable(False)
            parent.appendRow(item)
            for param in station[inst].parameters:
                paramitem = gui.QStandardItem(param)
                paramitem.setEditable(False)
                item.appendRow(paramitem)

    def loadimage(self):
        """
        Select an image from disk.
        """
        fd = qt.QFileDialog()
        filename = fd.getOpenFileName(self, 'Select device image',
                                      os.getcwd(),
                                      "Image files(*.jpg *.png *.jpeg)")
        self.filename = filename[0]
        self.pixmap = gui.QPixmap(filename[0])
        width = self.pixmap.width()
        height = self.pixmap.height()

        self.imageCanvas.setPixmap(self.pixmap)

        # fix the image scale, so that the pixel values of the mouse are
        # unambiguous
        self.imageCanvas.setMaximumWidth(width)
        self.imageCanvas.setMaximumHeight(height)

    def set_label_or_annotation(self, event):


        # verify valid
        if not self.treeView.selectedIndexes():
            return
        selected = self.treeView.selectedIndexes()[0]
        selected_instrument = selected.parent().data()
        selected_parameter = selected.data()
        self.click_x = event.pos().x()
        self.click_y = event.pos().y()

        # update the data
        if selected_instrument not in self._data.keys():
            self._data[selected_instrument] = {}
        if selected_parameter not in self._data[selected_instrument].keys():
            self._data[selected_instrument][selected_parameter] = {}

        if self.labelButton.isChecked():
            self._data[selected_instrument][selected_parameter]['labelpos'] = (self.click_x, self.click_y)
        elif self.annotButton.isChecked():
            self._data[selected_instrument][selected_parameter]['annotationpos'] = (self.click_x, self.click_y)
        self._data[selected_instrument][selected_parameter]['value'] = 'NaN'

        # draw it
        self.imageCanvas, _ = self._renderImage(self._data,
                                                self.imageCanvas,
                                                self.filename)

    def saveAndClose(self):
        """
        Save and close
        """
        if self.filename is None:
            return

        fileformat = self.filename.split('.')[-1]
        rawpath = os.path.join(self.folder, 'deviceimage_raw.'+fileformat)
        copyfile(self.filename, rawpath)

        # Now forget about the original
        self.filename = rawpath

        self.close()

    @staticmethod
    def _renderImage(data, canvas, filename):
        """
        Render an image
        """

        pixmap = gui.QPixmap(filename)
        width = pixmap.width()
        height = pixmap.height()

        label_size = min(height/30, width/30)
        spacing = int(label_size * 0.2)

        painter = gui.QPainter(pixmap)
        for instrument, parameters in data.items():
            for parameter, positions in parameters.items():

                if 'labelpos' in positions:
                    label_string = "{}_{} ".format(instrument, parameter)
                    (lx, ly) = positions['labelpos']
                    painter.setBrush(gui.QColor(255, 255, 255, 100))

                    textfont = gui.QFont('Decorative', label_size)
                    textwidth = gui.QFontMetrics(textfont).width(label_string)
                    rectangle_start_x = lx - spacing
                    rectangle_start_y = ly - spacing
                    rectangle_width = textwidth+2*spacing
                    rectangle_height = label_size+2*spacing
                    painter.drawRect(rectangle_start_x,
                                     rectangle_start_y,
                                     rectangle_width,
                                     rectangle_height)
                    painter.setBrush(gui.QColor(25, 25, 25))

                    painter.setFont(textfont)
                    painter.drawText(core.QRectF(rectangle_start_x, rectangle_start_y,
                                                 rectangle_width, rectangle_height),
                                     core.Qt.AlignCenter,
                                     label_string)

                if 'annotationpos' in positions:
                    (ax, ay) = positions['annotationpos']
                    annotationstring = data[instrument][parameter]['value']

                    textfont = gui.QFont('Decorative', label_size)
                    textwidth = gui.QFontMetrics(textfont).width(annotationstring)
                    rectangle_start_x = ax - spacing
                    rectangle_start_y = ay - spacing
                    rectangle_width = textwidth + 2 * spacing
                    rectangle_height = label_size + 2 * spacing
                    painter.setBrush(gui.QColor(255, 255, 255, 100))
                    painter.drawRect(rectangle_start_x,
                                     rectangle_start_y,
                                     rectangle_width,
                                     rectangle_height)
                    painter.setBrush(gui.QColor(50, 50, 50))
                    painter.setFont(textfont)
                    painter.drawText(core.QRectF(rectangle_start_x, rectangle_start_y,
                                                 rectangle_width, rectangle_height),
                                     core.Qt.AlignCenter,
                                     annotationstring)

            canvas.setPixmap(pixmap)

        return canvas, pixmap


class DeviceImage:

    """
    Manage an image of a device
    """

    def __init__(self, folder, station):

        self._data = {}
        self.filename = None
        self.folder = folder
        self.station = station

    def annotateImage(self):
        """
        Launch a Qt Widget to click
        """
        if not qt.QApplication.instance():
            app = qt.QApplication(sys.argv)
        else:
            app = qt.QApplication.instance()
        imagedrawer = MakeDeviceImage(self.folder, self.station)
        app.exec_()
        imagedrawer.close()
        self._data = imagedrawer._data
        self.filename = imagedrawer.filename
        self.saveAnnotations()

    def saveAnnotations(self):
        """
        Save annotated image to disk (image+instructions)
        """
        filename = os.path.join(self.folder, 'deviceimage_annotations.json')
        with open(filename, 'w') as fid:
            json.dump(self._data, fid)

    def loadAnnotations(self):
        """
        Get the annotations. Only call this if the files exist
        Need to load png/jpeg too
        """

        json_filename = os.path.join(self.folder, 'deviceimage_annotations.json')
        self.filename = glob.glob(os.path.join(self.folder, 'deviceimage_raw.*'))[0]
        # this assumes there is only on of deviceimage_raw.*
        with open(json_filename, 'r') as fid:
            self._data = json.load(fid)

    def updateValues(self, station):
        """
        Update the data with actual voltages from the QDac
        """

        for instrument, parameters in self._data.items():
            for parameter in parameters.keys():
                value = station.components[instrument][parameter].get_latest()
                try:
                    floatvalue = float(station.components[instrument][parameter].get_latest())
                    if floatvalue > 1000 or floatvalue < 0.1:
                        valuestr = "{:.2e}".format(floatvalue)
                    else:
                        valuestr = "{:.2f}".format(floatvalue)
                except ValueError:
                    valuestr = str(value)
                self._data[instrument][parameter]['value'] = valuestr

    def makePNG(self, counter, path=None):
        """
        Render the image with new voltage values and save it to disk

        Args:
            counter (int): A counter for the experimental run number
        """
        if self.filename is None:
            raise ValueError('No image selected!')
        if not qt.QApplication.instance():
            app = qt.QApplication(sys.argv)
        else:
            app = qt.QApplication.instance()
        win = qt.QWidget()
        grid = qt.QGridLayout()
        win.setLayout(grid)
        win.imageCanvas = qt.QLabel()
        grid.addWidget(win.imageCanvas)
        win.imageCanvas, pixmap = MakeDeviceImage._renderImage(self._data,
                                                               win.imageCanvas,
                                                               self.filename)
        filename = '{:03d}_deviceimage.png'.format(counter)
        if path:
            filename = os.path.join(path, filename)
        pixmap.save(filename, 'png')
