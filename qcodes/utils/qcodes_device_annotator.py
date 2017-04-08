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
        self.imageCanvas.setToolTip("Left click to insert a label and right click to insert an annotation.")
        self.loadButton = qt.QPushButton('Load image')

        self.okButton = qt.QPushButton('Save and close')
        self.removeButton = qt.QPushButton('Remove')
        self.removeButton.setToolTip("Remove annotation and label for this parameter")
        self.labeltitle = qt.QLabel('Label:')
        self.labelfield = qt.QLineEdit()
        self.labelfield.setText('')
        self.labelfield.setToolTip("String to be used as label. Defaults to instrument_parameter")
        self.formattertitle = qt.QLabel('Formatter:')
        self.formatterfield = qt.QLineEdit()
        self.formatterfield.setText('')
        self.formatterfield.setToolTip("Formatter to be used for this parameter:\n"
                                       "Uses new style python formatters. I.e.\n"
                                       "':.4f' standard formatter with 4 digits after the decimal point\n"
                                       "':.2e' exponential formatter with 2 digits after the decimal point\n"
                                       "Leave blank for default. \n"
                                       "See www.pyformat.info for more examples")
        self.loadButton.clicked.connect(self.loadimage)
        self.imageCanvas.mousePressEvent = self.set_label_or_annotation
        self.imageCanvas.setStyleSheet('background-color: white')
        self.okButton.clicked.connect(self.saveAndClose)
        self.removeButton.clicked.connect(self.remove_label_and_annotation)

        self.treeView = qt.QTreeView()
        self.model = gui.QStandardItemModel()
        self.model.setHorizontalHeaderLabels([self.tr("Instruments")])
        self.addStation(self.model, station)
        self.treeView.setModel(self.model)
        self.treeView.sortByColumn(0, core.Qt.AscendingOrder)
        self.treeView.setSortingEnabled(True)
        self.treeView.clicked.connect(self.selection_changed)
        grid.addWidget(self.imageCanvas, 0, 0, 4, 6)
        grid.addWidget(self.loadButton, 4, 0)
        grid.addWidget(self.labeltitle, 4, 1)
        grid.addWidget(self.labelfield, 4, 2)
        grid.addWidget(self.formattertitle, 4, 4)
        grid.addWidget(self.formatterfield, 4, 5)
        grid.addWidget(self.removeButton, 4, 8)
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

    def selection_changed(self):
        if not self.treeView.selectedIndexes():
            return
        selected = self.treeView.selectedIndexes()[0]
        selected_instrument = selected.parent().data()
        selected_parameter = selected.data()
        self.labelfield.setText("{}_{} ".format(selected_instrument, selected_parameter))

    def set_label_or_annotation(self, event):
        insertlabel = False
        insertannotation = False
        if event.button() == core.Qt.LeftButton:
            insertlabel = True
        elif event.button() == core.Qt.RightButton:
            insertannotation = True
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

        if insertlabel:
            self._data[selected_instrument][selected_parameter]['labelpos'] = (self.click_x, self.click_y)
            self._data[selected_instrument][selected_parameter]['labelstring'] = self.labelfield.text()
        elif insertannotation:
            self._data[selected_instrument][selected_parameter]['annotationpos'] = (self.click_x, self.click_y)
            if self.formatterfield.text():
                formatstring  = '{' + self.formatterfield.text() + "}"
                self._data[selected_instrument][selected_parameter]['annotationformatter'] = formatstring
                self._data[selected_instrument][selected_parameter]['value'] = formatstring
            else:
                self._data[selected_instrument][selected_parameter]['value'] = 'NaN'

        # draw it
        self.imageCanvas, _ = self._renderImage(self._data,
                                                self.imageCanvas,
                                                self.filename)

    def remove_label_and_annotation(self):
        selected = self.treeView.selectedIndexes()[0]
        selected_instrument = selected.parent().data()
        selected_parameter = selected.data()
        if selected_parameter in self._data[selected_instrument].keys():
            self._data[selected_instrument][selected_parameter] = {}
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
    def _renderImage(data, canvas, filename, title=None):
        """
        Render an image
        """

        pixmap = gui.QPixmap(filename)
        width = pixmap.width()
        height = pixmap.height()

        label_size = min(height/30, width/30)
        spacing = int(label_size * 0.2)

        painter = gui.QPainter(pixmap)
        if title:
            painter.setBrush(gui.QColor(255, 255, 255, 100))
            textfont = gui.QFont('Decorative', label_size)
            textwidth = gui.QFontMetrics(textfont).width(title)
            rectangle_width = textwidth + 2 * spacing
            rectangle_height = label_size + 2 * spacing
            painter.drawRect(0,
                             0,
                             rectangle_width,
                             rectangle_height)
            painter.drawText(core.QRectF(spacing, spacing,
                                         rectangle_width, rectangle_height),
                             core.Qt.AlignTop + core.Qt.AlignLeft,
                             title)

        for instrument, parameters in data.items():
            for parameter, paramsettings in parameters.items():

                if 'labelpos' in paramsettings:
                    if paramsettings.get('labelstring'):
                        label_string = paramsettings.get('labelstring')
                    else:
                        label_string = "{}_{} ".format(instrument, parameter)
                    if paramsettings.get('update'):
                        #parameters that are sweeped should be red.
                        painter.setBrush(gui.QColor(255, 0, 0, 100))
                    else:
                        painter.setBrush(gui.QColor(255, 255, 255, 100))
                    (lx, ly) = paramsettings['labelpos']
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

                if 'annotationpos' in paramsettings:
                    (ax, ay) = paramsettings['annotationpos']
                    annotationstring = paramsettings['value']

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

    def updateValues(self, station, sweeptparameters=None):
        """
        Update the data with actual voltages from the QDac
        """

        for instrument, parameters in self._data.items():
            for parameter in parameters.keys():
                value = station.components[instrument][parameter].get_latest()
                try:
                    floatvalue = float(station.components[instrument][parameter].get_latest())
                    if self._data[instrument][parameter].get('annotationformatter'):
                        valuestr = self._data[instrument][parameter].get('annotationformatter').format(floatvalue)
                    elif floatvalue > 1000 or floatvalue < 0.1:
                        valuestr = "{:.2e}".format(floatvalue)
                    else:
                        valuestr = "{:.2f}".format(floatvalue)
                except (ValueError, TypeError):
                    valuestr = str(value)
                self._data[instrument][parameter]['value'] = valuestr
                if sweeptparameters:
                    for sweeptparameter in sweeptparameters:
                        if sweeptparameter._instrument.name == instrument and sweeptparameter.name == parameter:
                            self._data[instrument][parameter]['update'] = True

    def makePNG(self, counter, path=None, title=None):
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
                                                               self.filename,
                                                               title)
        filename = '{:03d}_deviceimage.png'.format(counter)
        if path:
            filename = os.path.join(path, filename)
        pixmap.save(filename, 'png')
