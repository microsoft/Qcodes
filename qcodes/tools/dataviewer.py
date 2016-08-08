#%% Load packages

import os
import re
import logging

import qtpy.QtGui as QtGui
import qtpy.QtWidgets as QtWidgets
import pyqtgraph as pg
import argparse

import qcodes
from qcodes.plots.pyqtgraph import QtPlot

#%% Helper functions


def findfilesR(p, patt):
    """ Get a list of files (recursive)

    Arguments
    ---------

    p (string): directory
    patt (string): pattern to match

    """
    lst = []
    rr = re.compile(patt)
    for root, _, files in os.walk(p, topdown=False):
        lst += [os.path.join(root, f) for f in files if re.match(rr, f)]
    return lst

#%% Main class


class DataViewer(QtWidgets.QWidget):

    ''' Simple viewer for Qcodes data

    Arugments
    ---------

        datadir (string or None): directory to scan for experiments
        default_parameter (string): name of default parameter to plot
    '''

    def __init__(self, datadir=None, window_title='Data browser', default_parameter='amlitude'):
        super(DataViewer, self).__init__()

        self.default_parameter = default_parameter

        if datadir is None:
            datadir = qcodes.DataSet.default_io.base_location
        self.datadir = datadir

        qcodes.DataSet.default_io = qcodes.DiskIO(datadir)
        logging.info('DataViewer: data directory %s' % datadir)

        # setup GUI
        self.text = QtWidgets.QLabel()
        self.text.setText('Log files at %s' %
                          self.datadir)
        self.logtree = QtWidgets.QTreeView()  # QTreeWidget
        self.logtree.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows)
        self._treemodel = QtGui.QStandardItemModel()
        self.logtree.setModel(self._treemodel)
        self.__debug = dict()
        self.qplot = QtPlot(remote=False, interval=0)
        self.plotwindow = self.qplot.win

        vertLayout = QtWidgets.QVBoxLayout()
        vertLayout.addWidget(self.text)
        vertLayout.addWidget(self.logtree)
        vertLayout.addWidget(self.plotwindow)
        self.setLayout(vertLayout)

        self._treemodel.setHorizontalHeaderLabels(['Log', 'Comments'])
        self.setWindowTitle(window_title)
        self.logtree.header().resizeSection(0, 240)

        # disable edit
        self.logtree.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers)
        self.logtree.doubleClicked.connect(self.logCallback)

        # get logs from disk
        self.updateLogs()

    def updateLogs(self):
        ''' Update the list of measurements '''
        model = self._treemodel
        dd = findfilesR(self.datadir, '.*dat')
        print('found %d files' % (len(dd)))
        # print(dd)

        logs = dict()
        for i, d in enumerate(dd):
            try:
                datetag, logtag = d.split(os.sep)[-3:-1]
                if not datetag in logs:
                    logs[datetag] = dict()
                logs[datetag][logtag] = d
            except Exception:
                pass
        self.logs = logs

        for i, datetag in enumerate(sorted(logs.keys())[::-1]):
            parent1 = QtGui.QStandardItem(datetag)
            for j, logtag in enumerate(sorted(logs[datetag])):
                child1 = QtGui.QStandardItem(logtag)
                child2 = QtGui.QStandardItem('info about plot')
                child3 = QtGui.QStandardItem(os.path.join(datetag, logtag))
                parent1.appendRow([child1, child2, child3])
            model.appendRow(parent1)
            # span container columns
            self.logtree.setFirstColumnSpanned(
                i, self.logtree.rootIndex(), True)

    def plot_parameter(self, data):
        ''' Return parameter to be plotted '''
        arraynames = data.arrays.keys()
        if self.default_parameter in arraynames:
            return self.default_parameter
        vv = [v for v in arraynames if v.endswith('default_parameter')]
        if (len(vv) > 0):
            return vv[0]
        vv = [v for v in arraynames if v.endswith('amplitude')]
        if (len(vv) > 0):
            return vv[0]

        if 'amplitude' in data.arrays.keys():
            return 'amplitude'

        try:
            key = next(iter(data.arrays.keys()))
            return key
        except Exception:
            return None

    def logCallback(self, index):
        ''' Function called when a log entry is selected '''
        logging.info('logCallback!')
        logging.debug('logCallback: index %s' % str(index))
        self.__debug['last'] = index
        pp = index.parent()
        row = index.row()

        tag = pp.child(row, 2).data()

        # load data
        if tag is not None:
            print('logCallback! tag %s' % tag)
            try:
                logging.debug('load tag %s' % tag)
                data = qcodes.load_data(tag)

                self.qplot.clear()

                infotxt = 'arrays: ' + ', '.join(list(data.arrays.keys()))
                q = pp.child(row, 1).model()
                q.setData(pp.child(row, 1), infotxt)

                param_name = self.plot_parameter(data)

                if param_name is not None:
                    logging.info(
                        'using parameter %s for plotting' % param_name)
                    self.qplot.add(getattr(data, param_name))
                else:
                    logging.info('could not find parameter for DataSet')
            except Exception as e:
                print('logCallback! error ...')
                print(e)
                logging.warning(e)
        pass


#%% Run the GUI as a standalone program


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', default=1, help="verbosity level")
    parser.add_argument(
        '-d', '--datadir', type=str, default=None, help="data directory")
    args = parser.parse_args()
    verbose = args.verbose
    datadir = args.datadir

    app = pg.mkQApp()

    dataviewer = DataViewer(datadir=datadir)
    dataviewer.setGeometry(1280, 60, 700, 800)
    dataviewer.qplot.win.setMaximumHeight(400)
    dataviewer.show()
    self = dataviewer

    app.exec()
