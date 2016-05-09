import sys,os
import re
import logging

import qcodes
import qcodes as qc
import qtpy.QtCore as QtCore
import qtpy.QtGui as QtGui
import pyqtgraph as pg
#import pmatlab

def findfilesR(p, patt):
    """ Get a list of files (recursive)

    Arguments
    ---------

    p (string): directory
    patt (string): pattern to match

    """
    lst = []
    rr = re.compile(patt)
    for root, dirs, files in os.walk(p, topdown=False):
        lst += [os.path.join(root, f) for f in files if re.match(rr, f)]
    return lst

class LogViewer(QtGui.QWidget):

    def __init__(self, window_title='Log Viewer', debugdict=dict()):
        super(LogViewer, self).__init__()

        self.text= QtGui.QLabel()
        self.text.setText('Log files at %s' %  qcodes.DataSet.default_io.base_location)
        self.logtree= QtGui.QTreeView() # QTreeWidget
        self.logtree.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self._treemodel = QtGui.QStandardItemModel()
        self.logtree.setModel(self._treemodel)
        self.__debug = debugdict
        self.qplot= qc.QtPlot(remote=False)
        self.plotwindow= self.qplot.win
        #self.plotwindow = pg.GraphicsWindow(title='dummy')

        vertLayout = QtGui.QVBoxLayout()
        vertLayout.addWidget(self.text)
        vertLayout.addWidget(self.logtree)
        vertLayout.addWidget(self.plotwindow)
        self.setLayout(vertLayout)

        self._treemodel.setHorizontalHeaderLabels(['Log', 'Comments'])
        self.setWindowTitle(window_title)        
        self.logtree.header().resizeSection(0, 240)


        # disable edit
        self.logtree.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)

        self.logtree.doubleClicked.connect(self.logCallback)
             
        self.updateLogs()
    def updateLogs(self):
        pass

        model=self._treemodel
        dd=findfilesR(qcodes.DataSet.default_io.base_location, '.*dat')        
        print(dd)
        
        logs=dict()        
        for i, d in enumerate(dd):
            tag= os.path.basename(d)
            datetag, logtag=d.split('/')[-2:]
            if not datetag in logs:
                logs[datetag]=dict()
            logs[datetag][logtag]=d

        for i, datetag in enumerate(sorted(logs.keys())[::-1]):             
            parent1 = QtGui.QStandardItem(datetag)
            for j, logtag in enumerate(logs[datetag]):
                child1 = QtGui.QStandardItem(logtag)
                child2 = QtGui.QStandardItem('info about plot')
                child3 = QtGui.QStandardItem(os.path.join(datetag, logtag) )
                parent1.appendRow([child1, child2, child3])
            model.appendRow(parent1)
            # span container columns
            self.logtree.setFirstColumnSpanned(i, self.logtree.rootIndex(), True)

    def logCallback(self, index):
        print('logCallback!')
        logging.debug('index %s'% str(index))
        self.__debug['last']=index
        pp=index.parent()
        row=index.row()

        
        tag=pp.child(row,2).data()
        
        # load data
        if tag is not None:
            print('logCallback! tag %s' % tag)
            try:
                logging.debug('load tag %s' % tag) 
                data=qc.load_data(tag)
        
                self.qplot.clear(); 
                self.qplot.add(data.amplitude); 
        
            except Exception as e:
                print('logCallback! error ...' )
                logging.debug(e)
                pass
        pass


