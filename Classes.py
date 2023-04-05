import os
import sys
import glob
import re
# can handle multiple samples data; reference spectrum
import h5py
import numpy as np
import numpy.ma as ma
import pandas as pd
from numpy import isnan
import scipy.signal
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import pyqtgraph.dockarea
import pyqtgraph as pg
from pyqtgraph import functions as fn # haha
from draggabletabwidget_new import *
from datetime import datetime
import time
import pickle

import larch
# from larch.io import read...
from larch.xafs import *
from larch import *
from larch.fitting import *
import larch.math

# from azint import AzimuthalIntegrator

from struct import unpack
import struct

from scipy.signal import savgol_filter, savgol_coeffs
from scipy.signal import find_peaks, peak_widths
import shutil
from paramiko import SSHClient

# sys.path.insert(0,r"C:\Users\jialiu\gsas2full\GSASII")
gsas_path = os.path.join(os.path.expanduser('~'), 'gsas2full', 'GSASII')
sys.path.insert(0,gsas_path)
import GSASIIindex
import GSASIIspc
import GSASIIlattice
import GSASIIscriptable as G2sc
np.seterr(divide = 'ignore')

# add a stripped space group list
tight_spg_f = os.path.join(gsas_path, 'tight_spg_file')
if os.path.isfile(tight_spg_f):
    with open(tight_spg_f, 'rb') as f:
        tight_spg = pickle.load(f)

else:
    tight_spg = []
    for spg in GSASIIspc.spgbyNum:
        if spg != None:
            tight_spg.append(spg.replace(' ',''))

    with open(tight_spg_f, 'wb') as f:
        pickle.dump(tight_spg, f)


ev2nm = 1239.8419843320025

# warning
# pypowder is not available - profile calcs. not allowed
# pydiffax is not available for this platform
# pypowder is not available - profile calcs. not allowed

class MyGraphicsLayoutWidget(pg.GraphicsView): # only for methodobj == 'xrd' and (TabGraph).label == 'time series'
    def __init__(self, tabgraphobj, methodobj, parent=None, show=False, size=None, title=None, **kargs):
        pg.mkQApp()
        pg.GraphicsView.__init__(self, parent)
        self.ci = MyGraphicsLayout(tabgraphobj, methodobj, **kargs)
        for n in ['nextRow', 'nextCol', 'nextColumn', 'addPlot', 'addViewBox', 'addItem', 'getItem', 'addLayout',
                  'addLabel', 'removeItem', 'itemIndex', 'clear']:
            setattr(self, n, getattr(self.ci, n))
        self.setCentralItem(self.ci)

        if size is not None:
            self.resize(*size)

        if title is not None:
            self.setWindowTitle(title)

        if show is True:
            self.show()

class MyGraphicsLayout(pg.GraphicsLayout):
    def __init__(self, tabgraphobj, methodobj):
        super(MyGraphicsLayout, self).__init__()
        self.tabgraphobj = tabgraphobj
        self.methodobj = methodobj

    def addPlot(self, row=None, col=None, rowspan=1, colspan=1, **kargs):
        plot = MyPlotItem(self.tabgraphobj, self.methodobj, **kargs)
        self.addItem(plot, row, col, rowspan, colspan)
        return plot

class MyPlotItem(pg.PlotItem):
    def __init__(self, tabgraphobj, methodobj):
        super(MyPlotItem, self).__init__(viewBox=MyViewBox(tabgraphobj, methodobj))

class MyViewBox(pg.ViewBox):
    def __init__(self, tabgraphobj, methodobj):
        super(MyViewBox, self).__init__()
        self.act_list = []
        for act in self.menu.actions():
            self.menu.removeAction(act)
            self.act_list.append(act)

        self.updatecontextmenu = True
        self.tabgraphobj = tabgraphobj
        self.methodobj = methodobj

    def getMenu(self, event):
        act_ass_from = QAction('assign from', self.menu)
        act_ass_from.triggered.connect(lambda : self.assign_from(self.tabgraphobj, self.methodobj))
        act_ass_to = QAction('assign to', self.menu)
        act_ass_to.triggered.connect(lambda: self.assign_to(self.tabgraphobj, self.methodobj))
        if self.updatecontextmenu:
            self.menu.addAction(act_ass_from)
            self.menu.addAction(act_ass_to)
            self.updatecontextmenu = False
            for act in self.act_list:
                self.menu.addAction(act)

        return self.menu

    def assign_from(self, tabgraphobj, methodobj):
        peak_x = tabgraphobj.mousePoint.x()  # x and y are all data number, x is not actuall x!
        peak_y = tabgraphobj.mousePoint.y()
        # the following could be problematic: what if there is no integrated checkbox checked?!
        q_start = int(methodobj.parameters['integrated']['clip head'].setvalue)  # q is also data num
        # peak catalog step
        if hasattr(methodobj, 'peaks_catalog_select'):
            if methodobj.peaks_catalog_map != []:
                index = 0
                for peak in methodobj.peaks_catalog_select:
                    # diff = peak[::, 0:2] - np.array([peak_y, peak_x - q_start])
                    # dist = np.sqrt(np.diag(np.dot(diff, diff.T)))
                    # the limit is .01 of the maximum of peak x position, about 30 data num in x in our case (3000 bin in q)
                    # ignore dist in y direction (only a few hundred data usually)
                    # if min(dist) < int(methodobj.linewidgets['time series']['assign proximity'].text()):
                    diff_x = abs(peak[::,1] - (peak_x - q_start))
                    diff_y = abs(peak[::,0] - peak_y)
                    if min(diff_x) < int(methodobj.linewidgets['time series']['assign proximity x'].text()) and \
                        min(diff_y) < int(methodobj.linewidgets['time series']['assign proximity y'].text()):
                        methodobj.peak_ass_from = index
                        print('find peak {} to assign from'.format(index))
                        break

                    index += 1
        # phase assign step
        if hasattr(methodobj, 'phases'):
            if methodobj.phases_map != []:
                index = 0
                for phase in methodobj.phases: # each element of phases contains the index of peak in peaks_catalog_select
                    for peak in phase:
                        diff_x = abs(methodobj.peaks_catalog_select[peak][::, 1] - (peak_x - q_start))
                        diff_y = abs(methodobj.peaks_catalog_select[peak][::, 0] - peak_y)
                        if min(diff_x) < int(methodobj.linewidgets['time series']['assign proximity x'].text()) and \
                                min(diff_y) < int(methodobj.linewidgets['time series']['assign proximity y'].text()):
                            methodobj.phase_ass_from = [index, peak]
                            print('find peak {} of phase {} to assign from'.format(peak, index))
                            # break

                    index += 1
                    # if methodobj.phase_ass_from != []:
                    #     break

    # to add: export peaks_catalog_select and import at a later session
    # to modify: peaks belong to multi phase

    def assign_to(self, tabgraphobj, methodobj):
        peak_x = tabgraphobj.mousePoint.x()  # x and y are all data number, x is not actuall x!
        peak_y = tabgraphobj.mousePoint.y()
        q_start = int(methodobj.parameters['integrated']['clip head'].setvalue)  # q is also data num
        # the peak catalog step
        if hasattr(methodobj, 'peaks_catalog_select'):
            index_from = methodobj.peak_ass_from
            if methodobj.peaks_catalog_map != [] and index_from != []:
                index = 0
                methodobj.peak_ass_to = []
                for peak in methodobj.peaks_catalog_select:
                    diff_x = abs(peak[::, 1] - (peak_x - q_start))
                    diff_y = abs(peak[::, 0] - peak_y)
                    if min(diff_x) < int(methodobj.linewidgets['time series']['assign proximity x'].text()) and \
                            min(diff_y) < int(methodobj.linewidgets['time series']['assign proximity y'].text()):
                        methodobj.peak_ass_to = index
                        print('find peak {} to assign to'.format(index))
                        break

                    index += 1

                index_to = methodobj.peak_ass_to
                if index_to != []:
                    temp = np.concatenate([methodobj.peaks_catalog_select[index_from], methodobj.peaks_catalog_select[index_to]]) # array
                    methodobj.peaks_catalog_select[index_to] = temp[temp[:,0].argsort()]

                    # methodobj.peaks_catalog_map[index_from].setSymbolBrush(methodobj.peaks_color[index_to])
                    methodobj.peaks_catalog_map[index_to].setData(methodobj.peaks_catalog_select[index_to][::, 1] + q_start,
                                                                  methodobj.peaks_catalog_select[index_to][::, 0],
                                                                  symbol='o', symbolBrush=methodobj.peaks_color[index_to],
                                                                  symbolSize=methodobj.parameters['time series']['symbol size'].setvalue)
                    # # now delete it safely
                    methodobj.peaks_catalog_select.pop(index_from)
                    tabgraphobj.tabplot.removeItem(methodobj.peaks_catalog_map[index_from])  # correct?
                    methodobj.peaks_catalog_map.pop(index_from)
                    methodobj.peaks_color.pop(index_from) # for next assign to

                methodobj.peak_ass_from = []  # prevent from repeat index

        # the phase assign step
        # give the found peak of phase i to phase j and delete phase i if there is no other peaks
        if hasattr(methodobj, 'phases'):
            index_from = methodobj.phase_ass_from  # two element: phase and peak
            if methodobj.phases_map != [] and index_from != []:
                index = 0
                methodobj.phase_ass_to = []
                # find the phase to assign
                for phase in methodobj.phases:
                    for peak in phase:
                        diff_x = abs(methodobj.peaks_catalog_select[peak][::, 1] - (peak_x - q_start))
                        diff_y = abs(methodobj.peaks_catalog_select[peak][::, 0] - peak_y)
                        if min(diff_x) < int(methodobj.linewidgets['time series']['assign proximity x'].text()) and \
                                min(diff_y) < int(methodobj.linewidgets['time series']['assign proximity y'].text()):
                            methodobj.phase_ass_to = index
                            print('find phase %i to assign to' % index)
                            break

                    index += 1
                    # if methodobj.phase_ass_to != []:
                    #     break

                index_to = methodobj.phase_ass_to
                if index_to != []: # should be no confusion to the previous step
                    # list merge:
                    methodobj.phases[index_to] = [index_from[1]] + methodobj.phases[index_to] # list
                    # set data for index_to phase_map, hope the color is kept
                    phase_peaks = methodobj.peaks_catalog_select[methodobj.phases[index_to][0]]
                    if len(methodobj.phases[index_to]) > 1:
                        for k in range(len(methodobj.phases[index_to]) - 1):
                            phase_peaks = np.concatenate((phase_peaks,
                                                          methodobj.peaks_catalog_select[methodobj.phases[index_to][k + 1]]))  # another shock!

                    methodobj.phases_map[index_to].setData(phase_peaks[::, 1] + q_start, phase_peaks[::, 0],
                                                           symbol='d', symbolBrush=methodobj.phases_color[index_to], pen=None,
                                                           symbolSize=methodobj.parameters['time series']['symbol size'].setvalue)
                    # delete the whole phase if there is only one peak
                    if len(methodobj.phases[index_from[0]]) == 1:
                        methodobj.phases.pop(index_from[0])
                        tabgraphobj.tabplot.removeItem(methodobj.phases_map[index_from[0]])  # correct?
                        # methodobj.phases_map[index_from[0]].setSymbolBrush(methodobj.phases_color[index_from[0]])
                        methodobj.phases_map.pop(index_from[0])
                        methodobj.phases_color.pop(index_from[0])
                    else: # delete the peak found only
                        # delete only one peak, FIRST!
                        methodobj.phases[index_from[0]].remove(index_from[1])
                        # update the phase_map of assign_from phase:
                        phase_peaks = methodobj.peaks_catalog_select[methodobj.phases[index_from[0]][0]]
                        if len(methodobj.phases[index_from[0]]) > 1:
                            for k in range(len(methodobj.phases[index_from[0]]) - 1):
                                phase_peaks = np.concatenate((phase_peaks,
                                                              methodobj.peaks_catalog_select[methodobj.phases[index_from[0]][k + 1]]))  # another shock!

                        # set data phase_map of index_from, hope the color is kept
                        methodobj.phases_map[index_from[0]].setData(phase_peaks[::, 1] + q_start, phase_peaks[::, 0],
                                                                    symbol='d', symbolBrush=methodobj.phases_color[index_from[0]], pen=None,
                                                                    symbolSize=methodobj.parameters['time series']['symbol size'].setvalue)

                        # methodobj.phases_map.pop(index_from[0])

                    methodobj.phase_ass_from = [] # prevent from repeat index

class Dataclass():
    def __init__(self):
        self.data = None
        self.pen = 'r'
        # not successful:
        # np.array([(0, 0, 0, 0, 0)], dtype=[('red', np.ubyte), ('green', np.ubyte),
        #                                    ('blue', np.ubyte), ('alpha', np.ubyte), ('width', float)])
        self.symbol = None
        self.symbolsize = None
        self.symbolbrush = 'b'
        self.image = None

class Paraclass():
    def __init__(self, **kwargs):
        for key in kwargs:
            self.identity = key # only accept one key now!
            if key == 'values':
                self.setvalue = kwargs[key][0]
                self.upper = kwargs[key][2]
                self.lower = kwargs[key][1]
                self.step = kwargs[key][3]
                # bear in mind that the actual step is 1, the nominal value needs a multiplier, 10

            if key == 'strings':
                self.choice = kwargs[key][0]
                self.choices = kwargs[key][1]

class DockGraph():
    # top level docking graph widget
    def __init__(self, name):
        self.label = name # e.g. xas, xrd,...
        self.tabdict = {} # a dic for this level tab objects, important!

    def gendock(self, winobj):
        self.dockobj = QDockWidget(self.label, winobj)
        # if self.label[0:3] == 'xrd': self.dockobj.setMinimumWidth(winobj.screen_width * .5)
        # else: self.dockobj.setMinimumWidth(winobj.screen_width * .2)
        self.dockobj.setMinimumWidth(winobj.screen_width * .3)
        winobj.addDockWidget(Qt.BottomDockWidgetArea, self.dockobj)
        if len(winobj.gdockdict) > 3: # only accommodate two docks
            self.dockobj.setFloating(True)
        else: self.dockobj.setFloating(False)

        self.docktab = DraggableTabWidget()
        self.dockobj.setWidget(self.docktab)

    def deldock(self, winobj):
        winobj.removeDockWidget(self.dockobj)
        DraggableTabBar.dragging_widget_ = None
        DraggableTabBar.tab_bar_instances_ = [] # these two lines work!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        del self.docktab # these two lines caused fatal error!!! access violation !!!!!!!!!!!
        del self.dockobj # be careful when using del !!!!!!!!!!!!!!!!

    def gencontroltab(self, winobj):
        self.tooltab = QToolBox()
        # self.tooltab.setObjectName(self.label)
        winobj.controltabs.addTab(self.tooltab,self.label)

    def delcontroltab(self, winobj):
        index = winobj.controltabs.indexOf(self.tooltab)
        winobj.controltabs.removeTab(index)
        DraggableTabBar.dragging_widget_ = None
        DraggableTabBar.tab_bar_instances_ = []  # these two lines work!!!
        # del self.tooltab

class TabGraph_index(): # for xrd index
    def __init__(self, name):
        self.label = name # e.g. phase 1, phase 2,...

    def gentab(self, dockobj, methobj, winobj):
        self.itemwidget = QWidget()
        horilayout = QHBoxLayout()

        self.graph = pg.GraphicsLayoutWidget()
        self.table = QTableWidget(selectionBehavior=QtWidgets.QTableView.SelectRows,
            selectionMode=QtWidgets.QTableView.SingleSelection,)
        verti_layout1 = QVBoxLayout()
        verti_layout1.addWidget(self.table)
        verti_layout1.addWidget(self.graph)

        self.cbox_layout = QVBoxLayout()
        for name in methobj.bravaisNames:
            self.cbox_layout.addWidget(QCheckBox(name))

        self.cbox_layout.addWidget(QCheckBox('use M20/(X20 + 1)'))

        layout_form = QFormLayout()
        self.ncno = QLineEdit('4')
        layout_form.addRow('Max N_cal / N_obs', self.ncno)
        self.vol_start = QLineEdit('20')
        layout_form.addRow('<font> start volume/ &#8491; </font> <sup> 3 </sup>', self.vol_start)

        verti_layout2 = QVBoxLayout()
        verti_layout2.addLayout(self.cbox_layout)
        verti_layout2.addLayout(layout_form)

        self.qbt_index = QPushButton('index (Ctrl+N)')
        self.qbt_index.setShortcut('Ctrl+N')
        self.qbt_index.clicked.connect(lambda: methobj.indexing(self, winobj))
        self.qbt_keep = QPushButton('keep selected (Ctrl+k)')
        self.qbt_keep.setShortcut('Ctrl+K')
        self.qbt_keep.clicked.connect(lambda: methobj.keep_selected(self))
        verti_layout2.addWidget(self.qbt_index)
        verti_layout2.addWidget(self.qbt_keep)

        horilayout.addLayout(verti_layout1)
        horilayout.addLayout(verti_layout2)
        self.itemwidget.setLayout(horilayout)
        dockobj.docktab.addTab(self.itemwidget, self.label)

    def deltab(self, dockobj):
        index = dockobj.docktab.indexOf(self.itemwidget)
        dockobj.docktab.removeTab(index)
        # del self.itemwidget

class TabGraph():
    def __init__(self, name):
        self.label = name # e.g. raw, norm,...

    def mouseMoved(self, evt): # surprise!
        self.mousePoint = self.tabplot.vb.mapSceneToView(evt) # not evt[0]
        self.tabplot_label.setText("<span style='font-size: 10pt; color: black'> "
                "x = %0.3f, y = %0.3f</span>" % (self.mousePoint.x(), self.mousePoint.y()))
        if len(self.tabplot.items) > 0: # read z value of an image
            for item in self.tabplot.items:
                if hasattr(item, 'image'):
                    i_x = int(self.mousePoint.x())
                    i_y = int(self.mousePoint.y())
                    if (0 <= i_x < item.image.shape[0]) and (0 <= i_y < item.image.shape[1]):
                        self.tabplot_label_z.setText("<span style='font-size: 10pt; color: black'> z = %0.3f</span>" %
                                                     item.image[i_x, i_y])

    def gentab(self, dockobj, methodobj): # generate a tab for a docking graph
        if self.label == 'time series' and dockobj.label[0:3] == 'xrd':
            self.graphtab = MyGraphicsLayoutWidget(self, methodobj)
        else:
            self.graphtab = pg.GraphicsLayoutWidget()

        dockobj.docktab.addTab(self.graphtab, self.label)
        self.tabplot_label = pg.LabelItem(justify='right')
        self.tabplot_label_z = pg.LabelItem(justify='left')
        self.graphtab.addItem(self.tabplot_label)
        self.graphtab.addItem(self.tabplot_label_z)
        self.tabplot = self.graphtab.addPlot(row=1, col=0)
        # pg.SignalProxy(self.tabplot.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved) # this is outdated!
        self.tabplot.scene().sigMouseMoved.connect(self.mouseMoved) # this is correct !
        self.tabplot.setLabel('bottom',methodobj.axislabel[self.label]['bottom'])
        self.tabplot.setLabel('left', methodobj.axislabel[self.label]['left'])
        if methodobj.axislabel[self.label]['left'] is not 'Data number':
            self.tabplot.addLegend(labelTextSize='9pt')

    def deltab(self, dockobj):
        # print('del graph tab 1')
        index = dockobj.docktab.indexOf(self.graphtab)
        dockobj.docktab.removeTab(index)
        # print('del graph tab 2')
        # del self.graphtab
        # print('del graph tab 3')

    def gencontrolitem(self, dockobj):
        self.itemwidget = QWidget()
        self.itemwidget.setObjectName(self.label)
        self.itemwidget.setAccessibleName(dockobj.label)
        self.itemlayout = QVBoxLayout() # add control options to this layout
        self.itemwidget.setLayout(self.itemlayout)
        dockobj.tooltab.addItem(self.itemwidget, self.label)

    def delcontrolitem(self, dockobj):
        self.itemlayout.setParent(None)
        # index = dockobj.tooltab.indexOf(self.itemwidget)
        # dockobj.tooltab.removeItem(index)
        self.itemwidget.setParent(None)
        # del self.itemlayout
        # del self.itemwidget

    def delcurvechecks(self, tabname, methodobj): # curvedict is a dict for all curve checkboxes
        # actions
        if tabname in methodobj.actions:
            for key in methodobj.actions[tabname]:
                methodobj.actwidgets[tabname][key].setParent(None)

        # parameters
        if tabname in methodobj.parameters: # normalized, chi(k)
            for key in methodobj.parameters[tabname]: # rbkg, kmin,...
                methodobj.parawidgets[tabname][key].setParent(None)
                methodobj.paralabel[tabname][key].setParent(None)

        # checkboxes
        for key in methodobj.availablecurves[tabname]:
            if methodobj.curvedict[tabname][key].isChecked():
                methodobj.curvedict[tabname][key].setChecked(False)

            methodobj.curvedict[tabname][key].setParent(None)

        # LineEdit
        if tabname in methodobj.linedit:
            for key in methodobj.linedit[tabname]:
                # methodobj.linewidgets[tabname][key].setParent(None)
                label = self.range_select.labelForField(methodobj.linewidgets[tabname][key])
                if label is not None: label.setParent(None)
                methodobj.linewidgets[tabname][key].setParent(None)

            self.itemlayout.removeItem(self.itemlayout.itemAt(1))
        # the spacer:
        self.itemlayout.removeItem(self.itemlayout.itemAt(0))

    # tabname, e.g. raw, norm,...; methodobj, e.g. an XAS obj; for 'I0', 'I1',...
    def curvechecks(self, tabname, methodobj, winobj):
        # checkboxes
        for key in methodobj.availablecurves[tabname]:
            methodobj.curvedict[tabname][key] = QCheckBox(key)
            methodobj.curvedict[tabname][key].stateChanged.connect(winobj.graphcurve)
            self.itemlayout.addWidget(methodobj.curvedict[tabname][key])

        # refinement case
        if tabname == 'refinement single':
            for key in methodobj.availablecurves[tabname]:
                if key[0:5] == 'phase': methodobj.curvedict[tabname][key].setChecked(True)

        # parameters
        if tabname in methodobj.parameters:
            methodobj.parawidgets[tabname] = {}
            methodobj.paralabel[tabname] = {}
            for key in methodobj.parameters[tabname]: # rbkg, kmin,...
                temppara = methodobj.parameters[tabname][key]
                if temppara.identity == 'values':
                    methodobj.parawidgets[tabname][key] = QSlider(Qt.Horizontal)
                    tempwidget = methodobj.parawidgets[tabname][key]
                    tempwidget.setObjectName(key)
                    tempwidget.setRange(0, (temppara.upper - temppara.lower) / temppara.step)
                    # tempwidget.setSingleStep(temppara.step)
                    tempwidget.setValue((temppara.setvalue - temppara.lower) / temppara.step)
                    tempwidget.valueChanged.connect(winobj.update_parameters)
                    methodobj.paralabel[tabname][key] = QLabel(key + ':' + '{:.1f}'.format(temppara.setvalue))
                else:
                    methodobj.parawidgets[tabname][key] = QComboBox()
                    tempwidget = methodobj.parawidgets[tabname][key]
                    tempwidget.setObjectName(key)
                    tempwidget.addItems(temppara.choices)
                    tempwidget.currentTextChanged.connect(winobj.update_parameters)
                    methodobj.paralabel[tabname][key] = QLabel(key)

                self.itemlayout.addWidget(methodobj.paralabel[tabname][key])
                self.itemlayout.addWidget(tempwidget)

        # actions
        if tabname in methodobj.actions:
            methodobj.actwidgets[tabname] = {}
            for key in methodobj.actions[tabname]:
                methodobj.actwidgets[tabname][key] = QPushButton(key)
                if key[-1] == ')':
                    methodobj.actwidgets[tabname][key].setShortcut(key[-7:-1])
                # tempfunc = getattr(methodobj, methodobj.actions[tabname][key])

                if key == 'update y,z range (Ctrl+0)': # for update the range of 2D plot
                    methodobj.actwidgets[tabname][key].clicked.connect(
                        lambda state, t = tabname, k = key: methodobj.actions[t][k](winobj, methodobj, tabname))  # e,g, MainWin, XAS, munorm-T
                    # what is this state!!!???
                    # The QPushButton.clicked signal emits an argument that indicates the state of the button.
                    # When you connect to your lambda slot, the optional argument you assign to is being overwritten by the state of the button.
                    # This way the button state is ignored and the correct value is passed to your method.

                else:
                    methodobj.actwidgets[tabname][key].clicked.connect(
                        lambda state, t = tabname, k = key: methodobj.actions[t][k](winobj))  # shock !!!
                        # lambda state, k = key: methodobj.actions[tabname][k](winobj)) # shock !!!
                self.itemlayout.addWidget(methodobj.actwidgets[tabname][key])

        # lineEdit
        if tabname in methodobj.linedit:
            methodobj.linewidgets[tabname] = {}
            self.range_select = QFormLayout()
            for key in methodobj.linedit[tabname]:
                methodobj.linewidgets[tabname][key] = QLineEdit(methodobj.linedit[tabname][key])
                self.range_select.addRow(key, methodobj.linewidgets[tabname][key])
                
            self.itemlayout.addLayout(self.range_select)

        self.curvespacer = QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.itemlayout.addItem(self.curvespacer)

class Methods_Base():
    def __init__(self):
        self.availablemethods = []
        self.availablecurves = {}
        self.filename = []
        self.axislabel = {}
        self.dynamictitle = []
        self.checksdict = {}  # QCheckBox obj for raw, norm,...
        self.curvedict = {}  # QCheckBox obj for curves belong to raw, norm,...
        # the first element in the time series list, the following elements can be memorized
        self.data_timelist = []
        self.data_timelist.append({})
        # self.data_timelist = [{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}]
        self.curve_timelist = []
        self.curve_timelist.append({})  # the curve, corresponding to each data obj
        # self.curve_timelist = [{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}] # the curves
        self.colorhue = {}
        self.index = 0 # based on this to update the curves/data or not
        self.index_ref = -1 # based on this to update the curves/data or not
        self.update = True
        self.timediff = []  # based on this to decide the index from the slidervalue
        self.parawidgets = {} # to control parameters after curve checkboxes
        self.actwidgets = {}
        self.linewidgets = {}
        self.parameters = {} # control parameters
        self.actions = {}
        self.linedit = {}
        self.paralabel = {} # control para labels widget
        self.pre_ts_btn_text = 'Update by parameters(Ctrl+D)'
        # self.slideradded = False

        self.maxhue = 100  # 100 colorhues
        self.huestep = 3 # color rotation increment

    def update_range(self, winobj, methodobj, tabname):
        try:
            # print('done')
            time_start = methodobj.entrytimesec[max(0,int(methodobj.linewidgets[tabname]['y min'].text())),0]
            time_end = methodobj.entrytimesec[min(methodobj.entrytimesec.shape[0] - 1,
                                                  int(methodobj.linewidgets[tabname]['y max'].text())), 1]
        except:
            print('check your input y values')
        else:
            for key in winobj.methodict: # xas_, xrd_, refl_
                for sub_key in winobj.methodict[key].checksdict: # norm-T, chi-T, ...
                    if (sub_key[-2:] == '-T' or sub_key == 'time series') and winobj.methodict[key].checksdict[sub_key].isChecked():
                        pw = winobj.gdockdict[key].tabdict[sub_key].tabplot
                        try:
                            y_min = np.where(winobj.methodict[key].entrytimesec[:,0] - time_start <= 0)[0][-1]
                        except:
                            y_min = 0
                        try:
                            y_max = np.where(winobj.methodict[key].entrytimesec[:,1] - time_end >= 0)[0][0]
                        except:
                            y_max = winobj.methodict[key].entrytimesec.shape[0]
                        try:
                            pw.setYRange(y_min, y_max)
                            winobj.methodict[key].linewidgets[sub_key]['y min'].setText(str(y_min))
                            winobj.methodict[key].linewidgets[sub_key]['y max'].setText(str(y_max))
                            if os.path.isfile(winobj.methodict[key].exportfile):
                                with h5py.File(winobj.methodict[key].exportfile, 'a') as f:
                                    if 'y min' in list(f.keys()): del f['y min']
                                    if 'y max' in list(f.keys()): del f['y max']
                                    f.create_dataset('y min', data=y_min, dtype='int')
                                    f.create_dataset('y max', data=y_max, dtype='int')

                        except:
                            print('check your input y values')

            pw = winobj.gdockdict[methodobj.method_name].tabdict[tabname].tabplot
            for item in pw.childItems():
                if type(item).__name__ == 'ColorBarItem':
                    item_colorbar = item

            key = methodobj.method_name
            sub_key = tabname
            try:
                if item_colorbar and 'z min' in winobj.methodict[key].linewidgets[sub_key]:
                    item_colorbar.setLevels((float(winobj.methodict[key].linewidgets[sub_key]['z min'].text()),
                                             float(winobj.methodict[key].linewidgets[sub_key]['z max'].text())))
            except: print('there is no color bar yet')

    def plot_pointer(self, tabname, p_x, p_y, symbol, size):
        self.data_timelist[0][tabname]['pointer'].data = np.array([[p_x], [p_y]]).transpose()
        self.data_timelist[0][tabname]['pointer'].symbol = symbol
        self.data_timelist[0][tabname]['pointer'].symbolsize = size

    def ini_data_curve_color(self):
        for key in self.availablemethods: # raw, norm,...
            self.colorhue[key] = len(self.availablecurves[key]) * self.huestep # ini the colorhue for time series
            self.curvedict[key] = {}  # QCheckBox obj for curves belong to raw, norm,...
            self.data_timelist[0][key] = {}  # e.g. self.data_timelist[0]['raw']
            self.curve_timelist[0][key] = {}  # just initialize here
            for subkey in self.availablecurves[key]:
                self.data_timelist[0][key][subkey] = Dataclass()  # e.g. self.data_timelist[0]['raw']['I0']

    def data_update(self, slidervalue):
        self.index_from_time(slidervalue) # update self.index

        if self.index == self.index_ref: # no need to update if the index does not change
            self.update = False
        else:
            self.index_ref = self.index # data number
            self.update = True
            self.startime = datetime.fromtimestamp(self.entrytimesec[self.index, 0]).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
            self.endtime = datetime.fromtimestamp(self.entrytimesec[self.index, 1]).strftime("%H:%M:%S.%f")[:-3]
            # colors and processes
            for key in self.data_timelist[0]:
                nstep = 0
                for subkey in self.data_timelist[0][key]:
                    if self.data_timelist[0][key][subkey].pen is not None:
                        self.data_timelist[0][key][subkey].pen = pg.mkPen(pg.intColor(nstep * self.huestep, self.maxhue),
                                                                      width=1.5)
                        nstep += 1


                self.data_process(False)

    def index_from_time(self, slidervalue): # update self.index
        self.timediff = slidervalue - self.entrytimesec[::,0]
        if self.timediff[0] < 0:
            self.index = 0
        else:
            self.index = np.where(self.timediff >= 0)[0][-1] # return a tuple!

        self.index = int(self.index)

    def data_curve_copy(self, data): # data and curve are, e.g., the first obj in the data_timelist and curve_timelist
        # used for copying ...list[0] to next list element.
        newdata = {}
        newcurve = {}
        for key in self.availablemethods: # raw, norm,...
            newdata[key] = {}
            newcurve[key] = {}
            for subkey in self.availablecurves[key]:
                newdata[key][subkey] = Dataclass()
                # this error proof should always be present to separate, say curve from image
                if subkey in data[key]:
                    if hasattr(data[key][subkey], 'data'):
                        if data[key][subkey].data is not None:
                            newdata[key][subkey].data = data[key][subkey].data
                            if data[key][subkey].symbol is not None:
                                newdata[key][subkey].symbol = data[key][subkey].symbol
                                newdata[key][subkey].symbolsize = data[key][subkey].symbolsize
                                newdata[key][subkey].symbolbrush = pg.intColor(self.colorhue[key], self.maxhue)
                            elif data[key][subkey].pen is not None:
                                newdata[key][subkey].pen = pg.mkPen(pg.intColor(self.colorhue[key], self.maxhue), width=1.5)

                            self.colorhue[key] += self.huestep

        return newdata, newcurve

    def tabchecks(self, winobj, method): # generate all checks for the second level methods: raw, norm,.
        for key in self.availablemethods: # raw, norm,...
            self.checksdict[key] = QCheckBox(key) # e.g. self.checksdict['raw']
            self.checksdict[key].stateChanged.connect(winobj.graph_tab)
            winobj.subcboxverti[method].addWidget(self.checksdict[key])

        self.prep_progress = QProgressBar(minimum=0, maximum=99)
        self.prep_progress.setValue(0)
        self.pre_ts_btn = QPushButton(self.pre_ts_btn_text)
        self.pre_ts_btn.setShortcut('Ctrl+D')
        self.pre_ts_btn.clicked.connect(lambda: self.plot_from_prep(winobj))
        self.load_ts_btn = QPushButton('Load time series(Ctrl+1)')
        self.load_ts_btn.setShortcut('Ctrl+1')
        self.load_ts_btn.clicked.connect(lambda: self.plot_from_load(winobj))
        winobj.subcboxverti[method].addWidget(self.pre_ts_btn)
        winobj.subcboxverti[method].addWidget(self.prep_progress)
        winobj.subcboxverti[method].addWidget(self.load_ts_btn)
        self.itemspacer = QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
        winobj.subcboxverti[method].addItem(self.itemspacer)

    def deltabchecks(self, winobj, method): # del above checks
        for key in self.availablemethods:  # raw, norm,...
            if self.checksdict[key].isChecked(): self.checksdict[key].setChecked(False)
            winobj.subcboxverti[method].removeWidget(self.checksdict[key])

        self.pre_ts_btn.setParent(None)
        self.prep_progress.setParent(None)
        self.load_ts_btn.setParent(None)
        winobj.subcboxverti[method].removeWidget(self.pre_ts_btn)
        winobj.subcboxverti[method].removeWidget(self.prep_progress)
        winobj.subcboxverti[method].removeWidget(self.load_ts_btn)
        winobj.subcboxverti[method].removeItem(self.itemspacer)
        winobj.delslider()  # del the slider
        self.checksdict = {}

class XAS(Methods_Base):
    def __init__(self, path_name_widget):
        # below are needed for each method (xas, xrd,...)
        super(XAS, self).__init__()
        self.availablemethods = ['raw', 'normalizing', 'normalized', 'chi(k)', 'chi(r)',
                                 'E0-t', 'Jump-t',
                                 'mu_norm-T', 'chi(k)-T',
                                 'chi(r)-T',
                                 'LCA(internal) single', 'LCA(internal)-t']
        self.availablecurves['raw'] = ['I0', 'I1']
        self.availablecurves['normalizing'] = ['mu','filter by time','filter by energy',
                                               'pre-edge', 'pre-edge points',
                                               'post-edge', 'post-edge points']
        self.availablecurves['normalized'] = ['normalized mu','filter by time','filter by energy',
                                              'reference','post-edge bkg']
        self.availablecurves['chi(k)'] = ['chi-k','window']
        self.availablecurves['chi(r)'] = ['chi-r','Re chi-r','Im chi-r']
        self.availablecurves['E0-t'] = ['pointer']
        self.availablecurves['Jump-t'] = ['pointer']
        self.availablecurves['mu_norm-T'] = ['pointer']
        self.availablecurves['chi(k)-T'] = ['pointer']
        self.availablecurves['chi(r)-T'] = ['pointer']
        self.availablecurves['LCA(internal) single'] = ['mu_norm', 'component 1',
                                                        'component 2', 'component 3', 'errors']
        self.availablecurves['LCA(internal)-t'] = ['pointer', 'Components',
                                                   'chisqr.', 'redchi.', 'redchi. norm.']
        self.directory = path_name_widget['directory'].text()
        self.fileshort = path_name_widget['raw file'].text()
        self.filename = os.path.join(self.directory, 'raw', self.fileshort + '.h5')
        self.axislabel = {'raw':{'bottom':'Energy/eV',
                                 'left':'log(Intensity)/a.u.'},
                          'normalizing':{'bottom':'Energy/eV',
                                       'left':'<font> &mu; </font>'},
                          'normalized':{'bottom':'Energy/eV',
                                       'left':'<font> &mu; </font>'},
                          'chi(k)':{'bottom':'<font> k / &#8491; </font> <sup> -1 </sup>',
                                    'left':' k <sup> 2 </sup> <font> &chi; </font>'},
                          'chi(r)':{'bottom':'<font> R / &#8491; </font>',
                                    'left':''},
                          'E0-t':{'bottom':'Data number',
                                  'left':'Energy/eV'},
                          'Jump-t': {'bottom': 'Data number',
                                   'left': ''},
                          'mu_norm-T': {'bottom': 'Energy/eV',
                                       'left': 'Data number'},
                          'chi(k)-T': {'bottom': '<font> k / &#8491; </font> <sup> -1 </sup>',
                                       'left': 'Data number'},
                          'chi(r)-T':{'bottom':'<font> R / &#8491; </font>',
                                      'left':'Data number'},
                          'LCA(internal) single':{'bottom':'Energy/eV',
                                                  'left':'<font> &mu; </font>'},
                          'LCA(internal)-t':{'bottom':'data number',
                                             'left':'fraction'}}

        self.ini_data_curve_color()

        # below are xas specific attributes
        self.grouplist = [] # put each of the larch Group obj that's displayed on the active tabgraph
        self.entrytimesec = []
        self.entrydata = []
        # self.parameters = {'rbkg':1,
        #                    'kmin':1,
        #                    'kmax':6,
        #                    'dk':1,
        #                    'window':'hanning',
        #                    'kweight':2}
        # remember to set error proof for kmax, the max is the highest available k, not 20
        self.parameters = {'normalizing':{'Savitzky-Golay window (time)':Paraclass(values=(1,1,100,2)),
                                          'Savitzky-Golay order (time)':Paraclass(values=(1, 1, 7, 2)),
                                          'Savitzky-Golay window (energy)': Paraclass(values=(1, 1, 100, 2)),
                                          'Savitzky-Golay order (energy)': Paraclass(values=(1, 1, 7, 2)),
                                          'pre-edge para 1': Paraclass(values=(.01, 0, 1, .01)), # 0.01 of e0 - first point
                                          'pre-edge para 2': Paraclass(values=(.33, .05, .95, .01)), # 1/3 of e0 - pre 1
                                          'post-edge para 2': Paraclass(values=(.01, 0, 1, .01)), # 0.01 of last point - e0
                                          'post-edge para 1': Paraclass(values=(.33, .05, .95, .01))}, # 1/3 of post 1 - e0
                           'normalized':{'rbkg':Paraclass(values=(1,1,3,.1)),
                                         'Savitzky-Golay window (time)':Paraclass(values=(1,1,100,2)),
                                         'Savitzky-Golay order (time)':Paraclass(values=(1, 1, 7, 2)),
                                         'Savitzky-Golay window (energy)': Paraclass(values=(1, 1, 100, 2)),
                                         'Savitzky-Golay order (energy)': Paraclass(values=(1, 1, 7, 2)),},
                           'chi(k)':{'kmin':Paraclass(values=(.9,0,3,.1)),
                                     'kmax':Paraclass(values=(6,3,20,.1)),
                                     'dk':Paraclass(values=(0,0,1,1)),
                                     'window':Paraclass(strings=('Hanning',['Hanning','Parzen'])),
                                     'kweight':Paraclass(values=(1,0,2,1))},
                           'mu_norm-T':{'flat norm':Paraclass(strings=('flat',['flat','norm'])),
                                        'all half': Paraclass(strings=('all',['all','odd','even']))},
                           'LCA(internal)-t':{'all half': Paraclass(strings=('all',['all','odd','even']))},
                           }

        self.actions = {'normalizing':{'filter all': self.filter_all_normalizing},
                        'normalized': {'filter all': self.filter_all_normalized},
                        'mu_norm-T':{'x, y select start (Ctrl+R)': self.range_select,
                                     'do internal LCA (Ctrl+Y)': self.lca_internal,
                                     'Export time series (Ctrl+X)': self.export_ts,
                                     'update y,z range (Ctrl+0)': self.update_range},
                        'chi(k)-T':{'update y,z range (Ctrl+0)': self.update_range},
                        'chi(r)-T':{'update y,z range (Ctrl+0)': self.update_range},
                        'LCA(internal)-t':{'update (Ctrl+U)': self.lca_result_update,
                                           'load saved LCA (Ctrl+J)': self.lca_load}}

        self.linedit = {'mu_norm-T': {'z max':'102',
                                      'z min':'98',
                                      'y min':'0',
                                      'y max': '100',
                                      'component 1 (internal LCA)': '0',
                                      'component 2 (internal LCA)': '',
                                      'component 3 (internal LCA)': '',
                                      'select start (y)': '100',
                                      'select end (y)': '200',
                                      'select start (x)': '0',
                                      'select end (x)': '100',
                                      'average component 1 by':'10',
                                      'average component 2 by':'10',
                                      'average component 3 by':''},
                        'chi(k)-T': {'z max':'0.3',
                                     'z min':'-0.3',
                                     'y min': '0',
                                     'y max': '100'},
                        'chi(r)-T': {'z max': '0.05',
                                     'z min': '0',
                                     'y min': '0',
                                     'y max': '100'},
                        'LCA(internal)-t': {'savgol win': '5',
                                            'savgol order':'1'}}

                                        # self.average = int(path_name_widget['average (time axis)'].text())  # number of average data points along time axis, an odd number
        self.range = path_name_widget['energy range (eV)'].text() # useful for time_range!
        self.energy_range = [int(self.range.partition("-")[0]), int(self.range.partition("-")[2])]  # for Pb L and Br K edge combo spectrum
        self.exportdir = os.path.join(self.directory, 'process', self.fileshort + \
                               '_range_{}_{}eV'.format(self.energy_range[0], self.energy_range[1]))#, '')
        self.ref_mu = []
        self.filter_flag_normalizing = False
        self.filter_flag_normalized = False

        # self.filtered = False
        self.data_error = .001 # user defined data error

    def export_ts(self, winobj): # this could add to the principle data h5 file that's already there
        resultfile = h5py.File(os.path.join(self.exportdir, 'mu_norm_time_series.h5'), 'w')
        resultfile.create_dataset('mu_norm', data=np.array(self.espace))
        resultfile.create_dataset('Energy', data=self.entrydata[0,0,:])
        resultfile.close()

    def lca_load(self, winobj):
        fn = os.path.join(self.exportdir, self.fileshort + '_LCA_result')
        if os.path.isfile(fn):
            with open(fn, 'rb') as f:
                self.lca_result_group, y_start, y_end = pickle.load(f)

            print('load and begin plot saved lca results')

        self.lca_plot(winobj, y_start, y_end)

    def lca_result_update(self, winobj):
        y_start = int(self.linewidgets['mu_norm-T']['select start (y)'].text())
        y_end = int(self.linewidgets['mu_norm-T']['select end (y)'].text())
        self.lca_plot(winobj, y_start, y_end)

    def lca_plot(self, winobj, y_start, y_end):
        winobj.setslider()
        pw = winobj.gdockdict[self.method_name].tabdict['LCA(internal)-t'].tabplot
        for index in reversed(range(len(pw.items))):  # shocked!
            if pw.items[index].name()[0:8] in ['Componen', 'residue_']:
                pw.removeItem(pw.items[index])

        weights = {}
        residue = []
        for y in np.arange(y_start, y_end):
            for c_name in self.lca_result_group[y - y_start].weights:
                if y == y_start: weights[c_name] = []
                weights[c_name].append(self.lca_result_group[y - y_start].weights[c_name])

            residue.append([self.lca_result_group[y - y_start].chisqr,
                            self.lca_result_group[y - y_start].redchi])

        residue = np.array(residue)
        residue = np.column_stack((residue, residue[:, 1] / residue[:, 1].max()))

        sg_win = int(self.linewidgets['LCA(internal)-t']['savgol win'].text())
        sg_order = int(self.linewidgets['LCA(internal)-t']['savgol order'].text())
        if sg_win > sg_order + 1 and sg_order % 2:
            for c_name in self.lca_result_group[y - y_start].weights:
                weights[c_name] = scipy.signal.savgol_filter(weights[c_name], sg_win, sg_order)

        if self.parameters['LCA(internal)-t']['all half'].choice == 'all':
            data_sel = np.s_[:]
        elif self.parameters['LCA(internal)-t']['all half'].choice == 'odd':
            data_sel = np.s_[1::2]
        else:
            data_sel = np.s_[0::2]

        if self.curvedict['LCA(internal)-t']['Components'].isChecked():
            c_index = 0
            color = ['r', 'b', 'g']
            for c_name in self.lca_result_group[y - y_start].weights:
                pw.plot(np.arange(y_start, y_end)[data_sel], weights[c_name][data_sel],
                        symbol='+', symbolSize=10, symbolPen=color[c_index], name='Component {}'.format(c_index + 1))
                c_index += 1

        curves = self.availablecurves['LCA(internal)-t'][2:] # the first two are pointer and Components
        for curve in curves:
            if self.curvedict['LCA(internal)-t'][curve].isChecked():
                index = curves.index(curve)
                pw.plot(np.arange(y_start, y_end)[data_sel], residue[:, index][data_sel],
                        pen=pg.mkPen('r', width=5),name='residue_' + curve)

    def lca_internal(self, winobj): # currently only two components
        # establish errors
        # running plot_from_prep to have correct errors with each group/mu data
        # do lca
        # at least two components
        # according to larch minimize
        data_choice = self.parameters['mu_norm-T']['all half'].choice
        if data_choice != 'all':
            groups = []
            for i in np.arange(0,len(self.grouplist),2):
                groups.append(self.grouplist[i])
                groups.append(self.grouplist[i + 1])
                if data_choice == 'odd':
                    self.grouplist[i] = self.grouplist[i + 1]
                else:
                    self.grouplist[i + 1] = self.grouplist[i]

        c_all = []
        for i in range(3):
            t = self.linewidgets['mu_norm-T']['component %i (internal LCA)' % (i + 1)].text()
            try:
                t = int(t)
            except:
                print('component %i is not activated' % (i + 1))
            else:
                a = int(self.linewidgets['mu_norm-T']['average component %i by' % (i + 1)].text())
                if t - np.ceil(a / 2) > 0 and t + np.ceil(a / 2) < len(self.grouplist) - 1:
                    norm = self.grouplist[t].norm
                    for j in range(int(np.ceil(a / 2))):
                        norm += (self.grouplist[t - j - 1].norm + self.grouplist[t + j + 1].norm) / 2

                    self.grouplist[t].norm = norm / (np.ceil(a / 2) + 1)
                    c_all.append(self.grouplist[t]) # larch accept group as component

        y_start = int(self.linewidgets['mu_norm-T']['select start (y)'].text())
        y_end = int(self.linewidgets['mu_norm-T']['select end (y)'].text())
        x_start = float(self.linewidgets['mu_norm-T']['select start (x)'].text())
        x_end = float(self.linewidgets['mu_norm-T']['select end (x)'].text())

        self.lca_result_group = []
        if c_all != []:
            for k in np.arange(y_start, y_end):
                print('fitting data %i' % k)
                self.lca_result_group.append(larch.math.lincombo_fit(group = self.grouplist[k],
                                                                     components = c_all,
                                                                     minvals = [0] * len(c_all),
                                                                     maxvals = [1] * len(c_all),
                                                                     arrayname = 'norm',
                                                                     xmin = x_start, xmax = x_end))
                if k == y_end - 1: print('done with LCA for {}'.format(data_choice))

            # save it
            fn = os.path.join(self.exportdir, self.filename + '_LCA_result')
            with open(fn, 'wb') as f:
                pickle.dump([self.lca_result_group, y_start, y_end], f, -1)

            print('output lca results')

            if not self.checksdict['LCA(internal)-t'].isChecked():
                self.checksdict['LCA(internal)-t'].setChecked(True)

            self.parawidgets['LCA(internal)-t']['all half'].setCurrentText(data_choice)

        if data_choice != 'all':
            for i in range(len(groups)):
                self.grouplist[i] = groups[i]

    def range_select(self, winobj):
        # to select range for peaks sorting
        pw = winobj.gdockdict[self.method_name].tabdict['mu_norm-T'].tabplot
        tempwidget = self.actwidgets['mu_norm-T']['x, y select start (Ctrl+R)']
        pw.scene().sigMouseClicked.connect(lambda evt, p=pw: self.range_clicked(evt, p))
        if tempwidget.text() == 'x, y select start (Ctrl+R)':
            tempwidget.setText('x, y select end (Ctrl+R)')
            tempwidget.setShortcut('Ctrl+R')
        elif tempwidget.text() == 'x, y select end (Ctrl+R)':
            tempwidget.setText('done (Ctrl+R)')
            tempwidget.setShortcut('Ctrl+R')
        else:
            tempwidget.setText('x, y select start (Ctrl+R)')
            tempwidget.setShortcut('Ctrl+R')
            pw.scene().sigMouseClicked.disconnect()

    def range_clicked(self, evt, pw): # watch out the difference between the nominal and actual x position
        if pw.sceneBoundingRect().contains(evt.scenePos()):
            mouse_point = pw.vb.mapSceneToView(evt.scenePos())  # directly, we have a viewbox!!!
            temptext = self.actwidgets['mu_norm-T']['x, y select start (Ctrl+R)'].text()
            actual_x = mouse_point.x() / self.entrydata.shape[2] * \
                       (self.entrydata[0,0,-1] - self.entrydata[0,0,0]) + self.entrydata[0,0,0]
            if temptext == 'x, y select end (Ctrl+R)':
                self.linewidgets['mu_norm-T']['select start (y)'].setText(str(int(mouse_point.y())))
                self.linewidgets['mu_norm-T']['select start (x)'].setText('{:.1f}'.format(actual_x))
                self.linewidgets['mu_norm-T']['component 1 (internal LCA)'].setText(str(int(mouse_point.y())))
            elif temptext == 'done (Ctrl+R)':
                self.linewidgets['mu_norm-T']['select end (y)'].setText(str(int(mouse_point.y())))
                self.linewidgets['mu_norm-T']['select end (x)'].setText('{:.1f}'.format(actual_x))
                self.linewidgets['mu_norm-T']['component 2 (internal LCA)'].setText(str(int(mouse_point.y())))

    def filter_all_normalizing(self, winobj): # make some warning sign here
        qbtn = winobj.sender()
        if self.filter_flag_normalizing:
            qbtn.setText('filter all')
            self.filter_flag_normalizing = False
        else:
            qbtn.setText('do not filter all')
            self.filter_flag_normalizing = True

    def filter_all_normalized(self, winobj):
        qbtn = winobj.sender()
        if self.filter_flag_normalized:
            self.filter_flag_normalized = False
            qbtn.setText('filter all')
        else:
            self.filter_flag_normalized = True
            qbtn.setText('do not filter all')

    def plot_from_prep(self, winobj): # generate larch Group for all data ! not related to 'load from prep' any more
        for index in range(self.entrydata.shape[0]): # 0,1,...,self.entrydata.shape[0] - 1
            if (self.entrydata[index, 1:2, ::] <= 0).any():
                mu = np.zeros(self.entrydata.shape[2])
            else:
                mu = np.log(self.entrydata[index, 1, ::] / self.entrydata[index, 2, ::])
            # a way to prevent mu become nan ! when the electrometer readings are so small

            if len(self.grouplist) != self.entrydata.shape[0]:
                self.grouplist.append(Group(name='spectrum' + str(index)))

            if self.filter_flag_normalizing and not self.filter_flag_normalized:
                mu, mu_error = self.filter_single_point(index, 'normalizing')
                self.exafs_process_single(index, mu, mu_error)
            elif self.filter_flag_normalized and not self.filter_flag_normalizing:
                mu, mu_error = self.filter_single_point(index, 'normalized')
                self.exafs_process_single(index, mu, mu_error)
            else:
                self.exafs_process_single(index, mu, self.data_error)

            self.prep_progress.setValue(int((index + 1) / self.entrydata.shape[0] * 100))
            # self.parameters['chi(k)']['kmax'].upper = self.grouplist[index].k[-1]
            # rspacexlen = len(self.grouplist[index].r)
        # self.plot_from_load(winobj)
        # if self.filter_flag_normalizing: # always update the binary files
        #     with open(os.path.join(self.exportdir, self.fileshort + '_Group_List_Smoothed'), 'wb') as f:
        #         pickle.dump(self.grouplist, f, -1)  # equals to pickle.HIGHEST_PROTOCOL
        # else: # prevent filtered data to be saved
        output = os.path.join(self.exportdir, self.fileshort + '_Group_List')
        if not os.path.isfile(output):
            with open(output, 'wb') as f:
                pickle.dump(self.grouplist, f, -1)

    def exafs_process_single(self, index, mu, mu_error):
        Energy = self.entrydata[index, 0, ::]
        if not hasattr(self.grouplist[index], 'energy'):
            self.grouplist[index].energy = Energy

        self.grouplist[index].mu_error = mu_error # this is customized attribute?
        # print(isnan(mu) == True)
        try:
            e0 = 13040
            if e0 < Energy[1] or e0 > Energy[-2]:
                e0 = find_e0(Energy, mu, group=self.grouplist[index])
            else:
                self.grouplist[index].e0 = e0 # watch out! this is a hard coded line, not so good!!!
                print('hard coded e0 = 13040 eV! watch out!!!')
        except:
            print(str(index) + ' bad data, can not find e0')
            e0 = Energy[int(len(Energy) / 5)]

        pre_edge_point_1 = (e0 - Energy[0]) * (1 - self.parameters['normalizing']['pre-edge para 1'].setvalue)
        post_edge_point_2 = (Energy[-1] - e0) * (1 - self.parameters['normalizing']['post-edge para 2'].setvalue)
        try:
            pre_edge(Energy, mu,
                     e0 = e0, # watch out! this is a hard coded line, not so good!!!
                     group=self.grouplist[index],
                     pre1= - pre_edge_point_1,
                     pre2= - pre_edge_point_1 * self.parameters['normalizing']['pre-edge para 2'].setvalue,
                     norm2=post_edge_point_2,
                     norm1=post_edge_point_2 * self.parameters['normalizing']['post-edge para 1'].setvalue)
            print(str(index) + 'pre_edge')
        except:
            print(str(index) + ' bad data, can not find pre/post edge')
            self.grouplist[index].norm = np.ones(len(Energy))
            self.grouplist[index].flat = np.ones(len(Energy))
            self.grouplist[index].edge_step = 0
        # do autobk when chi(k) or chi(k)-T is checked
        if self.checksdict['chi(k)'].isChecked() or self.checksdict['chi(k)-T'].isChecked() or \
                self.checksdict['chi(r)'].isChecked() or self.checksdict['chi(r)-T'].isChecked():
            try:
                autobk(Energy, mu,
                       e0 = e0, # watch out! this is a hard coded line, not so good!!!
                       rbkg=self.parameters['normalized']['rbkg'].setvalue, group=self.grouplist[index])
                # print(str(index) + 'autobak')
            except:
                print(str(index) + ' bad data, can not find background')
                self.grouplist[index].chi = np.zeros(int(len(self.grouplist[index].k))) # will this work
        # do xftf when chi(r) or chi(r)-T is checked
        if self.checksdict['chi(k)'].isChecked() or self.checksdict['chi(r)-T'].isChecked():
            try:
                xftf(self.grouplist[index].k, self.grouplist[index].chi,
                     kmin=self.parameters['chi(k)']['kmin'].setvalue, kmax=self.parameters['chi(k)']['kmax'].setvalue,
                     dk=self.parameters['chi(k)']['dk'].setvalue, window=self.parameters['chi(k)']['window'].choice,
                     kweight=self.parameters['chi(k)']['kweight'].setvalue,
                     group=self.grouplist[index])
            except:
                print(str(index) + ' bad data, can not find background')
                self.grouplist[index].chir_mag = np.zeros(int(len(self.grouplist[index].r)))
                self.grouplist[index].chir_re = np.zeros(int(len(self.grouplist[index].r)))
                self.grouplist[index].chir_im = np.zeros(int(len(self.grouplist[index].r)))

    def plot_from_load(self, winobj): # for time series
        winobj.setslider()
        # self.curvedict['time series']['pointer'].setChecked(True)
        for key in self.curvedict:
            if self.checksdict[key].isChecked() and 'pointer' in self.curvedict[key]:
                self.curvedict[key]['pointer'].setChecked(True)

        self.E0 = []  # time series
        self.Jump = []
        self.espace = []
        self.kspace = []
        self.rspace = []  # time series
        kspace_length = [] # shocked! this naughty chi-k has irregular shape!
        if hasattr(self.grouplist[0], 'chi'):
            for index in range(self.entrydata.shape[0]):
                kspace_length.append(len(self.grouplist[index].chi))

        for index in range(self.entrydata.shape[0]):
            self.E0.append(self.grouplist[index].e0)
            self.Jump.append(self.grouplist[index].edge_step)

            if self.parameters['mu_norm-T']['flat norm'].choice == 'flat':
                self.espace.append(self.grouplist[index].flat)
            else:
                self.espace.append(self.grouplist[index].norm)

            if hasattr(self.grouplist[index], 'chi'):
                self.kspace.append(np.concatenate((self.grouplist[index].chi * np.square(self.grouplist[index].k),
                                                   np.zeros(max(kspace_length) - kspace_length[index]))))

            if hasattr(self.grouplist[index], 'chir_mag'):
                self.rspace.append(self.grouplist[index].chir_mag)

            self.prep_progress.setValue(int((index + 1) / self.entrytimesec.shape[0] * 100))

        if self.checksdict['E0-t'].isChecked():
            pw = winobj.gdockdict[self.method_name].tabdict['E0-t'].tabplot
            for index in reversed(range(len(pw.items))): # shocked!
                if pw.items[index].name() == 'E0-t': pw.removeItem(pw.items[index])

            pw.plot(range(self.entrytimesec.shape[0]), self.E0, symbol='o', symbolSize=10, symbolPen='r', name='E0-t')

        if self.checksdict['Jump-t'].isChecked():
            pw = winobj.gdockdict[self.method_name].tabdict['Jump-t'].tabplot
            for index in reversed(range(len(pw.items))):  # shocked!
                if pw.items[index].name() == 'Jump-t': pw.removeItem(pw.items[index])

            pw.plot(range(self.entrytimesec.shape[0]), self.Jump, symbol='o', symbolSize=10, symbolPen='r', name='Jump-t')
            
        if self.checksdict['mu_norm-T'].isChecked(): # shocked! this naughty chi-k has irregular shape!
            pw = winobj.gdockdict[self.method_name].tabdict['mu_norm-T'].tabplot
            for index in reversed(range(len(pw.items))):  # shocked!
                if isinstance(pw.items[index], pg.ImageItem): pw.removeItem(pw.items[index])

            im = np.array(self.espace) * 100
            if self.parameters['mu_norm-T']['all half'].choice == 'even':
                im[1::2,:] = im[0::2,:]
            elif self.parameters['mu_norm-T']['all half'].choice == 'odd':
                im[0::2, :] = im[1::2, :]

            self.plot_chi_2D(self.entrydata.shape[2], self.entrydata[0, 0, 0], self.entrydata[0, 0, -1], 100, pw, im, 'mu_norm-T')

        if self.checksdict['chi(k)-T'].isChecked() and hasattr(self.grouplist[index], 'chi'):
            pw = winobj.gdockdict[self.method_name].tabdict['chi(k)-T'].tabplot
            for index in reversed(range(len(pw.items))):  # shocked!
                if isinstance(pw.items[index], pg.ImageItem): pw.removeItem(pw.items[index])

            self.plot_chi_2D(max(kspace_length), 0, self.grouplist[kspace_length.index(max(kspace_length))].k[-1], 2, pw,
                             np.array(self.kspace), 'chi(k)-T')

        if self.checksdict['chi(r)-T'].isChecked() and hasattr(self.grouplist[index], 'chir_mag'):
            pw = winobj.gdockdict[self.method_name].tabdict['chi(r)-T'].tabplot
            for index in reversed(range(len(pw.items))):  # shocked!
                if isinstance(pw.items[index], pg.ImageItem): pw.removeItem(pw.items[index])

            self.plot_chi_2D(len(self.grouplist[0].r), 0, self.grouplist[0].r[-1], 2, pw,
                             np.array(self.rspace), 'chi(r)-T')

    def plot_chi_2D(self, r_range, r_min, r_max, step, pw, rspace_array, mode):
        xticklabels = []
        for tickvalue in np.arange(r_min, r_max,step):
            # here must use range to get integers, as in an image, integers works for position purpose.
            xticklabels.append((int(r_range / (r_max - r_min) * (tickvalue - r_min)), '{:.1f}'.format(tickvalue)))

        xticks = pw.getAxis('bottom')
        xticks.setTicks([xticklabels])

        # if hasattr(self, 'color_bar_ts'):
        #     pw.removeItem(self.img_ts)
        #     self.color_bar_ts.close()

        for item in pw.childItems():
            if type(item).__name__ == 'ViewBox':
                item.clear()

            if type(item).__name__ == 'ColorBarItem':
                item.close()

        img_ts = pg.ImageItem(image=np.transpose(rspace_array))
        pw.addItem(img_ts)
        color_map = pg.colormap.get('CET-R4')
        color_bar_ts = pg.ColorBarItem(values=(float(self.linewidgets[mode]['z min'].text()),
                                                    float(self.linewidgets[mode]['z max'].text())), colorMap=color_map)
        # color_bar_ts = pg.ColorBarItem(values=(rspace_array.max() * .9, rspace_array.max()), colorMap=color_map)
        color_bar_ts.setImageItem(img_ts, pw)
        if hasattr(self, 'y_range'):
            pw.setYRange(self.y_range[0], self.y_range[1])

    def filter_single_point(self, data_index, mode):
        sg_win = int(self.parameters[mode]['Savitzky-Golay window (time)'].setvalue)
        sg_order = int(self.parameters[mode]['Savitzky-Golay order (time)'].setvalue)
        sg_win_e = int(self.parameters[mode]['Savitzky-Golay window (energy)'].setvalue)
        sg_order_e = int(self.parameters[mode]['Savitzky-Golay order (energy)'].setvalue)

        if mode == 'normalizing':
            mu = lambda index: np.log(self.entrydata[index, 1, ::] / self.entrydata[index, 2, ::])

        if mode == 'normalized':
            mu = lambda index: self.grouplist[index].norm  # this one is dangerous as it uses smoothed/processed data

        if sg_win == 1 and sg_win_e == 1:
            return mu(data_index), self.grouplist[data_index].mu_error
            # is this a good way? error should be different at different stage???

        else:
            # time
            if sg_win > sg_order + 1:
                sg_data = []
                if data_index < (sg_win - 1) / 2:# padding with the first data
                    for index in np.arange(0, (sg_win - 1) / 2 - data_index, dtype=int): sg_data.append(mu(0))
                    for index in np.arange(0, (sg_win + 1) / 2 + data_index, dtype=int): sg_data.append(mu(index))

                elif data_index >  self.entrydata.shape[0] - (sg_win + 1) / 2: # padding with the last data
                    for index in np.arange(self.entrydata.shape[0], data_index + (sg_win + 1) / 2, dtype=int): sg_data.append(mu(-1))
                    for index in np.arange(data_index - (sg_win - 1) / 2, self.entrydata.shape[0], dtype=int): sg_data.append(mu(index))

                else:
                    for index in np.arange(data_index - (sg_win - 1) / 2, data_index + (sg_win + 1) / 2, dtype=int): sg_data.append(mu(index))

                sg_data = np.array(sg_data)
                mu_filtered = savgol_coeffs(sg_win, sg_order, use='dot').dot(sg_data)
                # estimate the uncertainty of the data according to "ERROR	ANALYSIS	2:	LEAST-SQUARES	FITTING"
                mu_filtered_error = 0
                for k in range(sg_win):
                    mu_filtered_error = mu_filtered_error + (sg_data[k] - savgol_coeffs(sg_win, sg_order, pos=k, use='dot').dot(sg_data))**2

                mu_filtered_error = np.sqrt(mu_filtered_error / (sg_win - sg_order) + self.data_error**2) # actually the latter can be neglected
            else:
                mu_filtered = np.log(self.entrydata[data_index, 1, :] / self.entrydata[data_index, 2, :])
                mu_filtered_error = self.data_error

            if 'filter by time' in self.curve_timelist[0][mode]:
                self.data_timelist[0][mode]['filter by time'].data = np.transpose([self.entrydata[data_index, 0, :], mu_filtered])

            # energy
            if sg_win_e > sg_order_e + 1:
                # mu_filtered = scipy.signal.savgol_filter(mu_filtered, sg_win_e, sg_order_e, mode='nearest')
                # construct a matrix based on mu_filtered
                sg_data = []
                for k in range(sg_win_e):
                    sg_data.append(np.concatenate((np.ones(k) * mu_filtered[0], mu_filtered, np.ones(sg_win_e - k) * mu_filtered[-1])))

                sg_data = np.array(sg_data)[:, (sg_win_e - 1) / 2 : -(sg_win_e + 1) / 2 - 1]
                mu_filtered = savgol_coeffs(sg_win_e, sg_order_e, use='dot').dot(sg_data)
                mu_filtered_error = 0
                for k in range(sg_win_e):
                    mu_filtered_error = mu_filtered_error + (sg_data[k] - savgol_coeffs(sg_win_e, sg_order_e, pos=k, use='dot').dot(sg_data))**2

                mu_filtered_error = np.sqrt(mu_filtered_error / (sg_win_e - sg_order_e) + self.data_error**2) # actually the latter can be neglected

            if 'filter by energy' in self.curve_timelist[0][mode]:
                self.data_timelist[0][mode]['filter by energy'].data = np.transpose([self.entrydata[data_index, 0, :], mu_filtered])

            return mu_filtered, mu_filtered_error

    def output_txt(self, data_single, index):
        try: data_start = np.where(data_single[:, 0] - self.energy_range[0] >= 0)[0][0]
        except: data_start = 0
        try: data_end = np.where(self.energy_range[1] - data_single[:, 0] >= 0)[0][-1]
        except: data_end = -1
        data_range = np.s_[data_start:data_end]
        tempdata = np.array([data_single[data_range, 0], data_single[data_range, 1], data_single[data_range, 2]])
        self.entrydata.append(tempdata)
        with open(os.path.join(self.exportdir, self.fileshort + '_{}_spectrum'.format(index + 1)), 'w') as f:
            np.savetxt(f, tempdata.T, header='Energy, I0, I1')

    def sel_by_energy_range(self, data_single):
        try:
            data_start = np.where(data_single - self.energy_range[0] >= 0)[0][0]
        except:
            data_start = 0
        try:
            data_end = np.where(self.energy_range[1] - data_single <= 0)[0][0]
        except:
            data_end = data_single.shape[0]
        # data_range = np.s_[data_start:data_end + 1] # critical!!!
        data_range = np.s_[data_start:data_end - 1] # do not slice more than the index
        return data_range

    def load_group(self,winobj):
        min_len = 1e5
        for index in range(len(self.entrydata)): # find the min array length, four is not enough
            if self.entrydata[index].shape[1] < min_len: min_len = self.entrydata[index].shape[1]

        for index in range(len(self.entrydata)): # make data same length
            self.entrydata[index] = self.entrydata[index][:,0:min_len]

        self.entrydata = np.array(self.entrydata) # now make a good array!

        tempfile_smoothed = os.path.join(self.exportdir, self.fileshort + '_Group_List_Smoothed')
        tempfile = os.path.join(self.exportdir, self.fileshort + '_Group_List')
        if os.path.isfile(tempfile_smoothed):
            with open(tempfile_smoothed, 'rb') as f:
                self.grouplist = pickle.load(f)
        elif os.path.isfile(tempfile):
            with open(tempfile, 'rb') as f:
                self.grouplist = pickle.load(f)

            # if no energy add energy list
            if not hasattr(self.grouplist[0], 'energy'):
                for k in range(len(self.grouplist)):
                    self.grouplist[k].energy = self.entrydata[k,0,::]
        else:
            self.plot_from_prep(winobj)

    def data_process(self, para_update): # for curves, embody data_timelist, if that curve exists
        # Energy, I0, I1 = self.read_data_time() # need to know self.slidervalue
        Energy = self.entrydata[self.index, 0, ::]
        I0 = self.entrydata[self.index, 1, ::]
        I1 = self.entrydata[self.index, 2, ::]
        mu = np.log(I0 / I1)
        mu_filtered = mu
        mu_filtered_error = self.grouplist[self.index].mu_error
        # this ensures that smoothed data get processed once you adjust those filtering parameters.
        # this function also sets data_timelist for filtered curves.
        # by abandon 'normalized' filter, there is less complexity.
        # when moving the slider, curves in other tabs change back to unfiltered state. this is actually good--fast!

        # if self.filtered: # detected by MainWin; but will the following work?
        if self.checksdict['normalizing'].isChecked():
            mu_filtered, mu_filtered_error = self.filter_single_point(self.index, 'normalizing')
            # if not (mu_filtered == mu).all():
            #     self.exafs_process_single(self.index, mu_filtered, mu_filtered_error)

        if self.checksdict['normalized'].isChecked():
            mu_filtered, mu_filtered_error = self.filter_single_point(self.index, 'normalized')
            # if not (mu_filtered_2 == mu_filtered).all():
            #     self.exafs_process_single(self.index, mu_filtered, mu_filtered_error)

        # for other parameters, the following both, which is better?
        if para_update: self.exafs_process_single(self.index, mu_filtered, mu_filtered_error)
        # self.exafs_process_single(self.index, mu_filtered, mu_filtered_error)

        self.dynamictitle = self.fileshort + '\n data' + str(self.index + 1) + '\t start:' + self.startime + '\t end:' + self.endtime

        # raw
        if 'I0' in self.curve_timelist[0]['raw']:
            self.data_timelist[0]['raw']['I0'].data = np.transpose([Energy, I0])
        if 'I1' in self.curve_timelist[0]['raw']:
            self.data_timelist[0]['raw']['I1'].data = np.transpose([Energy, I1])
        # norm
        if 'mu' in self.curve_timelist[0]['normalizing']: self.data_timelist[0]['normalizing']['mu'].data = np.transpose([Energy, mu])

        if 'pre-edge' in self.curve_timelist[0]['normalizing']:
            self.data_timelist[0]['normalizing']['pre-edge'].data = np.transpose([Energy, self.grouplist[self.index].pre_edge])

        if 'pre-edge points' in self.curve_timelist[0]['normalizing']:
            pre1 = (self.grouplist[self.index].e0 - Energy[0]) * (1 - self.parameters['normalizing']['pre-edge para 1'].setvalue)
            pre1_index = np.where(Energy - (self.grouplist[self.index].e0 - pre1) >= 0)[0][0]
            pre2_index = np.where(Energy - (self.grouplist[self.index].e0 -
                                            pre1 * self.parameters['normalizing']['pre-edge para 2'].setvalue) > 0)[0][0]
            # pre_index = np.s_[pre1_index, pre2_index]
            self.data_timelist[0]['normalizing']['pre-edge points'].data = \
                np.transpose([[Energy[pre1_index], Energy[pre2_index]],
                              [self.grouplist[self.index].pre_edge[pre1_index], self.grouplist[self.index].pre_edge[pre2_index]]])
            self.data_timelist[0]['normalizing']['pre-edge points'].pen = None
            self.data_timelist[0]['normalizing']['pre-edge points'].symbol = 'x'
            self.data_timelist[0]['normalizing']['pre-edge points'].symbolsize = 20

        if 'post-edge points' in self.curve_timelist[0]['normalizing']:
            post2 = (Energy[-1] - self.grouplist[self.index].e0) * (1 - self.parameters['normalizing']['post-edge para 2'].setvalue)
            post2_index = np.where(Energy - (self.grouplist[self.index].e0 + post2) >= 0)[0][0]
            post1_index = np.where(Energy - (self.grouplist[self.index].e0 +
                                             post2 * self.parameters['normalizing']['post-edge para 1'].setvalue) > 0)[0][0]
            # post_index = np.s_[post1_index, post2_index]
            self.data_timelist[0]['normalizing']['post-edge points'].data = \
                np.transpose([[Energy[post2_index], Energy[post1_index]],
                              [self.grouplist[self.index].post_edge[post2_index], self.grouplist[self.index].post_edge[post1_index]]])
            self.data_timelist[0]['normalizing']['post-edge points'].pen = None
            self.data_timelist[0]['normalizing']['post-edge points'].symbol = 'x'
            self.data_timelist[0]['normalizing']['post-edge points'].symbolsize = 20

        if 'post-edge' in self.curve_timelist[0]['normalizing']:
            self.data_timelist[0]['normalizing']['post-edge'].data = np.transpose([Energy, self.grouplist[self.index].post_edge])

        if 'normalized mu' in self.curve_timelist[0]['normalized']:
            if 'reference' in self.curve_timelist[0]['normalized']:
                self.data_timelist[0]['normalized']['normalized mu'].data = \
                    np.transpose([Energy, self.grouplist[self.index].norm - self.ref_mu]) # can also be .flat
            else:
                self.data_timelist[0]['normalized']['normalized mu'].data = \
                    np.transpose([Energy, self.grouplist[self.index].norm])  # can also be .flat
                self.ref_mu = self.grouplist[self.index].norm

        if 'post-edge bkg' in self.curve_timelist[0]['normalized']: self.data_timelist[0]['normalized']['post-edge bkg'].data = \
            np.transpose([Energy, (self.grouplist[self.index].bkg - self.grouplist[self.index].pre_edge) / self.grouplist[self.index].edge_step])

        # chi(k)
        if 'chi-k' in self.curve_timelist[0]['chi(k)']: self.data_timelist[0]['chi(k)']['chi-k'].data = \
            np.transpose([self.grouplist[self.index].k, np.square(self.grouplist[self.index].k) * self.grouplist[self.index].chi])
        if 'window' in self.curve_timelist[0]['chi(k)']:
            self.data_timelist[0]['chi(k)']['window'].data = np.transpose([self.grouplist[self.index].k, self.grouplist[self.index].kwin])

        # chi(r)
        if hasattr(self.grouplist[self.index], 'chir_mag'):
            if 'chi-r' in self.curve_timelist[0]['chi(r)']:
                self.data_timelist[0]['chi(r)']['chi-r'].data = np.transpose([self.grouplist[self.index].r, self.grouplist[self.index].chir_mag])
            if 'Re chi-r' in self.curve_timelist[0]['chi(r)']:
                self.data_timelist[0]['chi(r)']['Re chi-r'].data = np.transpose([self.grouplist[self.index].r, self.grouplist[self.index].chir_re])
            if 'Im chi-r' in self.curve_timelist[0]['chi(r)']:
                self.data_timelist[0]['chi(r)']['Im chi-r'].data = np.transpose([self.grouplist[self.index].r, self.grouplist[self.index].chir_im])

        # E0-t
        if 'pointer' in self.curve_timelist[0]['E0-t']:
            try: self.plot_pointer('E0-t', self.index, self.E0[self.index], '+', 30)
            except: print('Load time series first')

        # Jump-t
        if 'pointer' in self.curve_timelist[0]['Jump-t']:
            try: self.plot_pointer('Jump-t', self.index, self.Jump[self.index], '+', 30)
            except: print('Load time series first')

        # mu_norm-T
        if 'pointer' in self.curve_timelist[0]['mu_norm-T']:
            self.plot_pointer('mu_norm-T', 0, self.index, 't2', 15)

        # chi(k)-T
        if 'pointer' in self.curve_timelist[0]['chi(k)-T']:
            self.plot_pointer('chi(k)-T', 0, self.index, 't2', 15)

        # chi(r)-T
        if hasattr(self.grouplist[self.index], 'chir_mag'):
            if 'pointer' in self.curve_timelist[0]['chi(r)-T']:
                self.plot_pointer('chi(r)-T', 0, self.index, 't2', 15)

        # LCA t
        if 'pointer' in self.curve_timelist[0]['LCA(internal)-t']:
            self.plot_pointer('LCA(internal)-t', self.index, 1, 't', 15)

        # LCA single
        if hasattr(self,'lca_result_group') and self.checksdict['LCA(internal) single'].isChecked():
            y_start = int(self.linewidgets['mu_norm-T']['select start (y)'].text())
            y_end = int(self.linewidgets['mu_norm-T']['select end (y)'].text())

            if y_start <= self.index < y_end:
                if 'mu_norm' in self.curve_timelist[0]['LCA(internal) single']:
                    self.data_timelist[0]['LCA(internal) single']['mu_norm'].data = \
                        np.transpose([Energy, self.grouplist[self.index].norm])  # can also be .flat

                lca = self.lca_result_group[self.index - y_start]
                c_index = 1
                for c_name in lca.weights:
                    if ('component %i' % c_index) in self.curve_timelist[0]['LCA(internal) single']:
                        self.data_timelist[0]['LCA(internal) single']['component %i' % c_index].data = \
                            np.transpose([lca.xdata, lca.ycomps[c_name]]) # name of Group
                        c_index += 1

                if 'computed' in self.curve_timelist[0]['LCA(internal) single']:
                    self.data_timelist[0]['LCA(internal) single']['computed'].data = \
                        np.transpose([lca.xdata, lca.yfit])

                if 'errors' in self.curve_timelist[0]['LCA(internal) single']:
                    self.data_timelist[0]['LCA(internal) single']['errors'].data = \
                        np.transpose([lca.xdata, lca.yfit - lca.ydata])

class XAS_INFORM_1(XAS):
    def __init__(self, path_name_widget):
        super(XAS_INFORM_1, self).__init__(path_name_widget)

    # rewrite for different mode of collection; feed read_data_index the whole data in memory: entrydata (xas, pl, refl),
    # and/or a file in \process (xrd)
    def time_range(self, winobj):
        for key in winobj.path_name_widget:  # to distinguish xas_1, xas_2
            if self.fileshort == winobj.path_name_widget[key]['raw file'].text():
                if 'energy range (eV)' in winobj.path_name_widget[key] and \
                        self.range == winobj.path_name_widget[key]['energy range (eV)'].text():
                    self.method_name = key

        # read in time
        if self.entrytimesec == []:  # start, end time in seconds
            tempfile = os.path.join(self.exportdir, self.fileshort + '_time_in_seconds')
            if os.path.isfile(tempfile):
                with open(tempfile, 'r') as f:
                    self.entrytimesec = np.loadtxt(f)
            else:
                self.file = h5py.File(self.filename, 'r')
                self.filekeys = list(self.file.keys())

                # for SDC data in \balder\20220720\
                del self.filekeys[self.filekeys.index('ColumnInfo')]
                del self.filekeys[self.filekeys.index('spectrum0')]
                del self.filekeys[self.filekeys.index('timestamp0')]
                for index in range(1, int(len(self.filekeys) / 2 + 1)):
                    timesecond = time.mktime(
                        datetime.strptime(self.file['timestamp' + str(index)][()].decode(), '%c').timetuple())
                    self.entrytimesec.append([timesecond, timesecond + 1])  # one second collection time
                    del self.filekeys[self.filekeys.index('timestamp' + str(index))]

                self.entrytimesec = np.array(self.entrytimesec)
                # self.entrytimesec = self.entrytimesec[self.entrytimesec[:,0].argsort()]

        # read in data
        if self.entrydata == []:  # Energy, I0, I1
            if os.path.isdir(self.exportdir):  # can also do deep comparison from here
                for tempname in sorted(glob.glob(self.exportdir + '*_spectrum'),
                                       key=os.path.getmtime):  # in ascending order
                    with open(tempname, 'r') as f:
                        self.entrydata.append(np.loadtxt(f).transpose())

            else:
                os.mkdir(self.exportdir)
                with open(tempfile, 'w') as f:
                    np.savetxt(f, self.entrytimesec)

                for index in range(self.entrytimesec.shape[0]):
                    # a hidden average feature
                    # if index < (self.average + 1) / 2 or index > self.entrytimesec.shape[0] - (self.average - 1) / 2:
                    #     data = self.file['spectrum' + str(index)] # specific !
                    #     data_single = np.zeros((data.shape[0], data.shape[1]), dtype='float32')
                    #     data.read_direct(data_single)
                    # else:
                    #     data = self.file['spectrum' + str(index)]
                    #     data_single = np.zeros((data.shape[0], data.shape[1]), dtype='float32')
                    #     for k in range(int(index - (self.average - 1) / 2), int(index + (self.average - 1) / 2 + 1)):
                    #         data = self.file['spectrum' + str(k)]
                    #         data_neighbour = np.zeros((data.shape[0], data.shape[1]), dtype='float32')
                    #         data.read_direct(data_neighbour)
                    #         data_single = data_single + data_neighbour
                    #
                    #     data_single = data_single / self.average

                    # for SDC data in \balder\20220720\
                    data = self.file['spectrum' + str(index + 1)]
                    data_single = np.zeros((data.shape[0], data.shape[1]), dtype='float32')
                    data.read_direct(data_single)
                    self.output_txt(data_single, index)

            self.load_group(winobj)

        return [self.entrytimesec[0, 0], self.entrytimesec[-1, 1]]


class XAS_INFORM_2(XAS):
    def __init__(self, path_name_widget):
        super(XAS_INFORM_2, self).__init__(path_name_widget)

    # rewrite for different mode of collection; feed read_data_index the whole data in memory: entrydata (xas, pl, refl),
    # and/or a file in \process (xrd)
    def time_range(self, winobj):
        for key in winobj.path_name_widget:  # to distinguish xas_1, xas_2
            if self.fileshort == winobj.path_name_widget[key]['raw file'].text():
                if 'energy range (eV)' in winobj.path_name_widget[key] and \
                        self.range == winobj.path_name_widget[key]['energy range (eV)'].text():
                    self.method_name = key

        # read in time
        if self.entrytimesec == []:  # start, end time in seconds
            tempfile = os.path.join(self.directory, 'process', self.fileshort + '_time_in_seconds')
            if os.path.isfile(tempfile):
                with open(tempfile, 'r') as f:
                    self.entrytimesec = np.loadtxt(f)
            else:
                try:
                    self.file = h5py.File(self.filename, 'r')
                except:
                    print('double check your file name')

                self.filekeys = list(self.file.keys())
                timesecond_all = []
                if 'time' in self.filekeys:
                    for timetext in range(len(self.file['time'])):
                        timesecond = time.mktime(
                            datetime.strptime(self.file['time'][timetext].decode(), '%Y-%m-%d %H:%M:%S.%f').timetuple())
                        timesecond_all.append(timesecond)

                    if len(timesecond_all) == 1: print('you do not need to process this xas data, go to xrd data directly!')
                    # make for half time spectrum, as this is back and forth measurement
                    timesecond_all.append(2 * timesecond_all[-1] - timesecond_all[-2])
                    timesecond_array = np.array(timesecond_all)
                    # watch out the following two lines
                    # timesecond_start = np.sort(np.concatenate((timesecond_array[0:-2], (timesecond_array[0:-2] + timesecond_array[1:-1]) / 2)))
                    # timesecond_end = np.sort(np.concatenate((timesecond_array[1:-1], (timesecond_array[0:-2] + timesecond_array[1:-1]) / 2)))
                    timesecond_start = np.sort(np.concatenate((timesecond_array[0:-1], (timesecond_array[0:-1] + timesecond_array[1::]) / 2)))
                    timesecond_end = np.sort(np.concatenate((timesecond_array[1::], (timesecond_array[0:-1] + timesecond_array[1::]) / 2)))
                    self.entrytimesec = np.array([timesecond_start,timesecond_end]).T # so critical!!!

        # read in data
        if self.entrydata == []:  # Energy, I0, I1
            self.exportfile = os.path.join(self.exportdir, self.fileshort + '_spectrum_all.h5')
            if os.path.isdir(self.exportdir) and os.path.isfile(self.exportfile):
                f = h5py.File(self.exportfile, 'r')
                if 'y min' in list(f.keys()):
                    self.y_range = [f['y min'][()], f['y max'][()]]

                for spectrum in list(f.keys()):
                    if spectrum[0:8] == 'spectrum':
                        data = np.zeros((f[spectrum].shape[0], f[spectrum].shape[1]), dtype = 'float32')
                        f[spectrum].read_direct(data)
                        self.entrydata.append(data.T)

                f.close()
                # for tempname in sorted(glob.glob(self.exportdir + '*_spectrum'),
                #                        key=os.path.getmtime):  # in ascending order
                #     with open(tempname, 'r') as f:
                #         self.entrydata.append(np.loadtxt(f).transpose())

            else:
                if not os.path.isdir(self.exportdir):
                    os.mkdir(self.exportdir)

                if not os.path.isfile(tempfile):
                    with open(tempfile, 'w') as f:
                        np.savetxt(f, self.entrytimesec)

                if not hasattr(self, 'file'):
                    self.file = h5py.File(self.filename, 'r')

                self.filekeys = list(self.file.keys()) # I0a, I0b, I1, It, energy
                data_all = {}
                for key in self.filekeys:
                    if key != 'time':
                        data = self.file[key]
                        data_single = np.zeros((data.shape[0], data.shape[1]), dtype='float32')
                        data.read_direct(data_single)                        
                        data_all[key] = np.array(data_single)

                if data:
                    f = h5py.File(self.exportfile, 'w')
                    data4xrd = []

                    # if it is only up energy scan
                    temp = data_all['energy'][0,:]
                    only_up = False
                    # if it is not around the center, so set a loose condition: above 3 / 4
                    if np.where(temp == temp.max())[0][0] >= temp.shape[0] * 3 / 4:
                        only_up = True
                        print(only_up)
                    # added above
                    print_data = True
                    for index in range(data.shape[0]): # output format has to be Energy, I0, I1

                        if only_up: # only up scan
                            data_range = self.sel_by_energy_range(data_all['energy'][index, :])
                            data_temp = np.array([
                                data_all['energy'][index, data_range],  # E0
                                data_all['I0a'][index, data_range] + data_all['I0b'][index, data_range], # I0
                                data_all['I1a'][index, data_range] + data_all['I1b'][index, data_range]  # I1
                                ])
                            self.entrydata.append(data_temp)
                            self.entrydata.append(data_temp) # this is to trick the self.entrytimesec
                            if print_data:
                                print(data_range)
                                # print(data_temp)
                                print_data = False

                            f.create_dataset('spectrum_{:04d}'.format(2 * index), data=data_temp.T)
                            f.create_dataset('spectrum_{:04d}'.format(2 * index + 1), data=data_temp.T)
                            # this is to trick the self.entrytimesec
                            half_index = 1
                            # added above

                        else: # up and down scan
                            if data.shape[1] % 2 == 0:
                                half_index = int(data.shape[1] / 2 - 1)
                                # print(half_index)
                                # self.output_txt(np.array([data_all['energy'][index,0:half_index + 1], # E0
                                #                           data_all['I0a'][index,0:half_index + 1] + data_all['I0b'][index,0:half_index + 1], # I0
                                #                           data_all['I1'][index,0:half_index + 1]]).T, 2 * index) # I1
                                back_sel = np.s_[data.shape[1] - 1:half_index:-1]
                                # self.output_txt(np.array([data_all['energy'][index, back_sel], # E0
                                #                           data_all['I0a'][index,back_sel] + data_all['I0b'][index,back_sel], # I0
                                #                           data_all['I1'][index,back_sel]]).T, 2 * index + 1) # I1

                            else: # most of data is odd number of energy points
                                half_index = int((data.shape[1] + 1) / 2 - 1) # middle point by index
                                # self.output_txt(np.array([data_all['energy'][index, 1:half_index + 1],  # E0
                                #                           data_all['I0a'][index, 1:half_index + 1] + data_all['I0b'][index, 1:half_index + 1],  # I0
                                #                           data_all['I1'][index, 1:half_index + 1]]).T, 2 * index)  # I1
                                back_sel = np.s_[data.shape[1] - 1:half_index:-1] # not include the middle point, because the it is xrd!
                                # self.output_txt(np.array([data_all['energy'][index, back_sel],  # E0
                                #                           data_all['I0a'][index, back_sel] + data_all['I0b'][index, back_sel], # I0
                                #                           data_all['I1'][index, back_sel]]).T, 2 * index + 1)  # I1

                            data_range = self.sel_by_energy_range(data_all['energy'][index, 0:half_index + 1])

                            try:
                                data_half_1 = np.array([data_all['energy'][index, data_range],  # E0
                                                      data_all['I0a'][index, data_range] + data_all['I0b'][index, data_range],  # I0
                                                      data_all['I1'][index, data_range]])  # I1
                            except:
                                data_half_1 = np.array([data_all['energy'][index, data_range],  # E0
                                                        data_all['I0a'][index, data_range] + data_all['I0b'][index, data_range], # I0
                                                        data_all['I1a'][index, data_range] + data_all['I1b'][index, data_range]])  # I1

                            self.entrydata.append(data_half_1)
                            f.create_dataset('spectrum_{:04d}'.format(index * 2), data=data_half_1.T)

                            data_range = self.sel_by_energy_range(data_all['energy'][index, back_sel])

                            try:
                                data_half_2 = np.array([data_all['energy'][index, back_sel][data_range],  # E0
                                                      data_all['I0a'][index, back_sel][data_range] +
                                                      data_all['I0b'][index, back_sel][data_range],  # I0
                                                      data_all['I1'][index, back_sel][data_range]])  # I1
                            except:
                                data_half_2 = np.array([data_all['energy'][index, back_sel][data_range],  # E0
                                                        data_all['I0a'][index, back_sel][data_range] +
                                                        data_all['I0b'][index, back_sel][data_range],  # I0
                                                        data_all['I1a'][index, back_sel][data_range] +
                                                        data_all['I1b'][index, back_sel][data_range]])  # I1

                            self.entrydata.append(data_half_2)
                            f.create_dataset('spectrum_{:04d}'.format(index * 2 + 1), data=data_half_2.T)

                        # extract data for xrd normalization
                        try:
                            # the first xrd
                            data4xrd.append(np.array([data_all['energy'][index, 0],  # E0
                                                       data_all['I0a'][index, 0] + data_all['I0b'][index, 0],  # I0
                                                       data_all['I1'][index, 0]]).T  # I1
                                             )
                            # the second xrd
                            data4xrd.append(np.array([data_all['energy'][index, half_index],  # E0
                                                      data_all['I0a'][index, half_index] + data_all['I0b'][index, half_index],  # I0
                                                      data_all['I1'][index, half_index]]).T  # I1
                                            )
                        except:
                            # the first xrd
                            data4xrd.append(np.array([data_all['energy'][index, 0],  # E0
                                                      data_all['I0a'][index, 0] + data_all['I0b'][index, 0],  # I0
                                                      data_all['I1a'][index, 0] + data_all['I1b'][index, 0]]).T  # I1
                                            )
                            # the second xrd
                            data4xrd.append(np.array([data_all['energy'][index, half_index],  # E0
                                                      data_all['I0a'][index, half_index] + data_all['I0b'][index, half_index],  # I0
                                                      data_all['I1a'][index, half_index] + data_all['I1b'][index, 0]]).T  # I1
                                            )

                    f.close()
                else:
                    print('data not read in')

                # output data for xrd
                with open(os.path.join(self.directory, 'process', self.fileshort + '_xas4xrd'), 'w') as f:
                    np.savetxt(f, np.array(data4xrd))

            self.load_group(winobj)

        return [self.entrytimesec[0, 0], self.entrytimesec[-1, 1]]

class XAS_BATTERY_1(XAS):
    def __init__(self, path_name_widget):
        super(XAS_BATTERY_1, self).__init__(path_name_widget)

    def time_range(self, winobj):
        for key in winobj.path_name_widget:  # to distinguish xas_1, xas_2
            if self.fileshort == winobj.path_name_widget[key]['raw file'].text():
                if 'energy range (eV)' in winobj.path_name_widget[key] and \
                        self.range == winobj.path_name_widget[key]['energy range (eV)'].text():
                    self.method_name = key

        # read in time
        if self.entrytimesec == []:  # start, end time in seconds
            tempfile = os.path.join(self.exportdir, self.fileshort + '_time_in_seconds')
            if os.path.isfile(tempfile):
                with open(tempfile, 'r') as f:
                    self.entrytimesec = np.loadtxt(f)
            else:
                self.file = h5py.File(self.filename, 'r')
                self.filekeys = list(self.file.keys())

                # for battery data in \balder\20210738
                for key in self.filekeys:  # this prevent some entry without an end_time!
                    if 'start_time' in self.file[key] and 'end_time' in self.file[key] \
                            and 'albaem-02_ch1' in self.file[key + '/measurement']:
                        self.entrytimesec.append([
                            time.mktime(datetime.strptime(self.file[key + '/start_time'][()].decode(),
                                                          '%Y-%m-%dT%H:%M:%S.%f').timetuple()),
                            time.mktime(datetime.strptime(self.file[key + '/end_time'][()].decode(),
                                                          '%Y-%m-%dT%H:%M:%S.%f').timetuple())])
                    else:  # error proof, and only chance of effective error proof: filter the bad entries when setslider
                        del self.filekeys[self.filekeys.index(key)]

                self.entrytimesec = np.array(self.entrytimesec)

        # read in data
        if self.entrydata == []:  # Energy, I0, I1
            if os.path.isdir(self.exportdir):  # can also do deep comparison from here
                for tempname in sorted(glob.glob(self.exportdir + '*_spectrum'),
                                       key=os.path.getmtime):  # in ascending order
                    with open(tempname, 'r') as f:
                        self.entrydata.append(np.loadtxt(f).transpose())

            else:
                os.mkdir(self.exportdir)
                with open(tempfile, 'w') as f:
                    np.savetxt(f, self.entrytimesec)

                # here filekeys are cleaned in previous step
                # also be careful with the energy range defined in energy_range
                # also to prevent unequal number of data! really weird!
                fixed_length = len(list(self.file[self.filekeys[0] + '/measurement/mono1_energy']))
                for index in range(self.entrytimesec.shape[0]):
                    # for battery data in \balder\20210738
                    energy = np.array(list(self.file[self.filekeys[index] + '/measurement/mono1_energy']))
                    i0_1 = np.array(list(self.file[self.filekeys[index] + '/measurement/albaem-02_ch1']))
                    i0_2 = np.array(list(self.file[self.filekeys[index] + '/measurement/albaem-02_ch2']))
                    i1_1 = np.array(list(self.file[self.filekeys[index] + '/measurement/albaem-02_ch3']))
                    i1_2 = np.array(list(self.file[self.filekeys[index] + '/measurement/albaem-02_ch4']))
                    data_single = np.array([energy, i0_1 + i0_2, i1_1 + i1_2]).transpose()

                    if energy.shape[0] < fixed_length:  # I can't imagine how it could larger than fixed_length
                        l = fixed_length - energy.shape[0]
                        make_up = np.array([[data_single[-1,0]] * l, [data_single[-1,1]] * l, [data_single[-1,2]] * l])
                        data_single = np.concatenate((data_single,make_up.transpose()),axis=0)

                    self.output_txt(data_single, index)

            self.load_group(winobj)

        return [self.entrytimesec[0, 0], self.entrytimesec[-1, 1]]

class XRD(Methods_Base):
    def __init__(self, path_name_widget):
        super(XRD, self).__init__()
        self.availablemethods = ['raw', 'integrated', 'time series', 'single peak int.', 'refinement single',
                                 'centroid-T', 'integrated area-T', 'segregation degree-T']
        self.availablecurves['raw'] = ['show image']
        self.availablecurves['integrated'] = ['original', 'normalized to 1', # 'normalized to I0 and <font> &mu; </font>d',
                                              'truncated', 'smoothed', 'find peaks']
        self.availablecurves['time series'] = ['pointer']
        self.availablecurves['single peak int.'] = ['pointer', 'peak position', 'estimated phase frac.',
                                                    'center of mass', 'phase frac. by c.o.m.',
                                                    'integration area', 'norm. int. area']
        self.availablecurves['refinement single'] = ['observed', 'calculated', 'difference']
        self.availablecurves['centroid-T'] = ['pointer']
        self.availablecurves['integrated area-T'] = ['pointer']
        self.availablecurves['segregation degree-T'] = ['pointer']
        self.directory = path_name_widget['directory'].text()
        self.fileshort = path_name_widget['raw file'].text()
        self.intfile_appendix = path_name_widget['integration file appendix'].text()
        self.ponifile = os.path.join(self.directory, 'process', path_name_widget['PONI file'].text())
        if os.path.isfile(self.ponifile):
            with open(self.ponifile, 'r') as f:
                self.wavelength = 1e10 * float(f.readlines()[-1].splitlines()[0].partition(' ')[2]) # now in Angstrom

        else: print('wrong poni file, re-load xrd')

        self.filename = os.path.join(self.directory, 'raw', self.fileshort + '.h5')
        # for refinement, to activate the following part,
        # also activate the last part in self.parameters, one part in path_name_dict in MainWindow,
        # and the last part in self.read_data_time and in self.data_process
        # self.refinedir = os.path.join(path_name_widget['refine dir'].text(), path_name_widget['refine subdir'].text())
        # self.refinedata = glob.glob(os.path.join(self.refinedir, path_name_widget['data files'].text() + '.csv'))
        # self.refinephase = glob.glob(os.path.join(self.refinedir, path_name_widget['data files'].text() + '_calc'))
        # refine_file = os.path.join(self.refinedir, path_name_widget['refinement file'].text())
        # if os.path.isfile(refine_file): self.refinegpx = G2sc.G2Project(refine_file)
        # if self.refinephase:
        #     pf = np.loadtxt(self.refinephase[0])
        #     for ph in range(pf.shape[0]): self.availablecurves['refinement single'].append('phase' + str(ph))

        # self.colormax_set = False
        self.axislabel = {'raw':{'bottom':'',
                                 'left':''},
                          'integrated': {'bottom': '<font> q / &#8491; </font> <sup> -1 </sup>,'
                                                   '<font> 2 &#952; / </font> <sup> o </sup>, or'
                                                   '<font> d / &#8491; </font>',
                                        'left': 'Intensity'},
                          'time series': {'bottom': '<font> q / &#8491; </font> <sup> -1 </sup>',
                                        'left': 'Data number'},
                          'single peak int.':{'bottom': 'Data number',
                                              'left': 'a. u. or <font> &#8491; </font>'},
                          'refinement single': {'bottom': '<font> q / &#8491; </font> <sup> -1 </sup>,'
                                                   '<font> 2 &#952; / </font> <sup> o </sup>, or'
                                                   '<font> d / &#8491; </font>',
                                                'left': 'Intensity'},
                          'centroid-T':{'bottom':'data number',
                                        'left':'<font> d / &#8491; </font>'},
                          'integrated area-T':{'bottom':'data number',
                                               'left': 'Fcalc / a. u.'},
                          'segregation degree-T':{'bottom':'data number',
                                                  'left': 'degree of segregation <font> / &#8491; </font>'}}

        self.ini_data_curve_color()

        # unique to xrd
        # note that these parameter boundaries can be changed
        self.bravaisNames = ['Cubic-F', 'Cubic-I', 'Cubic-P', 'Trigonal-R', 'Trigonal/Hexagonal-P', # 5
                             'Tetragonal-I', 'Tetragonal-P', 'Orthorhombic-F', 'Orthorhombic-I', 'Orthorhombic-A', # 5
                             'Orthorhombic-B', 'Orthorhombic-C',
                             'Orthorhombic-P', 'Monoclinic-I', 'Monoclinic-A', 'Monoclinic-C', 'Monoclinic-P',
                             'Triclinic']

        self.parameters = {'integrated': {'scale': Paraclass(strings=('log10', ['log10', 'sqrt', 'linear'])),
                                          'x axis': Paraclass(strings=('q',['q','2th','d'])),
                                          'clip head': Paraclass(values=(0,0,1000,1)),
                                          'clip tail': Paraclass(values=(1, 1, 1000, 1)),
                                          'Savitzky-Golay window': Paraclass(values=(1,1,101,2)),
                                          'Savitzky-Golay order': Paraclass(values=(1,1,5,2)),
                                          'peak prominence min': Paraclass(values=(0,0,100,.1)),
                                          'peak prominence max': Paraclass(values=(1000,100,10000,10)),
                                          'peak width min': Paraclass(values=(0,0,20,1)),
                                          'peak width max': Paraclass(values=(20,10,1000,1)),
                                          'window length': Paraclass(values=(101,10,1000,2))},
                           # these parameters are special: they do not call data_process.
                           # change them also in ShowData.update_parameters!
                           'time series': {'scale': Paraclass(strings=('log10', ['log10', 'sqrt', 'linear'])),
                                           'normalization': Paraclass(strings=('not normalized',
                                                                               ['not normalized',
                                                                                'normalized to I0 and \u03BC d'])),
                                           'gap y tol.': Paraclass(values=(1,1,20,1)),
                                           'gap x tol.': Paraclass(values=(1,1,100,1)),
                                           'min time span': Paraclass(values=(5,1,50,1)),
                                           '1st deriv control': Paraclass(values=(5,1,100,.1)), # x is pixel number for a peak?
                                           'max diff time span': Paraclass(values=(5,0,50,1)),
                                           'max diff start time': Paraclass(values=(3,0,50,1)),
                                           'symbol size': Paraclass(values=(1,1,20,1)),
                                           'single peak int. width': Paraclass(values=(2,.1,20,.05)),
                                           'phases': Paraclass(strings=('choose a phase',
                                                                        ['choose a phase','phase1','phase2','phase3',
                                                                         'phase4','phase5','phase6','phase7','phase8',
                                                                         'phase9','phase10','phase11','phase12']))},
                           # 'refinement single': {'data number': Paraclass(values=(0,0,len(self.refinedata) - 1,1)),
                           #                        'scale': Paraclass(strings=('log10', ['log10', 'sqrt', 'linear'])),
                           #                        'x axis': Paraclass(strings=('q',['q','2th','d']))}
                           }
        self.actions = {'integrated':{"interpolate along q": self.interpolate_data,
                                      "find peaks for all (Ctrl+F)": self.find_peak_all,
                                      'clear peaks (Ctrl+P)': self.clear_reflections},
                       'time series':{'update y,z range (Ctrl+0)': self.update_range,
                                      "clear rainbow map (Ctrl+E)": self.show_clear_ts,
                                      "select start (Ctrl+R)": self.range_select,
                                      "catalog peaks (Ctrl+T)": self.catalog_peaks,
                                      "assign phases (Ctrl+A)": self.assign_phases,
                                      "clear peaks (Ctrl+P)": self.show_clear_peaks,
                                      "index phases (Ctrl+I)": self.index_phases,
                                      'add ref. phase (Ctrl+2)': self.add_ref_phase,
                                      'save peaks/phases (Ctrl+H)': self.save_peaks_phases,
                                      'load peaks/phases (Ctrl+J)': self.load_peaks_phases,
                                      'pick single peak for int. (Ctrl+Q)': self.single_peak_int},
                        'single peak int.':{'update (Ctrl+U)': self.single_peak_update}} # button name and function name

        self.linedit = {'time series': {'y min':'0',
                                        'y max': '100',
                                        'select start': '100',
                                        'select end': '200',
                                        'exclude from':'0',
                                        'exclude to':'0',
                                        'assign proximity x':'5',
                                        'assign proximity y':'2',
                                        'ref. phase 1':'TOPAS_local_gof_vol_unindex_1_0',
                                        'ref. phase 2':''},
                        'raw':{'color max':'2.5'},
                        'single peak int.':{'phase 1 peak pos. (<font> &#8491; </font>)':'5.9119',
                                            'phase 2 peak pos. (<font> &#8491; </font>)':'6.3621'}}

        self.pre_ts_btn_text = 'Do Batch Integration(Ctrl+D)'
        if hasattr(self, 'wavelength'):
            energy = ev2nm / self.wavelength * 10 / 1000
        else:
            print('reload xrd!')

        if ('20221562' in self.directory.split('\\')) or ('20221478' in self.directory.split('\\')):
            self.exportdir = os.path.join(self.directory, 'process', self.fileshort + '_{}keV'.format(int(energy)))
            self.exportfile = os.path.join(self.exportdir,self.fileshort + '_{}keV'.format(int(energy)) + self.intfile_appendix)
        else:
            self.exportdir = os.path.join(self.directory, 'process', self.fileshort + '_{:.1f}keV'.format(energy))
            self.exportfile = os.path.join(self.exportdir, self.fileshort + '_{:.1f}keV'.format(energy) + self.intfile_appendix)

        self.entrytimesec = []
        self.cells_sort = {}
        self.raw_tot = []

        # glitches in q, this should be dependent on experiment!
        # self.glitches = [[1.94804, 1.95192],
        #                  [2.41443, 2.41831]]  # based on reading on plot
        # ref
        self.ref_phase = {'FAPbI3': {'color': 'y',  # no black color, light color is better
                                     'Bravais': 2,
                                     'cell': np.array([6.3621, 6.3621, 6.3621, 90, 90, 90])}, #6.3621 6.3621 6.3621 90 90 90
                           'MAPbBr3': {'color': 'r',  # drawing color
                                       'Bravais': 2,  # choose from self.bravaisNames
                                       'cell': np.array([5.9119, 5.9119, 5.9119, 90, 90, 90])},
                          }
        # ref 1:
        # The phase diagram of a mixed halide (Br, I) hybrid perovskite obtained by synchrotron X-ray diffraction†
        # Frederike Lehmann, *ab Alexandra Franz,a Daniel M. Tobbens, ¨ a Sergej Levcenco,a Thomas Unold,a Andreas Taubertb and Susan Schorrac
        # ref 2:
        # Cubic Perovskite Structure of Black Formamidinium Lead Iodide, α‑[HC(NH2)2]PbI3, at 298 K
        # Mark T. Weller,* Oliver J. Weber, Jarvist M. Frost, and Aron Walsh
        for phase in self.ref_phase:
            self.availablecurves['time series'].append(phase)

        if os.path.isfile(self.exportfile):
            try:
                self.read_intg() # read in data at the beginning
            except:
                print('redo the integration!')

        # assign to peak X by right click
        self.peak_ass_from = []
        self.peak_ass_to = []
        # to add/do:
        # delete peaks function by right click
        # multi select peaks to assign from

        # assign to phase X by right click
        self.phase_ass_from = []
        self.phase_ass_to = []
        self.single_peak = []

    def single_peak_int(self, winobj):
        widgetname_1 = 'pick single peak for int. (Ctrl+Q)'
        widgetname_2 = 'single peak int. (Ctrl+Q)'
        tempwidget = self.actwidgets['time series'][widgetname_1]
        pw = winobj.gdockdict[self.method_name].tabdict['time series'].tabplot
        pw.scene().sigMouseClicked.connect(lambda evt, p=pw: self.single_peak_select(evt, p))
        if tempwidget.text() == widgetname_1:
            tempwidget.setText(widgetname_2)
            tempwidget.setShortcut('Ctrl+Q')
        else:
            tempwidget.setText(widgetname_1)
            tempwidget.setShortcut('Ctrl+Q')
            pw.scene().sigMouseClicked.disconnect()
            if not self.checksdict['single peak int.'].isChecked():
                self.checksdict['single peak int.'].setChecked(True)

            if hasattr(self, 'peaks_all') and self.single_peak != []:
                peaks = np.array(self.peaks_catalog_select[self.single_peak]) # y, x, ?, entry, k
                width_factor = self.parameters['time series']['single peak int. width'].setvalue
                # q_start = int(self.parameters['integrated']['clip head'].setvalue)
                single_peak_info = []
                for index in range(peaks.shape[0]):# double check
                    single_peak_info.append([])
                    peak_start = int(peaks[index,1] - width_factor * abs(peaks[index,1] -
                                         self.peaks_properties_all[peaks[index,3]]['left_ips'][peaks[index,4]]))
                    peak_end = int(peaks[index,1] + width_factor * abs(peaks[index,1] -
                                         self.peaks_properties_all[peaks[index,3]]['right_ips'][peaks[index,4]]))

                    d = 2 * np.pi / self.intqaxis[peaks[index,1]]
                    d1 = float(self.linewidgets['single peak int.']['phase 1 peak pos. (<font> &#8491; </font>)'].text())
                    d2 = float(self.linewidgets['single peak int.']['phase 2 peak pos. (<font> &#8491; </font>)'].text())
                    fraction = abs(d - d2) / abs(d2 - d1)
                    sg_win = int(self.parameters['integrated']['Savitzky-Golay window'].setvalue)
                    sg_order = int(self.parameters['integrated']['Savitzky-Golay order'].setvalue)
                    q_start = int(self.parameters['integrated']['clip head'].setvalue)
                    q_ending = int(self.parameters['integrated']['clip tail'].setvalue)
                    smoothed = scipy.signal.savgol_filter(self.intdata_ts[peaks[index,0]][q_start: -q_ending], sg_win, sg_order)
                    int_sum = sum(smoothed[peak_start:peak_end])
                    # int_sum = sum(self.intdata_ts[peaks[index, 0]][peak_start:peak_end])
                    bkg = (smoothed[peak_start] + smoothed[peak_end]) * (peak_end - peak_start) / 2
                    # center of mass
                    k = (smoothed[peak_end - 1] - smoothed[peak_start]) / (peak_end - peak_start)
                    bkg_array = np.arange(0, peak_end - peak_start) * k + smoothed[peak_start]
                    center = (int_sum - bkg_array) * 2 * np.pi / self.intqaxis[peak_start + q_start:peak_end + q_start]
                    center = center.sum() / (int_sum - bkg_array).sum()
                    center_frac = abs(center - d2) / abs(d2 - d1)
                    # degree of segregation
                    dos = (2 * np.pi / self.intqaxis[peak_start + q_start:peak_end + q_start] - center) ** 2
                    dos = np.sqrt(((int_sum - bkg_array) * dos).sum() / (int_sum - bkg_array).sum() * (1 - 1 / len(dos)))
                    single_peak_info[-1] = [peaks[index, 0], peak_start, peak_end, d, fraction,
                                            center, center_frac, int_sum - bkg, 0, dos, 0]

                self.single_peak_info = np.array(single_peak_info)
                self.single_peak_info[:,-3] = self.single_peak_info[:,-4] / max(self.single_peak_info[:,-4])
                self.single_peak_info[:, -1] = self.single_peak_info[:, -2] / max(self.single_peak_info[:, -2])
                # y, d, fraction, int., norm int.
                # output: peak y, start x, end x in pixel, peak d in Angstrom, frac. by d, com, frac. by com,
                # int., norm. int, dos., norm. dos.
                fn = os.path.join(self.exportdir, self.fileshort + '_single_peak_info')
                with open(fn, 'w') as f:
                    np.savetxt(f, self.single_peak_info, '%-10.5f', header='peak y, start x, end x in pixel,'
                                                                           'peak d in Angstrom, frac. by d, com, frac. by com,'
                                                                           'int., norm. int, dos., norm. dos.')
                    print('out put a single peak integrateion information file')


    def single_peak_select(self, evt, pw):
        if pw.sceneBoundingRect().contains(evt.scenePos()):
            mouse_point = pw.vb.mapSceneToView(evt.scenePos())  # directly, we have a viewbox!!!
            # the following could be made into a function
            # x and y are all data number, x is not actuall x!
            peak_x = mouse_point.x()
            peak_y = mouse_point.y()
            # the following could be problematic: what if there is no integrated checkbox checked?!
            q_start = int(self.parameters['integrated']['clip head'].setvalue)  # q is also data num
            # peak catalog step
            if hasattr(self, 'peaks_catalog_select'):
                if self.peaks_catalog_map != []:
                    index = 0
                    for peak in self.peaks_catalog_select:
                        diff_x = abs(peak[::, 1] - (peak_x - q_start))
                        diff_y = abs(peak[::, 0] - peak_y)
                        if min(diff_x) < int(self.linewidgets['time series']['assign proximity x'].text()) and \
                                min(diff_y) < int(self.linewidgets['time series']['assign proximity y'].text()):
                            self.single_peak = index
                            print('find peak {} to integrate'.format(index))
                            break

                        index += 1

    def single_peak_update(self, winobj):
        pw = winobj.gdockdict[self.method_name].tabdict['single peak int.'].tabplot
        curves = self.availablecurves['single peak int.'][1:]
        symbols = ['x', 'o', 't', 's', 'h', 'p']
        colors = ['r', 'g', 'b', 'm', 'c', 'k']
        for index in reversed(range(len(pw.items))):  # shocked!
            if isinstance(pw.items[index], pg.PlotDataItem): # pg.PlotCurveItem):
                if pw.items[index].name() in curves:
                    pw.removeItem(pw.items[index])

        if hasattr(self, 'single_peak_info'):
            for curve in curves:
                if self.curvedict['single peak int.'][curve].isChecked():
                    index = curves.index(curve)
                    pw.plot(self.single_peak_info[:,0], self.single_peak_info[:, index + 3],
                            symbol=symbols[index], symbolSize=10, name=curve, pen=pg.mkPen(colors[index], width=5))

    def single_peak_width(self, winobj):
        pw = winobj.gdockdict[self.method_name].tabdict['time series'].tabplot
        q_start = int(self.parameters['integrated']['clip head'].setvalue)  # q is also data num
        for index in reversed(range(len(pw.items))):  # shocked!
            if isinstance(pw.items[index], pg.PlotDataItem): # pg.PlotCurveItem):
                if pw.items[index].name() in ['left_ips','right_ips']:
                    pw.removeItem(pw.items[index])

        if hasattr(self, 'peaks_all') and self.single_peak != []:
            peaks = np.array(self.peaks_catalog_select[self.single_peak])  # y, x, ?, entry, k
            width_factor = self.parameters['time series']['single peak int. width'].setvalue
            peak_start = []
            peak_end = []
            for index in range(peaks.shape[0]):
                peak_start.append(int(peaks[index, 1] - width_factor * abs(peaks[index, 1] -
                    self.peaks_properties_all[peaks[index, 3]]['left_ips'][peaks[index, 4]])))
                peak_end.append(int(peaks[index, 1] + width_factor * abs(peaks[index, 1] -
                    self.peaks_properties_all[peaks[index, 3]]['right_ips'][peaks[index, 4]])))

            peak_start = np.array(peak_start)
            peak_end = np.array(peak_end)
            pw.plot(peak_start + q_start, peaks[:, 0], pen=pg.mkPen('w', width=3), name='left_ips')
            pw.plot(peak_end + q_start, peaks[:, 0], pen=pg.mkPen('w', width=3), name='right_ips')
            # return peak_start, peak_end

    def integrate_Cluster(self, ponifile_short, clusterfile):
        clusterfolder = '/data/visitors/' + self.directory.replace(os.sep, '/').replace('//', '/')[2::]
        client = SSHClient()
        client.load_system_host_keys()
        try:
            client.connect('clu0-fe-1.maxiv.lu.se', username='balder-user', password='BeamPass!')
        except:
            print('check network and password')
        else:
            cmd = "sbatch --export=ALL,argv=\'%s %s %s %s %s\' " \
                  "/mxn/home/balder-user/BalderProcessingScripts/balder-xrd-processing/submit_stream_MR.sh" % (
                      clusterfolder, self.fileshort[0:-6], ponifile_short, self.rawdata_path, self.eiger_mark)

            client.exec_command('mv ./*.log ./azint_logs')
            stdin, stdout, stderr = client.exec_command(cmd)
            print(stdout.readlines())
            client.close()

            # add status check
            t0 = time.time()
            t = 0
            while t < 10000:
                if os.path.exists(clusterfile):
                    print('Integration completed by HPC cluster in ', int(t), ' seconds.')
                    break
                time.sleep(1)
                t = time.time() - t0

            # clusterfile_linux = '/data/visitors/' + clusterfile.replace(os.sep, '/').replace('//', '/')[2::]
            # client.exec_command('chmod g+w {}'.format(clusterfile_linux))

            print(stderr.readlines())

    def read_intg(self): # it's raw, not normal here, can make a choice in the future
        with h5py.File(self.exportfile, 'r') as f:
            if self.parameters['time series']['normalization'].choice == 'not normalized':
                intdata_all = f['rawresult']
            else:
                intdata_all = f['normresult']

            self.intdata_ts = np.zeros((intdata_all.shape[0], intdata_all.shape[1]), dtype='float32')
            intdata_all.read_direct(self.intdata_ts)
            self.intqaxis = np.array(list(f['info/q(Angstrom)']))
            tth = np.array(list(f['info/2theta']))
            print(self.wavelength)
            self.wavelength = 4 * np.pi * np.sin(tth[0] / 2 / 180 * np.pi) / self.intqaxis[0] # check?
            print(self.wavelength)
            if 'y min' in list(f.keys()):
                self.y_range = [f['y min'][()], f['y max'][()]]

        if hasattr(self, 'glitches'):
            try:
                for glitch in self.glitches:
                    glitch_start = np.where(self.intqaxis - glitch[0] < 0)[0][-1]
                    glitch_end = np.where(self.intqaxis - glitch[1] > 0)[0][0]
                    self.intqaxis = np.delete(self.intqaxis, np.s_[glitch_start:glitch_end])
                    self.intdata_ts = np.delete(self.intdata_ts, np.s_[glitch_start:glitch_end], axis=1)
            except:
                print('redo the integration, may be caused by an incorrect first data in xas4xrd file')

    def output_intg(self, q, result, norm_result, wavelength, interval):
        if not os.path.isdir(self.exportdir): os.mkdir(self.exportdir)
        resultfile = h5py.File(self.exportfile, 'w')
        tempgroup = resultfile.create_group('info')
        # attention to this, q in nm-1, divided by 10 to get A ^-1
        tempgroup.create_dataset('q(Angstrom)', data=q / 10)
        data_2theta = np.arcsin(q * wavelength / 10 / np.pi / 4) * 2 / np.pi * 180
        tempgroup.create_dataset('2theta', data=data_2theta)
        tempgroup.create_dataset('abs_time_in_sec', data=self.entrytimesec)
        resultfile.create_dataset('rawresult', data=np.array(result))
        resultfile.create_dataset('normresult', data=np.array(norm_result))
        resultfile.close()
        # txt
        for k in range(0,len(norm_result),interval):
            with open(os.path.join(self.exportdir, self.fileshort + '_data{:04d}.xy'.format(k)), 'w') as f:
                np.savetxt(f, np.array([data_2theta, norm_result[k]]).transpose())

    def interpolate_data(self, winobj):
        # to densify data along q to make following peaks cataloguing more effective incase q is not enough
        pass

    def find_peak_all(self, winobj): # only works in q right now; 1st step
        q_start = int(self.parameters['integrated']['clip head'].setvalue)
        q_ending = int(self.parameters['integrated']['clip tail'].setvalue)
        peaks_q = []
        peaks_number = []
        self.peaks_index = [] # data number for each data where you can find some peaks, for this reason delta y should be based on this
        self.peaks_all = [] # x positions of all peaks within each xrd data, and for all xrd data
        self.peaks_properties_all = []
        for index in range(self.entrytimesec.shape[0]):
            intdata_clipped = self.intdata_ts[index][q_start: -q_ending]
            if 'smoothed' in self.curve_timelist[0]['integrated']:
                sg_win = int(self.parameters['integrated']['Savitzky-Golay window'].setvalue)
                sg_order = int(self.parameters['integrated']['Savitzky-Golay order'].setvalue)
                if sg_win > sg_order + 1:
                    self.intdata_smoothed = scipy.signal.savgol_filter(intdata_clipped,sg_win, sg_order)

            if hasattr(self, 'intdata_smoothed'):
                peaks, peak_properties = self.find_peak_conditions(self.intdata_smoothed)
            else:
                peaks, peak_properties = self.find_peak_conditions(intdata_clipped)

            if peaks is not []:
                self.peaks_index.append(index)
                self.peaks_all.append(peaks) # useful for catalog_peaks
                self.peaks_properties_all.append(peak_properties)
                for sub_index in range(len(peaks)):
                    peaks_q.append(peaks[sub_index]) # x, index
                    peaks_number.append(index)            # y, index in 2D contour plot

        self.find_peak_plot(peaks_q, peaks_number, winobj)

    def find_peak_plot(self, peaks_q, peaks_number, winobj):
        if not hasattr(self, 'peak_map'):
            if 'time series' not in winobj.gdockdict[self.method_name].tabdict:
                self.checksdict['time series'].setChecked(True)

            self.peak_map = winobj.gdockdict[self.method_name].tabdict['time series'].tabplot.plot(name='find peaks')

        # draw all peak positions overlap with 2D contour plot.
        self.peak_map.setData(np.array(peaks_q) + self.parameters['integrated']['clip head'].setvalue, peaks_number,
                              symbol='o', symbolPen='k', pen=None, symbolBrush=None,
                              symbolSize=self.parameters['time series']['symbol size'].setvalue)

    def show_clear_ts(self, winobj):
        tempwidget = self.actwidgets['time series']['clear rainbow map (Ctrl+E)']
        if "clear rainbow map (Ctrl+E)" == tempwidget.text():
            tempwidget.setText("show rainbow map (Ctrl+E)")
            tempwidget.setShortcut('Ctrl+E')
            if hasattr(self, 'color_bar_ts'):
                winobj.gdockdict[self.method_name].tabdict['time series'].tabplot.removeItem(self.img_ts)
                self.color_bar_ts.close()

        else:
            tempwidget.setText("clear rainbow map (Ctrl+E)")
            tempwidget.setShortcut('Ctrl+E')
            self.plot_from_load(winobj)

    def clear_reflections(self, winobj):
        pw = winobj.gdockdict[self.method_name].tabdict['integrated'].tabplot
        # clear plots
        for index in reversed(range(len(pw.items))):  # shocked!
            if isinstance(pw.items[index], pg.PlotDataItem):
                if pw.items[index].name()[0:4] in ['inde', 'ref.']:
                    pw.removeItem(pw.items[index])

        # clear text
        for index in reversed(range(len(pw.items))):  # shocked! must do this, forward order makes a mess!
            if isinstance(pw.items[index], pg.TextItem): pw.removeItem(pw.items[index])

    def show_clear_peaks(self, winobj):
        pw = winobj.gdockdict[self.method_name].tabdict['time series'].tabplot
        tempwidget = self.actwidgets['time series']['clear peaks (Ctrl+P)']
        if 'clear peaks (Ctrl+P)' == tempwidget.text():
            tempwidget.setText('show peaks (Ctrl+P)')
            tempwidget.setShortcut('Ctrl+P')
            for index in reversed(range(len(pw.items))): # shocked!
                if isinstance(pw.items[index], pg.PlotDataItem):
                    if pw.items[index].name()[0:4] in ['find', 'cata', 'assi', 'phas', 'inde', 'ref.', 'left', 'right']:
                        pw.removeItem(pw.items[index])

            # delete old text
            for index in reversed(range(len(pw.items))):  # shocked! must do this, forward order makes a mess!
                if isinstance(pw.items[index], pg.TextItem): pw.removeItem(pw.items[index])
                    # if pw.items[index].color == fn.mkColor(color): pw.removeItem(pw.items[index])
                        
        else:
            tempwidget.setText('clear peaks (Ctrl+P)')
            tempwidget.setShortcut('Ctrl+P')
            if hasattr(self, 'peak_map'): pw.addItem(self.peak_map)
            if hasattr(self, 'peaks_catalog_map'): # add the old plot
                for index in range(len(self.peaks_catalog_map)): pw.addItem(self.peaks_catalog_map[index])

            if hasattr(self, 'phases_map'): # add the old plot
                for index in range(len(self.phases_map)): pw.addItem(self.phases_map[index])

    def show_phase(self, winobj):
        pw = winobj.gdockdict[self.method_name].tabdict['time series'].tabplot
        phase_name = self.parameters['time series']['phases'].choice
        for index in reversed(range(len(pw.items))):
            if isinstance(pw.items[index], pg.PlotDataItem):
                if pw.items[index].name()[0:4] in ['find', 'cata', 'assi']: pw.removeItem(pw.items[index])

        if hasattr(self, 'phases_map') and phase_name != 'choose a phase': # shock! must use string compare
            if int(phase_name[5:]) <= len(self.phases):
                pw.addItem(self.phases_map[int(phase_name[5:]) - 1])
                # print('showing phase')

    def range_select(self, winobj):
        # to select range for peaks sorting
        pw = winobj.gdockdict[self.method_name].tabdict['time series'].tabplot
        tempwidget = self.actwidgets['time series']['select start (Ctrl+R)']
        pw.scene().sigMouseClicked.connect(lambda evt, p=pw: self.range_clicked(evt, p))
        if tempwidget.text() == 'select start (Ctrl+R)':
            tempwidget.setText('select end (Ctrl+R)')
            tempwidget.setShortcut('Ctrl+R')
        elif tempwidget.text() == 'select end (Ctrl+R)':
            tempwidget.setText('exclude from (Ctrl+R)')
            tempwidget.setShortcut('Ctrl+R')
        elif tempwidget.text() == 'exclude from (Ctrl+R)':
            tempwidget.setText('exclude to (Ctrl+R)')
            tempwidget.setShortcut('Ctrl+R')
        elif tempwidget.text() == 'exclude to (Ctrl+R)':
            tempwidget.setText('done (Ctrl+R)')
            tempwidget.setShortcut('Ctrl+R')
        else:
            tempwidget.setText('select start (Ctrl+R)')
            tempwidget.setShortcut('Ctrl+R')
            pw.scene().sigMouseClicked.disconnect()

    def range_clicked(self, evt, pw):
        if pw.sceneBoundingRect().contains(evt.scenePos()):
            mouse_point = pw.vb.mapSceneToView(evt.scenePos()) # directly, we have a viewbox!!!
            temptext = self.actwidgets['time series']['select start (Ctrl+R)'].text()
            if temptext == 'select end (Ctrl+R)':
                self.linewidgets['time series']['select start'].setText(str(int(mouse_point.y())))
            elif temptext == 'exclude from (Ctrl+R)':
                self.linewidgets['time series']['select end'].setText(str(int(mouse_point.y())))
            elif temptext == 'exclude to (Ctrl+R)':
                self.linewidgets['time series']['exclude from'].setText(str(int(mouse_point.y())))
            else:
                self.linewidgets['time series']['exclude to'].setText(str(int(mouse_point.y())))

    def catalog_peaks(self, winobj): # 2nd step
        i_start = self.linewidgets['time series']['select start'].text()
        i_end = self.linewidgets['time series']['select end'].text()
        i_ex_start = self.linewidgets['time series']['exclude from'].text()
        i_ex_end = self.linewidgets['time series']['exclude to'].text()

        if i_start is not '' and i_end is not '' and i_ex_start is not '' and i_ex_end is not '':
            i_start = np.where(np.array(self.peaks_index) - int(i_start) >= 0)[0][0]
            i_end = np.where(np.array(self.peaks_index) - int(i_end) >= 0)[0][0]
            i_ex_start = np.where(np.array(self.peaks_index) - int(i_ex_start) >= 0)[0][0]
            i_ex_end = np.where(np.array(self.peaks_index) - int(i_ex_end) >= 0)[0][0]

        # i_start = np.min([i_start, i_end])  # to prevent disorder
        # i_end = np.max([i_start, i_end])
        if i_start < i_ex_start < i_ex_end < i_end: # to insure the correct order
            peaks_index_sel = self.peaks_index[i_start:i_ex_start] + self.peaks_index[i_end:i_ex_end]
        else:
            print('ignore excluded region')
            peaks_index_sel = self.peaks_index[i_start:i_end]

        # start cataloging peaks, the most exciting part
        if peaks_index_sel != []:
            self.peaks_catalog = []
            for index in range(len(self.peaks_all[i_start])): # start from the first data in selected region
                # first level: peaks; second level: index in y in full data, index in x in full data (excl. clipped head), index in peaks_index_sel
                self.peaks_catalog.append([[self.peaks_index[i_start],self.peaks_all[i_start][index], 0, i_start, index]])

            gap_y = int(self.parameters['time series']['gap y tol.'].setvalue)
            gap_x = int(self.parameters['time series']['gap x tol.'].setvalue)

            for index in range(len(peaks_index_sel) - 1): # index on data number peaks selected (self.peaks_index)
                # for j in range(min([gap_y, i_end - index])): # index on gap_y tolerence
                index = index + 1
                entry = self.peaks_index.index(peaks_index_sel[index]) # the next available data index
                search_range = len(self.peaks_catalog) # search through each established peak catalog
                for k in range(len(self.peaks_all[entry])): # index on all peaks detected within one data
                    add_group = [] # indicator whether to add a new peaks group or not
                    # matched = [] # applies when you have Y shaped peak trajectory in time series plot
                    for i in range(search_range): # index on existing peaks groups, this one is constantly changing!!!
                        # the following condition is very tricky in y direction, it needs to be [2] not [0] of self.peak_catalog[i][-1]
                        distance_x = np.abs(self.peaks_all[entry][k] - self.peaks_catalog[i][-1][1])
                        distance_y = np.abs(index - self.peaks_catalog[i][-1][2])
                        if distance_x <= gap_x and 0 < distance_y <= gap_y: # add to existing peaks group
                            # matched.append([i, distance_x,
                            #                 self.peaks_index[entry], self.peaks_all[entry][k], index, entry, k])
                            self.peaks_catalog[i].append([self.peaks_index[entry],
                                                          self.peaks_all[entry][k], index, entry, k])
                            # data number y, x, index in peaks_index_sel, data number _0 in peaks_all (very similar to y) and _1
                            add_group.append(1) # 1 means not to add new group
                        else: add_group.append(0)

                    if np.sum(add_group) == 0: # add a new catalog if this peak belongs to no old catalogs
                        self.peaks_catalog.append([[self.peaks_index[entry],
                                                    self.peaks_all[entry][k], index, entry, k]])

            # show peaks that are longer-lasting in time, i.e. life span larger than lenght_limit
            length_limit = self.parameters['time series']['min time span'].setvalue
            self.peaks_catalog_select = [] # peak number, data number (y,x); still, you need q_start
            for index in range(len(self.peaks_catalog)):
                self.peaks_catalog[index] = np.array(self.peaks_catalog[index])
                if self.peaks_catalog[index].shape[0] > length_limit:
                    diff_average = np.average(abs(np.diff(self.peaks_catalog[index][::, 1]))) # dx only
                    # diff_average = np.average(abs(np.diff(self.peaks_catalog[index][::,1]) /
                    #                               np.diff(self.peaks_catalog[index][::,0]))) # dx / dy
                    # a real peak will have a smaller diff_average, for a fake peak, it appears more waggled, hence larger diff_average
                    if diff_average < float(self.parameters['time series']['1st deriv control'].setvalue):
                        self.peaks_catalog_select.append(self.peaks_catalog[index])

            # merge peaks
            peak_to_pop = []
            for i in range(len(self.peaks_catalog_select) - 1):
                for j in np.arange(i + 1, len(self.peaks_catalog_select)):
                    if self.peaks_catalog_select[i].shape[0] != 0 and self.peaks_catalog_select[j].shape[0] != 0:
                        merged = np.unique(np.concatenate((self.peaks_catalog_select[i],self.peaks_catalog_select[j]),axis=0), axis=0)
                        if merged.shape[0] < self.peaks_catalog_select[i].shape[0] + self.peaks_catalog_select[j].shape[0]:
                            # version 2: the longer one survives, the shorter one get even shorter and may not survive
                            if self.peaks_catalog_select[j].shape[0] >= self.peaks_catalog_select[i].shape[0]:
                                self.peaks_catalog_select[i] = np.array([p for p in self.peaks_catalog_select[i]
                                                                         if p not in self.peaks_catalog_select[j]])
                                # self.peaks_catalog_select[j] = merged
                                if self.peaks_catalog_select[i].shape[0] < length_limit:
                                    peak_to_pop.append(i)
                            else:
                                self.peaks_catalog_select[j] = np.array([p for p in self.peaks_catalog_select[j]
                                                                         if p not in self.peaks_catalog_select[i]])
                                # self.peaks_catalog_select[i] = merged
                                if self.peaks_catalog_select[j].shape[0] < length_limit:
                                    peak_to_pop.append(j)

                        # version 1: merge
                        # self.peaks_catalog_select[j] = merged
                        # peak_to_pop.append(i)
                        # print('peak {} and peak {} will be merged'.format(i,j))

            # this is cool: list comprehensions, all other methods is incorrect (pop remove del)... not a good way either!!!
            # self.peaks_catalog_select = [peak for peak in self.peaks_catalog_select if not self.peaks_catalog_select.index(peak) in peak_to_pop]
            peaks = []
            for index in range(len(self.peaks_catalog_select)):
                if index not in peak_to_pop:
                    peaks.append(self.peaks_catalog_select[index])

            self.peaks_catalog_select = peaks

            self.catalog_peaks_plot(winobj)

    def catalog_peaks_plot(self, winobj):
        pw = winobj.gdockdict[self.method_name].tabdict['time series'].tabplot
        # delete old plot and draw new ones
        if hasattr(self, 'peaks_catalog_map'):  # clear the old plot
            for index in range(len(self.peaks_catalog_map)):
                pw.removeItem(self.peaks_catalog_map[index])

        if hasattr(self, 'phases_map'):  # clear the old plot
            for index in range(len(self.phases_map)):
                pw.removeItem(self.phases_map[index])

            self.phases_map = []

        self.peaks_catalog_map = [None] * len(self.peaks_catalog_select)
        q_start = int(self.parameters['integrated']['clip head'].setvalue)
        self.peaks_color = []
        i_start = min([min(self.peaks_catalog_select[index][:,0]) for index in range(len(self.peaks_catalog_select))])
        i_end = max([max(self.peaks_catalog_select[index][:, 0]) for index in range(len(self.peaks_catalog_select))])
        self.linewidgets['time series']['y min'].setText(str(i_start))
        self.linewidgets['time series']['y max'].setText(str(i_end))
        pw.setYRange(i_start, i_end)
        for index in range(len(self.peaks_catalog_select)):
            color = pg.intColor(index * self.huestep, 100)
            self.peaks_color.append(color)
            self.peaks_catalog_map[index] = \
                pw.plot(name='catalog peaks {}'.format(index))
            self.peaks_catalog_map[index].setData(self.peaks_catalog_select[index][::, 1] + q_start,
                                                  self.peaks_catalog_select[index][::, 0], symbol='o',
                                                  symbolBrush=color,
                                                  symbolSize=self.parameters['time series']['symbol size'].setvalue)

    def save_peaks_phases(self, winobj):
        tempwidget = self.actwidgets['time series']['save peaks/phases (Ctrl+H)']
        if tempwidget.text() == 'save peaks/phases (Ctrl+H)': # save peaks found
            tempwidget.setText('save peaks catalogued (Ctrl+H)')
            tempwidget.setShortcut('Ctrl+H')
            if all(hasattr(self, attribute) for attribute in ['peaks_index', 'peaks_all', 'peaks_properties_all']):
                with open(os.path.join(self.exportdir, self.fileshort + '_peaks'), 'wb') as f:
                    pickle.dump([self.peaks_index, self.peaks_all, self.peaks_properties_all], f)

        elif tempwidget.text() == 'save peaks catalogued (Ctrl+H)': # save peaks catalogued
            tempwidget.setText('save phases catalogued (Ctrl+H)')
            tempwidget.setShortcut('Ctrl+H')
            if hasattr(self, 'peaks_catalog_select'):
                with open(os.path.join(self.exportdir, self.fileshort + '_peaks_catalog_select'), 'wb') as f:
                    pickle.dump(self.peaks_catalog_select, f)

        else: # save phases
            tempwidget.setText('save peaks/phases (Ctrl+H)')
            tempwidget.setShortcut('Ctrl+H')
            if hasattr(self,'phases'):
                with open(os.path.join(self.exportdir, self.fileshort + '_phases'), 'wb') as f:
                    pickle.dump(self.phases, f)

    def load_peaks_phases(self, winobj):
        tempwidget = self.actwidgets['time series']['load peaks/phases (Ctrl+J)']
        if tempwidget.text() == 'load peaks/phases (Ctrl+J)': # load peaks found
            tempwidget.setText('load peaks catalogued (Ctrl+J)')
            tempwidget.setShortcut('Ctrl+J')
            peaks_file = os.path.join(self.exportdir, self.fileshort + '_peaks')
            if os.path.isfile(peaks_file):
                with open(peaks_file, 'rb') as f:
                    self.peaks_index, self.peaks_all, self.peaks_properties_all = pickle.load(f)

                # draw all peaks
                peaks_q = []  # x
                peaks_number = []  # y
                for index in self.peaks_index:  # y
                    for sub_index in self.peaks_all[index]:  # x
                        peaks_q.append(sub_index)
                        peaks_number.append(index)

                try: self.find_peak_plot(peaks_q, peaks_number, winobj)
                except: print('what is wrong with find peak plot?')

        elif tempwidget.text() == 'load peaks catalogued (Ctrl+J)': # load peaks catalogued
            tempwidget.setText('load phases catalogued (Ctrl+J)')
            tempwidget.setShortcut('Ctrl+J')
            pcs_file = os.path.join(self.exportdir, self.fileshort + '_peaks_catalog_select')
            if os.path.isfile(pcs_file):
                with open(pcs_file, 'rb') as f:
                    self.peaks_catalog_select = pickle.load(f)

                try: self.catalog_peaks_plot(winobj)
                except: print('what is wrong with catalog peaks plot?')

        else: # load phases
            tempwidget.setText('load peaks/phases (Ctrl+J)')
            tempwidget.setShortcut('Ctrl+J')
            phases_file = os.path.join(self.exportdir, self.fileshort + '_phases')
            if os.path.isfile(phases_file):
                with open(phases_file, 'rb') as f:
                    self.phases = pickle.load(f)

                try: self.assign_phases_plot(winobj)
                except: print('what is wrong with assign phases plot?')

    def assign_phases(self, winobj):
        if hasattr(self, 'peaks_catalog_select'):
            time_diff = self.parameters['time series']['max diff time span'].setvalue
            start_diff = self.parameters['time series']['max diff start time'].setvalue
            self.phases = [[0]] # peak 0 belong to phase 0, next is peak 1; self.phases stores peak numbers for each phase
            # before assign phases, make sure that each peak is ordered in y direction, i.e. to peak_catalog_select[:,0]
            for index in range(len(self.peaks_catalog_select) - 1):
                # self.peaks_catalog_select[index + 1] = \
                #     self.peaks_catalog_select[index + 1][self.peaks_catalog_select[index + 1][:,0].argsort()]
                add_group = []
                for k in range(len(self.phases)):
                    time_span_index = np.abs(self.peaks_catalog_select[index + 1][-1][0] - \
                                      self.peaks_catalog_select[index + 1][0][0]) # time span in actual y number
                    time_span_k = np.abs(self.peaks_catalog_select[self.phases[k][-1]][-1][0] - \
                                  self.peaks_catalog_select[self.phases[k][-1]][0][0]) # time span in already established phases
                    time_start_index = self.peaks_catalog_select[index + 1][0][0]
                    time_start_k = self.peaks_catalog_select[self.phases[k][-1]][0][0]
                    if time_diff > np.abs(time_span_index - time_span_k) and \
                        start_diff > np.abs(time_start_index - time_start_k): # may release the first condition
                        self.phases[k].append(index + 1)
                        add_group.append(1)
                        break # work?
                    else: add_group.append(0)

                if np.sum(add_group) == 0:
                    self.phases.append([index + 1])

            print('{} phases found'.format(len(self.phases)))
            self.assign_phases_plot(winobj)

    def assign_phases_plot(self, winobj):
        pw = winobj.gdockdict[self.method_name].tabdict['time series'].tabplot
        if hasattr(self, 'phases_map'):  # clear the old plot
            for index in range(len(self.phases_map)):
                pw.removeItem(self.phases_map[index])

        if hasattr(self, 'peaks_catalog_map'):  # clear the old plot
            for index in range(len(self.peaks_catalog_map)):
                pw.removeItem(self.peaks_catalog_map[index])

            self.peaks_catalog_map = []

        self.phases_map = [None] * len(self.phases)
        q_start = int(self.parameters['integrated']['clip head'].setvalue)
        self.phases_color = []
        for index in range(len(self.phases)):
            temp = self.huestep
            self.huestep = 7
            color = pg.intColor(index * self.huestep, 100)
            self.phases_color.append(color)
            phase_peaks = self.peaks_catalog_select[self.phases[index][0]]
            if len(self.phases[index]) > 1:
                for k in range(len(self.phases[index]) - 1):
                    phase_peaks = np.concatenate(
                        (phase_peaks, self.peaks_catalog_select[self.phases[index][k + 1]]))  # another shock!

            self.phases_map[index] = \
                pw.plot(name='assign phases' + ' ' + str(index))
            self.phases_map[index].setData(phase_peaks[::, 1] + q_start, phase_peaks[::, 0],
                                           symbol='d', symbolBrush=color, pen=None,
                                           symbolSize=self.parameters['time series']['symbol size'].setvalue)
            self.huestep = temp

    def index_phases(self, winobj):
        if hasattr(self, 'phases'):
            if not hasattr(self, 'index_win'):
                self.index_win = DockGraph('Indexing') # index window
                self.index_win.gendock(winobj)

            if len(self.index_win.tabdict) is not 0: # index tabs in the index window
                for index in range(len(self.index_win.tabdict)):
                    phase_name = 'phase' + str(index + 1)
                    self.index_win.tabdict[phase_name].deltab(self.index_win)

                self.index_win.tabdict = {}

            for index in range(len(self.phases)):
                phase_name = 'phase' + str(index + 1)
                self.index_win.tabdict[phase_name] = TabGraph_index(phase_name)
                self.index_win.tabdict[phase_name].gentab(self.index_win, self, winobj)

    def indexing(self, tabobj, winobj): # this is also a cool part!
        q_start = int(self.parameters['integrated']['clip head'].setvalue)
        phase_num = int(tabobj.label[5:]) # assuming all names are 'phaseXX', so the actual index is phase_num - 1
        phase_d = [] # d_obs of all peaks
        phase_int = [] # intensity of all peaks
        q_peak = [] # q (or x) data number of all peaks
        common_data = set(range(self.entrytimesec.shape[0] + 1)) # this is smart!
        for index in self.phases[phase_num - 1]: # average the d-value, intensity, on common data only!
            peak_data = np.array(self.peaks_catalog_select[index])
            common_data = common_data & set(peak_data[:,0])

        common_data = list(common_data) # indexing the common data number!
        for index in self.phases[phase_num - 1]: # it is quite certain that everything is in timely order, i.e. sorted
            peak_data = np.array(self.peaks_catalog_select[index])
            peak_data = peak_data[np.searchsorted(peak_data[:,0],common_data),:]
            q_peak.append(int(peak_data[:,1].sum() / peak_data.shape[0] + q_start)) # average over the common data number
            phase_d.append(2 * np.pi / self.intqaxis[q_peak[-1]]) # in Angstrom, d spaces of a phase
            peak_int = 0 # intensity of one peak
            for peak in peak_data:
                data_num = self.peaks_index.index(peak[0])
                peak_num = np.where(self.peaks_all[data_num] == peak[1])[0][0]
                peak_prop = self.peaks_properties_all[data_num]
                peak_int += peak_prop['prominences'][peak_num] * (peak_prop['right_ips'][peak_num] -
                                                                   peak_prop['left_ips'][peak_num]) / 2

            phase_int.append(peak_int / peak_data.shape[0]) # average peak intensity

        # plot the averaged q for each peak for this phase
        self.plot_reflections(winobj=winobj, positions=q_peak, phase_name=tabobj.label, symbol='t',
                              offset=0, hkl=[], color=[])

        # with open(self.ponifile, 'r') as f:
        #     wavelength = float(f.readlines()[-1].splitlines()[0].partition(' ')[2])

        peaks = [] # tth, peak intensity, ... , d space, ...
        for index in range(len(phase_int)):
            peaks.append([np.arcsin(self.wavelength / np.array(phase_d[index]) / 2) * 2 / np.pi * 180,
                          phase_int[index], True, True, 0, 0, 0, phase_d[index], 0])

        peaks = np.array(peaks)
        peaks = peaks[peaks[:,0].argsort()]
        # print(peaks)
        # output d space and intensity for index
        d_space_file = os.path.join(self.exportdir, self.fileshort + '_d_space_phase_%i' % phase_num) # the actual phase number is phase_num - 1
        with open(d_space_file, 'w') as f:
            np.savetxt(f, peaks[:,[7,1]], '%-10.5f', header='index_d index_I') # must keep the order
            print('out put d space file for phase %i' % phase_num)

        bravais = [0] * len(self.bravaisNames)
        for index in range(tabobj.cbox_layout.count()): # get bravais
            item = tabobj.cbox_layout.itemAt(index).widget()
            if isinstance(item, QCheckBox):
                if item.isChecked():
                    if item.text() in self.bravaisNames:
                        bravais[self.bravaisNames.index(item.text())] = 1

        index_done,dmin,cells = GSASIIindex.DoIndexPeaks(peaks, [0, 0, int(tabobj.ncno.text()),
                                                                 int(tabobj.vol_start.text())], bravais, 0)
        # ifX20=True,timeout=None,M20_min=2.0,X20_max=None,return_Nc=False, cctbx_args=None)
        # cells[x] = [M20,X20,ibrav,a,b,c,alp,bet,gam,V,False,False]
        if index_done:
            headers = ['M20', 'X20', 'Bravais', 'a', 'b', 'c', 'alpha', 'beta', 'gamma', 'Volumn']
            format_all = ["{:.2f}","{}","{}","{:.4f}","{:.4f}","{:.4f}","{:.2f}","{:.2f}","{:.2f}","{:.4f}"]
            self.cells_sort[tabobj.label] = np.array([(*cells[row],) for row in range(len(cells))],
                                  dtype='f4, u1, u1, f8, f8, f8, f8, f8, f8, f8, b, b')
            self.cells_sort[tabobj.label][::-1].sort(axis=0, order='f0')
            tabobj.table.setRowCount(len(self.cells_sort[tabobj.label]))
            tabobj.table.setColumnCount(len(self.cells_sort[tabobj.label][0]) - 2)
            tabobj.table.setHorizontalHeaderLabels(headers)
            for row in range(len(self.cells_sort[tabobj.label])):
                for col in range(len(self.cells_sort[tabobj.label][0]) - 2):
                    tabobj.table.setItem(row,col,QTableWidgetItem(format_all[col].format(self.cells_sort[tabobj.label][row][col])))

            tabobj.table.selectionModel().selectionChanged.connect(lambda: self.show_reflections(tabobj, winobj))

    def plot_reflections(self, winobj, positions, phase_name, symbol, offset, hkl, color): # the first five letters of name must be distinguishable
        pw = winobj.gdockdict[self.method_name].tabdict['time series'].tabplot
        if color == []: # reflection dots
            y_start = int(self.linewidgets['time series']['select start'].text())
            y_end = int(self.linewidgets['time series']['select end'].text())

        elif symbol in ['max', 'min']: # on single data
            pw = winobj.gdockdict[self.method_name].tabdict['integrated'].tabplot
            try:
                if 'smoothed' not in self.curve_timelist[0]['integrated']:
                    data = self.curve_timelist[0]['integrated']['original'].yData
                else:
                    data = self.curve_timelist[0]['integrated']['smoothed'].yData

                y_start = max(data) if symbol == 'max' else min(data) / 2
                y_end = min(data) if symbol == 'min' else max(data) + min(data) / 2
            except:
                print('what is wrong with plot reflection')

        else: # ref_phases
            # y_start = int(self.linewidgets['time series']['y min'].text())
            # y_end = int(self.linewidgets['time series']['y max'].text())
            y_start = self.entrytimesec.shape[0] / 10
            y_end = self.entrytimesec.shape[0] / 4

        data_center = (y_start + y_end) / 2
        y = np.array([data_center] * len(positions))
        offset = abs(y_end - y_start) * offset

        # delete old plot
        for index in reversed(range(len(pw.items))):  # shocked!
            if isinstance(pw.items[index], pg.PlotDataItem):
                if pw.items[index].name()[0:4] == phase_name[0:4]: pw.removeItem(pw.items[index])

        if color == []:  # reflection dots
            pw.plot(positions, y - offset, pen=None, symbol=symbol, symbolSize=15, name=phase_name).setZValue(100)
            color = 'k'
        elif symbol in ['max', 'min']: # single data
            for x in positions:
                if x < self.intqaxis.shape[0]:
                    pw.plot([x, x], [y_start, y_end],
                            pen=pg.mkPen(color, width=1),name=phase_name)
            for item in winobj.gdockdict[self.method_name].tabdict['integrated'].graphtab.items():  # shocked!
                if isinstance(item, pg.LegendItem):
                    for lgd in reversed(item.items):
                        if lgd[1].text == phase_name:
                            item.removeItem(lgd[0].item)

        else: # ref_phases
            for x in positions:
                if x < self.intqaxis.shape[0]:
                    # pw.plot([x,x],[0,self.intdata_ts.shape[0]],pen=pg.mkPen(color,width=3),name=phase_name).setZValue(100) # work?
                    pw.plot([x, x], [(y_start + data_center) / 2, (data_center + y_end) / 2],
                            pen=pg.mkPen(color, width=3),name=phase_name).setZValue(100)  # work?

        # delete old text
        for index in reversed(range(len(pw.items))):  # shocked!
            if isinstance(pw.items[index], pg.TextItem):
                if pw.items[index].color == fn.mkColor(color): pw.removeItem(pw.items[index])

        if hkl != []:
            for index in range(hkl.shape[0]):
                hkl_text = pg.TextItem(str(hkl[index])[1:-1], angle=90, anchor=(1,1), color=color)
                pw.addItem(hkl_text)
                hkl_text.setPos(positions[index], data_center - offset * 2)
                hkl_text.setZValue(100)

    def show_reflections(self, tabobj, winobj):
        index = tabobj.table.selectionModel().currentIndex().row()
        A = GSASIIlattice.cell2A(np.array(list(self.cells_sort[tabobj.label][index]))[3:9])
        dmin = 2 * np.pi / self.intqaxis[-1]
        Bravais = self.cells_sort[tabobj.label][index][2]
        HKLD = np.array(GSASIIlattice.GenHBravais(dmin, Bravais, A)) # generate [h, k, l, d, -1]
        positions = (2 * np.pi / HKLD[:,3] - self.intqaxis[0]) / (self.intqaxis[-1] - self.intqaxis[0]) * self.intqaxis.shape[0]
        self.plot_reflections(winobj, positions, 'index_' + str(index), 't1', 0.03, HKLD[:,0:3], [])

    def keep_selected(self, tabobj):
        pass

    def add_ref_phase(self, winobj): # add and delete ref. phase lines
        tempwidget = self.actwidgets['time series']['add ref. phase (Ctrl+2)']
        pw = winobj.gdockdict[self.method_name].tabdict['time series'].tabplot
        dmin = 2 * np.pi / self.intqaxis[-1]
        if tempwidget.text() == 'add ref. phase (Ctrl+2)':
            tempwidget.setText('delete ref. phase (Ctrl+2)')
            tempwidget.setShortcut('Ctrl+2')
            for line_edit in self.linedit['time series']:
                if line_edit[0:10] == 'ref. phase':
                    # add new lines
                    phase = self.linewidgets['time series'][line_edit].text().split(',')
                    if phase != ['']:
                        if len(phase) == 1: # by outsourced index software
                            try:
                                phase_num = int(phase[0].split('_')[-1])
                                index_phase = int(phase[0].split('_')[-2])
                                index_method, drive, order1, order2, order3 = phase[0].split('_')[0:5]
                                if drive == 'Z': index_dir = r'Z:'
                                else: index_dir = self.exportdir

                                ndx_file = os.path.join(index_dir, self.fileshort + '_d_space_phase_%i.ndx' % index_phase)
                                if index_method == 'TOPAS' and os.path.isfile(ndx_file):
                                    with open(ndx_file, 'r') as f:
                                        ndx = f.readlines()

                                    for row in range(20): # judge the start:
                                        if ndx[row][0:18] == 'Indexing_Solutions': start_num = row + 2

                                    # get 100 phases out and sort them
                                    ndx = pd.DataFrame([item.split() for item in ndx[start_num:start_num + 100]]) # by gof
                                    # 3 unindexed, 4 volume, 5 gof
                                    order = ['unindex', 'vol', 'gof']
                                    ascend = [True, True, False]
                                    order_1 = order.index(order1)
                                    order_2 = order.index(order2)
                                    order_3 = order.index(order3)
                                    ndx = ndx.astype({3:float, 4:float, 5:float}).\
                                        sort_values(by=[order_1 + 3, order_2 + 3, order_3 + 3],
                                                    ascending=[ascend[order_1], ascend[order_2], ascend[order_3]])

                                    SGError, SGData = GSASIIspc.SpcGroup(GSASIIspc.spgbyNum[tight_spg.index(ndx.iloc[phase_num, 1]) + 1])
                                    A = GSASIIlattice.cell2A(np.array(ndx.iloc[phase_num, 6:12], dtype=float)) # attention to loc !!!
                                    HKLD = np.array(self.gen_refl(dmin, SGData, A))
                                    print(' '.join(list(ndx.astype({3:str, 4:str, 5:str}).iloc[phase_num])))
                                else:
                                    print('check your file name')
                            except:
                                print('pls put TOPAS_<drive num>_<gof, unindex, vol>_<num>_<num> (last num starts from zero)')

                        elif len(phase[0].split(' ')) == 1: # by Bravais
                            try:
                                Bravais = int(phase[0])
                                A = GSASIIlattice.cell2A(np.array([float(x) for x in phase[1:]]))
                                HKLD = np.array(GSASIIlattice.GenHBravais(dmin, Bravais, A))  # generate [h, k, l, d, -1]
                            except:
                                print('pls put a number for Bravais')
                                print('pls put 3 lattice and 3 angle values (,)')

                        else: # by space group H-M symbol
                            try:
                                SGError, SGData = GSASIIspc.SpcGroup(phase[0])
                                A = GSASIIlattice.cell2A(np.array([float(x) for x in phase[1:]]))
                                HKLD = np.array(self.gen_refl(dmin, SGData, A))
                            except:
                                print('pls put a H-M space group separated by space (No space in the head)')
                                print('pls put 3 lattice and 3 angle values (,)')

                        if 'HKLD' in locals():
                            positions = (2 * np.pi / HKLD[:, 3] - self.intqaxis[0]) / \
                                        (self.intqaxis[-1] - self.intqaxis[0]) * self.intqaxis.shape[0]
                            self.plot_reflections(winobj=winobj, positions=positions, phase_name=line_edit, symbol=[],
                                                  offset=0.05, hkl=HKLD[:, 0:3], color='w')
                            if self.checksdict['integrated'].isChecked() and \
                                    'original' in self.curve_timelist[0]['integrated'] or \
                                    'smoothed' in self.curve_timelist[0]['integrated']:
                                self.plot_reflections(winobj=winobj, positions=2 * np.pi / HKLD[:, 3], phase_name=line_edit,
                                                      symbol='max', offset=0.05, hkl=HKLD[:, 0:3], color='k')
                                print(HKLD)
                        # the following is the same as Bravais
                        # try:
                        #     A = GSASIIlattice.cell2A(np.array([float(x) for x in phase[1:]]))
                        #     SG = {}
                        #     sg_info = phase[0].split('_')
                        #     SG['SGLatt'] = sg_info[0][0]
                        #     SG['SGLaue'] = sg_info[0][1:]
                        #     if len(sg_info) > 1: SG['SGUniq'] = sg_info[1]
                        #     else: SG['SGUniq'] = ''
                        # except:
                        #     print('pls put e.g. C2/m or C2/m_c (_c for unique monoclinic axis) for correct Laue groups:')
                        #     Laue = ['-1','2/m','mmm','4/m','6/m','4/mmm','6/mmm', '3m1', '31m', '3', '3R', '3mR', 'm3', 'm3m']
                        #     print(Laue)
                        # else:
                        #     dmin = 2 * np.pi / self.intqaxis[-1]
                        #     HKLD = np.array(GSASIIlattice.GenHLaue(dmin, SG, A))  # generate [h, k, l, d, -1]
                        #     positions = (2 * np.pi / HKLD[:, 3] - self.intqaxis[0]) / (self.intqaxis[-1] - self.intqaxis[0]) * \
                        #                 self.intqaxis.shape[0]
                        #     self.plot_reflections(winobj=winobj, positions=positions, phase_name=line_edit, symbol=[],
                        #                           offset=0.03, hkl=HKLD[:, 0:3], color='w')
        else:
            tempwidget.setText('add ref. phase (Ctrl+2)')
            tempwidget.setShortcut('Ctrl+2')
            for line_edit in self.linedit['time series']:
                if line_edit[0:10] == 'ref. phase':
                    # delete old lines
                    for index in reversed(range(len(pw.items))):  # shocked!
                        if isinstance(pw.items[index], pg.PlotDataItem):
                            if pw.items[index].name() == line_edit: pw.removeItem(pw.items[index])

                    # hkl text
                    for index in reversed(range(len(pw.items))):  # shocked!
                        if isinstance(pw.items[index], pg.TextItem):
                            if pw.items[index].color == fn.mkColor('w'):
                                pw.removeItem(pw.items[index])

    def gen_refl(self, dmin, SGData, A):
        # modified from GenHBravais in GSASIIlattice
        Hmax = GSASIIlattice.MaxIndex(dmin, A)
        dminsq = 1. / (dmin ** 2)
        HKL = []
        Cent = SGData['SGLatt']
        SGSys = SGData['SGSys']
        Ops = []
        for ops in SGData['SpGrp'].split(' ')[1:]:
            Ops.append(ops.split('/'))
        # x = np.array([.11,.13,.17])  # a random position
        # x_all = []
        # for ops in SGData['SGOps']: # this should be all of equivalent positions? maybe not for some: R-3m
        #     x_all.append(ops[0].dot(x) + ops[1])
        #
        # x_unique = np.unique(np.array(x_all), axis=0) # should be unique already

        if SGSys == 'triclinic':
            for l in range(-Hmax[2], Hmax[2] + 1):
                for k in range(-Hmax[1], Hmax[1] + 1):
                    hmin = 0
                    if (k < 0): hmin = 1
                    if (k == 0 and l < 0): hmin = 1
                    for h in range(hmin, Hmax[0] + 1):
                        H = [h, k, l]
                        rdsq = GSASIIlattice.calc_rDsq(H, A)
                        if 0 < rdsq <= dminsq:
                            HKL.append([h, k, l, GSASIIlattice.rdsq2d(rdsq, 6), -1])

        elif SGSys == 'monoclinic': # - b unique, second setting, do you need the first setting, i.e. a or c unique?
            Hmax = GSASIIlattice.SwapIndx(2, Hmax)
            for h in range(Hmax[0] + 1):
                for k in range(-Hmax[1], Hmax[1] + 1):
                    lmin = 0
                    if k < 0: lmin = 1
                    for l in range(lmin, Hmax[2] + 1):
                        [h, k, l] = GSASIIlattice.SwapIndx(-2, [h, k, l])
                        H = []
                        screw_axis = True
                        glide_plane = True
                        if '21' in Ops[0] and h == 0 and l == 0 and k % 2: screw_axis = False
                        if k == 0:
                            if 'a' in Ops[0] and h % 2: glide_plane = False
                            if 'c' in Ops[0] and l % 2: glide_plane = False
                            if 'n' in Ops[0] and (h + l) % 2: glide_plane = False

                        if GSASIIlattice.CentCheck(Cent, [h, k, l]) and screw_axis and glide_plane: H = [h, k, l]
                        if H:
                            rdsq = GSASIIlattice.calc_rDsq(H, A)
                            if 0 < rdsq <= dminsq:
                                HKL.append([h, k, l, GSASIIlattice.rdsq2d(rdsq, 6), -1])
                        [h, k, l] = GSASIIlattice.SwapIndx(2, [h, k, l])

        elif SGSys == 'orthorhombic':
            for h in range(Hmax[0] + 1):
                for k in range(Hmax[1] + 1):
                    for l in range(Hmax[2] + 1):
                        H = []
                        screw_axis = True
                        glide_plane = True
                        if '21' in Ops[0] and k == 0 and l == 0 and h % 2: screw_axis = False
                        if '21' in Ops[1] and h == 0 and l == 0 and k % 2: screw_axis = False
                        if '21' in Ops[2] and k == 0 and h == 0 and l % 2: screw_axis = False
                        if h == 0:
                            if 'b' in Ops[0] and k % 2: glide_plane = False
                            if 'c' in Ops[0] and l % 2: glide_plane = False
                            if 'n' in Ops[0] and (k + l) % 2: glide_plane = False
                            if 'd' in Ops[0] and (k + l) % 4: glide_plane = False

                        if k == 0:
                            if 'a' in Ops[1] and h % 2: glide_plane = False
                            if 'c' in Ops[1] and l % 2: glide_plane = False
                            if 'n' in Ops[1] and (h + l) % 2: glide_plane = False
                            if 'd' in Ops[1] and (h + l) % 4: glide_plane = False

                        if l == 0:
                            if 'b' in Ops[2] and k % 2: glide_plane = False
                            if 'a' in Ops[2] and h % 2: glide_plane = False
                            if 'n' in Ops[2] and (k + h) % 2: glide_plane = False
                            if 'd' in Ops[2] and (k + h) % 4: glide_plane = False

                        if GSASIIlattice.CentCheck(Cent, [h, k, l]) and screw_axis and glide_plane: H = [h, k, l]
                        if H:
                            rdsq = GSASIIlattice.calc_rDsq(H, A)
                            if 0 < rdsq <= dminsq:
                                HKL.append([h, k, l, GSASIIlattice.rdsq2d(rdsq, 6), -1])

        elif SGSys == 'tetragonal': # c axis, a axis and ab diagonal axis
            for l in range(Hmax[2] + 1):
                for k in range(Hmax[1] + 1):
                    for h in range(k, Hmax[0] + 1):
                        H = []
                        screw_axis = True
                        glide_plane = True
                        if '42' in Ops[0] and h == 0 and k == 0 and l % 2: screw_axis = False
                        if ('41' or '43') in Ops [0] and h == 0 and k == 0 and l % 4: screw_axis = False
                        if l == 0:
                            if 'b' in Ops[0] and k % 2: glide_plane = False
                            if 'a' in Ops[0] and h % 2: glide_plane = False
                            if 'n' in Ops[0] and (k + h) % 2: glide_plane = False

                        if len(Ops) > 1:
                            if '21' in Ops[1] and k == 0 and l == 0 and h % 2: screw_axis = False
                            if h == 0:
                                if 'b' in Ops[1] and k % 2: glide_plane = False
                                if 'c' in Ops[1] and l % 2: glide_plane = False
                                if 'n' in Ops[1] and (k + l) % 2: glide_plane = False
                                if 'd' in Ops[1] and (k + l) % 4: glide_plane = False

                        if len(Ops) > 2:
                            if h == k:
                                if ('n' or 'c') in Ops[2] and l % 2: glide_plane = False
                                if 'd' in Ops[2] and (2 * h + l) % 4: glide_plane = False

                        if GSASIIlattice.CentCheck(Cent, [h, k, l]) and screw_axis and glide_plane: H = [h, k, l]
                        if H:
                            rdsq = GSASIIlattice.calc_rDsq(H, A)
                            if 0 < rdsq <= dminsq:
                                HKL.append([h, k, l, GSASIIlattice.rdsq2d(rdsq, 6), -1])

        elif SGSys == 'trigonal': # those with R center
            lmin = -Hmax[2]
            for l in range(lmin, Hmax[2] + 1):
                for k in range(Hmax[1] + 1):
                    hmin = k
                    if l < 0: hmin += 1
                    for h in range(hmin, Hmax[0] + 1):
                        H = []
                        glide_plane = True
                        if len(Ops) > 1:
                            if ('c' or 'n') in Ops[1] and h == k and l % 2: glide_plane = False

                        if GSASIIlattice.CentCheck(Cent, [h, k, l]) and glide_plane: H = [h, k, l]
                        if H:
                            rdsq = GSASIIlattice.calc_rDsq(H, A)
                            if 0 < rdsq <= dminsq:
                                HKL.append([h, k, l, GSASIIlattice.rdsq2d(rdsq, 6), -1])

        elif SGSys == 'hexagonal': # those starts with P
            lmin = 0
            for l in range(lmin, Hmax[2] + 1):
                for k in range(Hmax[1] + 1):
                    hmin = k
                    if l < 0: hmin += 1
                    for h in range(hmin, Hmax[0] + 1):
                        H = []
                        screw_axis = True
                        glide_plane = True
                        if h == 0 and k == 0:
                            if ('31' or '32' or '62' or '64') in Ops[0] and l % 3: screw_axis = False
                            if '63' in Ops[0] and l % 2: screw_axis = False
                            if ('61' or '65') in Ops[0] and l % 6: screw_axis = False

                        if len(Ops) > 1:
                            if 'c' in Ops[1] and (h + k) == 0 and l % 2: glide_plane = False

                        if len(Ops) > 2:
                            if 'c' in Ops[2] and h == k and l % 2: glide_plane = False

                        if GSASIIlattice.CentCheck(Cent, [h, k, l]) and screw_axis and glide_plane: H = [h, k, l]
                        if H:
                            rdsq = GSASIIlattice.calc_rDsq(H, A)
                            if 0 < rdsq <= dminsq:
                                HKL.append([h, k, l, GSASIIlattice.rdsq2d(rdsq, 6), -1])

        else:  # cubic
            for l in range(Hmax[2] + 1):
                for k in range(l, Hmax[1] + 1):
                    for h in range(k, Hmax[0] + 1):
                        H = []
                        screw_axis = True
                        glide_plane = True
                        if k == 0 and l == 0:
                            if ('21' or '42') in Ops[0] and h % 2: screw_axis = False
                            if ('41' or '43') in Ops[0] and h % 4: screw_axis = False

                        if h == 0:
                            if 'b' in Ops[0] and k % 2: glide_plane = False
                            if 'c' in Ops[0] and l % 2: glide_plane = False
                            if 'n' in Ops[0] and (k + l) % 2: glide_plane = False
                            if 'd' in Ops[0] and (k + l) % 4: glide_plane = False

                        if h == k and len(Ops) > 2:
                            if ('c' or 'n') in Ops[2] and l % 2: glide_plane = False
                            if 'd' in Ops[2] and (2 * h + l) % 4: glide_plane = False

                        if GSASIIlattice.CentCheck(Cent, [h, k, l]) and screw_axis and glide_plane: H = [h, k, l]
                        if H:
                            rdsq = GSASIIlattice.calc_rDsq(H, A)
                            if 0 < rdsq <= dminsq:
                                HKL.append([h, k, l, GSASIIlattice.rdsq2d(rdsq, 6), -1])
        return GSASIIlattice.sortHKLd(HKL, True, False)

    def plot_ref_phases(self, winobj):
        if hasattr(self, 'ref_phase'):
            for phase in self.ref_phase:
                if self.curvedict['time series'][phase].isChecked():
                    A = GSASIIlattice.cell2A(self.ref_phase[phase]['cell'])
                    dmin = 2 * np.pi / self.intqaxis[-1]
                    Bravais = self.ref_phase[phase]['Bravais']  # a number
                    HKLD = np.array(GSASIIlattice.GenHBravais(dmin, Bravais, A))  # generate [h, k, l, d, -1]
                    positions = (2 * np.pi / HKLD[:, 3] - self.intqaxis[0]) / (self.intqaxis[-1] - self.intqaxis[0]) * \
                                self.intqaxis.shape[0]
                    self.plot_reflections(winobj=winobj, positions=positions, phase_name=phase, symbol=[],
                                          offset=-.05, hkl=HKLD[:, 0:3],color=self.ref_phase[phase]['color'])
                    if self.checksdict['integrated'].isChecked() and \
                            'original' in self.curve_timelist[0]['integrated'] or \
                            'smoothed' in self.curve_timelist[0]['integrated']:
                        self.plot_reflections(winobj=winobj, positions=2 * np.pi / HKLD[:, 3], phase_name=phase,
                                              symbol='min', offset=0.05, hkl=HKLD[:, 0:3], color='b')

            for phase in self.ref_phase:
                if not self.curvedict['time series'][phase].isChecked():
                    # delete old plot
                    pw = winobj.gdockdict[self.method_name].tabdict['time series'].tabplot
                    for index in reversed(range(len(pw.items))):  # shocked!
                        if isinstance(pw.items[index], pg.PlotDataItem):
                            if pw.items[index].name() == phase: pw.removeItem(pw.items[index])

                    # delete old text
                    for index in reversed(range(len(pw.items))):  # shocked!
                        if isinstance(pw.items[index], pg.TextItem):
                            if pw.items[index].color == fn.mkColor(self.ref_phase[phase]['color']):
                                pw.removeItem(pw.items[index])

    def plot_from_load(self, winobj): # some part can be cut out for a common function
        # if winobj.slideradded == False:
        winobj.setslider()
        # self.slideradded = True
        # winobj.slideradded = True
        # if not hasattr(self,'read_intg'):
        try:
            self.read_intg() # issue?
        except Exception as e:
            print(e)
            print('do integration first')
        else:
            if self.checksdict['time series'].isChecked():
                self.curvedict['time series']['pointer'].setChecked(True)
                if self.parameters['time series']['scale'].choice == 'log10':
                    intdata_ts = np.log10(self.intdata_ts)
                if self.parameters['time series']['scale'].choice == 'sqrt':
                    intdata_ts = np.sqrt(self.intdata_ts)
                if self.parameters['time series']['scale'].choice == 'linear':
                    intdata_ts = self.intdata_ts

                pw = winobj.gdockdict[self.method_name].tabdict['time series'].tabplot
                for index in reversed(range(len(pw.items))):  # shocked!
                    if isinstance(pw.items[index], pg.ImageItem): pw.removeItem(pw.items[index])

                xticklabels = []
                for tickvalue in np.arange(self.intqaxis[0], self.intqaxis[-1], 0.2):  # hope it works
                    xticklabels.append((self.intqaxis.shape[0] / (self.intqaxis[-1] - self.intqaxis[0])
                                        * (tickvalue - self.intqaxis[0]), "{:4.1f}".format(tickvalue))) # 10 is the total width

                xticks = pw.getAxis('bottom')
                xticks.setTicks([xticklabels])

                # if hasattr(self, 'color_bar_ts'):
                #     pw.removeItem(self.img_ts)
                #     self.color_bar_ts.close()

                for item in pw.childItems():
                    if type(item).__name__ == 'ViewBox':
                        item.clear()

                    if type(item).__name__ == 'ColorBarItem':
                        item.close()

                self.img_ts = pg.ImageItem(image=np.transpose(intdata_ts)) # need transpose here ?
                self.img_ts.setZValue(-100) # as long as less than 0
                pw.addItem(self.img_ts)
                color_map = pg.colormap.get('CET-R4')
                intdata_ts[isnan(intdata_ts)] = 0 # surprise!
                intdata_ts[np.isinf(intdata_ts)] = 0
                # this one has grains on plot
                # self.color_bar_ts = pg.ColorBarItem(values=(intdata_ts.min(), intdata_ts.max()), colorMap=color_map)
                # this one is more lovely
                self.color_bar_ts = pg.ColorBarItem(values=(0, intdata_ts.max()), colorMap=color_map)
                self.color_bar_ts.setImageItem(self.img_ts, pw)
                if hasattr(self, 'y_range'):
                    pw.setYRange(self.y_range[0], self.y_range[1])

                # ref_phases
                if hasattr(self,'ref_phase'):
                    for phase in self.ref_phase:
                        self.curvedict['time series'][phase].stateChanged.connect(lambda:self.plot_ref_phases(winobj))
                        self.curvedict['time series'][phase].setChecked(True)

        if hasattr(self, 'refinegpx'):
            cell_para = []
            refl_fcalc = []
            data_num = []
            for hist in self.refinegpx.histograms():
                reflist = self.refinegpx[hist.name]['Reflection Lists']
                if len(reflist) > 1: # if it is more than one main phase, of course this number can be 2, 3, ...
                    # in this case the main phase is the cubic phase
                    data_num.append(int(hist.name[-7:-3])) # only applies to the naming ends with data_num.xy
                    refl_fcalc.append([]) # the hist level
                    cell_para.append([])
                    for phase in reflist: # the phase level
                        refl_fcalc[-1].append([])
                        # cell_para[-1].append([])
                        for k in range(reflist[phase]['RefList'].shape[0]): # the reflection level
                            refl_fcalc[-1][-1].append(reflist[phase]['RefList'][k,9])# * reflist[phase]['RefList'][k,3]) # Fcalc * Multiplicity
                            if (reflist[phase]['RefList'][k,0:3] == [1,0,0]).all(): # d-space, this only holds for cubic phase
                                cell_para[-1].append(reflist[phase]['RefList'][k,4]) # next time, remember to output cell para. etc

            refl_fcalc = np.array(refl_fcalc) # 3-d
            cell_para = np.array(cell_para) # 2-d
            data_num = np.array(data_num) # 1-d
            symbols = ['o', 't', 't1', 't2', 't3', 's', 'p', 'h', 'star', '+', 'd', 'x'] # limited by number of reflections

        if self.checksdict['integrated area-T'].isChecked():
            pw = winobj.gdockdict[self.method_name].tabdict['integrated area-T'].tabplot
            for index in reversed(range(len(pw.items))):  # shocked!
                if isinstance(pw.items[index], pg.ImageItem): pw.removeItem(pw.items[index])

            refl_fsum_phase = [] # sum over all phases for each reflection
            for hist in range(refl_fcalc.shape[0]):
                refl_fsum_phase.append([])
                for refl in range(refl_fcalc.shape[2]):
                    refl_fsum_phase[-1].append(refl_fcalc[hist,:,refl].sum())

            refl_fsum_phase = np.array(refl_fsum_phase)

            for refl in range(refl_fsum_phase.shape[1]): # need to distinguish the symbols !!!
                pw.plot(data_num, refl_fsum_phase[:,refl], symbol = symbols[refl], symbolSize=10, symbolPen='b',
                        name=str(self.refinegpx.histogram(0)['Reflection Lists'][self.refinegpx.phase(0).name]['RefList'][refl,0:3]))
                # above name only applies to this one cubic main phase case

            refl_faverage = [] # average over reflections
            for hist in range(refl_fsum_phase.shape[0]):
                refl_faverage.append(refl_fsum_phase[hist,:].sum() / refl_fsum_phase.shape[1]) # average

            refl_faverage = np.array(refl_faverage)

            pw.plot(data_num, refl_faverage, symbol='x', symbolSize=10, symbolPen='r', name='average', pen=pg.mkPen('r', width=5))
        
        if self.checksdict['centroid-T'].isChecked():
            if not self.checksdict['segregation degree-T'].isChecked(): self.checksdict['segregation degree-T'].setChecked(True)
            pw = winobj.gdockdict[self.method_name].tabdict['centroid-T'].tabplot
            for index in reversed(range(len(pw.items))):  # shocked!
                if isinstance(pw.items[index], pg.PlotDataItem): # pg.PlotCurveItem):
                    pw.removeItem(pw.items[index])

            centroid_refl = [] # centre of mass for each reflection
            for hist in range(cell_para.shape[0]):
                centroid_refl.append([])
                for refl in range(refl_fcalc.shape[2]):
                    centroid_refl[-1].append(np.dot(cell_para[hist,:], refl_fcalc[hist,:,refl]) / refl_fcalc[hist,:,refl].sum())

            centroid_refl = np.array(centroid_refl) # 2-d

            for refl in range(centroid_refl.shape[1]):
                pw.plot(data_num, centroid_refl[:,refl], symbol = symbols[refl], symbolSize=10, symboPen='b',
                        name=str(self.refinegpx.histogram(0)['Reflection Lists'][self.refinegpx.phase(0).name]['RefList'][refl,0:3]))

            refl_fsum_refl = [] # sum over all reflections for each phase
            for hist in range(cell_para.shape[0]):
                refl_fsum_refl.append([])
                for phase in range(cell_para.shape[1]):
                    refl_fsum_refl[-1].append(refl_fcalc[hist,phase,:].sum())

            refl_fsum_refl = np.array(refl_fsum_refl) # 2-d

            centroid_average = []  # centre of mass averaged over reflections
            for hist in range(cell_para.shape[0]):
                centroid_average.append(np.dot(cell_para[hist,:], refl_fsum_refl[hist,:]) / refl_fsum_refl[hist,:].sum())

            centroid_average = np.array(centroid_average)

            pw.plot(data_num, centroid_average, symbol='x', symbolSize=10, symbolPen='r', name='average', pen=pg.mkPen('r', width=5))

        if self.checksdict['segregation degree-T'].isChecked():
            # if not self.checksdict['centroid-T'].isChecked(): self.checksdict['centroid-T'].setChecked(True)
            pw = winobj.gdockdict[self.method_name].tabdict['segregation degree-T'].tabplot
            for index in reversed(range(len(pw.items))):  # shocked!
                if isinstance(pw.items[index], pg.LegendItem): pw.removeItem(pw.items[index])

            segregation_refl = [] # segregation for each refl
            for hist in range(cell_para.shape[0]):
                segregation_refl.append([])
                for refl in range(refl_fcalc.shape[2]):
                    segregation_refl[-1].append(np.dot(np.abs(cell_para[hist,:] - centroid_refl[hist,refl]),
                                                       refl_fcalc[hist,:,refl]) / refl_fcalc[hist,:,refl].sum())

            segregation_refl = np.array(segregation_refl) # 2-d

            for refl in range(segregation_refl.shape[1]):
                pw.plot(data_num, segregation_refl[:,refl], symbol = symbols[refl], symbolSize=10, symboPen='b',
                        name=str(self.refinegpx.histogram(0)['Reflection Lists'][self.refinegpx.phase(0).name]['RefList'][refl,0:3]))

            segregation = [] # average over reflections
            for hist in range(cell_para.shape[0]):
                segregation.append(np.dot(np.abs(cell_para[hist,:] - centroid_average[hist]),
                                          refl_fsum_refl[hist]) / refl_fsum_refl[hist].sum())

            segregation = np.array(segregation)

            pw.plot(data_num, segregation, symbol='x', symbolSize=10, symbolPen='r', name='average', pen=pg.mkPen('r', width=5))

        if hasattr(self, 'slider'):
            if not winobj.slider.value():
                winobj.slider.setValue(winobj.slider.minimum() + 1) # may be not the best way

    def data_scale(self, mode, sub_mode, data_x, data_y):  # for data_process
        if self.parameters[mode]['x axis'].choice == 'q':
            data_x = data_x
        if self.parameters[mode]['x axis'].choice == '2th':
            # with open(self.ponifile, 'r') as f:
            #     wavelength = float(f.readlines()[-1].splitlines()[0].partition(' ')[2]) * 1e10  # in Angstrom
            data_x = np.arcsin(data_x * self.wavelength / 4 / np.pi) * 2 / np.pi * 180
        if self.parameters[mode]['x axis'].choice == 'd':
            data_x = 2 * np.pi / data_x
        if self.parameters[mode]['scale'].choice == 'log10':
            self.data_timelist[0][mode][sub_mode].data = np.transpose([data_x, np.log10(np.abs(data_y))])
        if self.parameters[mode]['scale'].choice == 'sqrt':
            self.data_timelist[0][mode][sub_mode].data = np.transpose([data_x, np.sqrt(data_y)])
        if self.parameters[mode]['scale'].choice == 'linear':
            self.data_timelist[0][mode][sub_mode].data = np.transpose([data_x, data_y])

    def find_peak_conditions(self, data):
        return find_peaks(data,
                          prominence = (self.parameters['integrated']['peak prominence min'].setvalue,
                                        self.parameters['integrated']['peak prominence max'].setvalue),
                          width = (self.parameters['integrated']['peak width min'].setvalue,
                                   self.parameters['integrated']['peak width max'].setvalue),
                          wlen = int(self.parameters['integrated']['window length'].setvalue))

    def read_data_time(self): # for new data collection
        if self.timediff[0] < 0:
            if self.checksdict['raw'].isChecked():
                self.read_data_index(0)

            if self.checksdict['integrated'].isChecked() or self.checksdict['time series'].isChecked():
                if hasattr(self, 'intdata_ts'):
                    self.intdata = self.intdata_ts[0]

        else: # there could be a few xrd data within one index
            if self.checksdict['raw'].isChecked():
                self.read_data_index(self.index)

            if self.checksdict['integrated'].isChecked() or self.checksdict['time series'].isChecked():
                if hasattr(self, 'intdata_ts'):
                    try:
                        self.intdata = self.intdata_ts[self.index]
                    except:
                        print('not matched array size between time and data number?')

        # if self.refinedata: # if not [], read according to para value, not global slider value
        #     self.refine_df = pd.read_csv(self.refinedata[self.parameters['refinement single']['data number'].setvalue],
        #         sep=',', comment='\"', header=None, names=["x", "y_obs", "weight", "y_calc", "y_bkg", "Q"])
        #
        # if self.refinephase:
        #     self.refine_pf = np.loadtxt(self.refinephase[self.parameters['refinement single']['data number'].setvalue])


    def read_data_index(self, sub_index): # for raw img
        file = h5py.File(self.filename, 'r')
        rawdata = file['entry/data/data']
        self.rawdata = np.zeros((rawdata.shape[1],rawdata.shape[2]), dtype='uint32')
        rawdata.read_direct(self.rawdata, source_sel=np.s_[sub_index, :, :])
        self.rawdata = np.log10(self.rawdata).transpose()
        file.close()

    def data_process(self, para_update): # for curves, embody data_timelist, if that curve exists
        # energy/wavelength can be acquired from poni file
        self.read_data_time()  # gives raw and integrated data depending on checks

        self.dynamictitle = self.fileshort + '\n data' + str(self.index + 1) + '\t start:' + self.startime + '\t end:' + self.endtime

        # raw
        if 'show image' in self.curve_timelist[0]['raw']:
            # if self.colormax_set:  # the max color value of time series data
            #     pass
            # else:
            #     self.colormax = np.ceil(self.rawdata[0:500, 0].max())  # part of the detetor area
            #     self.colormax_set = True
            self.colormax = float(self.linewidgets['raw']['color max'].text())
            self.data_timelist[0]['raw']['show image'].image = pg.ImageItem(image=self.rawdata)
            # to avoid the central mask, the size of rawdata is 1065, 1030

        # integrated
        if self.checksdict['integrated'].isChecked() or self.checksdict['time series'].isChecked():
            q_start = int(self.parameters['integrated']['clip head'].setvalue)
            q_ending = int(self.parameters['integrated']['clip tail'].setvalue)
            intqaxis_clipped = self.intqaxis[q_start: -q_ending]
            intdata_clipped = self.intdata[q_start: -q_ending]

        if 'original' in self.curve_timelist[0]['integrated']:
            self.data_scale('integrated', 'original', self.intqaxis, self.intdata)

        if 'normalized to 1' in self.curve_timelist[0]['integrated']:
            self.data_scale('integrated', 'normalized to 1', self.intqaxis, self.intdata / max(self.intdata))

        # if 'normalized to I0 and <font> &mu; </font>d' in self.curve_timelist[0]['integrated']:
        #     pass

        if 'truncated' in self.curve_timelist[0]['integrated']:
            self.data_scale('integrated', 'truncated', [intqaxis_clipped[0],intqaxis_clipped[-1]],
                            [intdata_clipped[0], intdata_clipped[-1]])
            self.data_timelist[0]['integrated']['truncated'].pen = None
            self.data_timelist[0]['integrated']['truncated'].symbol = 'x'
            self.data_timelist[0]['integrated']['truncated'].symbolsize = 20

        if 'smoothed' in self.curve_timelist[0]['integrated']:
            sg_win = int(self.parameters['integrated']['Savitzky-Golay window'].setvalue)
            sg_order = int(self.parameters['integrated']['Savitzky-Golay order'].setvalue)
            if sg_win > sg_order + 1:
                self.intdata_smoothed = scipy.signal.savgol_filter(intdata_clipped,sg_win,sg_order)
                self.data_scale('integrated', 'smoothed', intqaxis_clipped, self.intdata_smoothed)

        if 'find peaks' in self.curve_timelist[0]['integrated']:
            try:
                if hasattr(self, 'intdata_smoothed'):
                    peaks, peak_properties = self.find_peak_conditions(self.intdata_smoothed)
                    self.data_scale('integrated', 'find peaks', intqaxis_clipped[peaks], self.intdata_smoothed[peaks])
                else:
                    peaks, peak_properties = self.find_peak_conditions(self.intdata)
                    self.data_scale('integrated', 'find peaks', intqaxis_clipped[peaks], intdata_clipped[peaks])

                self.data_timelist[0]['integrated']['find peaks'].pen = None
                self.data_timelist[0]['integrated']['find peaks'].symbol = 't'
                self.data_timelist[0]['integrated']['find peaks'].symbolsize = 20
            except:
                print('find peak after you are fine with other steps')
            
        # time series
        if 'pointer' in self.curve_timelist[0]['time series']:
            self.plot_pointer('time series', 0, self.index, 't2', 15)

        # single int.
        if 'pointer' in self.curve_timelist[0]['single peak int.']:
            self.plot_pointer('single peak int.', self.index, 1, 'd', 15)

        # refinement single
        if 'observed' in self.curve_timelist[0]['refinement single']:
            self.data_scale('refinement single', 'observed',
                            4 * np.pi * np.sin(self.refine_df['x'] / 2 / 180 * np.pi) / self.wavelength,
                            self.refine_df['y_obs'])
            self.data_timelist[0]['refinement single']['observed'].pen = None
            self.data_timelist[0]['refinement single']['observed'].symbol = 'x'
            self.data_timelist[0]['refinement single']['observed'].symbolsize = 5

        if 'calculated' in self.curve_timelist[0]['refinement single']:
            self.data_scale('refinement single', 'calculated',
                            4 * np.pi * np.sin(self.refine_df['x'] / 2 / 180 * np.pi) / self.wavelength,
                            self.refine_df['y_calc'])

        if 'difference' in self.curve_timelist[0]['refinement single']:
            self.data_scale('refinement single', 'difference',
                            4 * np.pi * np.sin(self.refine_df['x'] / 2 / 180 * np.pi) / self.wavelength,
                            self.refine_df['y_calc'] - self.refine_df['y_obs'])

        # if self.refinephase:
        #     for ph in range(self.refine_pf.shape[0]):
        #         if 'phase' + str(ph) in self.curve_timelist[0]['refinement single']:
        #             nonzero = np.where(self.refine_pf[ph,::] != 0)
        #             self.data_scale('refinement single', 'phase' + str(ph),
        #                             4 * np.pi * np.sin(np.array(self.refine_df['x'])[nonzero] / 2 / 180 * np.pi) / self.wavelength,
        #                             self.refine_pf[ph, nonzero][0])
                    # self.data_scale('refinement single', 'phase' + str(ph),
                    #                 ma.array(4 * np.pi * np.sin(self.refine_df['x'] / 2 / 180 * np.pi) / self.wavelength,
                    #                          mask=(self.refine_pf[ph,::] == 0)),
                    #                 ma.array(self.refine_pf[ph,::], mask=(self.refine_pf[ph,::] == 0)))

class XRD_INFORM_1(XRD):
    def __init__(self, path_name_widget):
        super(XRD_INFORM_1, self).__init__(path_name_widget)

    def plot_from_prep(self, winobj):  # do integration; some part should be cut out to make a new function
        # if winobj.slideradded == False:
        winobj.setslider()
        # self.slideradded = True
        # winobj.slideradded = True

        # add: if there is already the file, load it directly
        ai = AzimuthalIntegrator(self.ponifile, (1065, 1030), 75e-6, 4, [3000, ], solid_angle=True)
        result = []
        norm_result = []
        file = h5py.File(self.filename, 'r')
        rawdata = file['entry/data/data']
        mask = np.zeros((rawdata.shape[1], rawdata.shape[2]))
        rawimg = np.zeros((rawdata.shape[1], rawdata.shape[2]), dtype='uint32')
        # this xrd data has to be bind with xas
        # with open(self.ponifile, 'r') as f:
        #     wavelength = float(f.readlines()[-1].splitlines()[0].partition(' ')[2])

        energy = ev2nm / self.wavelength * 10  # wavelength in A, change to nm

        hasxas = False
        for key in winobj.methodict:  # for xas-xrd correlation
            if key[0:3] == 'xas' and key[3:] == self.method_name[3:]:
                self.sync_xas_name = key
                hasxas = True

        if hasxas:
            # position = np.where(winobj.methodict[self.sync_xas_name].entrydata[0, 0, ::] - energy > 0)[0][0]
            # the data point is always the first one in this experiment setup
            I0 = winobj.methodict[self.sync_xas_name].entrydata[::, 1, 0]
            I1 = winobj.methodict[self.sync_xas_name].entrydata[::, 2, 0]
            for index in range(I0.shape[0]):
                rawdata.read_direct(rawimg, source_sel=np.s_[index, :, :])
                mask[rawimg == 2 ** 32 - 1] = 1
                intdata = ai.integrate(rawimg, mask=mask)[0]
                result.append(intdata)
                norm_result.append(
                    intdata / I0[index] / (np.log(I0[index] / I1[index]) + 2.6))
                # Justus method to normalize. the 2.6 is based on if the observation background can be smoothed successfully
                # or basically to make the absorption of the materials reach calculated value, in this case 0.6, assuming
                # thickness and components.
                self.prep_progress.setValue(int((index + 1) / I0.shape[0] * 100))

            self.output_intg(ai.q, result, norm_result, self.wavelength, 1)
            # self.read_intg()
            self.plot_from_load(winobj)
        else:
            print('you need to import a bundled XAS data')

        file.close()

    def time_range(self, winobj): # for new data collection, linked to xas, through winobj--which is added only for this purpose
        for key in winobj.path_name_widget: # to distinguish xrd_1, xrd_2
            if self.fileshort == winobj.path_name_widget[key]['raw file'].text() and \
                    self.ponifile.split('\\')[-1] == winobj.path_name_widget[key]['PONI file'].text():
                self.method_name = key

        if self.entrytimesec == []:
            if os.path.isfile(self.exportfile):
                with h5py.File(self.exportfile, 'r') as f:
                    self.entrytimesec = np.zeros((f['info/abs_time_in_sec'].shape[0], f['info/abs_time_in_sec'].shape[1]), dtype='float')
                    f['info/abs_time_in_sec'].read_direct(self.entrytimesec)
            else:
                hasxas = False
                for key in winobj.methodict: # for xrd-xas correlation
                    if key[0:3] == 'xas' and key[3:] == self.method_name[3:]:
                        self.sync_xas_name = key
                        hasxas = True

                if hasxas: self.entrytimesec = winobj.methodict[self.sync_xas_name].entrytimesec
                else: print('you need to import a bundled XAS data')

        return [self.entrytimesec[0, 0], self.entrytimesec[-1, 1]]

class XRD_INFORM_2(XRD): # 20220660
    def __init__(self, path_name_widget):
        super(XRD_INFORM_2, self).__init__(path_name_widget)
        self.energy_div = 13000
        self.eiger_mark = '_eiger*'
        self.rawdata_path = 'entry/instrument/eiger/data'
        if self.intfile_appendix == '_resultFile.h5': # this mode use the cluster
            self.intfile_appendix = '_resultCluster.h5'
            self.exportfile.replace('File', 'Cluster')

    def read_data_index(self, index): # for raw img
        files = glob.glob(os.path.join(self.directory,'raw',
                                        self.fileshort[0:-6] + self.eiger_mark))
        # if len(self.raw_tot) == 0:
        #     tot_num = [0]
        #     for fi in files:
        #         with h5py.File(fi,'r') as f:
        #             tot_num.append(f[self.rawdata_path].shape[0] + tot_num[-1])
        #
        #     self.raw_tot = tot_num

        if ev2nm / self.wavelength * 10 > self.energy_div: index = index * 2 + 1
        else: index = index * 2

        # for i in range(len(self.raw_tot) - 1):
        #     if self.raw_tot[i] <= index < self.raw_tot[i + 1]:
        # with h5py.File(files[i], 'r') as f:
        with h5py.File(files[0], 'r') as f:
            rawdata = f[self.rawdata_path]
            self.rawdata = np.zeros((rawdata.shape[1],rawdata.shape[2]), dtype='uint32')
            # rawdata.read_direct(self.rawdata, source_sel=np.s_[index - self.raw_tot[i], :, :])
            rawdata.read_direct(self.rawdata, source_sel=np.s_[index, :, :])
            self.rawdata = np.log10(self.rawdata).transpose()

    def plot_from_prep(self, winobj):  # do integration; some part should be cut out to make a new function
        # if winobj.slideradded == False:
        winobj.setslider()
        # result = []
        # norm_result = []
        # file = h5py.File(self.filename, 'r')
        # rawdata = file['entry/instrument/eiger/data']
        # mask = np.zeros((rawdata.shape[1], rawdata.shape[2]))
        # rawimg = np.zeros((rawdata.shape[1], rawdata.shape[2]), dtype='uint32')

        # the following file stores the data needed for normalizationn of xrd according to I0 and mu d
        data4xrd_file = os.path.join(self.directory, 'process', self.fileshort[0:-6] + '_xas4xrd')

        if os.path.isfile(data4xrd_file):
            # the xrd data point is always the first one and the middle one of a xas spectrum
            # the ratio of log(I0/I1) is 1.5 for the middle one and 1.42 for the first one. to make it 0.6, add 2.02
            # Justus method to normalize. the constant is based on if the observation background can be smoothed successfully
            # or basically to make the absorption of the materials reach calculated value, in this case 0.6, assuming
            # thickness and components.
            # in essence, this method allow the intensity normalized by the flux, I0, and the amount of material, mu*d

            with open(data4xrd_file, 'r') as f:
                data4xrd = np.loadtxt(f)

            I0 = data4xrd[:, 1]
            I1 = data4xrd[:, 2]

            with open(self.ponifile, 'r') as f:
                wavelength = float(f.readlines()[-1].splitlines()[0].partition(' ')[2]) * 1e10 # in Angstrom

            if ev2nm / wavelength * 10 < self.energy_div: # low or high energy
                index_sel = np.arange(0,I0.shape[0],2)
                E0 = data4xrd[2,0] # was [0,0] before, but is not always correct !
            else:
                index_sel = np.arange(1, I0.shape[0] + 1, 2)
                E0 = data4xrd[1,0]

            self.wavelength = ev2nm / E0 * 10 # in Angstrom

            if not os.path.isdir(self.exportdir): os.mkdir(self.exportdir)

            # if np.abs(self.wavelength - wavelength) > .00001:
            #     ponifile_update = os.path.join(self.exportdir, self.fileshort + '_{}eV.poni'.format(int(E0)))
            #     ponifile_short = ponifile_update.split('\\')[-1]
            #     with open(self.ponifile, 'r') as f:
            #         poni = f.readlines()
            #
            #     poni[-1] = 'Wavelength: {:16e}'.format(self.wavelength * 1e-10)
            #
            #     with open(ponifile_update, 'w') as f:
            #         f.writelines(poni)
            #
            # else:
            #     ponifile_short = self.ponifile.split('\\')[-1]
            #
            # ai = AzimuthalIntegrator(ponifile_short, (1065, 1030), 75e-6, 4, [3000, ], solid_angle=True)
            # for index in index_sel:
            #     rawdata.read_direct(rawimg, source_sel=np.s_[index, :, :])
            #     mask[rawimg == 2 ** 32 - 1] = 1
            #     intdata = ai.integrate(rawimg, mask=mask)[0] # time consuming
            #     result.append(intdata)
            #     norm_result.append(intdata / I0[index] / (np.log(I0[index] / I1[index]) + 2.02))
            #     self.prep_progress.setValue(int((index + 1) / index_sel.shape[0] * 100))
            #
            # self.output_intg(ai.q, result, norm_result, self.wavelength)

            clusterfile = os.path.join(self.directory, 'process', self.fileshort[0:-6] + self.intfile_appendix)
            if not os.path.isfile(clusterfile):
                self.integrate_Cluster(self.ponifile.split('\\')[-1], clusterfile)
                # with h5py.File(clusterfile, 'r+') as f:
                #     f.create_dataset('wavelength from poni', dtype='float32', data=wavelength)

            with h5py.File(clusterfile, 'r') as f:
                q = np.zeros(f['q'].shape[-1], dtype='float32')
                f['q'].read_direct(q)
                wavelength = f['wavelength'][()] * 1e10  # critical, in Angstrom
                # shit! the detector records one more data for MAFAPbI_DMF_coat021!
                # attention that for this data, the normalization might not be accurate
                # there are 1182 xas while there are 1183 xrd
                if self.fileshort == 'MAFAPbI_DMF_coat021_eiger':
                    turning_point = 262 * 2 # read from the incorrectly output xrd data
                    r1 = np.zeros((turning_point, q.shape[0]), dtype='float32')
                    r2 = np.zeros((I0.shape[0] - turning_point, q.shape[0]), dtype='float32')
                    f['results'].read_direct(r1, source_sel=np.s_[0:turning_point,::])
                    f['results'].read_direct(r2, source_sel=np.s_[turning_point + 1::,::])
                    raw_result = np.concatenate((r1,r2), axis=0)
                elif self.fileshort == 'MAPbBrI_2ME_DMF_coat006a_eiger':
                    turning_point = 48 * 2 # read from the incorrectly output xrd data
                    r1 = np.zeros((turning_point, q.shape[0]), dtype='float32')
                    r12 = np.zeros((2, q.shape[0]), dtype='float32')
                    r2 = np.zeros((I0.shape[0] - turning_point - 2, q.shape[0]), dtype='float32')
                    f['results'].read_direct(r1, source_sel=np.s_[0:turning_point,::])
                    f['results'].read_direct(r12, source_sel=np.s_[turning_point - 2:turning_point,::])
                    f['results'].read_direct(r2, source_sel=np.s_[turning_point + 1::,::])
                    raw_result = np.concatenate((r1,r12,r2), axis=0)           
                elif self.fileshort == 'Br_I_2syringe_coat017c_XAS-XRD_eiger':
                    turning_point = 26 * 2 # read from the incorrectly output xrd data
                    r1 = np.zeros((turning_point, q.shape[0]), dtype='float32')
                    r2 = np.zeros((I0.shape[0] - turning_point, q.shape[0]), dtype='float32')
                    f['results'].read_direct(r1, source_sel=np.s_[0:turning_point,::])
                    f['results'].read_direct(r2, source_sel=np.s_[turning_point + 1::,::])
                    raw_result = np.concatenate((r1,r2), axis=0)
                elif self.fileshort == 'MAPbBrI_2ME_DMSO_coat013a_eiger': # same length of I0 and diffraction data
                    turning_point = 276 * 2  # read from the incorrectly output xrd data
                    r1 = np.zeros((turning_point, q.shape[0]), dtype='float32') # sum of r must be length of I0
                    r2 = np.zeros((1, q.shape[0]), dtype='float32')
                    r12 = np.zeros((I0.shape[0] - turning_point - 1, q.shape[0]), dtype='float32')
                    f['results'].read_direct(r1, source_sel=np.s_[0:turning_point, ::])
                    f['results'].read_direct(r2, source_sel=np.s_[-2, ::])
                    f['results'].read_direct(r12, source_sel=np.s_[turning_point + 1::, ::])
                    raw_result = np.concatenate((r1, r12, r2), axis=0)
                else:
                    raw_result = np.zeros((f['results'].shape[0], q.shape[0]), dtype='float32')
                    f['results'].read_direct(raw_result)

            # shutil.copy(clusterfile, self.exportfile)
            # q = 2 pi / d = 4 pi sin theta / lambda; q lambda = q' lambda'
            self.output_intg(q * wavelength / self.wavelength * 10, # now in inverse nm, to be consistent with olde files
                             raw_result[index_sel, :],
                             raw_result[index_sel, :] / I0[index_sel, None] / \
                             (np.log(I0[index_sel, None] / I1[index_sel, None]) + 2.02), # so so critical
                             self.wavelength, 50)

            self.plot_from_load(winobj)

        else:
            print('you need to process a bundled XAS data first')

        # file.close()

    def time_range(self, winobj):
        for key in winobj.path_name_widget: # to distinguish xrd_1, xrd_2
            if self.fileshort == winobj.path_name_widget[key]['raw file'].text():
                if 'PONI file' in winobj.path_name_widget[key]:
                    if self.ponifile.split('\\')[-1] == winobj.path_name_widget[key]['PONI file'].text():
                        self.method_name = key

        if self.entrytimesec == []:
            if os.path.isfile(self.exportfile):
                with h5py.File(self.exportfile, 'r') as f:
                    self.entrytimesec = np.zeros((f['info/abs_time_in_sec'].shape[0], f['info/abs_time_in_sec'].shape[1]), dtype='float')
                    f['info/abs_time_in_sec'].read_direct(self.entrytimesec)
            else:
                # hasxas = False
                # for key in winobj.methodict: # for xrd-xas correlation
                #     if key[0:3] == 'xas' and key[3:] == self.method_name[3:]:
                #         self.sync_xas_name = key
                #         hasxas = True
                #
                # if hasxas: self.entrytimesec = winobj.methodict[self.sync_xas_name].entrytimesec
                # else: print('you need to import a bundled XAS data')

                xas_time = os.path.join(self.directory, 'process', self.fileshort[0:-6] + '_time_in_seconds')
                # with open(self.ponifile, 'r') as f:
                #     wavelength = float(f.readlines()[-1].splitlines()[0].partition(' ')[2])
                #
                # if wavelength < 13000:  # low or high energy
                #     index_sel = np.arange(0, I0.shape[0], 2)
                # else:
                #     index_sel = np.arange(1, I0.shape[0], 2)
                #
                if os.path.isfile(xas_time):
                    entrytimesec = np.loadtxt(xas_time)
                    with open(self.ponifile, 'r') as f:
                        wavelength = float(f.readlines()[-1].splitlines()[0].partition(' ')[2])

                    if wavelength < self.energy_div:  # low or high energy
                        self.entrytimesec = entrytimesec[0::2,:]
                    else:
                        self.entrytimesec = entrytimesec[1::2,:]
                else:
                    print('you need to process a bundled XAS data first')

        return [self.entrytimesec[0, 0], self.entrytimesec[-1, 1]]

class XRD_INFORM_3(XRD_INFORM_2):
    def __init__(self, path_name_widget):
        super(XRD_INFORM_3, self).__init__(path_name_widget)
        self.energy_div = 10000
        self.eiger_mark = '_eiger_data*'
        self.rawdata_path = 'entry/data/data'

class XRD_INFORM_2_ONLY(XRD_INFORM_2):
    def __init__(self, path_name_widget):
        super(XRD_INFORM_2_ONLY, self).__init__(path_name_widget)
        if self.intfile_appendix == '_resultFile.h5': # this mode use the cluster
            self.intfile_appendix = '_resultCluster.h5'
            self.exportfile.replace('File', 'Cluster')

    def read_data_index(self, index): # for raw img
        file = h5py.File(self.filename, 'r')
        rawdata = file[self.rawdata_path]
        if self.fileshort == 'MAPbBrI_DMSO_2ME_coat013_eiger':
            if ev2nm / self.wavelength * 10 > self.energy_div: index = index * 2 + 1
            else: index = index * 2

        self.rawdata = np.zeros((rawdata.shape[1],rawdata.shape[2]), dtype='uint32')
        rawdata.read_direct(self.rawdata, source_sel=np.s_[index, :, :])
        self.rawdata = np.log10(self.rawdata).transpose()
        file.close()

    def plot_from_prep(self, winobj):  # do integration; some part should be cut out to make a new function
        # if winobj.slideradded == False:
        winobj.setslider()
        # norm_result = []
        # the following file stores the data needed for normalizationn of xrd according to I0 and mu d
        data4xrd_file = os.path.join(self.directory, 'raw', self.fileshort[0:-6] + '.h5')

        if os.path.isfile(data4xrd_file):
            if os.stat(data4xrd_file).st_size > 1000: # Bytes
                # the xrd data point is always the first one and the middle one of a xas spectrum
                # the ratio of log(I0/I1) is 1.5 for the middle one and 1.42 for the first one. to make it 0.6, add 2.02
                # Justus method to normalize. the constant is based on if the observation background can be smoothed successfully
                # or basically to make the absorption of the materials reach calculated value, in this case 0.6, assuming
                # thickness and components.
                # in essence, this method allow the intensity normalized by the flux, I0, and the amount of material, mu*d
                with h5py.File(data4xrd_file, 'r') as f:
                    data = {}
                    for key in f.keys():
                        if key != 'time':
                            # data[key] = np.zeros(f[key].shape[-1], dtype='float')
                            data[key] = np.zeros((f[key].shape[0],f[key].shape[1]),dtype='float')
                            f[key].read_direct(data[key])

                I0 = (data['I0a'] + data['I0b']).flatten()

                try:
                    I1 = data['I1'].flatten()
                except:
                    I1 = (data['I1a'] + data['I1b']).flatten()

                energy = data['energy'].flatten()
                # with open(self.ponifile, 'r') as f:
                #     wavelength = float(f.readlines()[-1].splitlines()[0].partition(' ')[2]) * 1e10 # in Angstrom
                #
                self.wavelength = ev2nm / energy[0] * 10 # in Angstrom
                if not os.path.isdir(self.exportdir): os.mkdir(self.exportdir)
                # if np.abs(self.wavelength - wavelength) > .00001:
                #     ponifile_update = os.path.join(self.exportdir, self.fileshort + '_{}eV.poni'.format(int(energy[0])))
                #     ponifile_short = self.exportdir.split('\\')[-1] + '/' + ponifile_update.split('\\')[-1]
                #     with open(self.ponifile, 'r') as f:
                #         poni = f.readlines()
                #
                #     poni[-1] = 'Wavelength: {:16e}'.format(self.wavelength * 1e-10)
                #
                #     with open(ponifile_update, 'w') as f:
                #         f.writelines(poni)
                #
                # else:
                #     ponifile_short = self.ponifile.split('\\')[-1]
                clusterfile = os.path.join(self.directory, 'process', self.fileshort[0:-6] + self.intfile_appendix)

                if not os.path.isfile(clusterfile): self.integrate_Cluster(self.ponifile.split('\\')[-1], clusterfile)

                with h5py.File(clusterfile, 'r') as f:
                    q = np.zeros(f['q'].shape[-1], dtype='float32')
                    f['q'].read_direct(q)
                    raw_result = np.zeros((f['results'].shape[0], f['results'].shape[1]), dtype='float32')
                    f['results'].read_direct(raw_result)
                    wavelength = f['wavelength'][()] * 1e10  # critical, in Angstrom
                # for index in range(raw_result.shape[0]):
                #     norm_result.append(raw_result[index,:] / I0[index] / (np.log(I0[index] / I1[index]) + 2.02))
                #
                # shutil.copy(clusterfile,self.exportfile)
                if raw_result.shape[0] < I0.shape[0]: # this is another menifestation of unreliabel data collection
                    # let's discard the last one, maybe few, of I0
                    I0 = I0[0:raw_result.shape[0]]
                    I1 = I1[0:raw_result.shape[0]]
                    self.entrytimesec = self.entrytimesec[0:raw_result.shape[0],:]

                self.output_intg(q * wavelength / self.wavelength * 10, # now in inverse nm, to be consistent with olde files
                                 raw_result,
                                 raw_result / I0[:,None] / (np.log(I0[:,None] / I1[:,None]) + 2.02),
                                 self.wavelength, 50)
                # self.plot_from_load(winobj)
            else:
                print('be careful, this file has no binding xas file, the energy and time are both uncertain')
                print('the energy is taken directly from the poni file instead of xas file')
                print('time is based on the xrd file time and predefined time intervals')

                if not os.path.isdir(self.exportdir): os.mkdir(self.exportdir)
                clusterfile = os.path.join(self.directory, 'process', self.fileshort[0:-6] + self.intfile_appendix)
                if not os.path.isfile(clusterfile): self.integrate_Cluster(self.ponifile.split('\\')[-1], clusterfile)
                with h5py.File(clusterfile, 'r') as f:
                    q = np.zeros(f['q'].shape[-1], dtype='float32')
                    f['q'].read_direct(q)
                    raw_result = np.zeros((f['results'].shape[0], f['results'].shape[1]), dtype='float32')
                    f['results'].read_direct(raw_result)
                    wavelength = f['wavelength'][()] * 1e10  # critical, in Angstrom

                if self.fileshort == 'MAPbBrI_DMSO_2ME_coat013_eiger':
                    if ev2nm / (self.wavelength / 10) < self.energy_div: # here the self.wavelength depends on poni file, not xas file
                        self.output_intg(q * wavelength / self.wavelength * 10,
                                         raw_result[0::2,:], raw_result[0::2,:], self.wavelength, 50)
                    else:
                        self.output_intg(q * wavelength / self.wavelength * 10,
                                         raw_result[1::2, :], raw_result[1::2, :], self.wavelength, 50)

                else:
                    self.output_intg(q * wavelength / self.wavelength * 10, raw_result, raw_result, self.wavelength, 50)

            self.plot_from_load(winobj)
        else:
            print('you need to double check your file name')

        # file.close()

    def time_range(self, winobj):
        for key in winobj.path_name_widget: # to distinguish xrd_1, xrd_2
            if self.fileshort == winobj.path_name_widget[key]['raw file'].text():
                if 'PONI file' in winobj.path_name_widget[key]:
                    if self.ponifile.split('\\')[-1] == winobj.path_name_widget[key]['PONI file'].text():
                        self.method_name = key

        if self.entrytimesec == []:
            if os.path.isfile(self.exportfile):
                with h5py.File(self.exportfile, 'r') as f:
                    self.entrytimesec = np.zeros((f['info/abs_time_in_sec'].shape[0], f['info/abs_time_in_sec'].shape[1]), dtype='float')
                    f['info/abs_time_in_sec'].read_direct(self.entrytimesec)
            else: # read in entrytimesec from xas raw file
                data4xrd_file = os.path.join(self.directory, 'raw', self.fileshort[0:-6] + '.h5')
                if os.path.isfile(data4xrd_file):
                    if os.stat(data4xrd_file).st_size > 1000:
                        with h5py.File(data4xrd_file, 'r') as f:
                            endtime = time.mktime(
                                datetime.strptime(f['time'][-1].decode(), '%Y-%m-%d %H:%M:%S.%f').timetuple())
                            # f['integration_time'].read_direct(durations)

                            duration = float(winobj.path_name_widget[self.method_name]['time interval(ms)'].text()) # in ms
                            data_length = f['integration_time'].shape[-1]
                        # self.entrytimesec = (startime + np.array([np.arange(0,len(durations)),
                        #                                         np.arange(1,len(durations) + 1)]) * durations[0] * 1e-6).T # check the unit!!!
                        self.entrytimesec = (endtime - np.array([np.arange(data_length, 0, -1), # double check the time size!!!
                                                                  np.arange(data_length - 1, -1, -1 )]) * duration / 1e3).T  # check the unit!!!

                    else:
                        startime = os.path.getctime(self.filename) # in absolute seconds
                        duration = float(winobj.path_name_widget[self.method_name]['time interval(ms)'].text()) # in ms
                        with h5py.File(self.filename, 'r') as f:
                            data_length = f[self.rawdata_path].shape[0]

                        if self.fileshort == 'MAPbBrI_DMSO_2ME_coat013_eiger':
                             self.entrytimesec = (startime + np.array([np.arange(data_length / 2),
                                                                       np.arange(1,data_length / 2 + 1)]) * duration / 1e3).T
                        else:
                            self.entrytimesec = (startime + np.array([np.arange(data_length),
                                                                      np.arange(1,data_length + 1)]) * duration / 1e3).T

                else:
                    print('you need to double check your file name')

        return [self.entrytimesec[0, 0], self.entrytimesec[-1, 1]]

class XRD_INFORM_3_ONLY(XRD_INFORM_2_ONLY):
    def __init__(self, path_name_widget):
        super(XRD_INFORM_3_ONLY, self).__init__(path_name_widget)
        self.energy_div = 10000
        self.eiger_mark = '_eiger_data*'
        self.rawdata_path = 'entry/data/data'

    def read_data_index(self, index): # for raw img
        try:
            files = glob.glob(os.path.join(self.directory,'raw',
                                           self.fileshort.split('_eiger')[0]
                                           + self.eiger_mark))
        except: # caution: this may fail!
            files = glob.glob(os.path.join(self.directory,'raw',
                                           self.fileshort + self.eiger_mark))

        if len(self.raw_tot) == 0:
            tot_num = [0]
            for fi in files:
                with h5py.File(fi,'r') as f:
                    tot_num.append(f[self.rawdata_path].shape[0] + tot_num[-1])

            self.raw_tot = tot_num

        for i in range(len(self.raw_tot) - 1):
            if self.raw_tot[i] <= index < self.raw_tot[i + 1]:
                with h5py.File(files[i], 'r') as f:
                    rawdata = f[self.rawdata_path]
                    self.rawdata = np.zeros((rawdata.shape[1],rawdata.shape[2]), dtype='uint32')
                    rawdata.read_direct(self.rawdata, source_sel=np.s_[index - self.raw_tot[i], :, :])
                    self.rawdata = np.log10(self.rawdata).transpose()

class XRD_INFORM_1_ONLY(XRD):
    def __init__(self, path_name_widget):
        super(XRD_INFORM_1_ONLY, self).__init__(path_name_widget)

    def plot_from_prep(self, winobj):  # do integration; some part should be cut out to make a new function
        # if winobj.slideradded == False:
        winobj.setslider()
        # self.slideradded = True
        # winobj.slideradded = True

        # add: if there is already the file, load it directly
        ai = AzimuthalIntegrator(self.ponifile, (1065, 1030), 75e-6, 4, [3000, ], solid_angle=True)
        result = []
        norm_result = []

        # with open(self.ponifile, 'r') as f:
        #     wavelength = float(f.readlines()[-1].splitlines()[0].partition(' ')[2])

        energy = ev2nm / self.wavelength * 10  # wavelength in A, change to nm

        with h5py.File(self.filename[0:-17] + '1.h5', 'r') as f:  # this is so so so special a case!!!
            I0 = f['spectrum0'][:, 1]
            I1 = f['spectrum0'][:, 2]

        datafiles = glob.glob(self.filename[0:-6] + "*")
        index_total = len(I0)
        for k in range(len(datafiles)):
            with h5py.File(datafiles[k], 'r') as file:
                rawdata = file['entry/data/data']
                mask = np.zeros((rawdata.shape[1], rawdata.shape[2]))
                rawimg = np.zeros((rawdata.shape[1], rawdata.shape[2]), dtype='uint32')
                for index in range(min(1000, index_total)):
                    rawdata.read_direct(rawimg, source_sel=np.s_[index, :, :])
                    mask[rawimg == 2 ** 32 - 1] = 1
                    intdata = ai.integrate(rawimg, mask=mask)[0]
                    result.append(intdata)
                    total_index = index + len(I0) - index_total
                    norm_result.append(
                        intdata / I0[total_index] / np.log(I1[total_index] / I0[total_index]))  # Mahesh's method to normalize
                    self.prep_progress.setValue(int((total_index + 1) / I0.shape[0] * 100))

            index_total -= 1000

        self.output_intg(ai.q, result, norm_result, self.wavelength, 1)
        # self.read_intg()
        self.plot_from_load(winobj)

    def time_range(self, winobj): #
        for key in winobj.path_name_widget: # to distinguish xrd_1, xrd_2
            if self.fileshort == winobj.path_name_widget[key]['raw file'].text() and \
                    self.ponifile.split('\\')[-1] == winobj.path_name_widget[key]['PONI file'].text():
                self.method_name = key

        if self.entrytimesec == []:
            if os.path.isfile(self.exportfile):
                with h5py.File(self.exportfile, 'r') as f:
                    self.entrytimesec = np.zeros((f['info/abs_time_in_sec'].shape[0], f['info/abs_time_in_sec'].shape[1]), dtype='float')
                    f['info/abs_time_in_sec'].read_direct(self.entrytimesec)
            else:
                with h5py.File(self.filename[0:-17] + '1.h5', 'r') as f: # this is so so so special a case!!!
                    endtimesecond = time.mktime(datetime.strptime(f['timestamp0'][()].decode(), '%c').timetuple())
                    datalen = f['spectrum0'].shape[0]

                endtime_array = endtimesecond - np.flip(np.arange(datalen)) * 0.18 # 0.18 s per shot
                self.entrytimesec = np.array([endtime_array - 0.18, endtime_array]).transpose()

        return [self.entrytimesec[0, 0], self.entrytimesec[-1, 1]]

class XRD_BATTERY_1(XRD):
    def __init__(self, path_name_widget):
        super(XRD_BATTERY_1, self).__init__(path_name_widget)

    def read_data_time(self): # read the data according to self.index, the speed may be slower when reading both files
        timespan = self.entrytimesec[self.index, 1] - self.entrytimesec[self.index, 0]
        self.file = h5py.File(self.filename, 'r')  # not the best solution. close all files that opened!
        self.filekeys = list(self.file.keys())
        for key in self.filekeys:  # this prevent some entry without an end_time!
            if 'start_time' in self.file[key] and 'end_time' in self.file[key] \
                    and 'albaem-02_ch1' in self.file[key + '/measurement']:
                xrdnumber = int(self.file[key + '/measurement/albaem-02_ch1'].shape[0])
                break

        timeperxrd = timespan / xrdnumber
        if self.timediff[0] < 0:
            if self.checksdict['raw'].isChecked():
                self.read_data_index(0, 0)
            if self.checksdict['integrated'].isChecked() or self.checksdict['time series'].isChecked():
                self.intdata = self.intdata_ts[0]

        else: # there could be a few xrd data within one index
            sub_index = int(np.floor(self.timediff[self.index] / timeperxrd)) # start from 0
            if sub_index > xrdnumber - 1:
                sub_index = xrdnumber - 1

            if self.checksdict['raw'].isChecked():
                self.read_data_index(self.index, sub_index)

            if self.checksdict['integrated'].isChecked() or self.checksdict['time series'].isChecked():
                self.intdata = self.intdata_ts[self.index * xrdnumber + sub_index]

    def read_data_index(self, index, sub_index): # update self.rawdata, only accept multi data per entry, e.g. 2 data/entry
        rawfileentry = self.filekeys[index] + '/measurement/eiger_xrd_datafile'
        rawfilename = os.path.join(self.directory, 'raw', self.file[rawfileentry][()].decode())
        rawfile = h5py.File(rawfilename, 'r')
        rawdata = rawfile['entry/data/data']
        self.rawdata = np.zeros((rawdata.shape[1],rawdata.shape[2]), dtype='uint32')
        rawdata.read_direct(self.rawdata, source_sel=np.s_[sub_index, :, :])
        self.rawdata = np.log10(self.rawdata).transpose()
        rawfile.close()

    def plot_from_prep(self, winobj):  # do integration; some part should be cut out to make a new function
        # if winobj.slideradded == False:
        winobj.setslider()
        # self.slideradded = True
        # winobj.slideradded = True

        # add: if there is already the file, load it directly
        ai = AzimuthalIntegrator(self.ponifile, (1065, 1030), 75e-6, 4, [3000, ], solid_angle=True)
        result = []
        norm_result = []

        index = 0
        for key in self.filekeys:
            rawfileentry = key + '/measurement/eiger_xrd_datafile'
            rawfilename = os.path.join(self.directory, 'raw', self.file[rawfileentry][()].decode())
            rawfile = h5py.File(rawfilename, 'r')
            rawdata = rawfile['entry/data/data']
            rawimg = np.zeros((rawdata.shape[1], rawdata.shape[2]), dtype='uint32')
            I0 = np.array(list(self.file[key + '/measurement/albaem-02_ch1'])) + \
                 np.array(list(self.file[key + '/measurement/albaem-02_ch2']))
            I1 = np.array(list(self.file[key + '/measurement/albaem-02_ch3'])) + \
                 np.array(list(self.file[key + '/measurement/albaem-02_ch4']))
            for sub_index in range(rawdata.shape[0]):
                rawdata.read_direct(rawimg, source_sel=np.s_[sub_index, :, :])
                mask = np.zeros((rawdata.shape[1], rawdata.shape[2]))
                mask[rawimg == 2 ** 32 - 1] = 1
                intdata = ai.integrate(rawimg, mask=mask)[0]
                result.append(intdata)
                norm_result.append(
                    intdata / I0[sub_index] / np.log(I1[sub_index] / I0[sub_index]))  # Mahesh's method to normalize

            self.prep_progress.setValue(int((index + 1) / self.entrytimesec.shape[0] * 100))
            index += 1
            rawfile.close()

        # with open(self.ponifile, 'r') as f:
        #     wavelength = float(f.readlines()[-1].splitlines()[0].partition(' ')[2])

        self.output_intg(ai.q, result, norm_result, self.wavelength, 1) # output integrated data
        # self.read_intg() # ini self.intdata_ts and self.intqaxis
        self.plot_from_load(winobj)

    def time_range(self, winobj):  # need to rewrite for none-h5file
        for key in winobj.path_name_widget:  # to distinguish xrd_1, xrd_2
            if self.fileshort == winobj.path_name_widget[key]['raw file'].text() and \
                    self.ponifile.split('\\')[-1] == winobj.path_name_widget[key]['PONI file'].text():
                self.method_name = key

        if self.entrytimesec == []:
            if os.path.isfile(self.exportfile):
                with h5py.File(self.exportfile, 'r') as f:
                    self.entrytimesec = np.zeros((f['info/abs_time_in_sec'].shape[0], f['info/abs_time_in_sec'].shape[1]), dtype='float')
                    f['info/abs_time_in_sec'].read_direct(self.entrytimesec)
            else:
                self.file = h5py.File(self.filename, 'r')  # not the best solution. close all files that opened!
                self.filekeys = list(self.file.keys())
                for key in self.filekeys:  # this prevent some entry without an end_time!
                    if 'start_time' in self.file[key] and 'end_time' in self.file[key] \
                            and 'albaem-02_ch1' in self.file[key + '/measurement']:
                        self.entrytimesec.append([
                            time.mktime(datetime.strptime(self.file[key + '/start_time'][()].decode(),
                                                          '%Y-%m-%dT%H:%M:%S.%f').timetuple()),
                            time.mktime(datetime.strptime(self.file[key + '/end_time'][()].decode(),
                                                          '%Y-%m-%dT%H:%M:%S.%f').timetuple())])
                    else:  # error proof, and only chance of effective error proof: filter the bad entries when setslider
                        del self.filekeys[self.filekeys.index(key)] # really ???

                self.entrytimesec = np.array(self.entrytimesec)

        return [self.entrytimesec[0, 0], self.entrytimesec[-1, 1]]

class Optic(Methods_Base):
    def __init__(self, path_name_widget):
        super(Optic, self).__init__()
        self.directory = path_name_widget['directory'].text()
        self.fileshort = path_name_widget['raw file'].text()
        self.exportdir = os.path.join(self.directory, 'process', self.fileshort + '_Optic_data')
        if not os.path.isdir(self.exportdir): os.mkdir(self.exportdir)
        self.align_data = int(path_name_widget['align data number'].text())
        self.to_time = time.mktime(
            datetime.strptime(path_name_widget['to time'].text(), '%Y-%m-%dT%H:%M:%S').timetuple())  # in second
        self.aligned = False
        self.pre_ts_btn_text = 'Align data to time(Ctrl+D)'
        # if int(self.directory.split('202')[1][0]) > 1: # new spectrometer after 2021 (202 1)
        self.channelnum = 2068
        ch = np.arange(0, self.channelnum, dtype='float')
        A = 194.85605
        B = 0.5487633
        C = -5.854113E-05
        D = 3.1954566E-09
        self.channels = A + ch * B + ch ** 2 * C + ch ** 3 * D # calibrated numbers
        # else: # old spectrometer
        #     self.channelnum = 2048
        #     self.channels = np.linspace(200, 1100, self.channelnum) # an estimation

        self.entrytimesec = []
        self.method_name = []

    def read_data_time(self, datafile):
        if os.path.exists(os.path.join(self.directory, 'raw', 'PL_Reflection')):
            if 'coat010' in self.fileshort:
                filename = os.path.join(self.directory, 'raw', 'PL_Reflection',
                                        'MFPI_2MACl_coat010_' + self.fileshort.split('_')[-1] + '.qvd')
                timename = os.path.join(self.directory, 'raw', 'PL_Reflection',
                                        'MFPI_coat010_' + self.fileshort.split('_')[-1] + '.qvt')
                timefile = os.path.join(self.directory, 'raw', 'PL_Reflection', 'MFPI_coat010_info.txt')
            else:
                filename = os.path.join(self.directory, 'raw', 'PL_Reflection', self.fileshort + '.qvd')
                timename = os.path.join(self.directory, 'raw', 'PL_Reflection', self.fileshort + '.qvt')
                timefile = os.path.join(self.directory, 'raw', 'PL_Reflection', str(self.fileshort.split('_')[:-1]) + '_info.txt')

        else: # needs tailored for each experiment!!!
            filename = os.path.join(self.directory, 'raw', self.fileshort + '.qvd')
            timename = os.path.join(self.directory, 'raw', self.fileshort + '.qvt')
            timefile = os.path.join(self.directory, 'raw', self.fileshort + '_info.txt')

        if os.path.isfile(filename):
            data = self.read_qvd(filename)
            datatime = self.read_qvt(timename)
            self.data = (data.T / datatime[:,1]).T
            with h5py.File(datafile, 'w') as f:
                f.create_dataset('raw', data=np.array(self.data))
                f.create_dataset('wave length(nm)', data=self.channels)

            if self.entrytimesec == []:  # start, end time in seconds
                if os.path.isfile(timefile):  # new time stamp since 20220660
                    with open(timefile, 'r') as f:
                        lines = f.readlines()
                    start_time = time.mktime(
                        datetime.strptime(lines[1].split('stamp: ')[1][:-1], '%Y-%m-%d %H:%M:%S').timetuple())
                else:
                    end_time = os.path.getmtime(timename)  # estimation, in seconds
                    start_time = end_time - datatime[-1, 0] / 1000 - datatime[-1, 1] / 1000  # original in milisecond

                self.entrytimesec = start_time + np.stack((datatime[:, 0], datatime[:, 0] + datatime[:, 1])).T / 1000
                # for index in range(self.data.shape[0]):
                #     self.entrytimesec.append(start_time + datatime[index, 0] / 1000 + [0, datatime[index, 1] / 1000])

                self.entrytimesec = np.array(self.entrytimesec)  # , dtype=int)
                with h5py.File(datafile, 'a') as f:
                    f.create_dataset('time in seconds', data=self.entrytimesec, dtype='float64')
                    # here it will automatically choose a dtype which is float64!!! so critical!!!
        else:
            print('double check with your file name!')

    def read_qvt(self, qvtfilename):
        # qvt: double (time in ms), int16 (time span/data), double (dummy)
        struct_fmt = '=dHd'  # very important to get the format correct
        struct_len = struct.calcsize(struct_fmt)
        struct_unpack = struct.Struct(struct_fmt).unpack_from
        timedata = []
        with open(qvtfilename, "rb") as f:
            while True:
                data = f.read(struct_len)
                if not data: break
                s = list(struct_unpack(data))  # change from a tuple to a list
                timedata.append(s)
        return np.array(timedata)

    def read_qvd(self, qvdfilename):
        struct_fmt = '=' + str(self.channelnum) + 'd'
        # very important to get the format correct, the old spectrometer is 2068 pixel, the new one is also 2068!
        struct_len = struct.calcsize(struct_fmt)
        struct_unpack = struct.Struct(struct_fmt).unpack_from
        timedata = []
        with open(qvdfilename, "rb") as f:
            while True:
                data = f.read(struct_len)
                if not data: break
                s = list(struct_unpack(data))  # change from a tuple to a list
                timedata.append(s)
        return np.array(timedata)

    def plot_from_prep(self, winobj):
        winobj.setslider()
        if self.align_data: # and not self.aligned: # there is a little problem here: what if you want to align it again?
            try:
                time_diff = self.entrytimesec[self.align_data, 0] - self.to_time
                self.entrytimesec = self.entrytimesec - time_diff
                self.aligned = True
            except Exception as e:
                print(e)
                print('check your input')

    def plot_optic_2D(self, step, data_norm, pw):
        xticklabels = []
        for tickvalue in np.arange(self.channels[0], self.channels[-1],step):
            tickpos = np.where(self.channels - tickvalue >= 0)[0][0]
            if tickpos:
                xticklabels.append((tickpos, "{:4.1f}".format(tickvalue)))

        xticks = pw.getAxis('bottom')
        xticks.setTicks([xticklabels])

        # if hasattr(self, 'color_bar_ts'):
        #     pw.removeItem(self.img_ts)
        #     self.color_bar_ts.close()

        for item in pw.childItems():
            if type(item).__name__ == 'ViewBox':
                item.clear()

            if type(item).__name__ == 'ColorBarItem':
                item.close()

        # for t in range(data_norm.shape[0] - 1): only works for bad data!!!
        #     # if len(np.where(data_norm[t,:] == max(data_norm[t,:]))[0]) > \
        #     #     int(self.linedit['time series']['filter criterion']) and t > 0:
        #     if max(data_norm[t,:]) > np.log10(float(self.linedit['time series']['filter criterion'])) and t > 0 \
        #             and t < data_norm.shape[0]:
        #         data_norm[t,:] = (data_norm[t - 1,:] + data_norm[t + 1, :]) / 2 # linear interpolation between adjacent data

        img_ts = pg.ImageItem(image=data_norm.transpose()) # need log here?
        img_ts.setZValue(-100)  # as long as less than 0
        pw.addItem(img_ts)
        color_map = pg.colormap.get('CET-R4')
        # data_norm[isnan(data_norm)] = 0
        # data_norm = ma.array(data_norm, mask=np.isinf(data_norm))
        # mode = 'time series'
        # self.color_bar_ts = pg.ColorBarItem(values=(float(self.linedit[mode]['z min']),
        #                                             float(self.linedit[mode]['z max'])), colorMap=color_map)
        data_norm[isnan(data_norm)] = 0
        inf_ele = np.where(np.isfinite(data_norm) == False)
        for k in range(len(inf_ele[0])):
            data_norm[inf_ele[0][k],inf_ele[1][k]] = 0

        # limit the range to xas/xrd for more colorful
        view_range = np.s_[max(0,int(pw.viewRange()[1][0])):
                           min(data_norm.shape[0] - 1, int(pw.viewRange()[1][1])),:]
        self.color_bar_ts = pg.ColorBarItem(values=(data_norm[view_range].min(), data_norm[view_range].max()),colorMap=color_map)
        self.color_bar_ts.setImageItem(img_ts, pw)
        if hasattr(self, 'y_range'):
            pw.setYRange(self.y_range[0], self.y_range[1])

class PL(Optic):
    def __init__(self, path_name_widget):
        super(PL, self).__init__(path_name_widget)
        self.availablemethods = ['raw', 'time series', 'fit-T']
        self.availablecurves['raw'] = ['show','fit','dark']
        self.availablecurves['time series'] = ['pointer']
        self.availablecurves['fit-T'] = ['Height', 'FWHM']
        self.dark = []
        self.axislabel = {'raw':{'bottom':'wavelength/nm',
                                 'left':'log10(intensity)'},
                          'time series': {'bottom': 'wavelength/nm',
                                  'left': 'Data number'},
                          'fit-T': {'bottom':'Data number',
                                     'left':''}}
        self.actions = {'time series': {'update y,z range (Ctrl+0)': self.update_range,
                                        # 'Export reference (Ctrl+X)': self.export_norm
                                        }}
        self.linedit = {'time series': {'z min': '-2',
                                        'z max': '0.1',
                                        'y min':'0',
                                        'y max':'1000',}}

        self.ini_data_curve_color()  # this has to be at the end line of each method after a series of other attributes

    def export_norm(self, winobj):
        with h5py.File(os.path.join(self.exportdir, self.fileshort + '_pl_data.h5'), 'a') as f:
            if len(self.dark) > 0:
                if 'pl_dark' in list(f.keys()): del f['pl_dark']
                f.create_dataset('pl_dark', data=np.array(self.dark), dtype='float32')

            if hasattr(self, 'aligned'):
                if self.aligned:
                    del f['time in seconds']
                    f.create_dataset('time in seconds', self.entrytimesec)

    def plot_from_load(self, winobj):
        winobj.setslider()
        self.curvedict['time series']['pointer'].setChecked(True)
        pw = winobj.gdockdict[self.method_name].tabdict['time series'].tabplot
        if hasattr(self, 'entrytimesec'):
            for key in winobj.methodict:  # for xas-xrd-pl correlation
                if key[0:3] in ['xas', 'xrd'] and hasattr(winobj.methodict[key], 'entrytimesec'):
                    try:
                        y_low = np.where(self.entrytimesec[:, 0] > winobj.methodict[key].entrytimesec[0, 0])[0][0]
                    except:
                        y_low = 0
                    try:
                        y_up = np.where(self.entrytimesec[:, 1] > winobj.methodict[key].entrytimesec[-1, -1])[0][0]
                    except:
                        y_up = self.entrytimesec.shape[0] - 1

                    pw.setYRange(y_low, y_up)
                    continue

            if self.checksdict['time series'].isChecked():  # and 'reference' in self.curve_timelist[0]['raw']:
                self.curvedict['time series']['pointer'].setChecked(True)
                if self.checksdict['raw'].isChecked() and self.curvedict['raw']['dark'].isChecked():
                    self.plot_optic_2D(100, np.log10(self.data - self.dark), pw)
                else:
                    self.plot_optic_2D(100, np.log10(self.data), pw)

    def time_range(self, winobj):
        for key in winobj.path_name_widget:  # to distinguish _1, _2, ...
            if self.fileshort == winobj.path_name_widget[key]['raw file'].text():
                self.method_name = key

        self.exportfile = os.path.join(self.exportdir, self.fileshort + '_pl_data.h5')
        if os.path.isfile(self.exportfile):
            with h5py.File(self.exportfile, 'r') as f:
                self.data = np.zeros((f['raw'].shape[0], f['raw'].shape[1]), dtype='float32')
                f['raw'].read_direct(self.data)
                self.entrytimesec = np.zeros((f['time in seconds'].shape[0], 2), dtype='float64')
                f['time in seconds'].read_direct(self.entrytimesec)
                if 'y min' in list(f.keys()):
                    self.y_range = [f['y min'][()], f['y max'][()]]

                for key in list(f.keys()):
                    if key == 'pl_dark':
                        self.checksdict['raw'].setChecked(True)
                        self.curvedict['raw']['dark'].setChecked(True)
                        self.dark = np.zeros(f['pl_dark'].shape[0], dtype='float64')
                        f['pl_dark'].read_direct(self.dark)

        else:
            self.read_data_time(self.exportfile)

        return [self.entrytimesec[0, 0], self.entrytimesec[-1, 1]]

    def data_process(self, para_update):
        self.dynamictitle = self.fileshort + '\n data' + str(self.index + 1) + '\t start:' + self.startime + '\t end:' + self.endtime
        # raw
        if 'show' in self.curve_timelist[0]['raw']:
            if 'dark' in self.curve_timelist[0]['raw'] and len(self.dark) > 0:
                self.data_timelist[0]['raw']['show'].data = \
                    np.transpose([self.channels, self.data[self.index, :] - self.dark])  # 200 nm - 1.1 um
            else:
                self.data_timelist[0]['raw']['show'].data = \
                    np.transpose([self.channels, self.data[self.index, :]])  # 200 nm - 1.1 um
                self.dark = self.data[self.index, :]

        # time series
        if 'pointer' in self.curve_timelist[0]['time series']:
            self.plot_pointer('time series', 0, self.index, 't2', 15)

class Refl(Optic):
    def __init__(self, path_name_widget):
        super(Refl, self).__init__(path_name_widget)
        self.availablemethods = ['raw', 'time series', 'fit-T']
        self.availablecurves['raw'] = ['show','reference']
        self.availablecurves['time series'] = ['pointer']
        self.availablecurves['fit-T'] = ['band gap', 'thickness']
        self.axislabel = {'raw': {'bottom': 'wavelength/nm',
                                  'left': 'intensity'},
                          'time series': {'bottom': 'wavelength/nm',
                                          'left': 'Data number'},
                          'fit-T': {'bottom': 'Data number',
                                     'left': ''}}

        self.ini_data_curve_color()  # this has to be at the end line of each method after a series of other attributes

        # unique to Refl
        self.refcandidate = []
        # self.parameters = {'time series': {'normalize': Paraclass('original', ['original', 'normalized'])}}
        self.linedit = {'time series':{'z min':'-2',
                                       'z max':'0.1',
                                       'y min':'0',
                                       'y max':'1000'},
                        'raw':{'load reference from':''}} # ,
                                       # 'filter criterion':'2'}} # larger than 2 after referenced.

        self.actions = {'time series':{'update y,z range (Ctrl+0)': self.update_range,},
                        'raw':{'Export reference (Ctrl+X)': self.export_norm,
                               'Load refence (Ctrl+B)': self.load_norm}}
                                       # 'update z': self.update_z}}

    # def updata_z(self, winobj): # this function is currently redundant as plot_optic_2D does not involve z values any more
    #     pw = winobj.gdockdict[self.method_name].tabdict['time series'].tabplot
    #     if self.checksdict['raw'].isChecked() and self.curvedict['raw']['reference'].isChecked():
    #         self.plot_optic_2D(100, np.log10(self.data / self.refcandidate), pw)
    #     else:
    #         self.plot_optic_2D(100, np.log10(self.data), pw)  # / self.refcandidate

    def export_norm(self, winobj):
        # must guarantee reference is clicked...
        with h5py.File(os.path.join(self.exportdir, self.fileshort + '_refl_data.h5'), 'a') as f:
            if len(self.refcandidate) > 0:
                if 'refl_ref' in list(f.keys()): del f['refl_ref']
                f.create_dataset('refl_ref', data=np.array(self.refcandidate), dtype='float32')

            if hasattr(self, 'aligned'):
                if self.aligned:
                    del f['time in seconds']
                    f.create_dataset('time in seconds', self.entrytimesec)

    def load_norm(self, winobj):
        norm_file = self.linewidgets['raw']['load reference from'].text()
        nf = os.path.join(self.directory, 'process', norm_file + '_Optic_data', norm_file + '_refl_data.h5')
        if os.path.isfile(nf):
            with h5py.File(nf, 'r') as f:
                if 'refl_ref' in f.keys():
                    self.refcandidate = np.zeros(f['refl_ref'].shape[0], dtype=float)
                    f['refl_ref'].read_direct(self.refcandidate)

        # turn on ref. check
        self.curvedict['raw']['reference'].setChecked(True)

    def plot_from_load(self, winobj): # this needs pre-select a good reference
        winobj.setslider()
        pw = winobj.gdockdict[self.method_name].tabdict['time series'].tabplot
        if hasattr(self, 'entrytimesec'):
            for key in winobj.methodict:  # for xas-xrd-refl correlation
                if key[0:3] in ['xas', 'xrd'] and hasattr(winobj.methodict[key], 'entrytimesec'):
                    try:
                        y_low = np.where(self.entrytimesec[:,0] > winobj.methodict[key].entrytimesec[0,0])[0][0]
                    except:
                        y_low = 0
                    try:
                        y_up = np.where(self.entrytimesec[:, 1] > winobj.methodict[key].entrytimesec[-1,-1])[0][0]
                    except:
                        y_up = self.entrytimesec.shape[0]

                    pw.setYRange(y_low, y_up)
                    continue

            if self.checksdict['time series'].isChecked(): # and 'reference' in self.curve_timelist[0]['raw']:
                self.curvedict['time series']['pointer'].setChecked(True)
                if self.checksdict['raw'].isChecked() and self.curvedict['raw']['reference'].isChecked():
                    self.plot_optic_2D(100, self.data / self.refcandidate, pw)
                else:
                    self.plot_optic_2D(100, self.data, pw) # / self.refcandidate

    def time_range(self, winobj):
        for key in winobj.path_name_widget:  # to distinguish _1, _2, ...
            if self.fileshort == winobj.path_name_widget[key]['raw file'].text():
                self.method_name = key

        self.exportfile = os.path.join(self.exportdir, self.fileshort + '_refl_data.h5')
        if os.path.isfile(self.exportfile):
            with h5py.File(self.exportfile, 'r') as f:
                self.data = np.zeros((f['raw'].shape[0], f['raw'].shape[1]), dtype='float32')
                f['raw'].read_direct(self.data)
                self.entrytimesec = np.zeros((f['time in seconds'].shape[0], f['time in seconds'].shape[1]), dtype='float64')
                # how important is this dtype!!! as the abs time in seconds is a large number, you need to choose 64!!! so critical!!!
                f['time in seconds'].read_direct(self.entrytimesec)
                if 'y min' in list(f.keys()):
                    self.y_range = [f['y min'][()], f['y max'][()]]

                for key in list(f.keys()):
                    if key == 'refl_ref':
                        self.checksdict['raw'].setChecked(True)
                        self.curvedict['raw']['reference'].setChecked(True)
                        self.refcandidate = np.zeros(f['refl_ref'].shape[0], dtype='float64')
                        f['refl_ref'].read_direct(self.refcandidate)

        else:
            self.read_data_time(self.exportfile)

        return [self.entrytimesec[0, 0], self.entrytimesec[-1, 1]]

    def data_process(self, para_update):
        self.dynamictitle = self.fileshort + '\n data' + str(self.index + 1) + '\t start:' + self.startime + '\t end:' + self.endtime
        if 'show' in self.curve_timelist[0]['raw']:
            if 'reference' in self.curve_timelist[0]['raw'] and len(self.refcandidate) > 0:
                self.data_timelist[0]['raw']['show'].data = \
                    np.transpose([self.channels, self.data[self.index, :] / self.refcandidate])  # 200 nm - 1.1 um
            else:
                self.data_timelist[0]['raw']['show'].data = \
                    np.transpose([self.channels, self.data[self.index, :]])  # 200 nm - 1.1 um
                self.refcandidate = self.data[self.index, :]

        # time series
        if 'pointer' in self.curve_timelist[0]['time series']:
            self.plot_pointer('time series', 0, self.index, 't2', 15)


class XRF(Methods_Base):
    def __init__(self, path_name_widget):
        super(XRF, self).__init__()
        self.directory = path_name_widget['directory'].text()
        self.fileshort = path_name_widget['raw file'].text()
        self.channels = np.arange(1,4097) # number of channels in xrf
        self.exportdir = os.path.join(self.directory, 'process', self.fileshort + '_XRF_data')
        if not os.path.isdir(self.exportdir): os.mkdir(self.exportdir)
        self.entrytimesec = []
        self.method_name = []
        self.availablemethods = ['raw', 'time series', 'fit-T']
        self.availablecurves['raw'] = ['show', 'fit']
        self.availablecurves['time series'] = ['pointer']
        self.availablecurves['fit-T'] = ['Height', 'FWHM']
        self.axislabel = {'raw': {'bottom': 'channels',
                                  'left': 'log10(intensity)'},
                          'time series': {'bottom': 'channels',
                                          'left': 'Data number'},
                          'fit-T': {'bottom': 'Data number',
                                    'left': ''}}
        self.actions = {'time series': {'update y,z range (Ctrl+0)': self.update_range,
                                        # 'Export reference (Ctrl+X)': self.export_norm
                                        }}
        self.linedit = {'time series': {'z min': '-2',
                                        'z max': '0.1',
                                        'y min': '0',
                                        'y max': '1000', }}

        self.align_data = int(path_name_widget['align data number'].text())
        self.to_time = time.mktime(
            datetime.strptime(path_name_widget['to time'].text(), '%Y-%m-%dT%H:%M:%S').timetuple())  # in second
        self.aligned = False
        self.ini_data_curve_color()  # this has to be at the end line of each method after a series of other attributes

    def read_data_time(self, datafile):
        if os.path.exists(os.path.join(self.directory, 'raw', 'XRF')):
            filenames = sorted(glob.glob(os.path.join(self.directory, 'raw', 'XRF', self.fileshort + '*')))

        data = []
        datatime = []
        for f_name in filenames:
            data.append([])
            with open(f_name, 'r', encoding='windows-1252') as f:
                for line in f:
                    if 'START_TIME' in line:
                        datatime.append(time.mktime(datetime.strptime(line.split(' - ')[-1].split('\n')[0],
                                                                      '%m/%d/%Y %H:%M:%S').timetuple()))

                    if '<<DATA>>' in line:
                        for line in f:
                            if '<<END>>' in line:
                                break
                            else:
                                data[-1].append(int(line))

        if len(data) > 0:
            self.data = np.array(data)
            with h5py.File(datafile, 'w') as f:
                f.create_dataset('raw', data=self.data)

            datatime = np.array(datatime)
            self.entrytimesec = np.stack((datatime, datatime + datatime[1] - datatime[0])).T

            with h5py.File(datafile, 'a') as f:
                f.create_dataset('time in seconds', data=self.entrytimesec)

    def plot_from_prep(self, winobj):
        winobj.setslider()
        if self.align_data:  # and not self.aligned: # there is a little problem here: what if you want to align it again?
            time_diff = self.entrytimesec[self.align_data, 0] - self.to_time
            self.entrytimesec = self.entrytimesec - time_diff
            self.aligned = True

    def plot_optic_2D(self, step, data_norm, pw):
        # xticklabels = []
        # for tickvalue in np.arange(self.channels[0], self.channels[-1], step):
        #     xticklabels.append((int(data_norm.shape[1] * (tickvalue - self.channels[0]) /
        #                             (self.channels[-1] - self.channels[0])), "{:4.1f}".format(tickvalue)))
        #
        # xticks = pw.getAxis('bottom')
        # xticks.setTicks([xticklabels])

        for item in pw.childItems():
            if type(item).__name__ == 'ViewBox':
                item.clear()

            if type(item).__name__ == 'ColorBarItem':
                item.close()

        img_ts = pg.ImageItem(image=data_norm.transpose())  # need log here?
        img_ts.setZValue(-100)  # as long as less than 0
        pw.addItem(img_ts)
        color_map = pg.colormap.get('CET-R4')
        data_norm[isnan(data_norm)] = 0
        inf_ele = np.where(np.isfinite(data_norm) == False)
        for k in range(len(inf_ele[0])):
            data_norm[inf_ele[0][k], inf_ele[1][k]] = 0

        # limit the range to xas/xrd for more colorful
        view_range = np.s_[max(0, int(pw.viewRange()[1][0])):
                           min(data_norm.shape[0] - 1, int(pw.viewRange()[1][1])), :]
        self.color_bar_ts = pg.ColorBarItem(
            values=(data_norm[view_range].min(), data_norm[view_range].max()), colorMap=color_map)
        self.color_bar_ts.setImageItem(img_ts, pw)
        if hasattr(self, 'y_range'):
            pw.setYRange(self.y_range[0], self.y_range[1])

        pw.setXRange(0,len(self.channels) / 4) # since our interested region is at lower range

    def plot_from_load(self, winobj): # this needs pre-select a good reference
        winobj.setslider()
        pw = winobj.gdockdict[self.method_name].tabdict['time series'].tabplot
        if hasattr(self, 'entrytimesec'):
            for key in winobj.methodict:  # for xas-xrd-refl correlation
                if key[0:3] in ['xas', 'xrd'] and hasattr(winobj.methodict[key], 'entrytimesec'):
                    try:
                        y_low = np.where(self.entrytimesec[:,0] > winobj.methodict[key].entrytimesec[0,0])[0][0]
                    except:
                        y_low = 0
                    try:
                        y_up = np.where(self.entrytimesec[:, 1] > winobj.methodict[key].entrytimesec[-1,-1])[0][0]
                    except:
                        y_up = self.entrytimesec.shape[0]

                    pw.setYRange(y_low, y_up)
                    continue

            if self.checksdict['time series'].isChecked(): # and 'reference' in self.curve_timelist[0]['raw']:
                self.curvedict['time series']['pointer'].setChecked(True)
                self.plot_optic_2D(100, np.log10(self.data), pw) # / self.refcandidate

    def time_range(self, winobj):
        for key in winobj.path_name_widget:  # to distinguish _1, _2, ...
            if self.fileshort == winobj.path_name_widget[key]['raw file'].text():
                self.method_name = key

        self.exportfile = os.path.join(self.exportdir, self.fileshort + '_xrf_data.h5')
        if os.path.isfile(self.exportfile):
            with h5py.File(self.exportfile, 'r') as f:
                self.data = np.zeros((f['raw'].shape[0], f['raw'].shape[1]), dtype='float32')
                f['raw'].read_direct(self.data)
                self.entrytimesec = np.zeros((f['time in seconds'].shape[0], f['time in seconds'].shape[1]), dtype='float64')
                # how important is this dtype!!! as the abs time in seconds is a large number, you need to choose 64!!! so critical!!!
                f['time in seconds'].read_direct(self.entrytimesec)
                if 'y min' in list(f.keys()):
                    self.y_range = [f['y min'][()], f['y max'][()]]

        else:
            self.read_data_time(self.exportfile)

        return [self.entrytimesec[0, 0], self.entrytimesec[-1, 1]]

    def data_process(self, para_update):
        self.dynamictitle = self.fileshort + '\n data' + str(self.index + 1) + '\t start:' + self.startime + '\t end:' + self.endtime
        if 'show' in self.curve_timelist[0]['raw']:
            self.data_timelist[0]['raw']['show'].data = \
                np.transpose([self.channels, self.data[self.index, :]])  # 200 nm - 1.1 um

        # time series
        if 'pointer' in self.curve_timelist[0]['time series']:
            self.plot_pointer('time series', 0, self.index, 't2', 15)

class Example(Methods_Base):
    def __init__(self, path_name_widget):
        super(Example, self).__init__()
        self.directory = path_name_widget['directory'].text()
        self.fileshort = path_name_widget['raw file'].text()
        self.exportdir = os.path.join(self.directory, 'process', self.fileshort + '_XRF_data')
        if not os.path.isdir(self.exportdir): os.mkdir(self.exportdir)
        self.entrytimesec = []
        self.method_name = []
        self.availablemethods = ['raw', 'time series', 'fit-T']
        self.availablecurves['raw'] = ['show', 'fit']
        self.availablecurves['time series'] = ['pointer']
        self.availablecurves['fit-T'] = ['Height', 'FWHM']
        self.axislabel = {'raw': {'bottom': 'wavelength/nm',
                                  'left': 'log10(intensity)'},
                          'time series': {'bottom': 'wavelength/nm',
                                          'left': 'Data number'},
                          'fit-T': {'bottom': 'Data number',
                                    'left': ''}}
        self.actions = {'time series': {'update y,z range (Ctrl+0)': self.update_range,
                                        'Export reference (Ctrl+X)': self.export_norm}}

        self.linedit = {'time series': {'z min': '-2',
                                        'z max': '0.1',
                                        'y min': '0',
                                        'y max': '1000', }}

        self.ini_data_curve_color()  # this has to be at the end line of each method after a series of other attributes

    def read_data_time(self, datafile):
        pass

    def plot_from_prep(self, winobj):
        pass

    def plot_optic_2D(self, step, data_norm, pw):
        pass

    def plot_from_load(self, winobj):  # this needs pre-select a good reference
        pass

    def time_range(self, winobj):
        pass

    def data_process(self, para_update):
        pass