import os
import sys
import glob
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
from draggabletabwidget import *
from datetime import datetime
import time
import pickle

import larch
# from larch.io import read...
from larch.xafs import *
from larch import *
from larch.fitting import *

# from azint import AzimuthalIntegrator

from struct import unpack
import struct

from scipy.signal import savgol_filter, savgol_coeffs
from scipy.signal import find_peaks, peak_widths
import shutil
from paramiko import SSHClient

# sys.path.insert(0,r"C:\Users\jialiu\gsas2full\GSASII")
sys.path.insert(0,os.path.join(os.path.expanduser('~'), 'gsas2full', 'GSASII'))
import GSASIIindex
import GSASIIlattice
import GSASIIscriptable as G2sc
np.seterr(divide = 'ignore')

ev2nm = 1239.8419843320025

# warning
# pypowder is not available - profile calcs. not allowed
# pydiffax is not available for this platform
# pypowder is not available - profile calcs. not allowed

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
        if self.label[0:3] == 'xrd': self.dockobj.setMinimumWidth(winobj.screen_width * .5)
        else: self.dockobj.setMinimumWidth(winobj.screen_width * .2)
        # self.dockobj.setMinimumWidth(winobj.screen_width * .3)
        winobj.addDockWidget(Qt.BottomDockWidgetArea, self.dockobj)
        if len(winobj.gdockdict) > 3: # only accommodate two docks
            self.dockobj.setFloating(True)
        else: self.dockobj.setFloating(False)

        self.docktab = DraggableTabWidget()
        self.dockobj.setWidget(self.docktab)

    def deldock(self, winobj):
        winobj.removeDockWidget(self.dockobj)
        
    def gencontroltab(self, winobj):
        self.tooltab = QToolBox()
        # self.tooltab.setObjectName(self.label)
        winobj.controltabs.addTab(self.tooltab,self.label)

    def delcontroltab(self, winobj):
        index = winobj.controltabs.indexOf(self.tooltab)
        winobj.controltabs.removeTab(index)

class TabGraph_index():
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

class TabGraph():
    def __init__(self, name):
        self.label = name # e.g. raw, norm,...

    def mouseMoved(self, evt): # surprise!
        mousePoint = self.tabplot.vb.mapSceneToView(evt) # not evt[0]
        self.tabplot_label.setText("<span style='font-size: 10pt; color: black'> "
                "x = %0.2f, <span style='color: black'> y = %0.2f</span>" % (mousePoint.x(), mousePoint.y()))

    def gentab(self, dockobj, methodobj): # generate a tab for a docking graph
        self.graphtab = pg.GraphicsLayoutWidget()
        dockobj.docktab.addTab(self.graphtab, self.label)
        self.tabplot_label = pg.LabelItem(justify='right')
        self.graphtab.addItem(self.tabplot_label)
        self.tabplot = self.graphtab.addPlot(row=1, col=0)
        # pg.SignalProxy(self.tabplot.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved) # this is outdated!
        self.tabplot.scene().sigMouseMoved.connect(self.mouseMoved) # this is correct !
        self.tabplot.setLabel('bottom',methodobj.axislabel[self.label]['bottom'])
        self.tabplot.setLabel('left', methodobj.axislabel[self.label]['left'])
        if methodobj.axislabel[self.label]['left'] is not 'Data number':
            self.tabplot.addLegend(labelTextSize='9pt')

    def deltab(self, dockobj):
        index = dockobj.docktab.indexOf(self.graphtab)
        dockobj.docktab.removeTab(index)

    def gencontrolitem(self, dockobj):
        self.itemwidget = QWidget()
        self.itemwidget.setObjectName(self.label)
        self.itemwidget.setAccessibleName(dockobj.label)
        self.itemlayout = QVBoxLayout() # add control options to this layout
        self.itemwidget.setLayout(self.itemlayout)
        dockobj.tooltab.addItem(self.itemwidget, self.label)

    def delcontrolitem(self, dockobj):
        index = dockobj.tooltab.indexOf(self.itemwidget)
        dockobj.tooltab.removeItem(index)

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
                    methodobj.paralabel[tabname][key] = QLabel(key + ':' + str(temppara.setvalue))
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
                methodobj.actwidgets[tabname][key].clicked.connect(
                    lambda state, k = key: methodobj.actions[tabname][k](winobj)) # shock !!!
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
        self.huestep = 7 # color rotation increment

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
        self.timediff = slidervalue - self.entrytimesec[::,0] # 4294967289 1650706650
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
            self.checksdict[key].stateChanged.connect(winobj.graphtab)
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
        self.availablemethods = ['raw', 'normalizing', 'normalized', 'chi(k)', 'chi(r)', 'E0-T', 'Jump-T', 'mu_norm-T',
                                 'chi(k)-T', 'chi(r)-T', 'LCA(internal) single', 'LCA(internal)-T']
        self.availablecurves['raw'] = ['log(I0)', 'log(I1)']
        self.availablecurves['normalizing'] = ['mu','filter by time','filter by energy',
                                               'pre-edge', 'pre-edge points',
                                               'post-edge', 'post-edge points']
        self.availablecurves['normalized'] = ['normalized mu','reference','post-edge bkg']
        self.availablecurves['chi(k)'] = ['chi-k','window']
        self.availablecurves['chi(r)'] = ['chi-r','Re chi-r','Im chi-r']
        self.availablecurves['E0-T'] = ['pointer']
        self.availablecurves['Jump-T'] = ['pointer']
        self.availablecurves['mu_norm-T'] = ['pointer']
        self.availablecurves['chi(k)-T'] = ['pointer']
        self.availablecurves['chi(r)-T'] = ['pointer']
        self.availablecurves['LCA(internal) single'] = ['mu_norm', 'components', 'errors']
        self.availablecurves['LCA(internal)-T'] = ['pointer']
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
                          'E0-T':{'bottom':'Data number',
                                  'left':'Energy/eV'},
                          'Jump-T': {'bottom': 'Data number',
                                   'left': ''},
                          'mu_norm-T': {'bottom': 'Energy/eV',
                                       'left': 'Data number'},
                          'chi(k)-T': {'bottom': '<font> k / &#8491; </font> <sup> -1 </sup>',
                                       'left': 'Data number'},
                          'chi(r)-T':{'bottom':'<font> R / &#8491; </font>',
                                      'left':'Data number'},
                          'LCA(internal) single':{'bottom':'Energy/eV',
                                                  'left':'<font> &mu; </font>'},
                          'LCA(internal)-T':{'bottom':'data number',
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
                                          'Savitzky-Golay order (time)':Paraclass(values=(1,1,4,1)),
                                          'Savitzky-Golay window (energy)': Paraclass(values=(1, 1, 100, 2)),
                                          'Savitzky-Golay order (energy)': Paraclass(values=(1, 1, 4, 1)),
                                          'pre-edge para 1': Paraclass(values=(.01, 0, 1, .01)), # 0.01 of e0 - first point
                                          'pre-edge para 2': Paraclass(values=(.33, .05, .95, .01)), # 1/3 of e0 - pre 1
                                          'post-edge para 2': Paraclass(values=(.01, 0, 1, .01)), # 0.01 of last point - e0
                                          'post-edge para 1': Paraclass(values=(.33, .05, .95, .01))}, # 1/3 of post 1 - e0
                           'normalized':{'rbkg':Paraclass(values=(1,1,3,.1))},
                                         # 'Savitzky-Golay window (time)':Paraclass(values=(11,5,100,2)),
                                         # 'Savitzky-Golay order (time)':Paraclass(values=(1,1,4,1)),
                                         # 'Savitzky-Golay window (energy)': Paraclass(values=(11, 5, 100, 2)),
                                         # 'Savitzky-Golay order (energy)': Paraclass(values=(1, 1, 4, 1)), 
                           'chi(k)':{'kmin':Paraclass(values=(.9,0,3,.1)),
                                     'kmax':Paraclass(values=(6,3,20,.1)),
                                     'dk':Paraclass(values=(0,0,1,1)),
                                     'window':Paraclass(strings=('Hanning',['Hanning','Parzen'])),
                                     'kweight':Paraclass(values=(1,0,2,1))},
                           'mu_norm-T':{'flat norm':Paraclass(strings=('flat',['flat','norm']))}}

        self.actions = {'normalizing':{'filter all': self.filter_all_normalizing},
                        'mu_norm-T':{'x, y range start (Ctrl+R)': self.range_select,
                                     'do internal LCA (Ctrl+Y)': self.lca_internal,
                                     'Export time series (Ctrl+X)': self.export_ts}}

        self.linedit = {'mu_norm-T': {'z max':'102',
                                      'z min':'98',
                                      'components (internal LCA)': '2',
                                      'range start (y)': '100',
                                      'range end (y)': '200',
                                      'range start (x, data point)': '',
                                      'range end (x, data point)': ''},
                        'chi(k)-T': {'z max':'0.3',
                                     'z min':'-0.3'},
                        'chi(r)-T': {'z max': '0.05',
                                     'z min': '0'}}

                                        # self.average = int(path_name_widget['average (time axis)'].text())  # number of average data points along time axis, an odd number
        self.range = path_name_widget['energy range (eV)'].text() # useful for time_range!
        self.energy_range = [int(self.range.partition("-")[0]), int(self.range.partition("-")[2])]  # for Pb L and Br K edge combo spectrum
        self.exportdir = os.path.join(self.directory, 'process', self.fileshort + \
                               '_range_{}_{}eV'.format(self.energy_range[0], self.energy_range[1]), '')
        self.ref_mu = []
        self.filter_flag_normalizing = False
        self.filter_flag_normalized = False

        self.filtered = False
        self.data_error = .001

    def export_ts(self, winobj):
        resultfile = h5py.File(os.path.join(self.exportdir, 'mu_norm_time_series.h5'), 'w')
        resultfile.create_dataset('mu_norm', data=np.array(self.espace))
        resultfile.create_dataset('Energy', data=self.entrydata[0,0,:])
        resultfile.close()

    def lca_internal(self, winobj): # currently only two components
        # establish errors
        # running plot_from_prep to have correct errors with each group/mu data
        # do lca
        # at least two components
        component_start = int(self.linewidgets['mu_norm-T']['range start (y)'].text())
        component_end = int(self.linewidgets['mu_norm-T']['range end (y)'].text())
        amp_list = [param(.5, min=0, max=1), param(.5, min=0, max=1)]
        components_list = [self.grouplist[component_start].norm, self.grouplist[component_end].norm]
        components_num = int(self.linedit['mu_norm-T']['components (internal LCA)'].text())
        if components_num > 2:
            for k in range(components_num - 2):
                amp_list.append(param(0,min=1,max=1))

        def lca_cal(amp_list, data): pass


    def range_select(self, winobj):
        # to select range for peaks sorting
        pw = winobj.gdockdict[self.method_name].tabdict['mu_norm-T'].tabplot
        tempwidget = self.actwidgets['mu_norm-T']['x, y range start (Ctrl+R)']
        pw.scene().sigMouseClicked.connect(lambda evt, p=pw: self.range_clicked(evt, p))
        if tempwidget.text() == 'x, y range start (Ctrl+R)':
            tempwidget.setText('x, y range end (Ctrl+R)')
            tempwidget.setShortcut('Ctrl+R')
        elif tempwidget.text() == 'x, y range end (Ctrl+R)':
            tempwidget.setText('done (Ctrl+R)')
            tempwidget.setShortcut('Ctrl+R')
        else:
            tempwidget.setText('x, y range start (Ctrl+R)')
            tempwidget.setShortcut('Ctrl+R')
            pw.scene().sigMouseClicked.disconnect()

    def range_clicked(self, evt, pw): # watch out the difference between the nominal and actual x position
        if pw.sceneBoundingRect().contains(evt.scenePos()):
            mouse_point = pw.vb.mapSceneToView(evt.scenePos())  # directly, we have a viewbox!!!
            temptext = self.actwidgets['mu_norm_T']['x, y range start (Ctrl+R)'].text()
            actual_x = mouse_point.x() / self.entrydata.shape[2] * \
                       (self.entrydata[0,0,-1] - self.entrydata[0,0,0]) + self.entrydata[0,0,0]
            if temptext == 'x, y range end (Ctrl+R)':
                self.linewidgets['mu_norm-T']['range start (y)'].setText(str(int(mouse_point.y())))
                self.linewidgets['mu_norm-T']['range start (x)'].setText('{:.1f}'.format(actual_x))
            elif temptext == 'done (Ctrl+R)':
                self.linewidgets['mu_norm-T']['range end (y)'].setText(str(int(mouse_point.y())))
                self.linewidgets['mu_norm-T']['range end (x)'].setText('{:.1f}'.format(actual_x))

    def filter_all_normalizing(self, winobj): # make some warning sign here
        qbtn = winobj.sender()
        if self.filter_flag_normalizing:
            qbtn.setText('Go to click: Update by parameters')
            self.filter_flag_normalizing = False
        else:
            qbtn.setText('filter all')
            self.filter_flag_normalizing = True

    def filter_all_normalized(self, winobj):
        qbtn = winobj.sender()
        if self.filter_flag_normalized:
            self.filter_flag_normalized = False
            qbtn.setText('Go to click: Update by parameters')
        else:
            self.filter_flag_normalized = True
            qbtn.setText('filter all')

    def plot_from_prep(self, winobj): # generate larch Group for all data ! not related to 'load from prep' any more
        for index in range(self.entrydata.shape[0]): # 0,1,...,self.entrydata.shape[0] - 1
            mu = np.log(self.entrydata[index, 1, ::] / self.entrydata[index, 2, ::])

            if len(self.grouplist) != self.entrydata.shape[0]:
                self.grouplist.append(Group(name='spectrum' + str(index)))

            if self.filter_flag_normalizing and not self.filter_flag_normalized:
                mu, mu_error = self.filter_single_point(index)
                self.exafs_process_single(index, mu, mu_error)
            else: self.exafs_process_single(index, mu, self.data_error)

            # if self.filter_flag_normalized and not self.filter_flag_normalizing:
            #     mu = self.filter_single_point(index, Energy, self.grouplist[index].norm, 'normalized')
            #     self.exafs_process_single(index, Energy, mu)

            self.prep_progress.setValue(int((index + 1) / self.entrydata.shape[0] * 100))
            # self.parameters['chi(k)']['kmax'].upper = self.grouplist[index].k[-1]
            # rspacexlen = len(self.grouplist[index].r)
        # self.plot_from_load(winobj)
        if self.filter_flag_normalizing: # always update the binary files
            with open(os.path.join(self.exportdir, self.fileshort + '_Group_List_Smoothed'), 'wb') as f:
                pickle.dump(self.grouplist, f, -1)  # equals to pickle.HIGHEST_PROTOCOL
        else: # prevent filtered data to be saved
            with open(os.path.join(self.exportdir, self.fileshort + '_Group_List'), 'wb') as f:
                pickle.dump(self.grouplist, f, -1)

    def exafs_process_single(self, index, mu, mu_error):
        Energy = self.entrydata[index, 0, ::]
        self.grouplist[index].mu_error = mu_error
        try:
            e0 = find_e0(Energy, mu, group=self.grouplist[index])
        except: print('bad data')
        try:
            pre_edge_point_1 = (e0 - Energy[0]) * (1 - self.parameters['normalizing']['pre-edge para 1'].setvalue)
            post_edge_point_2 = (Energy[-1] - e0) * (1 - self.parameters['normalizing']['post-edge para 2'].setvalue)
        except: print('bad data')
        try:
            pre_edge(Energy, mu, group=self.grouplist[index],
                     pre1= - pre_edge_point_1,
                     pre2= - pre_edge_point_1 * self.parameters['normalizing']['pre-edge para 2'].setvalue,
                     norm2=post_edge_point_2,
                     norm1=post_edge_point_2 * self.parameters['normalizing']['post-edge para 1'].setvalue)
        except: print('bad data')
        try:
            autobk(Energy, mu, rbkg=self.parameters['normalized']['rbkg'].setvalue, group=self.grouplist[index])
        except: print('bad data')
        try:
            xftf(self.grouplist[index].k, self.grouplist[index].chi,
                 kmin=self.parameters['chi(k)']['kmin'].setvalue, kmax=self.parameters['chi(k)']['kmax'].setvalue,
                 dk=self.parameters['chi(k)']['dk'].setvalue, window=self.parameters['chi(k)']['window'].choice,
                 kweight=self.parameters['chi(k)']['kweight'].setvalue,
                 group=self.grouplist[index])
        except: print('bad data')

    def plot_from_load(self, winobj): # for time series
        winobj.setslider()
        # self.curvedict['time series']['pointer'].setChecked(True)
        self.E0 = []  # time series
        self.Jump = []
        self.espace = []
        self.kspace = []
        self.rspace = []  # time series
        kspace_length = [] # shocked! this naughty chi-k has irregular shape!
        for index in range(self.entrydata.shape[0]):
            kspace_length.append(len(self.grouplist[index].chi))

        for index in range(self.entrydata.shape[0]):
            self.E0.append(self.grouplist[index].e0)
            self.Jump.append(self.grouplist[index].edge_step)

            if self.parameters['mu_norm-T']['flat norm'].choice == 'flat':
                self.espace.append(self.grouplist[index].flat)
            else:
                self.espace.append(self.grouplist[index].norm)

            self.kspace.append(np.concatenate((self.grouplist[index].chi * np.square(self.grouplist[index].k),
                                               np.zeros(max(kspace_length) - kspace_length[index]))))
            self.rspace.append(self.grouplist[index].chir_mag)
            self.prep_progress.setValue(int((index + 1) / self.entrytimesec.shape[0] * 100))

        if self.checksdict['E0-T'].isChecked():
            pw = winobj.gdockdict[self.method_name].tabdict['E0-T'].tabplot
            for index in reversed(range(len(pw.items))): # shocked!
                if pw.items[index].name() == 'E0-T': pw.removeItem(pw.items[index])

            pw.plot(range(self.entrytimesec.shape[0]), self.E0, symbol='o', symbolSize=10, symbolPen='r', name='E0-T')

        if self.checksdict['Jump-T'].isChecked():
            pw = winobj.gdockdict[self.method_name].tabdict['Jump-T'].tabplot
            for index in reversed(range(len(pw.items))):  # shocked!
                if pw.items[index].name() == 'Jump-T': pw.removeItem(pw.items[index])

            pw.plot(range(self.entrytimesec.shape[0]), self.Jump, symbol='o', symbolSize=10, symbolPen='r', name='Jump-T')

        if self.checksdict['mu_norm-T'].isChecked(): # shocked! this naughty chi-k has irregular shape!
            pw = winobj.gdockdict[self.method_name].tabdict['mu_norm-T'].tabplot
            for index in reversed(range(len(pw.items))):  # shocked!
                if isinstance(pw.items[index], pg.ImageItem): pw.removeItem(pw.items[index])

            self.plot_chi_2D(self.entrydata.shape[2], self.entrydata[0,0,0], self.entrydata[0,0,-1], 100, pw,
                             np.array(self.espace) * 100, 'mu_norm-T')

        if self.checksdict['chi(k)-T'].isChecked(): # shocked! this naughty chi-k has irregular shape!
            pw = winobj.gdockdict[self.method_name].tabdict['chi(k)-T'].tabplot
            for index in reversed(range(len(pw.items))):  # shocked!
                if isinstance(pw.items[index], pg.ImageItem): pw.removeItem(pw.items[index])

            self.plot_chi_2D(max(kspace_length), 0, self.grouplist[kspace_length.index(max(kspace_length))].k[-1], 2, pw,
                             np.array(self.kspace), 'chi(k)-T')

        if self.checksdict['chi(r)-T'].isChecked():
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
        color_bar_ts = pg.ColorBarItem(values=(float(self.linedit[mode]['z min']),
                                                    float(self.linedit[mode]['z max'])), cmap=color_map)
        color_bar_ts.setImageItem(img_ts, pw)

    def filter_single_point(self, data_index):
        # if mode == 'normalizing':
        # time
        mode = 'normalizing'
        sg_win = int(self.parameters[mode]['Savitzky-Golay window (time)'].setvalue)
        sg_order = int(self.parameters[mode]['Savitzky-Golay order (time)'].setvalue)
        # below is a function!
        mu = lambda index: np.log(self.entrydata[index, 1, ::] / self.entrydata[index, 2, ::])

        # if mode == 'normalized':
        #     mu = lambda index: self.grouplist[index].norm # this one is dangerous as it uses smoothed/processed data

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
        sg_win = int(self.parameters[mode]['Savitzky-Golay window (energy)'].setvalue)
        sg_order = int(self.parameters[mode]['Savitzky-Golay order (energy)'].setvalue)

        if sg_win > sg_order + 1:
            # mu_filtered = scipy.signal.savgol_filter(mu_filtered, sg_win, sg_order, mode='nearest')
            # construct a matrix based on mu_filtered
            sg_data = []
            for k in range(sg_win):
                sg_data.append(np.concatenate((np.ones(k) * mu_filtered[0], mu_filtered, np.ones(sg_win - k) * mu_filtered[-1])))

            sg_data = np.array(sg_data)[:, (sg_win - 1) / 2 : -(sg_win + 1) / 2 - 1]
            mu_filtered = savgol_coeffs(sg_win, sg_order, use='dot').dot(sg_data)
            mu_filtered_error = 0
            for k in range(sg_win):
                mu_filtered_error = mu_filtered_error + (sg_data[k] - savgol_coeffs(sg_win, sg_order, pos=k, use='dot').dot(sg_data))**2

            mu_filtered_error = np.sqrt(mu_filtered_error / (sg_win - sg_order) + self.data_error**2) # actually the latter can be neglected
        
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
            data_end = np.where(self.energy_range[1] - data_single >= 0)[0][-1]
        except:
            data_end = -1
        data_range = np.s_[data_start:data_end]
        return data_range

    def load_group(self,winobj):
        min_len = 1e5
        for index in range(len(self.entrydata)): # find the min array length
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
        else:
            self.plot_from_prep(winobj)

    def data_process(self, para_update): # for curves, embody data_timelist, if that curve exists
        # Energy, I0, I1 = self.read_data_time() # need to know self.slidervalue
        Energy = self.entrydata[self.index, 0, ::]
        I0 = self.entrydata[self.index, 1, ::]
        I1 = self.entrydata[self.index, 2, ::]
        mu = np.log(I0 / I1)
        mu_filtered = mu
        
        # this ensures that smoothed data get processed once you adjust those filtering parameters.
        # this function also sets data_timelist for filtered curves.
        # by abandon 'normalized' filter, there is less complexity.
        # when moving the slider, curves in other tabs change back to unfiltered state. this is actually good--fast!
        if self.filtered: 
            mu_filtered, mu_filtered_error = self.filter_single_point(self.index)
            self.exafs_process_single(self.index, mu_filtered, mu_filtered_error)
        else:
            if para_update: self.exafs_process_single(self.index, mu_filtered, self.grouplist[self.index].mu_error)
        
        self.dynamictitle = 'data' + str(self.index + 1) + '\t start:' + self.startime + '\t end:' + self.endtime

        # raw
        if 'I0' in self.curve_timelist[0]['raw']:
            self.data_timelist[0]['raw']['I0'].data = np.transpose([Energy, np.log(I0 * 10)])
        if 'I1' in self.curve_timelist[0]['raw']:
            self.data_timelist[0]['raw']['I1'].data = np.transpose([Energy, np.log(I1 * 10)])
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
        if 'chi-r' in self.curve_timelist[0]['chi(r)']:
            self.data_timelist[0]['chi(r)']['chi-r'].data = np.transpose([self.grouplist[self.index].r, self.grouplist[self.index].chir_mag])
        if 'Re chi-r' in self.curve_timelist[0]['chi(r)']:
            self.data_timelist[0]['chi(r)']['Re chi-r'].data = np.transpose([self.grouplist[self.index].r, self.grouplist[self.index].chir_re])
        if 'Im chi-r' in self.curve_timelist[0]['chi(r)']:
            self.data_timelist[0]['chi(r)']['Im chi-r'].data = np.transpose([self.grouplist[self.index].r, self.grouplist[self.index].chir_im])

        # E0-T
        if 'pointer' in self.curve_timelist[0]['E0-T']:
            try: self.plot_pointer('E0-T', self.index, self.E0[self.index], '+', 30)
            except: print('Load time series first')

        # Jump-T
        if 'pointer' in self.curve_timelist[0]['Jump-T']:
            try: self.plot_pointer('Jump-T', self.index, self.Jump[self.index], '+', 30)
            except: print('Load time series first')

        # mu_norm-T
        if 'pointer' in self.curve_timelist[0]['mu_norm-T']:
            self.plot_pointer('mu_norm-T', 0, self.index, 't2', 15)

        # chi(k)-T
        if 'pointer' in self.curve_timelist[0]['chi(k)-T']:
            self.plot_pointer('chi(k)-T', 0, self.index, 't2', 15)

        # chi(r)-T
        if 'pointer' in self.curve_timelist[0]['chi(r)-T']:
            self.plot_pointer('chi(r)-T', 0, self.index, 't2', 15)

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
                    timesecond_start = np.sort(np.concatenate((timesecond_array[0:-2], (timesecond_array[0:-2] + timesecond_array[1:-1]) / 2)))
                    timesecond_end = np.sort(np.concatenate((timesecond_array[1:-1], (timesecond_array[0:-2] + timesecond_array[1:-1]) / 2)))
                    self.entrytimesec = np.array([timesecond_start,timesecond_end]).T # so critical!!!

        # read in data
        if self.entrydata == []:  # Energy, I0, I1
            datafile = os.path.join(self.exportdir, self.fileshort + '_spectrum_all.h5')
            if os.path.isdir(self.exportdir) and os.path.isfile(datafile):
                f = h5py.File(datafile, 'r')
                for spectrum in f.keys():
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
                
                f = h5py.File(datafile, 'w')
                data4xrd = []
                for index in range(data.shape[0]): # output format has to be Energy, I0, I1
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
                        back_sel = np.s_[data.shape[1] - 1:half_index - 1:-1] # include the middle point
                        # self.output_txt(np.array([data_all['energy'][index, back_sel],  # E0
                        #                           data_all['I0a'][index, back_sel] + data_all['I0b'][index, back_sel], # I0
                        #                           data_all['I1'][index, back_sel]]).T, 2 * index + 1)  # I1

                    data_range = self.sel_by_energy_range(data_all['energy'][index, 0:half_index + 1])
                    data_half = np.array([data_all['energy'][index, data_range],  # E0
                                          data_all['I0a'][index, data_range] + data_all['I0b'][index, data_range],  # I0
                                          data_all['I1'][index, data_range]])  # I1
                    self.entrydata.append(data_half)
                    f.create_dataset('spectrum_{:04d}'.format(index * 2), data=data_half.T)

                    data_range = self.sel_by_energy_range(data_all['energy'][index, back_sel])
                    data_half = np.array([data_all['energy'][index, back_sel][data_range],  # E0
                                          data_all['I0a'][index, back_sel][data_range] +
                                          data_all['I0b'][index, back_sel][data_range],  # I0
                                          data_all['I1'][index, back_sel][data_range]])  # I1
                    self.entrydata.append(data_half)
                    f.create_dataset('spectrum_{:04d}'.format(index * 2 + 1), data=data_half.T)

                    # extract data for xrd normalization
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

                f.close()

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
        self.availablemethods = ['raw', 'integrated', 'time series', 'refinement single',
                                 'centroid-T', 'integrated area-T', 'segregation degree-T']
        self.availablecurves['raw'] = ['show image']
        self.availablecurves['integrated'] = ['original', 'normalized to 1', # 'normalized to I0 and <font> &mu; </font>d',
                                              'truncated', 'smoothed', 'find peaks']
        self.availablecurves['time series'] = ['pointer']
        self.availablecurves['refinement single'] = ['observed', 'calculated', 'difference']
        self.availablecurves['centroid-T'] = ['pointer']
        self.availablecurves['integrated area-T'] = ['pointer']
        self.availablecurves['segregation degree-T'] = ['pointer']
        self.directory = path_name_widget['directory'].text()
        self.fileshort = path_name_widget['raw file'].text()
        self.intfile_appendix = path_name_widget['integration file appendix'].text()
        self.ponifile = os.path.join(self.directory, 'process', path_name_widget['PONI file'].text())
        with open(self.ponifile, 'r') as f: 
            self.wavelength = 1e10 * float(f.readlines()[-1].splitlines()[0].partition(' ')[2]) # now in Angstrom 
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

        self.colormax_set = False
        self.axislabel = {'raw':{'bottom':'',
                                 'left':''},
                          'integrated': {'bottom': '<font> q / &#8491; </font> <sup> -1 </sup>,'
                                                   '<font> 2 &#952; / </font> <sup> o </sup>, or'
                                                   '<font> d / &#8491; </font>',
                                        'left': 'Intensity'},
                          'time series': {'bottom': '<font> q / &#8491; </font> <sup> -1 </sup>',
                                        'left': 'Data number'},
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
        self.bravaisNames = ['Cubic-F', 'Cubic-I', 'Cubic-P', 'Trigonal-R', 'Trigonal/Hexagonal-P',
                             'Tetragonal-I', 'Tetragonal-P', 'Orthorhombic-F', 'Orthorhombic-I', 'Orthorhombic-A',
                             'Orthorhombic-B', 'Orthorhombic-C',
                             'Orthorhombic-P', 'Monoclinic-I', 'Monoclinic-A', 'Monoclinic-C', 'Monoclinic-P',
                             'Triclinic']

        self.parameters = {'integrated': {'scale': Paraclass(strings=('log10', ['log10', 'sqrt', 'linear'])),
                                          'normalization': Paraclass(strings=('not normalized',
                                                                             ['not normalized',
                                                                              'normalized to I0 and \u03BC d'])),
                                          'x axis': Paraclass(strings=('q',['q','2th','d'])),
                                          'clip head': Paraclass(values=(0,0,1000,1)),
                                          'clip tail': Paraclass(values=(1, 1, 1000, 1)),
                                          'Savitzky-Golay window': Paraclass(values=(11,3,101,2)),
                                          'Savitzky-Golay order': Paraclass(values=(1,1,2,1)),
                                          'peak prominence min': Paraclass(values=(0,0,1e2,.1)),
                                          'peak prominence max': Paraclass(values=(1e4,1e2,1e4,1)),
                                          'peak width min': Paraclass(values=(0,0,9,1)),
                                          'peak width max': Paraclass(values=(20,10,100,1)),
                                          'window length': Paraclass(values=(101,10,1000,2))},
                           # these parameters are special: they do not call data_process.
                           # change them also in ShowData.update_parameters!
                           'time series': {'scale': Paraclass(strings=('log10', ['log10', 'sqrt', 'linear'])),
                                           'gap y tol.': Paraclass(values=(1,1,20,1)),
                                           'gap x tol.': Paraclass(values=(1,1,100,1)),
                                           'min time span': Paraclass(values=(5,1,50,1)),
                                           'max diff time span': Paraclass(values=(5,0,50,1)),
                                           'max diff start time': Paraclass(values=(3,0,50,1)),
                                           'symbol size': Paraclass(values=(1,1,20,1)),
                                           'phases': Paraclass(strings=('choose a phase',
                                                                        ['choose a phase','phase1','phase2','phase3',
                                                                         'phase4','phase5','phase6','phase7','phase8',
                                                                         'phase9','phase10','phase11','phase12']))},
                           # 'refinement single': {'data number': Paraclass(values=(0,0,len(self.refinedata) - 1,1)),
                           #                        'scale': Paraclass(strings=('log10', ['log10', 'sqrt', 'linear'])),
                           #                        'x axis': Paraclass(strings=('q',['q','2th','d']))}
                           }
        self.actions = {'integrated':{"interpolate along q": self.interpolate_data,
                                      "find peaks (Ctrl+F)": self.find_peak_all},
                       'time series':{"clear rainbow map (Ctrl+E)": self.show_clear_ts,
                                      "range start (Ctrl+R)": self.range_select,
                                      "catalog peaks (Ctrl+T)": self.catalog_peaks,
                                      "assign phases (Ctrl+A)": self.assign_phases,
                                      "clear peaks (Ctrl+P)": self.show_clear_peaks,
                                      "index phases (Ctrl+I)": self.index_phases}} # button name and function name

        self.linedit = {'time series': {'range start': '100',
                                        'range end': '200',
                                        'exclude from':'0',
                                        'exclude to':'0'}}

        self.pre_ts_btn_text = 'Do Batch Integration(Ctrl+D)'
        energy = ev2nm / self.wavelength * 10 / 1000
        self.exportdir = os.path.join(self.directory, 'process', self.fileshort + '_{:.1f}keV'.format(energy))
        self.exportfile = os.path.join(self.exportdir, self.fileshort + '_{:.1f}keV'.format(energy) + self.intfile_appendix)
        self.entrytimesec = []
        self.cells_sort = {}

        if os.path.isfile(self.exportfile):
            try:
                self.read_intg() # read in data at the beginning
            except:
                print('redo the integration!')

    def integrate_Cluster(self, ponifile_short, clusterfile):
        clusterfolder = '/data/visitors/' + self.directory.replace(os.sep, '/').replace('//', '/')[2::]
        client = SSHClient()
        client.load_system_host_keys()
        client.connect('clu0-fe-1.maxiv.lu.se', username='balder-user', password='BeamPass!')

        cmd = "sbatch --export=ALL,argv=\'%s %s %s\' " \
              "/mxn/home/balder-user/BalderProcessingScripts/balder-xrd-processing/submit_stream_MR.sh" % (
                  clusterfolder, self.fileshort[0:-6], ponifile_short)

        client.exec_command('mv ./*.log ./azint_logs')
        stdin, stdout, stderr = client.exec_command(cmd)
        print(stdout.readlines())
        client.close()

        # add status check
        t0 = time.time()
        t = 0
        while t < 1000:
            if os.path.exists(clusterfile):
                print('Integration completed by HPC cluster in ', int(t), ' seconds.')
                break
            time.sleep(1)
            t = time.time() - t0

        # clusterfile_linux = '/data/visitors/' + clusterfile.replace(os.sep, '/').replace('//', '/')[2::]
        # client.exec_command('chmod g+w {}'.format(clusterfile_linux))

        print(stderr.readlines())

    def read_intg(self): # it's raw, not normal here, can make a choice in the future
        intfile = h5py.File(self.exportfile, 'r')
        if self.parameters['integrated']['normalization'].choice == 'not normalized':
            intdata_all = intfile['rawresult']
        else:
            intdata_all = intfile['normresult']

        self.intdata_ts = np.zeros((intdata_all.shape[0], intdata_all.shape[1]), dtype='float32')
        intdata_all.read_direct(self.intdata_ts)
        self.intqaxis = np.array(list(intfile['info/q(Angstrom)']))
        intfile.close()

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

    def find_peak_all(self, winobj):
        q_start = int(self.parameters['integrated']['clip head'].setvalue)
        q_ending = int(self.parameters['integrated']['clip tail'].setvalue)
        peaks_q = []
        peaks_number = []
        self.peaks_index = [] # data number for each data where you can find some peaks, for this reason delta y should be based on this
        self.peaks_all = []
        self.peaks_properties_all = []
        for index in range(self.entrytimesec.shape[0]):
            intdata_clipped = self.intdata_ts[index][q_start: -q_ending]
            if 'smoothed' in self.curve_timelist[0]['integrated']:
                self.intdata_smoothed = \
                    scipy.signal.savgol_filter(intdata_clipped,
                                               int(self.parameters['integrated']['Savitzky-Golay window'].setvalue),
                                               int(self.parameters['integrated']['Savitzky-Golay order'].setvalue))

            if hasattr(self, 'intdata_smoothed'):
                peaks, peak_properties = self.find_peak_conditions(self.intdata_smoothed)
            else: peaks, peak_properties = self.find_peak_conditions(intdata_clipped)

            if peaks is not []:
                self.peaks_index.append(index)
                self.peaks_all.append(peaks) # useful for catalog_peaks
                self.peaks_properties_all.append(peak_properties)
                for sub_index in range(len(peaks)):
                    peaks_q.append(peaks[sub_index]) # x, index
                    peaks_number.append(index)            # y, index

        if not hasattr(self, 'peak_map'):
            if 'time series' not in winobj.gdockdict[self.method_name].tabdict:
                self.checksdict['time series'].setChecked(True)

            self.peak_map = winobj.gdockdict[self.method_name].tabdict['time series'].tabplot.plot(name='find peaks')

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

    def show_clear_peaks(self, winobj):
        pw = winobj.gdockdict[self.method_name].tabdict['time series'].tabplot
        tempwidget = self.actwidgets['time series']['clear peaks (Ctrl+P)']
        if 'clear peaks (Ctrl+P)' == tempwidget.text():
            tempwidget.setText('show peaks (Ctrl+P)')
            tempwidget.setShortcut('Ctrl+P')
            for index in reversed(range(len(pw.items))): # shocked!
                if isinstance(pw.items[index], pg.PlotDataItem):
                    if pw.items[index].name()[0:4] in ['find', 'cata', 'assi']: pw.removeItem(pw.items[index])
                        
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

    def range_select(self, winobj):
        # to select range for peaks sorting
        pw = winobj.gdockdict[self.method_name].tabdict['time series'].tabplot
        tempwidget = self.actwidgets['time series']['range start (Ctrl+R)']
        pw.scene().sigMouseClicked.connect(lambda evt, p=pw: self.range_clicked(evt, p))
        if tempwidget.text() == 'range start (Ctrl+R)':
            tempwidget.setText('range end (Ctrl+R)')
            tempwidget.setShortcut('Ctrl+R')
        elif tempwidget.text() == 'range end (Ctrl+R)':
            tempwidget.setText('exclude from (Ctrl+R)')
            tempwidget.setShortcut('Ctrl+R')
        elif tempwidget.text() == 'exclude from (Ctrl+R)':
            tempwidget.setText('exclude to (Ctrl+R)')
            tempwidget.setShortcut('Ctrl+R')
        elif tempwidget.text() == 'exclude to (Ctrl+R)':
            tempwidget.setText('done (Ctrl+R)')
            tempwidget.setShortcut('Ctrl+R')
        else:
            tempwidget.setText('range start (Ctrl+R)')
            tempwidget.setShortcut('Ctrl+R')
            pw.scene().sigMouseClicked.disconnect()

    def range_clicked(self, evt, pw):
        if pw.sceneBoundingRect().contains(evt.scenePos()):
            mouse_point = pw.vb.mapSceneToView(evt.scenePos()) # directly, we have a viewbox!!!
            temptext = self.actwidgets['time series']['range start (Ctrl+R)'].text()
            if temptext == 'range end (Ctrl+R)':
                self.linewidgets['time series']['range start'].setText(str(int(mouse_point.y())))
            elif temptext == 'exclude from (Ctrl+R)':
                self.linewidgets['time series']['range end'].setText(str(int(mouse_point.y())))
            elif temptext == 'exclude to (Ctrl+R)':
                self.linewidgets['time series']['exclude from'].setText(str(int(mouse_point.y())))
            else:
                self.linewidgets['time series']['exclude to'].setText(str(int(mouse_point.y())))

    def catalog_peaks(self, winobj):
        pw = winobj.gdockdict[self.method_name].tabdict['time series'].tabplot
        i_start = self.linewidgets['time series']['range start'].text()
        i_end = self.linewidgets['time series']['range end'].text()
        i_ex_start = self.linewidgets['time series']['exclude from'].text()
        i_ex_end = self.linewidgets['time series']['exclude to'].text()

        if i_start is not '' and i_end is not '' and i_ex_start is not '' and i_ex_end is not '':
            i_start = np.where(np.array(self.peaks_index) - int(i_start) >= 0)[0][0]
            i_end = np.where(np.array(self.peaks_index) - int(i_end) >= 0)[0][0]
            i_ex_start = np.where(np.array(self.peaks_index) - int(i_ex_start) >= 0)[0][0]
            i_ex_end = np.where(np.array(self.peaks_index) - int(i_ex_end) >= 0)[0][0]

        i_start = np.min([i_start, i_end])  # to prevent disorder
        i_end = np.max([i_start, i_end])
        if i_start < i_ex_start < i_ex_end < i_end: # to insure the correct order
            peaks_index_sel = self.peaks_index[i_start:i_ex_start] + self.peaks_index[i_end:i_ex_end]
        else:
            print('ignore excluded region')
            peaks_index_sel = self.peaks_index[i_start:i_end]

        # start cataloging peaks, the most exciting part
        if peaks_index_sel != []:
            pw.setYRange(i_start, i_end)
            self.peaks_catalog = []
            for index in range(len(self.peaks_all[i_start])):
                # first level: peaks; second level: index in y in full data, index in x in full data, index in peaks_index_sel
                self.peaks_catalog.append([[self.peaks_index[i_start],self.peaks_all[i_start][index], 0]])

            gap_y = int(self.parameters['time series']['gap y tol.'].setvalue)
            gap_x = int(self.parameters['time series']['gap x tol.'].setvalue)

            for index in range(len(peaks_index_sel)): # index on data number peaks selected (self.peaks_index)
                # for j in range(min([gap_y, i_end - index])): # index on gap_y tolerence
                entry = self.peaks_index.index(peaks_index_sel[index])
                search_range = len(self.peaks_catalog)
                for k in range(len(self.peaks_all[entry])): # index on all peaks detected within one data
                    add_group = [] # indicator whether to add a new peaks group or not
                    for i in range(search_range): # index on existing peaks groups, this one is constantly changing!!!
                        # the following condition is very tricky in y direction, it has to be [2] not [0]
                        if np.abs(self.peaks_all[entry][k] - self.peaks_catalog[i][-1][1]) <= gap_x and \
                            np.abs(index - self.peaks_catalog[i][-1][2]) <= gap_y: # add to existing peaks group
                            self.peaks_catalog[i].append([self.peaks_index[entry],
                                                          self.peaks_all[entry][k], index]) # data number (y,x)
                            add_group.append(1)
                        else: add_group.append(0)

                    if np.sum(add_group) == 0: # add a new catalog if this peak belongs to no old catalogs
                        self.peaks_catalog.append([[self.peaks_index[entry],
                                                    self.peaks_all[entry][k], index]])

            # show peaks
            length_limit = self.parameters['time series']['min time span'].setvalue
            if hasattr(self, 'peaks_catalog_map'): # clear the old plot
                for index in range(len(self.peaks_catalog_map)):
                    pw.removeItem(self.peaks_catalog_map[index])

            if hasattr(self, 'phases_map'): # clear the old plot
                for index in range(len(self.phases_map)):
                    pw.removeItem(self.phases_map[index])

                self.phases_map = []

            self.peaks_catalog_select = [] # peak number, data number (y,x)
            for index in range(len(self.peaks_catalog)):
                self.peaks_catalog[index] = np.array(self.peaks_catalog[index])
                if self.peaks_catalog[index].shape[0] > length_limit:
                    self.peaks_catalog_select.append(self.peaks_catalog[index])

            self.peaks_catalog_map = [None] * len(self.peaks_catalog_select)
            q_start = int(self.parameters['integrated']['clip head'].setvalue)
            for index in range(len(self.peaks_catalog_select)):
                color = pg.intColor(index * self.huestep, 100)
                self.peaks_catalog_map[index] = \
                    pw.plot(name='catalog peaks' + ' ' + str(index))
                self.peaks_catalog_map[index].setData(self.peaks_catalog_select[index][::,1] + q_start,
                                                  self.peaks_catalog_select[index][::,0], symbol='o', symbolBrush=color,
                                                  symbolSize=self.parameters['time series']['symbol size'].setvalue)

    def assign_phases(self, winobj):
        pw = winobj.gdockdict[self.method_name].tabdict['time series'].tabplot
        if hasattr(self, 'peaks_catalog_select'):
            time_diff = self.parameters['time series']['max diff time span'].setvalue
            start_diff = self.parameters['time series']['max diff start time'].setvalue
            self.phases = [[0]] # peak 0 belong to phase 0
            for index in range(len(self.peaks_catalog_select) - 1):
                add_group = []
                for k in range(len(self.phases)):
                    time_span_index = np.abs(self.peaks_catalog_select[index + 1][-1][0] - \
                                      self.peaks_catalog_select[index + 1][0][0])
                    time_span_k = np.abs(self.peaks_catalog_select[self.phases[k][-1]][-1][0] - \
                                  self.peaks_catalog_select[self.phases[k][-1]][0][0])
                    time_start_index = self.peaks_catalog_select[index + 1][0][0]
                    time_start_k = self.peaks_catalog_select[self.phases[k][-1]][0][0]
                    if time_diff > np.abs(time_span_index - time_span_k) and \
                        start_diff > np.abs(time_start_index - time_start_k):
                        self.phases[k].append(index + 1)
                        add_group.append(1)
                    else: add_group.append(0)

                if np.sum(add_group) == 0:
                    self.phases.append([index + 1])

            if hasattr(self, 'phases_map'): # clear the old plot
                for index in range(len(self.phases_map)):
                    pw.removeItem(self.phases_map[index])

            if hasattr(self, 'peaks_catalog_map'):  # clear the old plot
                for index in range(len(self.peaks_catalog_map)):
                    pw.removeItem(self.peaks_catalog_map[index])

                self.peaks_catalog_map = []

            self.phases_map = [None] * len(self.phases)
            q_start = int(self.parameters['integrated']['clip head'].setvalue)
            for index in range(len(self.phases)):
                color = pg.intColor(index * self.huestep, 100)
                tempdata = self.peaks_catalog_select[self.phases[index][0]]
                if len(self.phases[index]) > 1:
                    for k in range(len(self.phases[index]) - 1):
                        tempdata = np.concatenate((tempdata, self.peaks_catalog_select[self.phases[index][k + 1]])) # another shock!

                self.phases_map[index] = \
                    pw.plot(name='assign phases'+ ' ' + str(index))
                self.phases_map[index].setData(tempdata[::,1] + q_start, tempdata[::,0],
                                               symbol='d', symbolBrush=color, pen=None,
                                               symbolSize=self.parameters['time series']['symbol size'].setvalue)

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

    def indexing(self, tabobj, winobj):
        q_start = int(self.parameters['integrated']['clip head'].setvalue)
        phase_num = int(tabobj.label[5:]) # assuming all names are 'phaseXX', so the actual index is phase_num - 1
        phase_d = [] # d_obs of all peaks
        phase_int = [] # intensity of all peaks
        q_peak = [] # q (or x) data number of all peaks
        common_data = set(range(self.entrytimesec.shape[0] + 1))
        for index in self.phases[phase_num - 1]: # average the d-value, intensity, on common data only!
            peak_data = np.array(self.peaks_catalog_select[index])
            common_data = common_data & set(peak_data[:,0])

        common_data = list(common_data)
        for index in self.phases[phase_num - 1]: # it is quite certain that everything is in timely order, i.e. sorted
            peak_data = np.array(self.peaks_catalog_select[index])
            peak_data = peak_data[np.searchsorted(peak_data[:,0],common_data),:]
            q_peak.append(int(peak_data[:,1].sum() / peak_data.shape[0] + q_start)) # average
            phase_d.append(2 * np.pi / self.intqaxis[q_peak[index]]) # in Angstrom
            peak_int = 0 # intensity of one peak
            for peak in peak_data:
                data_num = self.peaks_index.index(peak[0])
                peak_num = np.where(self.peaks_all[data_num] == peak[1])[0][0]
                peak_prop = self.peaks_properties_all[data_num]
                peak_int += peak_prop['prominences'][peak_num] * (peak_prop['right_ips'][peak_num] -
                                                                   peak_prop['left_ips'][peak_num]) / 2

            phase_int.append(peak_int / peak_data.shape[0]) # average

        # plot the averaged q for each peak for this phase
        self.plot_reflections(winobj, q_peak, tabobj.label, 't', 0, [])

        # with open(self.ponifile, 'r') as f:
        #     wavelength = float(f.readlines()[-1].splitlines()[0].partition(' ')[2])

        peaks = []
        for index in range(len(phase_int)):
            peaks.append([np.arcsin(self.wavelength / np.array(phase_d[index]) / 2) * 2 / np.pi * 180,
                          phase_int[index], True, True, 0, 0, 0, phase_d[index], 0])

        peaks = np.array(peaks)
        peaks = peaks[peaks[:,0].argsort()]
        print(peaks)
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

    def plot_reflections(self, winobj, positions, phase_name, symbol, offset, hkl): # the first five letters of name must be distinguishable
        pw = winobj.gdockdict[self.method_name].tabdict['time series'].tabplot
        data_center = (int(self.linewidgets['time series']['range start'].text()) +
                       int(self.linewidgets['time series']['range end'].text())) / 2
        for index in reversed(range(len(pw.items))):  # shocked!
            if isinstance(pw.items[index], pg.PlotDataItem):
                if pw.items[index].name()[0:5] == phase_name[0:5]: pw.removeItem(pw.items[index])

        y = np.array([data_center] * len(positions))
        offset = abs(int(self.linewidgets['time series']['range start'].text()) -
                     int(self.linewidgets['time series']['range end'].text())) * offset
        pw.plot(positions, y - offset, pen=None, symbol=symbol, symbolSize=15, name=phase_name).setZValue(100)

        for index in reversed(range(len(pw.items))):  # shocked!
            if isinstance(pw.items[index], pg.TextItem): pw.removeItem(pw.items[index])

        if hkl != []:
            for index in range(hkl.shape[0]):
                hkl_text = pg.TextItem(str(hkl[index])[1:-1], angle=90, anchor=(1,1), color='k')
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
        self.plot_reflections(winobj, positions, 'index_' + str(index), 't1', 0.03, HKLD[:,0:3])

    def keep_selected(self, tabobj):
        pass

    def plot_from_load(self, winobj): # some part can be cut out for a common function
        # if winobj.slideradded == False:
        winobj.setslider()
        # self.slideradded = True
        # winobj.slideradded = True
        # if not hasattr(self,'read_intg'):
        self.read_intg() # issue?
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

            if hasattr(self, 'color_bar_ts'):
                pw.removeItem(self.img_ts)
                self.color_bar_ts.close()

            self.img_ts = pg.ImageItem(image=np.transpose(intdata_ts)) # need transpose here ?
            self.img_ts.setZValue(-100) # as long as less than 0
            pw.addItem(self.img_ts)
            color_map = pg.colormap.get('CET-R4')
            intdata_ts[isnan(intdata_ts)] = 0 # surprise!
            self.color_bar_ts = pg.ColorBarItem(values=(0, intdata_ts.max()), cmap=color_map)
            self.color_bar_ts.setImageItem(self.img_ts, pw)

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
                if isinstance(pw.items[index], pg.PlotCurveItem): pw.removeItem(pw.items[index])

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

        if not winobj.slider.value() :winobj.slider.setValue(winobj.slider.minimum() + 1) # may be not the best way

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
                self.intdata = self.intdata_ts[0]

        else: # there could be a few xrd data within one index
            if self.checksdict['raw'].isChecked():
                self.read_data_index(self.index)

            if self.checksdict['integrated'].isChecked() or self.checksdict['time series'].isChecked():
                self.intdata = self.intdata_ts[self.index]

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

        self.dynamictitle = 'data' + str(self.index + 1) + '\t start:' + self.startime + '\t end:' + self.endtime

        # raw
        if 'show image' in self.curve_timelist[0]['raw']:
            if self.colormax_set:  # the max color value of time series data
                pass
            else:
                self.colormax = np.ceil(self.rawdata[0:500, 0].max())  # part of the detetor area
                self.colormax_set == True

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
            self.intdata_smoothed = scipy.signal.savgol_filter(intdata_clipped,
                                                  int(self.parameters['integrated']['Savitzky-Golay window'].setvalue),
                                                  int(self.parameters['integrated']['Savitzky-Golay order'].setvalue))
            self.data_scale('integrated', 'smoothed', intqaxis_clipped, self.intdata_smoothed)

        if 'find peaks' in self.curve_timelist[0]['integrated']:
            if hasattr(self, 'intdata_smoothed'):
                peaks, peak_properties = self.find_peak_conditions(self.intdata_smoothed)
                self.data_scale('integrated', 'find peaks', intqaxis_clipped[peaks], self.intdata_smoothed[peaks])
            else:
                peaks, peak_properties = self.find_peak_conditions(self.intdata)
                self.data_scale('integrated', 'find peaks', intqaxis_clipped[peaks], intdata_clipped[peaks])

            self.data_timelist[0]['integrated']['find peaks'].pen = None
            self.data_timelist[0]['integrated']['find peaks'].symbol = 't'
            self.data_timelist[0]['integrated']['find peaks'].symbolsize = 20
            
        # time series
        if 'pointer' in self.curve_timelist[0]['time series']:
            self.plot_pointer('time series', 0, self.index, 't2', 15)

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

class XRD_INFORM_2(XRD):
    def __init__(self, path_name_widget):
        super(XRD_INFORM_2, self).__init__(path_name_widget)
        if self.intfile_appendix == '_resultFile.h5': # this mode use the cluster
            self.intfile_appendix = '_resultCluster.h5'
            self.exportfile.replace('File', 'Cluster')

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

            if ev2nm / wavelength * 10 < 13000: # low or high energy
                index_sel = np.arange(0,I0.shape[0],2)
                E0 = data4xrd[0,0]
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
            if self.fileshort == winobj.path_name_widget[key]['raw file'].text() and \
                    self.ponifile.split('\\')[-1] == winobj.path_name_widget[key]['PONI file'].text():
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

                    if wavelength < 13000:  # low or high energy
                        self.entrytimesec = entrytimesec[0::2,:]
                    else:
                        self.entrytimesec = entrytimesec[1::2,:]
                else:
                    print('you need to process a bundled XAS data first')

        return [self.entrytimesec[0, 0], self.entrytimesec[-1, 1]]

class XRD_INFORM_2_ONLY(XRD):
    def __init__(self, path_name_widget):
        super(XRD_INFORM_2_ONLY, self).__init__(path_name_widget)
        if self.intfile_appendix == '_resultFile.h5': # this mode use the cluster
            self.intfile_appendix = '_resultCluster.h5'
            self.exportfile.replace('File', 'Cluster')

    def plot_from_prep(self, winobj):  # do integration; some part should be cut out to make a new function
        # if winobj.slideradded == False:
        winobj.setslider()
        # norm_result = []
        # the following file stores the data needed for normalizationn of xrd according to I0 and mu d
        data4xrd_file = os.path.join(self.directory, 'raw', self.fileshort[0:-6] + '.h5')

        if os.path.isfile(data4xrd_file):
            # the xrd data point is always the first one and the middle one of a xas spectrum
            # the ratio of log(I0/I1) is 1.5 for the middle one and 1.42 for the first one. to make it 0.6, add 2.02
            # Justus method to normalize. the constant is based on if the observation background can be smoothed successfully
            # or basically to make the absorption of the materials reach calculated value, in this case 0.6, assuming
            # thickness and components.
            # in essence, this method allow the intensity normalized by the flux, I0, and the amount of material, mu*d

            with h5py.File(data4xrd_file, 'r') as f:
                I1 = np.zeros(f['I1'].shape[-1], dtype = 'float32')
                f['I1'].read_direct(I1)
                I0a = np.zeros(f['I1'].shape[-1], dtype='float32')
                f['I0a'].read_direct(I0a)
                I0b = np.zeros(f['I1'].shape[-1], dtype='float32')
                f['I0a'].read_direct(I0b)
                I0 = I0a + I0b
                energy = np.zeros(f['I1'].shape[-1], dtype='float32')
                f['energy'].read_direct(energy)

            with open(self.ponifile, 'r') as f:
                wavelength = float(f.readlines()[-1].splitlines()[0].partition(' ')[2]) * 1e10 # in Angstrom

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

            self.output_intg(q * wavelength / self.wavelength * 10, # now in inverse nm, to be consistent with olde files
                             raw_result,
                             raw_result / I0[:,None] / (np.log(I0[:,None] / I1[:,None]) + 2.02),
                             self.wavelength, 50)

            self.plot_from_load(winobj)

        else:
            print('you need to double check your file name')

        # file.close()

    def time_range(self, winobj):
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
                data4xrd_file = os.path.join(self.directory, 'raw', self.fileshort[0:-6] + '.h5')
                if os.path.isfile(data4xrd_file):
                    with h5py.File(data4xrd_file, 'r') as f:
                        endtime = time.mktime(
                            datetime.strptime(f['time'][0].decode(), '%Y-%m-%d %H:%M:%S.%f').timetuple())
                        durations = np.zeros(f['integration_time'].shape[-1],dtype='float32')
                        f['integration_time'].read_direct(durations)

                    # self.entrytimesec = (startime + np.array([np.arange(0,len(durations)),
                    #                                         np.arange(1,len(durations) + 1)]) * durations[0] * 1e-6).T # check the unit!!!
                    self.entrytimesec = (endtime - np.array([np.arange(len(durations) + 1, 0, -1),
                                                              np.arange(len(durations), -1, -1 )]) * durations[0] * 1e-6).T  # check the unit!!!

                else:
                    print('you need to double check your file name')

        return [self.entrytimesec[0, 0], self.entrytimesec[-1, 1]]

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
                        del self.filekeys[self.filekeys.index(key)]

                self.entrytimesec = np.array(self.entrytimesec)

        return [self.entrytimesec[0, 0], self.entrytimesec[-1, 1]]

class Optic(Methods_Base):
    def __init__(self, path_name_widget):
        super(Optic, self).__init__()
        self.directory = path_name_widget['directory'].text()
        self.fileshort = path_name_widget['raw file'].text()
        self.exportdir = os.path.join(self.directory, 'process', self.fileshort)
        if not os.path.isdir(self.exportdir): os.mkdir(self.exportdir)
        filename = os.path.join(self.directory, 'raw', self.fileshort + '.qvd')
        # qvt: double (time in ms), int16 (time span/data), double (dummy)
        timename = os.path.join(self.directory, 'raw', self.fileshort + '.qvt')
        timefile = os.path.join(self.directory, 'raw', self.fileshort + '_info.txt')

        if int(self.directory.split(os.sep)[1][0:4]) > 2021: # new spectrometer
            self.channelnum = 2068
            ch = np.arange(0, self.channelnum)
            A = 194, 85605
            B = 0, 5487633
            C = -5, 854113E-05
            D = 3, 1954566E-09
            self.channels = A + ch * B + ch ** 2 * C + ch ** 3 * D # calibrated numbers
        else: # old spectrometer
            self.channelnum = 2048
            self.channels = np.linspace(200, 1100, self.channelnum) # an estimation

        if os.path.isfile(timefile): # new time stamp since 20220660
            with open(timefile, 'r') as f: lines = f.readlines()
            self.end_time = time.mktime(datetime.strptime(lines[1].split('stamp: ')[1][:-1], '%Y-%m-%d %H:%M:%S').timetuple())
        else:
            self.end_time = os.path.getmtime(os.path.join(self.directory, 'raw', self.fileshort + '.qvt'))  # in seconds

        self.data = self.read_qvd(filename)
        self.datatime = self.read_qvt(timename)
        self.entrytimesec = []
        self.method_name = []

    def time_range(self, winobj):
        for key in winobj.path_name_widget:  # to distinguish xrd_1, xrd_2
            if self.fileshort == winobj.path_name_widget[key]['raw file'].text():
                self.method_name = key

        if self.entrytimesec == []:  # start, end time in seconds
            start_time = self.end_time - self.datatime[-1, 0] / 1000 - self.datatime[-1, 1] / 1000  # original in milisecond

            for index in range(self.data.shape[0]):
                self.entrytimesec.append(start_time + self.datatime[index, 0] / 1000 + [0, self.datatime[index, 1] / 1000])

            self.entrytimesec = np.array(self.entrytimesec) # , dtype=int)

        # return [int(self.entrytimesec[0, 0]), int(self.entrytimesec[-1, 1])]
        return [self.entrytimesec[0, 0], self.entrytimesec[-1, 1]]

    def read_qvt(self, qvtfilename):
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

    def plot_optic_2D(self, step, data_norm, pw):
        xticklabels = []
        for tickvalue in np.arange(self.channels[0], self.channels[-1],step):
            xticklabels.append((int(data_norm.shape[1] * (tickvalue - self.channels[0]) /
                                    (self.channels[-1] - self.channels[0])), "{:4.1f}".format(tickvalue)))

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

        for t in range(data_norm.shape[0]):
            # if len(np.where(data_norm[t,:] == max(data_norm[t,:]))[0]) > \
            #     int(self.linedit['time series']['filter criterion']) and t > 0:
            if max(data_norm[t,:]) > np.log10(float(self.linedit['time series']['filter criterion'])) and t > 0:
                data_norm[t,:] = (data_norm[t - 1,:] + data_norm[t + 1, :]) / 2 # linear interpolation between adjacent data

        self.img_ts = pg.ImageItem(image=data_norm.transpose()) # need log here?
        self.img_ts.setZValue(-100)  # as long as less than 0
        pw.addItem(self.img_ts)
        color_map = pg.colormap.get('CET-R4')
        # data_norm[isnan(data_norm)] = 0
        # data_norm = ma.array(data_norm, mask=np.isinf(data_norm))
        mode = 'time series'
        self.color_bar_ts = pg.ColorBarItem(values=(float(self.linedit[mode]['z min']),
                                                    float(self.linedit[mode]['z max'])), cmap=color_map)
        self.color_bar_ts.setImageItem(self.img_ts, pw)

class PL(Optic):
    def __init__(self, path_name_widget):
        super(PL, self).__init__(path_name_widget)
        self.availablemethods = ['raw', 'time series', 'fit-T']
        self.availablecurves['raw'] = ['show','fit']
        self.availablecurves['time series'] = ['pointer']
        self.availablecurves['fit-T'] = ['Height', 'FWHM']
        self.axislabel = {'raw':{'bottom':'wavelength/nm',
                                 'left':'log10(intensity)'},
                          'time series': {'bottom': 'wavelength/nm',
                                  'left': 'Data number'},
                          'fit-T': {'bottom':'Data number',
                                     'left':''}}
        self.ini_data_curve_color()  # this has to be at the end line of each method after a series of other attributes

    def plot_from_prep(self, winobj):
        pass
        # tempfile = os.path.join(self.directory, 'process', self.fileshort + '_PL_data.txt')
        # if os.path.isfile(tempfile):
        #     np.savetxt(tempfile, self.data)
        # self.plot_from_load(winobj)

    def plot_from_load(self, winobj):
        winobj.setslider()
        self.curvedict['time series']['pointer'].setChecked(True)
        pw = winobj.gdockdict[self.method_name].tabdict['time series'].tabplot
        if self.checksdict['time series'].isChecked():
            self.plot_optic_2D(100, self.data, pw)

    def data_process(self, para_update):
        self.dynamictitle = 'data' + str(self.index + 1) + '\t start:' + self.startime + '\t end:' + self.endtime
        # raw
        if 'show' in self.curve_timelist[0]['raw']:
            self.data_timelist[0]['raw']['show'].data = \
                np.transpose([self.channels, np.log10(self.data[self.index,:])]) # 200 nm - 1.1 um

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
        self.actions = {'time series': {'Export time series (Ctrl+X)': self.export_ts}}

        self.align_data = int(path_name_widget['align data number'].text())
        self.to_time = time.mktime(datetime.strptime(path_name_widget['to time'].text(), '%Y-%m-%dT%H:%M:%S').timetuple()) # in second
        self.aligned = False
        self.ini_data_curve_color()  # this has to be at the end line of each method after a series of other attributes

        # unique to Refl
        self.refcandidate = []
        # self.parameters = {'time series': {'normalize': Paraclass('original', ['original', 'normalized'])}}
        self.linedit = {'time series':{'z min':'-2',
                                       'z max':'0.1',
                                       'filter criterion':'2'}} # larger than 2 after referenced.

    def export_ts(self, winobj):
        # must guarantee reference is clicked...
        resultfile = h5py.File(os.path.join(self.exportdir, 'refl_norm_time_series.h5'), 'w')
        if len(self.refcandidate) > 0:
            resultfile.create_dataset('refl_norm', data=np.array(self.data / self.refcandidate))
        resultfile.create_dataset('Wave length/nm', data=self.channels)
        resultfile.close()

    def plot_from_prep(self, winobj):
        winobj.setslider()
        if self.align_data and not self.aligned: # there is a little problem here: what if you want to align it again?
            time_diff = self.entrytimesec[self.align_data, 0] - self.to_time
            self.entrytimesec = self.entrytimesec - time_diff
            self.align_data = True
        # tempfile = os.path.join(self.directory, 'process', self.fileshort + '_Refl_data.txt')
        # if os.path.isfile(tempfile):
        #     np.savetxt(tempfile, self.data)
        # self.plot_from_load(winobj)

    def plot_from_load(self, winobj): # this needs pre-select a good reference
        winobj.setslider()
        self.curvedict['time series']['pointer'].setChecked(True)
        pw = winobj.gdockdict[self.method_name].tabdict['time series'].tabplot
        if self.checksdict['time series'].isChecked(): # and 'reference' in self.curve_timelist[0]['raw']:
            if self.checksdict['raw'].isChecked() and self.curvedict['raw']['reference'].isChecked():
                self.plot_optic_2D(100, np.log10(self.data / self.refcandidate), pw)
            else:
                self.plot_optic_2D(100, np.log10(self.data), pw) # / self.refcandidate

    def data_process(self, para_update):
        self.dynamictitle = 'data' + str(self.index + 1) + '\t start:' + self.startime + '\t end:' + self.endtime
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


