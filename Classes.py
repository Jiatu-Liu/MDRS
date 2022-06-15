import os
import sys
import glob
# can handle multiple samples data; reference spectrum
import h5py
import numpy as np
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

from azint import AzimuthalIntegrator

from struct import unpack
import struct

from scipy.signal import savgol_filter, savgol_coeffs
from scipy.signal import find_peaks, peak_widths

sys.path.insert(0,r"C:\Users\jialiu\gsas2full\GSASII")
import GSASIIindex
import GSASIIlattice

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
        self.dockobj.setMinimumWidth(winobj.screen_width * .3)
        winobj.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.dockobj)
        if len(winobj.gdockdict) > 2: # only accommodate two docks
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

    def gentab(self, dockobj, methodobj): # generate a tab for a docking graph
        self.graphtab = pg.GraphicsLayoutWidget()
        dockobj.docktab.addTab(self.graphtab, self.label)
        self.tabplot = self.graphtab.addPlot()
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
        # LineEdit
        if tabname in methodobj.linedit:
            for key in methodobj.linedit[tabname]:
                self.itemlayout.removeWidget(methodobj.linewidgets[tabname][key])

        # parameters
        if tabname in methodobj.parameters: # normalized, chi(k)
            for key in methodobj.parameters[tabname]: # rbkg, kmin,...
                self.itemlayout.removeWidget(methodobj.parawidgets[tabname][key])
                self.itemlayout.removeWidget(methodobj.paralabel[tabname][key])

        # actions
        if tabname in methodobj.actions:
            for key in methodobj.actions[tabname]:
                self.itemlayout.removeWidget(methodobj.actwidgets[tabname][key])

        # checkboxes
        for key in methodobj.availablecurves[tabname]:
            if methodobj.curvedict[tabname][key].isChecked: methodobj.curvedict[tabname][key].setChecked(False)
            self.itemlayout.removeWidget(methodobj.curvedict[tabname][key])

        methodobj.curvedict[tabname] = {} # del the whole dict, maybe not necessary, because this dict will be embodied

        self.itemlayout.removeItem(self.curvespacer)

    # tabname, e.g. raw, norm,...; methodobj, e.g. an XAS obj; for 'I0', 'I1',...
    def curvechecks(self, tabname, methodobj, winobj):
        # checkboxes
        for key in methodobj.availablecurves[tabname]:
            methodobj.curvedict[tabname][key] = QCheckBox(key)
            methodobj.curvedict[tabname][key].stateChanged.connect(winobj.graphcurve)
            self.itemlayout.addWidget(methodobj.curvedict[tabname][key])

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
                    self.itemlayout.addWidget(methodobj.paralabel[tabname][key])
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
            for key in methodobj.linedit[tabname]:
                methodobj.linewidgets[tabname][key] = QLineEdit(methodobj.linedit[tabname][key])
                methodobj.linewidgets[tabname][key].setPlaceholderText(key)
                self.itemlayout.addWidget(methodobj.linewidgets[tabname][key])

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
        self.pre_ts_btn_text = 'Update by parameters'

        self.maxhue = 100  # 100 colorhues
        self.huestep = 7 # color rotation increment

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
            self.startime = datetime.fromtimestamp(self.entrytimesec[self.index, 0]).strftime("%Y-%m-%dT%H:%M:%S")
            self.endtime = datetime.fromtimestamp(self.entrytimesec[self.index, 1]).strftime("%H:%M:%S")
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
        self.pre_ts_btn.clicked.connect(lambda: self.plot_from_prep(winobj))
        self.load_ts_btn = QPushButton('Load time series')
        self.load_ts_btn.clicked.connect(lambda: self.plot_from_load(winobj))
        winobj.subcboxverti[method].addWidget(self.pre_ts_btn)
        winobj.subcboxverti[method].addWidget(self.prep_progress)
        winobj.subcboxverti[method].addWidget(self.load_ts_btn)
        self.itemspacer = QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
        winobj.subcboxverti[method].addItem(self.itemspacer)

    def deltabchecks(self, winobj, method): # del above checks
        for key in self.availablemethods:  # raw, norm,...
            if self.checksdict[key].isChecked: self.checksdict[key].setChecked(False)
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
    def __init__(self, path_name_dict):
        # below are needed for each method (xas, xrd,...)
        super(XAS, self).__init__()
        self.availablemethods = ['raw', 'normalizing', 'normalized', 'chi(k)', 'chi(r)', 'E0-T', 'chi(r)-T']
        self.availablecurves['raw'] = ['I0', 'I1']
        self.availablecurves['normalizing'] = ['mu','filter by time','filter by energy','pre-edge','post-edge']
        self.availablecurves['normalized'] = ['normalized mu','filter by time','filter by energy','reference','post-edge bkg']
        self.availablecurves['chi(k)'] = ['chi-k','window']
        self.availablecurves['chi(r)'] = ['chi-r','Re chi-r','Im chi-r']
        self.availablecurves['E0-T'] = ['pointer']
        self.availablecurves['chi(r)-T'] = ['pointer']
        self.directory = path_name_dict['directory'].text()
        self.fileshort = path_name_dict['raw file'].text()
        self.filename = os.path.join(self.directory, 'raw', self.fileshort + '.h5')
        self.axislabel = {'raw':{'bottom':'Energy/eV',
                                 'left':'<font> &mu; </font>'},
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
                          'chi(r)-T':{'bottom':'<font> R / &#8491; </font>',
                                      'left':'Data number'}}

        self.ini_data_curve_color()

        # below are xas specific attributes
        self.E0 = [] # time series
        self.rspace = [] # time series
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
        self.parameters = {'normalizing':{'Savitzky-Golay window (time)':Paraclass(values=(11,5,100,2)),
                                          'Savitzky-Golay order (time)':Paraclass(values=(1,1,4,1)),
                                          'Savitzky-Golay window (energy)': Paraclass(values=(11, 5, 100, 2)),
                                          'Savitzky-Golay order (energy)': Paraclass(values=(1, 1, 4, 1))},
                           'normalized':{'Savitzky-Golay window (time)':Paraclass(values=(11,5,100,2)),
                                         'Savitzky-Golay order (time)':Paraclass(values=(1,1,4,1)),
                                         'Savitzky-Golay window (energy)': Paraclass(values=(11, 5, 100, 2)),
                                         'Savitzky-Golay order (energy)': Paraclass(values=(1, 1, 4, 1)),
                                         'rbkg':Paraclass(values=(1,1,3,.1))},
                           'chi(k)':{'kmin':Paraclass(values=(.9,0,3,.1)),
                                     'kmax':Paraclass(values=(6,3,20,.1)),
                                     'dk':Paraclass(values=(0,0,1,1)),
                                     'window':Paraclass(strings=('Hanning',['Hanning','Parzen'])),
                                     'kweight':Paraclass(values=(1,0,2,1))}}

        self.actions = {'normalizing':{'filter all': self.filter_all_normalizing},
                        'normalized':{'filter all': self.filter_all_normalized}}

        # self.average = int(path_name_dict['average (time axis)'].text())  # number of average data points along time axis, an odd number
        self.range = path_name_dict['energy range (eV)'].text() # useful for time_range!
        self.energy_range = [int(self.range.partition("-")[0]), int(self.range.partition("-")[2])]  # for Pb L and Br K edge combo spectrum
        self.exportdir = os.path.join(self.directory, 'process', self.fileshort + \
                               '_range_{}_{}eV'.format(self.energy_range[0], self.energy_range[1]), '')
        self.ref_mu = []
        self.filter_flag_normalizing = False
        self.filter_flag_normalized = False

    def filter_all_normalizing(self, winobj): # make some warning sign here
        qbtn = self.sender()
        if self.filter_flag_normalizing:
            self.filter_flag_normalizing = False
            qbtn.setText('Go to click Update by parameters')
        else:
            self.filter_flag_normalizing = True
            qbtn.setText('filter all')

    def filter_all_normalized(self, winobj):
        qbtn = self.sender()
        if self.filter_flag_normalized:
            self.filter_flag_normalized = False
            qbtn.setText('Go to click Update by parameters')
        else:
            self.filter_flag_normalized = True
            qbtn.setText('filter all')

    def plot_from_prep(self, winobj): # generate larch Group for all data ! not related to 'load from prep' any more
        for index in range(self.entrydata.shape[0]): # 0,1,...,self.entrydata.shape[0] - 1
            Energy = self.entrydata[index, 0, ::]
            mu = np.log10(self.entrydata[index, 1, ::] / self.entrydata[index, 2, ::])

            if len(self.grouplist) is not self.entrydata.shape[0]:
                self.grouplist.append(Group(name='spectrum' + str(index)))

            if self.filter_flag_normalizing:
                mu = self.filter_single_point(index, Energy, mu, 'normalizing')

            self.exafs_process_single(index, Energy, mu)

            if self.filter_flag_normalized: # this step will double the waiting time, very sad.
                mu = self.filter_single_point(index, Energy, self.grouplist[index].norm, 'normalized')
                self.exafs_process_single(index, Energy, mu)

            self.prep_progress.setValue(int((index + 1) / self.entrydata.shape[0] * 100))
            # self.parameters['chi(k)']['kmax'].upper = self.grouplist[index].k[-1]
            # rspacexlen = len(self.grouplist[index].r)
        # self.plot_from_load(winobj)
        with open(os.path.join(self.exportdir, self.fileshort + '_Group_List'), 'wb') as f:
            pickle.dump(self.grouplist, f, -1)  # equals to pickle.HIGHEST_PROTOCOL

    def exafs_process_single(self, index, Energy, mu):
        pre_edge(Energy, mu, group=self.grouplist[index])
        autobk(Energy, mu, rbkg=self.parameters['normalized']['rbkg'].setvalue, group=self.grouplist[index])
        xftf(self.grouplist[index].k, self.grouplist[index].chi,
             kmin=self.parameters['chi(k)']['kmin'].setvalue, kmax=self.parameters['chi(k)']['kmax'].setvalue,
             dk=self.parameters['chi(k)']['dk'].setvalue, window=self.parameters['chi(k)']['window'].choice,
             kweight=self.parameters['chi(k)']['kweight'].setvalue,
             group=self.grouplist[index])
    # rewrite for different mode of collection; feed read_data_index the whole data in memory: entrydata (xas, pl, refl),
    # and/or a file in \process (xrd)
    def time_range(self, winobj):
        for key in winobj.path_name_dict: # to distinguish xas_1, xas_2
            if 'energy range (eV)' in winobj.path_name_dict[key]:
                if self.range == winobj.path_name_dict[key]['energy range (eV)'].text():
                    self.method_name = key

        # read in time
        if self.entrytimesec == []:  # start, end time in seconds
            tempfile = os.path.join(self.exportdir, self.fileshort + '_time_in_seconds')
            if os.path.isfile(tempfile):
                with open(tempfile, 'r') as f:
                    self.entrytimesec = np.loadtxt(f)
            else: # for SDC data in \balder\20220720\
                self.file = h5py.File(self.filename, 'r')
                self.filekeys = list(self.file.keys())
                del self.filekeys[self.filekeys.index('ColumnInfo')]
                del self.filekeys[self.filekeys.index('spectrum0')]
                del self.filekeys[self.filekeys.index('timestamp0')]
                for index in range(1, int(len(self.filekeys) / 2 + 1)):
                    timesecond = time.mktime(datetime.strptime(self.file['timestamp' + str(index)][()].decode(),'%c').timetuple())
                    self.entrytimesec.append([timesecond, timesecond + 1]) # one second collection time
                    del self.filekeys[self.filekeys.index('timestamp' + str(index))]

                self.entrytimesec = np.array(self.entrytimesec)
                # self.entrytimesec = self.entrytimesec[self.entrytimesec[:,0].argsort()]

        # read in data
        if self.entrydata == []:  # Energy, I0, I1
            if os.path.isdir(self.exportdir): # can also do deep comparison from here
                for tempname in sorted(glob.glob(self.exportdir + '*_spectrum'), key = os.path.getmtime): # in ascending order
                    with open(tempname, 'r') as f:
                        self.entrydata.append(np.loadtxt(f).transpose())

            else: # for SDC data in \balder\20220720\
                os.mkdir(self.exportdir)
                with open(tempfile, 'w') as f:
                    np.savetxt(f, self.entrytimesec)

                for index in range(1, self.entrytimesec.shape[0] + 1): # 1, 2,...
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
                    data = self.file['spectrum' + str(index)]  # specific !
                    data_single = np.zeros((data.shape[0], data.shape[1]), dtype='float32')
                    data.read_direct(data_single)
                    data_start = np.where(data_single[::,0] - self.energy_range[0] >= 0)[0][0]
                    data_end = np.where(self.energy_range[1] - data_single[::,0] >= 0)[0][-1]
                    data_range = np.s_[data_start:data_end]
                    tempdata = [data_single[data_range,0], data_single[data_range,1], data_single[data_range,2]]
                    self.entrydata.append(tempdata)
                    with open(os.path.join(self.exportdir, self.fileshort + '_{}_spectrum'.format(index)), 'w') as f:
                        np.savetxt(f, np.array(tempdata).transpose(), header='Energy, I0, I1')

            self.entrydata = np.array(self.entrydata)
            tempfile = os.path.join(self.exportdir, self.fileshort + '_Group_List')
            if os.path.isfile(tempfile):
                with open(tempfile, 'rb') as f:
                    self.grouplist = pickle.load(f)
            else:
                self.plot_from_prep(winobj)

        return [int(self.entrytimesec[0, 0]), int(self.entrytimesec[-1, 1])]

    def plot_from_load(self, winobj): # for time series
        for index in range(self.entrydata.shape[0]):
            self.E0.append(self.grouplist[index].e0)
            self.rspace.append(self.grouplist[index].chir_mag)
            self.prep_progress.setValue(int((index + 1) / self.entrytimesec.shape[0] * 100))

        if self.checksdict['E0-T'].isChecked():
            winobj.gdockdict[self.method_name].tabdict['E0-T'].tabplot.plot(range(self.entrytimesec.shape[0]), self.E0,
                                                                 symbol='o', symbolSize=10, symbolPen='r', name='E0-T')

        if self.checksdict['chi(r)-T'].isChecked():
            r_range = len(self.grouplist[0].r) # set the xticklabels for the image
            r_max = self.grouplist[0].r[-1]
            xticklabels = []
            for tickvalue in range(0,r_max,2): # hope it works
                xticklabels.append((r_range / r_max * tickvalue, str(tickvalue)))

            xticks = winobj.gdockdict[self.method_name].tabdict['chi(r)-T'].tabplot.getAxis('bottom')
            xticks.setTicks([xticklabels])
            rspce_array = np.array(self.rspace)
            img = pg.ImageItem(image=np.transpose(rspce_array))
            winobj.gdockdict[self.method_name].tabdict['chi(r)-T'].tabplot.addItem(img)
            color_map = pg.colormap.get('CET-R4')
            color_bar = pg.ColorBarItem(values=(0, rspce_array.max()), cmap=color_map)  # the range needs to update for different data
            color_bar.setImageItem(img, winobj.gdockdict[self.method_name].tabdict['chi(r)-T'].tabplot)

    def filter_single_point(self, data_index, Energy, mu_filtered, mode):
        if mode == 'normalizing':
            mu = lambda index: np.log10(self.entrydata[index, 1, ::] / self.entrydata[index, 2, ::])

        if mode == 'normalized':
            mu = lambda index: self.grouplist[index].norm

        if 'filter by time' in self.curve_timelist[0][mode]:
            sg_win = int(self.parameters[mode]['Savitzky-Golay window (time)'].setvalue)
            sg_order = int(self.parameters[mode]['Savitzky-Golay order (time)'].setvalue)
            sg_data = []
            if data_index < (sg_win - 1) / 2:# padding with the first data
                for index in np.arange(0, (sg_win - 1) / 2 - data_index, dtype=int): sg_data.append(mu(0))
                for index in np.arange(0, (sg_win + 1) / 2 + data_index, dtype=int): sg_data.append(mu(index))
                mu_filtered = savgol_coeffs(sg_win, sg_order, use='dot').dot(np.array(sg_data))

            elif data_index >  self.entrydata.shape[2] - (sg_win + 1) / 2: # padding with the last data
                for index in np.arange(self.entrydata.shape[2], data_index + (sg_win + 1) / 2, dtype=int): sg_data.append(mu(-1))
                for index in np.arange(data_index - (sg_win - 1) / 2, self.entrydata.shape[2], dtype=int): sg_data.append(mu(index))
                mu_filtered = savgol_coeffs(sg_win, sg_order, use='dot').dot(np.array(sg_data))

            else:
                for index in np.arange(data_index - (sg_win - 1) / 2, data_index + (sg_win + 1) / 2, dtype=int): sg_data.append(mu(index))
                mu_filtered = savgol_coeffs(sg_win, sg_order, use='dot').dot(np.array(sg_data))

            self.data_timelist[0][mode]['filter by time'].data = np.transpose([Energy, mu_filtered])

        if 'filter by energy' in self.curve_timelist[0][mode]:
            sg_win = int(self.parameters[mode]['Savitzky-Golay window (energy)'].setvalue)
            sg_order = int(self.parameters[mode]['Savitzky-Golay order (energy)'].setvalue)
            mu_filtered = scipy.signal.savgol_filter(mu_filtered, sg_win, sg_order, mode='nearest')
            self.data_timelist[0][mode]['filter by energy'].data = np.transpose([Energy, mu_filtered])

        return mu_filtered

    def data_process(self, para_update): # for curves, embody data_timelist, if that curve exists
        # Energy, I0, I1 = self.read_data_time() # need to know self.slidervalue
        Energy = self.entrydata[self.index, 0, ::]
        I0 = self.entrydata[self.index, 1, ::]
        I1 = self.entrydata[self.index, 2, ::]
        mu = np.log10(I0 / I1)
        mu_filtered = mu
        if para_update: self.exafs_process_single(self.index, Energy, mu_filtered)
        self.dynamictitle = 'data' + str(self.index + 1) + '\t start:' + self.startime + '\t end:' + self.endtime

        # raw
        if 'I0' in self.curve_timelist[0]['raw']:
            self.data_timelist[0]['raw']['I0'].data = np.transpose([Energy, I0])
        if 'I1' in self.curve_timelist[0]['raw']:
            self.data_timelist[0]['raw']['I1'].data = np.transpose([Energy, I1])
        # norm
        if 'mu' in self.curve_timelist[0]['normalizing']: self.data_timelist[0]['normalizing']['mu'].data = np.transpose([Energy, mu])

        mu_filtered = self.filter_single_point(self.index, Energy, mu_filtered, 'normalizing')

        if 'pre-edge' in self.curve_timelist[0]['normalizing']:
            self.data_timelist[0]['normalizing']['pre-edge'].data = np.transpose([Energy, self.grouplist[self.index].pre_edge])
        if 'post-edge' in self.curve_timelist[0]['normalizing']:
            self.data_timelist[0]['normalizing']['post-edge'].data = np.transpose([Energy, self.grouplist[self.index].post_edge])

        # normalized
        mu_filtered = self.filter_single_point(self.index, Energy, mu_filtered, 'normalized')

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
            self.data_timelist[0]['E0-T']['pointer'].data = np.array([[self.index], [self.E0[self.index]]]).transpose()
            self.data_timelist[0]['E0-T']['pointer'].symbol = '+'
            self.data_timelist[0]['E0-T']['pointer'].symbolsize = 30
        # chi(r)-T
        if 'pointer' in self.curve_timelist[0]['chi(r)-T']:
            self.data_timelist[0]['chi(r)-T']['pointer'].data = np.array([[0], [self.index]]).transpose()
            self.data_timelist[0]['chi(r)-T']['pointer'].symbol = 't2'
            self.data_timelist[0]['chi(r)-T']['pointer'].symbolsize = 15

class XRD(Methods_Base):
    def __init__(self, path_name_dict):
        super(XRD, self).__init__()
        self.availablemethods = ['raw', 'integrated', 'time series']
        self.availablecurves['raw'] = ['show image']
        self.availablecurves['integrated'] = ['original', 'truncated', 'smoothed', 'find peaks']
        self.availablecurves['time series'] = ['pointer']
        self.directory = path_name_dict['directory'].text()
        self.fileshort = path_name_dict['raw file'].text()
        self.intfile_appendix = path_name_dict['integration file appendix'].text()
        self.start_index = int(path_name_dict['start from'].text()) - 1
        self.ponifile = os.path.join(self.directory, 'process', path_name_dict['PONI file'].text())
        self.filename = os.path.join(self.directory, 'raw', self.fileshort + '.h5')
        self.colormax_set = False
        self.axislabel = {'raw':{'bottom':'',
                                 'left':''},
                          'integrated': {'bottom': '<font> q / &#8491; </font> <sup> -1 </sup>',
                                  'left': 'Intensity'},
                          'time series': {'bottom': '<font> q / &#8491; </font> <sup> -1 </sup>',
                                  'left': 'Data number'}}

        self.ini_data_curve_color()

        # unique to xrd
        # note that these parameter boundaries can be changed
        self.bravaisNames = ['Cubic-F', 'Cubic-I', 'Cubic-P', 'Trigonal-R', 'Trigonal/Hexagonal-P',
                             'Tetragonal-I', 'Tetragonal-P', 'Orthorhombic-F', 'Orthorhombic-I', 'Orthorhombic-A',
                             'Orthorhombic-B', 'Orthorhombic-C',
                             'Orthorhombic-P', 'Monoclinic-I', 'Monoclinic-A', 'Monoclinic-C', 'Monoclinic-P',
                             'Triclinic']

        self.parameters = {'integrated': {'scale': Paraclass(strings=('log10', ['log10', 'sqrt', 'linear'])),
                                          'clip head': Paraclass(values=(0,0,100,1)),
                                          'clip tail': Paraclass(values=(1, 1, 100, 1)),
                                          'Savitzky-Golay window': Paraclass(values=(11,5,100,2)),
                                          'Savitzky-Golay order': Paraclass(values=(1,1,4,1)),
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
                                                                         'phase9','phase10','phase11','phase12']))}}
        self.actions = {'integrated':{"interpolate along q": self.interpolate_data,
                                      "find peaks (Ctrl+F)": self.find_peak_all},
                       'time series':{"clear rainbow map": self.show_clear_ts,
                                      "range start": self.range_select,
                                      "catalog peaks (Ctrl+T)": self.catalog_peaks,
                                      "assign phases (Ctrl+A)": self.assign_phases,
                                      "clear peaks": self.show_clear_peaks,
                                      "index phases (Ctrl+I)": self.index_phases}} # button name and function name

        self.linedit = {'time series': {'range start': '100',
                                        'range end': '200'}}

        self.pre_ts_btn_text = 'Do Batch Integration'
        self.exportfile = os.path.join(self.directory, 'process', self.fileshort + self.intfile_appendix)
        self.entrytimesec = []
        self.cells_sort = {}

        if os.path.isfile(self.exportfile): # read in data at the beginning
            intfile = h5py.File(self.exportfile, 'r')
            intdata_all = intfile['rawresult']
            self.intdata_ts = np.zeros((intdata_all.shape[0], intdata_all.shape[1]), dtype='float32')
            intdata_all.read_direct(self.intdata_ts)
            self.intqaxis = np.array(list(intfile['info/q(Angstrom)']))
            intfile.close()

    def interpolate_data(self, winobj):
        # to densify data along q to make following peaks cataloguing more effective incase q is not enough
        pass

    def find_peak_all(self, winobj):
        q_start = int(self.parameters['integrated']['clip head'].setvalue)
        q_ending = int(self.parameters['integrated']['clip tail'].setvalue)
        peaks_q = []
        peaks_number = []
        self.peaks_index = []
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
        if "clear rainbow map" == self.actwidgets['time series']['clear rainbow map'].text():
            self.actwidgets['time series']['clear rainbow map'].setText("show rainbow map")
            if hasattr(self, 'color_bar_ts'):
                winobj.gdockdict[self.method_name].tabdict['time series'].tabplot.removeItem(self.img_ts)
                self.color_bar_ts.close()

        else:
            self.actwidgets['time series']['clear rainbow map'].setText("clear rainbow map")
            self.plot_from_load(winobj)

    def show_clear_peaks(self, winobj):
        pw = winobj.gdockdict[self.method_name].tabdict['time series'].tabplot
        if 'clear peaks' == self.actwidgets['time series']['clear peaks'].text():
            self.actwidgets['time series']['clear peaks'].setText('show peaks')
            for index in reversed(range(len(pw.items))): # shocked!
                if isinstance(pw.items[index], pg.PlotDataItem):
                    if pw.items[index].name()[0:4] in ['find', 'cata', 'assi']: pw.removeItem(pw.items[index])
                        
        else:
            self.actwidgets['time series']['clear peaks'].setText('clear peaks')
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
        tempwidget = self.actwidgets['time series']['range start']
        if tempwidget.text() == 'range start':
            tempwidget.setText('range end')
            pw.scene().sigMouseClicked.connect(lambda evt, p = pw: self.range_clicked(evt, p))
        elif tempwidget.text() == 'range end':
            tempwidget.setText('Done')
        else:
            tempwidget.setText('range start')
            pw.scene().sigMouseClicked.disconnect()

    def range_clicked(self, evt, pw):
        if pw.sceneBoundingRect().contains(evt.scenePos()):
            mouse_point = pw.vb.mapSceneToView(evt.scenePos()) # directly, we have a viewbox!!!
            if self.actwidgets['time series']['range start'].text() == 'range end':
                self.linewidgets['time series']['range start'].setText(str(int(mouse_point.y())))
            else:
                self.linewidgets['time series']['range end'].setText(str(int(mouse_point.y())))

    def catalog_peaks(self, winobj):
        pw = winobj.gdockdict[self.method_name].tabdict['time series'].tabplot
        i_start = self.linewidgets['time series']['range start'].text()
        if i_start is not '':
            i_start = self.peaks_index.index(int(i_start))

        i_end = self.linewidgets['time series']['range end'].text()
        if i_end is not '':
            i_end = self.peaks_index.index(int(i_end))

        if i_start is not '' and i_end is not '':
            pw.setYRange(i_start, i_end)
            i_start = np.min([i_start, i_end]) # to prevent disorder
            i_end = np.max([i_start, i_end])
            self.peaks_catalog = []
            for index in range(len(self.peaks_all[i_start])):
                self.peaks_catalog.append([[self.peaks_index[i_start],self.peaks_all[i_start][index]]])

            gap_y = int(self.parameters['time series']['gap y tol.'].setvalue)
            gap_x = int(self.parameters['time series']['gap x tol.'].setvalue)
            for index in range(i_start + 1,i_end + 1): # index on data number peaks selected (self.peaks_index)
                # for j in range(min([gap_y, i_end - index])): # index on gap_y tolerence
                search_range = len(self.peaks_catalog)
                for k in range(len(self.peaks_all[index])): # index on all peaks detected within one data
                    add_group = [] # indicator whether to add a new peaks group or not
                    for i in range(search_range): # index on existing peaks groups, this one is constantly changing!!!
                        if np.abs(self.peaks_all[index][k] - self.peaks_catalog[i][-1][1]) <= gap_x and \
                            np.abs(index - self.peaks_catalog[i][-1][0]) <= gap_y: # add to existing peaks group
                            self.peaks_catalog[i].append([self.peaks_index[index],self.peaks_all[index][k]]) # data number (y,x)
                            add_group.append(1)
                        else: add_group.append(0)

                    if np.sum(add_group) == 0: # add a new catalog if this peak belongs to no old catalogs
                        self.peaks_catalog.append([[self.peaks_index[index], self.peaks_all[index][k]]])

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

        with open(self.ponifile, 'r') as f:
            wavelength = float(f.readlines()[-1].splitlines()[0].partition(' ')[2])

        peaks = []
        for index in range(len(phase_int)):
            peaks.append([np.arcsin(wavelength * 1e10 / np.array(phase_d[index]) / 2) * 2 / np.pi * 180,
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

    def read_data_index(self, sub_index): # for raw img
        file = h5py.File(self.filename, 'r')
        rawdata = file['entry/data/data']
        self.rawdata = np.zeros((rawdata.shape[1],rawdata.shape[2]), dtype='uint32')
        rawdata.read_direct(self.rawdata, source_sel=np.s_[sub_index + self.start_index, :, :])
        self.rawdata = np.log10(self.rawdata).transpose()
        file.close()

    def time_range(self, winobj): # for new data collection, linked to xas, through winobj--which is added only for this purpose
        for key in winobj.path_name_dict: # to distinguish xrd_1, xrd_2
            if self.fileshort == winobj.path_name_dict[key]['raw file'].text():
                self.method_name = key

        if self.entrytimesec == []:
            if os.path.isfile(self.exportfile):
                with h5py.File(self.exportfile, 'r') as f:
                    self.entrytimesec = np.zeros((f['info/abs_time_in_sec'].shape[0], f['info/abs_time_in_sec'].shape[1]), dtype='float')
                    f['info/abs_time_in_sec'].read_direct(self.entrytimesec)
            else:
                for key in winobj.methodict: # for xrd-xas correlation
                    hasxas = False
                    if key[0:3] == 'xas':
                        self.sync_xas_name = key
                        hasxas = True

                if hasxas: self.entrytimesec = winobj.methodict[self.sync_xas_name].entrytimesec
                else: print('check the file name, buddy')

        return [int(self.entrytimesec[0, 0]), int(self.entrytimesec[-1, 1])]

    def plot_from_prep(self, winobj): # do integration; some part should be cut out to make a new function
        if winobj.slideradded == False:
            winobj.setslider()

        # add: if there is already the file, load it directly
        ai = AzimuthalIntegrator(self.ponifile, (1065, 1030), 75e-6, 4, [3000,], solid_angle=True)
        result = []
        norm_result = []
        file = h5py.File(self.filename, 'r')
        rawdata = file['entry/data/data']
        mask = np.zeros((rawdata.shape[1], rawdata.shape[2]))
        rawimg = np.zeros((rawdata.shape[1], rawdata.shape[2]), dtype='uint32')
        # this xrd data has to be bind with xas
        with open(self.ponifile, 'r') as f:
            wavelength = float(f.readlines()[-1].splitlines()[0].partition(' ')[2])

        energy = 1239.8419843320025 / wavelength / 1e9 # wavelength in m, change to nm
        for key in winobj.methodict: # for xas-xrd correlation
            if key[0:3] == 'xas':
                self.sync_xas_name = key

        position = np.where(winobj.methodict[self.sync_xas_name].entrydata[0,0,::] - energy > 0)[0][0]
        I0 = winobj.methodict[self.sync_xas_name].entrydata[::, 1, position]
        I1 = winobj.methodict[self.sync_xas_name].entrydata[::, 2, position]

        for sub_index in range(I0.shape[0]):
            index = sub_index + self.start_index
            rawdata.read_direct(rawimg, source_sel=np.s_[index, :, :])
            mask[rawimg == 2 ** 32 - 1] = 1
            intdata = ai.integrate(rawimg, mask=mask)[0]
            result.append(intdata)
            norm_result.append(intdata / I0[sub_index] / np.log(I1[sub_index] / I0[sub_index])) # Mahesh's method to normalize
            self.prep_progress.setValue(int((sub_index + 1) / I0.shape[0] * 100))

        file.close()
        resultfile = h5py.File(self.exportfile, 'w')
        tempgroup = resultfile.create_group('info')
        tempgroup.create_dataset('q(Angstrom)', data=ai.q / 10) # attention to this, ai.q in nm-1, divided by 10 to get A ^-1
        tempgroup.create_dataset('2theta', data=np.arcsin(ai.q * wavelength * 1e9 / np.pi / 4) * 2 / np.pi * 180) # need to re-run!
        tempgroup.create_dataset('abs_time_in_sec', data=self.entrytimesec)
        resultfile.create_dataset('rawresult', data=np.array(result))
        resultfile.create_dataset('normresult',data=np.array(norm_result))
        resultfile.close()
        self.plot_from_load(winobj)

    def plot_from_load(self, winobj): # some part can be cut out for a common function
        #if winobj.slideradded == False:
        winobj.setslider()

        pw = winobj.gdockdict[self.method_name].tabdict['time series'].tabplot

        if self.parameters['time series']['scale'].choice == 'log10':
            intdata_ts = np.log10(self.intdata_ts)
        if self.parameters['time series']['scale'].choice == 'sqrt':
            intdata_ts = np.sqrt(self.intdata_ts)
        if self.parameters['time series']['scale'].choice == 'linear':
            intdata_ts = self.intdata_ts

        if self.checksdict['time series'].isChecked():
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
            self.color_bar_ts = pg.ColorBarItem(values=(0, intdata_ts.max()), cmap=color_map)
            self.color_bar_ts.setImageItem(self.img_ts, pw)

        winobj.slider.setValue(winobj.slider.minimum() + 1)

    def data_scale(self, method_string, data_x, data_y):  # for data_process
        if self.parameters['integrated']['scale'].choice == 'log10':
            self.data_timelist[0]['integrated'][method_string].data = np.transpose([data_x, np.log10(data_y)])
        if self.parameters['integrated']['scale'].choice == 'sqrt':
            self.data_timelist[0]['integrated'][method_string].data = np.transpose([data_x, np.sqrt(data_y)])
        if self.parameters['integrated']['scale'].choice == 'linear':
            self.data_timelist[0]['integrated'][method_string].data = np.transpose([data_x, data_y])

    def find_peak_conditions(self, data):
        return find_peaks(data,
                          prominence = (self.parameters['integrated']['peak prominence min'].setvalue,
                                        self.parameters['integrated']['peak prominence max'].setvalue),
                          width = (self.parameters['integrated']['peak width min'].setvalue,
                                   self.parameters['integrated']['peak width max'].setvalue),
                          wlen = int(self.parameters['integrated']['window length'].setvalue))

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
        q_start = int(self.parameters['integrated']['clip head'].setvalue)
        q_ending = int(self.parameters['integrated']['clip tail'].setvalue)
        intqaxis_clipped = self.intqaxis[q_start: -q_ending]
        intdata_clipped = self.intdata[q_start: -q_ending]

        if 'original' in self.curve_timelist[0]['integrated']:
            self.data_scale('original', self.intqaxis, self.intdata)

        if 'truncated' in self.curve_timelist[0]['integrated']:
            self.data_scale('truncated', [intqaxis_clipped[0],intqaxis_clipped[-1]],
                            [intdata_clipped[0], intdata_clipped[-1]])
            self.data_timelist[0]['integrated']['truncated'].pen = None
            self.data_timelist[0]['integrated']['truncated'].symbol = 'x'
            self.data_timelist[0]['integrated']['truncated'].symbolsize = 20

        if 'smoothed' in self.curve_timelist[0]['integrated']:
            self.intdata_smoothed = scipy.signal.savgol_filter(intdata_clipped,
                                                  int(self.parameters['integrated']['Savitzky-Golay window'].setvalue),
                                                  int(self.parameters['integrated']['Savitzky-Golay order'].setvalue))
            self.data_scale('smoothed', intqaxis_clipped, self.intdata_smoothed)

        if 'find peaks' in self.curve_timelist[0]['integrated']:
            if hasattr(self, 'intdata_smoothed'):
                peaks, peak_properties = self.find_peak_conditions(self.intdata_smoothed)
                self.data_scale('find peaks', intqaxis_clipped[peaks], self.intdata_smoothed[peaks])
            else:
                peaks, peak_properties = self.find_peak_conditions(self.intdata)
                self.data_scale('find peaks', intqaxis_clipped[peaks], intdata_clipped[peaks])

            self.data_timelist[0]['integrated']['find peaks'].pen = None
            self.data_timelist[0]['integrated']['find peaks'].symbol = 't'
            self.data_timelist[0]['integrated']['find peaks'].symbolsize = 20
            
        # time series
        if 'pointer' in self.curve_timelist[0]['time series']:
            self.data_timelist[0]['time series']['pointer'].data = np.array([[0],[self.index]]).transpose()
            self.data_timelist[0]['time series']['pointer'].symbol = 't2'
            self.data_timelist[0]['time series']['pointer'].symbolsize = 15

class Optic(Methods_Base):
    def __init__(self, path_name_dict):
        super(Optic, self).__init__()
        self.directory = path_name_dict['directory'].text()
        self.fileshort = path_name_dict['raw file'].text()
        self.filename = os.path.join(self.directory, 'raw', self.fileshort + '.qvd')
        # qvt: double (time in ms), int16 (time span/data), double (dummy)
        self.timename = os.path.join(self.directory, 'raw', self.fileshort + '.qvt')
        self.channelnum = 2068  # due to the data collection scheme, there are 2068 instead of 2048 channels
        self.channel_start = 200
        self.channel_end = 1100
        self.data = self.read_qvd(self.filename)
        self.datatime = self.read_qvt(self.timename)
        self.filetime = os.path.getmtime(self.timename)  # in seconds

    def time_range(self, winobj):
        if self.entrytimesec == []:  # start, end time in seconds
            start_time = self.filetime - self.datatime[-1, 0] / 1000 - self.datatime[-1, 1] / 1000  # original in milisecond
            for index in range(self.data.shape[0]):
                self.entrytimesec.append(start_time + self.datatime[index, 0] / 1000 + [0, self.datatime[index, 1] / 1000])

            self.entrytimesec = np.array(self.entrytimesec, dtype=int)

        return [int(self.entrytimesec[0, 0]), int(self.entrytimesec[-1, 1])]

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

class PL(Optic):
    def __init__(self, path_name_dict):
        super(PL, self).__init__(path_name_dict)
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
        if self.checksdict['time series'].isChecked():
            xticklabels = []
            for tickvalue in np.arange(self.channel_start, self.channel_end,100):  # need to change for different spectrometer
                xticklabels.append((int(self.data.shape[1] * (tickvalue - self.channel_start) /
                                        (self.channel_end - self.channel_start)), "{:4.1f}".format(tickvalue)))

            xticks = winobj.gdockdict['pl'].tabdict['time series'].tabplot.getAxis('bottom')
            xticks.setTicks([xticklabels])
            img = pg.ImageItem(image=np.log10(self.data).transpose()) # need log here?
            winobj.gdockdict['pl'].tabdict['time series'].tabplot.addItem(img)
            color_map = pg.colormap.get('CET-R4')
            color_bar = pg.ColorBarItem(values=(0, self.data.max()), cmap=color_map)
            color_bar.setImageItem(img, winobj.gdockdict['pl'].tabdict['time series'].tabplot)

    def data_process(self, para_update):
        self.dynamictitle = 'data' + str(self.index + 1) + '\t start:' + self.startime + '\t end:' + self.endtime
        # raw
        if 'show' in self.curve_timelist[0]['raw']:
            self.data_timelist[0]['raw']['show'].data = \
                np.transpose([np.linspace(self.channel_start, self.channel_end,num=self.channelnum),
                              np.log10(self.data[self.index,:])]) # 200 nm - 1.1 um

        # time series
        if 'pointer' in self.curve_timelist[0]['time series']:
            self.data_timelist[0]['time series']['pointer'] = np.array([[0], [self.index]]).transpose()
            self.data_timelist[0]['time series']['pointer'].symbol = 't2'
            self.data_timelist[0]['time series']['pointer'].symbolsize = 15

class Refl(Optic):
    def __init__(self, path_name_dict):
        super(Refl, self).__init__(path_name_dict)
        self.availablemethods = ['raw', 'time series', 'fit-T']
        self.availablecurves['raw'] = ['show','reference','fit']
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

    def plot_from_prep(self, winobj):
        pass
        # tempfile = os.path.join(self.directory, 'process', self.fileshort + '_Refl_data.txt')
        # if os.path.isfile(tempfile):
        #     np.savetxt(tempfile, self.data)
        # self.plot_from_load(winobj)

    def plot_from_load(self, winobj): # this needs pre-select a good reference
        if self.checksdict['time series'].isChecked() and 'reference' in self.curve_timelist[0]['raw']:
            xticklabels = []
            for tickvalue in np.arange(self.channel_start, self.channel_end, 100):  # need to change for different spectrometer
                xticklabels.append((int(self.data.shape[1] * (tickvalue - self.channel_start) /
                                        (self.channel_end - self.channel_start)), "{:4.1f}".format(tickvalue)))

            xticks = winobj.gdockdict['Refl' ].tabdict['time series'].tabplot.getAxis('bottom')
            xticks.setTicks([xticklabels])
            img = pg.ImageItem(image= (self.data / self.refcandidate).transpose())
            winobj.gdockdict['Refl' ].tabdict['time series'].tabplot.addItem(img)
            color_map = pg.colormap.get('CET-R4')
            color_bar = pg.ColorBarItem(values=(0, self.data.max()), cmap=color_map)
            color_bar.setImageItem(img, winobj.gdockdict['Refl' ].tabdict['time series'].tabplot)

    def data_process(self, para_update):
        self.dynamictitle = 'data' + str(self.index + 1) + '\t start:' + self.startime + '\t end:' + self.endtime
        if 'show' in self.curve_timelist[0]['raw']:
            if 'reference' in self.curve_timelist[0]['raw']:
                self.data_timelist[0]['raw']['show'].data = \
                    np.transpose([np.linspace(self.channel_start, self.channel_end, num=self.channelnum),
                                  self.data[self.index, :] / self.refcandidate])  # 200 nm - 1.1 um
            else:
                self.data_timelist[0]['raw']['show'].data = \
                    np.transpose([np.linspace(self.channel_start, self.channel_end, num=self.channelnum),
                                  self.data[self.index, :]])  # 200 nm - 1.1 um
                self.refcandidate = self.data[self.index, :]

        # time series
        if 'pointer' in self.curve_timelist[0]['time series']:
            self.data_timelist[0]['time series']['pointer'] = np.array([[0], [self.index]]).transpose()
            self.data_timelist[0]['time series']['pointer'].symbol = 't2'
            self.data_timelist[0]['time series']['pointer'].symbolsize = 15
