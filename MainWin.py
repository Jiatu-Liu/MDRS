"""
author: Jiatu Liu

notes:

1, install larch and gsas2 in your python environment
2, download Akihito Takeuchi's draggabletabwidget.py and put it in the same folder
3, install azint by Clemens Weninger: conda install -c maxiv azint; this might not work in some windows machine

acknowledgement:

Mahesh Ramakrishnan's code on XRD image integration

"""
import fileinput
import os
import sys
import time
import copy

from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import pyqtgraph.dockarea
import pyqtgraph as pg
import numpy as np
import h5py
import ctypes
import platform
from lmfit.lineshapes import gaussian
import pandas as pd
from struct import unpack
import struct
from scipy.signal import savgol_filter

import larch
# from larch.io import read...
from larch.xafs import pre_edge
from larch import Group
from larch.fitting import param, param_group, guess, minimize
from larch.xafs import autobk
from larch.xafs import xftf

from Classes import *
from draggabletabwidget_new import *

import ast
import faulthandler

def make_dpi_aware():
    if int(platform.release()) >= 8:
        ctypes.windll.shcore.SetProcessDpiAwareness(True)

# unfortunately it is not working for deeper level widget!
# from https://gist.github.com/eyllanesc/be8a476bb7038c7579c58609d7d0f031.js
# def restore(settings):
#     finfo = QFileInfo(settings.fileName())
#
#     if finfo.exists() and finfo.isFile():
#         for w in qApp.allWidgets():
#             mo = w.metaObject()
#             if w.objectName() != "":
#                 for i in range(mo.propertyCount()):
#                     name = mo.property(i).name()
#                     val = settings.value("{}/{}".format(w.objectName(), name), w.property(name))
#                     w.setProperty(name, val)
#
# def save(settings):
#     for w in qApp.allWidgets():
#         mo = w.metaObject()
#         if w.objectName() != "":
#             for i in range(mo.propertyCount()):
#                 name = mo.property(i).name()
#                 settings.setValue("{}/{}".format(w.objectName(), name), w.property(name))
#
# updated: https://stackoverflow.com/questions/60027853/loading-widgets-properly-while-restoring-from-qsettings/60028282#60028282
# to prevent QPixmap overwrites QLabel name
# important widget should be named before use

class ShowData(QMainWindow):
    def __init__(self):
        super(ShowData, self).__init__()

        myscreen = app.primaryScreen()
        self.screen_height = myscreen.geometry().height()
        self.screen_width = myscreen.geometry().width()
        self.setGeometry(0, int(self.screen_height * .1), int(self.screen_width),
                         int(self.screen_height * .85))

        cframe = QFrame()
        cframe.setMaximumHeight(int(self.screen_height * .05))

        self.horilayout = QHBoxLayout()
        self.sliderset = QPushButton("Set slider(Ctrl+>)")
        self.sliderset.setShortcut('Ctrl+>')
        self.horilayout.addWidget(self.sliderset)
        self.sliderset.clicked.connect(self.setslider)
        self.slideradded = False # may not the best way
        self.slidervalues = []
        self.initialized = False

        cframe.setLayout(self.horilayout)
        self.setCentralWidget(cframe)

        self.controls = QDockWidget('Parameters', self)
        self.controls.setMaximumWidth(int(self.screen_width * .2))
        self.controls.setMinimumWidth(int(self.screen_width * .1))

        self.controltabs = DraggableTabWidget()
        # self.test = QTabWidget()
        # self.test.addTab(self.controls)
        # self.test.indexOf()

        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        pg.setConfigOption('leftButtonPan', False)

        self.cboxes = QToolBox()
        self.cboxes.setMinimumHeight(int(self.screen_height * 2)) # hope this can make it a bit more spacious
        self.scroll_area = QScrollArea() #
        self.scroll_area.setWidget(self.cboxes) #
        self.scroll_area.setWidgetResizable(True) # this line is the key, OMG!!!
        self.controltabs.addTab(self.scroll_area, "Checkboxes") #
        # self.controltabs.addTab(self.cboxes,"Checkboxes")
        self.controls.setWidget(self.controltabs)
        self.controls.setFloating(False)

        self.addDockWidget(Qt.BottomDockWidgetArea, self.controls)

        self.gdockdict = dict() # a dic for all top-level graph dock widgets: xas, xrd,...
        self.methodict = dict() # a dic for all available methods: xas, xrd,... there should be only one type each!
        self.data_read_dict = {'20221478_inform': {'xas':XAS_INFORM_2, 'xrd': XRD_INFORM_3},
                               '20221478_inform_xrd_only': {'xas': XAS_INFORM_2, 'xrd': XRD_INFORM_3_ONLY},
                               '20220660_inform': {'xas':XAS_INFORM_2, 'xrd': XRD_INFORM_2},
                               '20220660_inform_xrd_only': {'xas': XAS_INFORM_2, 'xrd': XRD_INFORM_2_ONLY},
                               '20220720_inform': {'xas': XAS_INFORM_1, 'xrd': XRD_INFORM_1},
                               '20220720_inform_xrd_only': {'xas': XAS_INFORM_1, 'xrd': XRD_INFORM_1_ONLY},
                               '20210738_battery': {'xas': XAS_BATTERY_1, 'xrd': XRD_BATTERY_1},
                               'general': {'xas': XAS_general, 'xrd': XRD_general}
                               }

        data_read = QMenu('Data Read Mode', self)
        read_mode_group = QActionGroup(data_read)
        for read_mode in self.data_read_dict:
            action = QAction(read_mode,data_read,checkable=True)
            data_read.addAction(action)
            read_mode_group.addAction(action)

        read_mode_group.setExclusive(True)
        # read_mode_group.triggered.connect(self.choose_data_read_mode)

        bar = self.menuBar()
        bar.addMenu(data_read)

        file = bar.addMenu('File')
        actionload = QAction('Load setup', self)
        actionload.setShortcut('Ctrl+L')
        actionload.triggered.connect(self.file_load)
        file.addAction(actionload)
        actionsave = QAction('Save setup', self)
        actionsave.setShortcut('Ctrl+S')
        actionsave.triggered.connect(self.file_save)
        file.addAction(actionsave)

        self.checkmethods = {} # checkboxes of top level
        self.cboxwidget = {}
        self.subcboxverti = {}
        self.path_name_dict = {}

        # self.fill_path_name_dict_general()
        # self.fill_path_name_dict_2023Mar()
        self.fill_path_name_dict_2022Dec() # 20221526
        # self.fill_path_name_dict_2022Oct() # 20220660
        # self.fill_path_name_dict_2022April()

        self.path_name_widget = {}
        self.methodclassdict = {}  # a dic for preparing corresponding Class

        read_mode_group.triggered.connect(self.choose_data_read_mode)

    def fill_pnd(self, name_list, name_list_refl, name_list_pl, name_list_xrf,
                 poni_file1, poni_file2, time_intevals, file_directory, refl_directory, xrf_directory):
        file_apdx = '_resultCluster.h5'
        self.repeat = len(name_list)  # number of xas or xrd
        for k in range(len(name_list)):
            if len(name_list[k]) > 1:
                if name_list[k][0] == name_list[k][1]: # two xas
                    self.path_name_dict['xas_' + str(k + 1) + '_1'] = {'directory': file_directory,
                                                                       'raw file': name_list[k][0],
                                                                       'energy range (eV)': '12935-13435'}
                    self.path_name_dict['xas_' + str(k + 1) + '_2'] = {'directory': file_directory,
                                                                       'raw file': name_list[k][1],
                                                                       'energy range (eV)': '13435-13935'}
                else: # one xas
                    self.path_name_dict['xas_' + str(k + 1)] = {'directory': file_directory,
                                                                'raw file': name_list[k][0],
                                                                'energy range (eV)': '12935-13935'}

                if name_list[k][-1] == name_list[k][-2]: # two xrd
                    self.path_name_dict['xrd_' + str(k + 1) + '_1'] = {'directory': file_directory,
                                                                       'raw file': name_list[k][-2], # '_data_000001',
                                                                       'integration file appendix': file_apdx,
                                                                       'PONI file': poni_file1} # 'LaB6_12935eV.poni',
                    self.path_name_dict['xrd_' + str(k + 1) + '_2'] = {'directory': file_directory,
                                                                       'raw file': name_list[k][-1],
                                                                       'integration file appendix': file_apdx,
                                                                       'PONI file': poni_file2}
                else: # one xrd
                    self.path_name_dict['xrd_' + str(k + 1)] = {'directory': file_directory,
                                                                'raw file': name_list[k][-1],
                                                                'integration file appendix': file_apdx,
                                                                'PONI file': poni_file1}
                                                                # 'refine dir': r'C:\Users\jialiu\OneDrive - Lund University'
                                                                #                r'\Dokument\Data_20220720_Inform',
                                                                # 'refine subdir': 'Refine_Coat7_RT',
                                                                # 'data files': 'data_result*20phases',
                                                                # 'refinement file': 'refine_coat7_rt_all.gpx'}

            else: # only one xrd
                self.path_name_dict['xrd_' + str(k + 1)] = {'directory': file_directory,
                                                            'raw file': name_list[k][-1],
                                                            'integration file appendix': file_apdx,
                                                            'PONI file': poni_file1,
                                                            'time interval(ms)':str(time_intevals[k])}
        if name_list_refl != []:
            for index in range(len(name_list_refl)):
                self.path_name_dict['refl_' + str(index + 1)] = {'directory': refl_directory,
                                                                 'raw file': name_list_refl[index],
                                                                 # 'align data number':'360',
                                                                 # 'to time':'2023-03-05T22:44:44',
                                                                 'time diff in sec':'140',
                                                                 }
        if name_list_pl != []:
            for index in range(len(name_list_refl)):
                self.path_name_dict['pl_' + str(index + 1)] = {'directory': refl_directory,
                                                                 'raw file': name_list_pl[index],
                                                                 # 'align data number':'360',
                                                                 # 'to time':'2023-03-05T22:44:44',
                                                                 'time diff in sec': '140',
                                                                 }

        if name_list_xrf != []:
            for index in range(len(name_list_xrf)):
                self.path_name_dict['xrf' + str(index + 1)] = {'directory': xrf_directory,
                                                               'raw file': name_list_xrf[index],
                                                               # 'align data number':'360',
                                                               # 'to time':'2023-03-05T22:44:44',
                                                               'time diff in sec': '140',
                                                               }

        # self.path_name_dict['refl_1'] = {'directory': r"C:\Users\jialiu\OneDrive - Lund University\Skrivbordet\OpticData",
        #                                'raw file': 'test_1M_MACl_1M_FAPbI3'}
        # self.path_name_dict['refl_2'] = {'directory': r"C:\Users\jialiu\OneDrive - Lund University\Skrivbordet\OpticData",
        #                                'raw file': 'test_1M_MACl_1M_MAFAPbI3_2_Refl'}
        # self.path_name_dict['pl_1'] = {'directory':r"C:\Users\jialiu\OneDrive - Lund University\Skrivbordet\OpticData",
        #                                  'raw file': 'test_1M_MACl_1M_MAFAPbI3_PL'}
        # self.path_name_dict['pl_2'] = {'directory': r"C:\Users\jialiu\OneDrive - Lund University\Skrivbordet\OpticData",
        #                              'raw file': 'test_1M_MACl_1M_MAFAPbI3_2_PL'}

    # 20220720 in April 2022: might not have two xrd data
    def fill_path_name_dict_2022April(self):
        # one xas, one xrd
        name_ls01 = [['Coat1_FAPI_05MCl_2',
                      'Coat1_FAPI_05MCl_12'],
                     ['Coat1_FAPI_05MCl_Tramp_1',
                      'Coat1_FAPI_05MCl_Tramp_14'],
                     ['Coat4_2S_FAPI_MACl05M_2',
                      'Coat4_2S_FAPI_MACl05M_26'],
                     ['Coat4b_2S_FAPI_MACl05M_XAS_XRD_100C_1',
                      'Coat4b_2S_FAPI_MACl05M_XAS_XRD_100C_38'],
                     ['Coat4c_2S_FAPI_MACl05M_XAS_XRD_RT_1',
                      'Coat4c_2S_FAPI_MACl05M_XAS_XRD_RT_39'],
                     ['Coat5_12S_FAPI_MACl01M_XAS_XRD_3',
                      'Coat5_12S_FAPI_MACl01M_XAS_XRD_33'],
                     ['Coat5_12S_FAPI_MACl01M_XAS_XRD_100C_2',
                      'Coat5_12S_FAPI_MACl01M_XAS_XRD_100C_37'],
                     ['Coat5b_12S_FAPI_MACl01M_XAS_XRD_RT_1',
                      'Coat5b_12S_FAPI_MACl01M_XAS_XRD_RT_40'],
                     ['Coat5b_12S_FAPI_MACl01M_XAS_XRD_RT_2',
                      'Coat5b_12S_FAPI_MACl01M_XAS_XRD_RT_41']]

        # two xas, one xrd
        name_ls02 = [['Coat6_12S_FAPI_Br02M_Cl01M_RT_6',
                      'Coat6_12S_FAPI_Br02M_Cl01M_RT_6',
                      'Coat6_12S_FAPI_Br02M_Cl01M_RT_48'],
                     ['Coat6b_12S_FAPI_Br02M_Cl01M_heat_1',
                      'Coat6b_12S_FAPI_Br02M_Cl01M_heat_1',
                      'Coat6b_12S_FAPI_Br02M_Cl01M_heat_53'],
                     ['Coat7_12S_FAPI_Br06M_Cl01M_RT_1',
                      'Coat7_12S_FAPI_Br06M_Cl01M_RT_1',
                      'Coat7_12S_FAPI_Br06M_Cl01M_RT_51'],
                     ['Coat7_12S_FAPI_Br06M_Cl01M_heat_1',
                      'Coat7_12S_FAPI_Br06M_Cl01M_heat_1',
                      'Coat7_12S_FAPI_Br06M_Cl01M_heat_52']]

        for index in range(len(name_ls04)):
            name_ls04[index].append(name_ls04[index][0])
            name_ls04[index].append(name_ls04[index][0] + '_eiger')
            name_ls04[index].append(name_ls04[index][0] + '_eiger')

        name_list_ref1 = ['Coat1_FAPbI_Cl05M_Refl',
                          'Coat04_Refl',
                          'Coat04b_Refl',
                          'Coat04c_Refl',
                          'Coat05_Refl',
                          'Coat05b_Refl']

        name_list_ref2 = ['Coat_06_Br02_Refl',
                          'Coat_07_Br06_Refl',
                          'Coat_07B_Refl',
                          'Coat_07B_after_Refl']

        # test
        name_list_ref1 = ['Ref_test_Refl',
                          'Ref_airblade_Refl',
                          'Ref_heating_after_airblade_Refl',
                          'Ref_heat_Refl',
                          'Ref_heat_150_Refl',
                          'Ref_heat_150_again_Refl']

        # test
        name_list_ref1 = ['test_old_Br_ink_Refl',
                          'test_old_Br_ink_2_Refl']


        # to change: name_list, poni_file, directory
        # name_list = name_ls01 # one xrd
        # name_list = name_ls02 # two xas, one xrd
        name_list = name_ls03 # one xas, two xrd
        # name_list = name_ls04 # two xas, two xrd

        # name_list_refl = name_list_ref1
        # name_list_refl = name_list_ref3
        # name_list_refl = name_list_ref4
        name_list_refl = []
        name_list_pl = []

        # might not be necessary !!!
        # for index in range(len(name_list_refl)):
        #     name_list_refl[index] = name_list_refl[index] + '_Refl'
        #     name_list_pl[index] = name_list_pl[index] + '_PL'

        file_directory = r"W:balder\20220720\2022042008"
        refl_directory = r"C:\Users\jialiu\OneDrive - Lund University\Skrivbordet\OpticData"
        poni_file1 = 'LaB6_12935eV.poni'
        poni_file2 = 'LaB6_12935eV.poni'

        name_list_xrf = []
        xrf_directory = []

        self.fill_pnd(name_list, name_list_refl, name_list_pl, name_list_xrf,
                      poni_file1, poni_file2, time_intevals, file_directory,
                      refl_directory, xrf_directory)

    # 20220660 in Oct 2022
    def fill_path_name_dict_2022Oct(self):
        # xrd only, name and time
        name_ls01 = [['MACl1M_DMF_airblade_QXRD_coat006_eiger', ],
                     ['MACl1M_DMF_airblade_QXRD_coat006a_eiger'],
                     ['MACl1M_DMF_airblade_QXRD_coat007_eiger'],
                     ['Br_I_2syringe_coat017b_QXRD_eiger'],  # be careful with this one, no xas!
                     ['MAPbBrI_DMSO_2ME_coat011_eiger'],
                     ['MAPbBrI_DMSO_2ME_coat012_eiger'],
                     ['MAPbBrI_DMSO_2ME_coat013_eiger'],
                     ['MAPbBrI_DMSO_2ME_coat013_eiger'],
                     ['MAPbBrI_DMSO_2ME_coat016_eiger'],
                     ['MAPbBrI_DMSO_2ME_coat018_eiger'],  # no xas
                     ['MAFAPbI_DMF_coat020_eiger']]

        time_intevals = [10, 10, 10, 100, 100, 100, 4500, 4500, 10, 200, 10]  # mapping above one by one, in ms

        # one xas, two xrd
        name_ls03 = [['LaB6'],
                     ['MACl1M_DMF_slow_dry_coat004d'],
                     ['MACl1M_DMF_slow_dry_coat004fb'],
                     ['MACl1M_DMF_slow_dry_coat005'],
                     ['MACl1M_DMF_airblade_QXRD_coat007a'],
                     ['MAFAPbI_DMF_coat021'],
                     ['MAFAPbI_DMF_coat022g'],
                     ['MACl3M_coat023a']]

        for index in range(len(name_ls03)):
            name_ls03[index].append(name_ls03[index][0] + '_eiger')
            name_ls03[index].append(name_ls03[index][0] + '_eiger')

        # two xas, two xrd
        name_ls04 = [['MAPbBrI_DMF_coat005'],
                     ['MAPbBrI_DMF_coat001'],
                     ['MAPbBrI_DMSO_2ME_coat08b'],
                     ['MAPbBrI_DMSO_2ME_coat009b'],
                     ['MAPbBrI_DMSO_2ME_coat013'],  # no xas
                     ['MAPbBrI_DMSO_2ME_coat014b'],
                     ['MAPbBrI_DMSO_2ME_coat015b'],
                     ['MAPbBrI_DMSO_2ME_coat017a'],
                     ['test_thickness']
                     ]

        for index in range(len(name_ls04)):
            name_ls04[index].append(name_ls04[index][0])
            name_ls04[index].append(name_ls04[index][0] + '_eiger')
            name_ls04[index].append(name_ls04[index][0] + '_eiger')

        name_list_ref1 = ['MACl1M_DMF_airblade_QXRD_coat006',
                          'MACl1M_DMF_airblade_QXRD_coat006a',
                          'MACl1M_DMF_airblade_QXRD_coat007',
                          'MAPbI_DMSO_2ME_coat10',  # no xas
                          'MAPbI_DMSO_2ME_coat11',
                          'MAPbI_DMSO_2ME_coat12',
                          'MAPbI_DMSO_2ME_coat13',
                          'MAPbI_DMSO_2ME_coat16',
                          'MAPbI_DMSO_2ME_coat18',  # no xas
                          'MAFAPbI_DME_coat20']

        name_list_ref3 = ['MACl1M_DMF_slow_dry_coat004d',
                          'MACl1M_DMF_slow_dry_coat004f',
                          'MACl1M_DMF_slow_dry_coat005',
                          'MAFAPbI_DME_coat21',
                          'MAFAPbI_DME_coat22']

        name_list_ref4 = ['MAPbI_DMSO_2ME_coat8',
                          'MAPbI_DMSO_2ME_coat9b',
                          'MAPbI_DMSO_2ME_coat14',
                          'MAPbI_DMSO_2ME_coat15',
                          'MAPbI_DMSO_2ME_coat17',
                          'MAPbI_DMSO_2ME_coat19']

        # to change: name_list, poni_file, directory
        name_list = name_ls01 # one xrd
        # name_list = name_ls02 # two xas, one xrd
        # name_list = name_ls03  # one xas, two xrd
        # name_list = name_ls04 # two xas, two xrd

        # name_list_refl = name_list_ref1
        # name_list_refl = name_list_ref3
        # name_list_refl = name_list_ref4
        name_list_refl = []
        name_list_pl = []

        # might not be necessary !!!
        # for index in range(len(name_list_refl)):
        #     name_list_refl[index] = name_list_refl[index] + '_Refl'
        #     name_list_pl[index] = name_list_pl[index] + '_PL'

        poni_file1 = 'LaB6_12936p37eV_realCalib_sum.poni'
        poni_file2 = 'LaB6_13591p12eV_realCalib_sum.poni'
        # poni_file2 = 'LaB6_13875eV_realCalib_sum.poni' # for 2 xrd 2 xas data with Br edge covered
        file_directory = r"W:balder\20220660\2022101308"
        # file_directory = r'C:\Users\jialiu\OneDrive - Lund University\Dokument\Data_20220660_Inform'
        refl_directory = r'C:\Users\jialiu\OneDrive - Lund University\Dokument\Data_20220660_Inform'

        name_list_xrf = []
        xrf_directory = []

        self.fill_pnd(name_list, name_list_refl, name_list_pl, name_list_xrf,
                      poni_file1, poni_file2, time_intevals, file_directory,
                      refl_directory, xrf_directory)

    # 20221562 in Dec 2022
    def fill_path_name_dict_2022Dec(self):
        name_ls01 = [['MAPbBrI_DMF_coat002_QXRD_eiger'],
                     ['MAPbBrI_DMF_coat004_QXRD_eiger'],
                     ['MAPbBrI_2ME_DMF_coat009_QXRD_eiger'],
                     ['MAPbBrI_DMSO_coat011_QXRD_eiger'],
                     ['MAPbBrI_2ME_DMSO_coat016c_QXRD_eiger'],
                     ['Br_I_2syringe_coat017c_QXRD_eiger'],
                     ['Br_I_2syringe_coat019b_QXRD_eiger'],
                     ]

        time_intevals = [20, 20, 20, 20, 20, 20, 20]

        # one xas, two xrd
        name_ls03 = [['MAFAPbI3_MACl_2ME_NMP_coat021'],
                       # ['MAFAPbI3_MACl_DMF_coat022a']
                       ]

        for index in range(len(name_ls03)):
            name_ls03[index].append(name_ls03[index][0] + '_eiger')
            name_ls03[index].append(name_ls03[index][0] + '_eiger')

        # two xas, two xrd
        name_ls04 = [['MAPbBrI_DMF_coat001'],
                     ['MAPbBrI_DMF_coat002'],
                     ['MAPbBrI_DMF_coat003b'],
                     ['MAPbBrI_DMF_coat005'],
                     ['MAPbBrI_2ME_DMF_coat006a'],
                     ['MAPbBrI_2ME_DMF_coat007a'],
                     ['MAPbBrI_2ME_DMF_coat008'],
                     ['Br_I_2syringe_coat017c_XAS-XRD'],
                     ['Br_I_2syringe_coat018c'],
                     ['Br_I_2syringe_coat020'],
                     ['MAPbBrI_DMSO_coat010b'],
                     ['MAPbBrI_DMSO_coat010b_annealed_BeamDamage'],
                     ['MAPbBrI_DMSO_coat011_03'],
                     ['MAPbBrI_DMSO_coat012'],
                     ['MAPbBrI_2ME_DMSO_coat013a'],
                     ['MAPbBrI_2ME_DMSO_coat014'],
                     ['MAPbBrI_2ME_DMSO_coat015a'],
                     ]

        for index in range(len(name_ls04)):
            name_ls04[index].append(name_ls04[index][0])
            name_ls04[index].append(name_ls04[index][0] + '_eiger')
            name_ls04[index].append(name_ls04[index][0] + '_eiger')

        name_list_opti1 = ['Coat02_MAPbBrI_DMF',
                           'Coat03_MAPbBrI_DMF',
                           'Coat05_MAPbBrI_DMF_Refl',
                           'Coat06_MAPbBrI_2ME_DMF',
                           'Coat07_MAPbBrI_2ME_DMF',
                           'Coat08_MAPbBrI_2ME_DMF']

        name_list_opti2 = ['Coat10_MAPbBrI_DMSO',
                           'Coat11_MAPbBrI_DMSO',
                           'Coat12_MAPbBrI_DMSO',
                           'Coat13_MAPbBrI_2ME_DMSO',
                           'Coat14_MAPbBrI_2ME_DMSO',
                           'Coat15_MAPbBrI_2ME_DMSO', ]

        name_list_opti3 = ['Coat17_MAPbBrI_mix',
                           'Coat18_MAPbBrI_mix',
                           'Coat20_MAPbBrI_mix']
        # QXRD ???
        name_list_opti4 = ['Coat02_MAPbBrI_DMF',
                           'Coat11_MAPbBrI_DMSO',
                           'Coat16_MAPbBrI_2ME_DMSO',
                           'Coat17_MAPbBrI_mix',
                           'Coat19_MAPbBrI_mix']

        # to change: name_list, poni_file, directory
        # name_list = name_ls01 # one xrd
        # name_list = name_ls02 # two xas, one xrd
        # name_list = name_ls03  # one xas, two xrd
        name_list = name_ls04 # two xas, two xrd

        # name_list_refl = name_list_ref1
        # name_list_refl = name_list_ref3
        # name_list_refl = name_list_ref4

        name_list_refl = []
        name_list_pl = []

        poni_file1 = 'LaB6_12keV_accurate.poni'
        poni_file2 = 'LaB6_14keV_accurate.poni'
        file_directory = r'W:balder\20221562\2022120808'
        # file_directory = r'C:\Users\jialiu\OneDrive - Lund University\Dokument\Data_20221562_Inform'
        refl_directory = file_directory
        name_list_refl = name_list_opti3.copy()
        # name_list_pl = name_list_opti3.copy()
        add_one_refl = False
        if name_list_refl == name_list_opti1:
            add_one_refl = True

        # might not be necessary !!!
        # for index in range(len(name_list_refl)):
        #     name_list_refl[index] = name_list_refl[index] + '_Refl'
        #     name_list_pl[index] = name_list_pl[index] + '_PL'

        if add_one_refl:
            name_list_refl.append('Coat01_MAPbBrI_DMF_Refl') # correct???

        name_list_xrf = []
        xrf_directory = []

        self.fill_pnd(name_list, name_list_refl, name_list_pl, name_list_xrf,
                      poni_file1, poni_file2, time_intevals, file_directory,
                      refl_directory, xrf_directory)
    # 20221478 in Mar 2023
    def fill_path_name_dict_2023Mar(self):
        # xrd only, name and time
        name_ls01 = [  # ['testqxrd_01_eiger'], # caution: must be with _eiger and nothing else!!!
            ['MAPbBr_DMSO_coat18_qxrd_eiger'],
            ['MAPbBr_DMSO_coat19_qxrd_aboveEdge_eiger'], # ?
            ['MAPbBr_DMSO_coat20_01_qxrd_eiger'],
            ['MAPbBr_DMSO_coat20_02_qxrd_eiger'],
            ['MAPbBr_DMSO_coat20_03_qxrd_eiger'],
            ['MAPbBr_DMSO_coat20_04_qxrd_eiger'],
            ['MAPbBr_DMSO_coat20_05_qxrd_eiger'],
            ['MAPbBr_DMSO_coat20_06_qxrd_eiger'],
            ['MAPbBr_DMSO_coat20_07_qxrd_eiger'],
            ['MAPbBr_DMSO_coat20_08_qxrd_eiger'],
            ['MAPbBr_DMSO_coat20_09_qxrd_eiger'],
            ['MAPbBr05I05_DMSO_coat21_01_qxrd_eiger'],
            ['MAPbBr05I05_DMSO_coat21_02_qxrd_eiger'],
            ['MAPbBr05I05_DMSO_coat21_03_qxrd_eiger'],
            ['MAPbBr05I05_DMSO_coat21_04_qxrd_eiger'],
            ['MAPbBr05I05_DMSO_coat21_05_qxrd_eiger'],
            ['MAPbBr05I05_DMSO_coat21_06_qxrd_eiger'],
            ['MAPbI_DMSO_coat22_01_qxrd_1_eiger'],
            ['MAPbI_DMSO_coat22_02_qxrd_eiger'],
            ['MAPbI_DMSO_coat22_03_qxrd_eiger'],
            ['MAPbI_DMSO_coat22_04_qxrd_eiger'],
            ['MAPbI_DMSO_coat22_05_qxrd_eiger'],
            ['MAPbI_DMSO_coat22_06_qxrd_eiger'],
            ['MAPbBr06I04_DMSO_coat23_01_qxrd_eiger'],
            ['MAPbBr06I04_DMSO_coat23_02_qxrd_eiger'],
            ['MAPbBr06I04_DMSO_coat23_03_qxrd_eiger'],
            ['MAPbBr06I04_DMSO_coat23_04_qxrd_eiger'],
            ['MAPbBr06I04_DMSO_coat23_05_qxrd_eiger'],  # this one is 100 ms, watch out, and has two timestamp, beam damage!
            ['MAPbBr04I06_DMSO_coat24_01_qxrd_eiger'],
            ['MAPbBr04I06_DMSO_coat24_02_qxrd_eiger'],
            ['MAPbBr04I06_DMSO_coat24_03_qxrd_eiger'],
            ['FPI50_MPBr50_coat025_01_qxrd_eiger'],
            ['FPI50_MPBr50_coat025_02_qxrd_eiger'],
            ['FPI50_MPBr50_coat025_03_qxrd_eiger'],
            ['FPI40_MPBr60_coat026_01_qxrd_eiger'],
            ['FPI40_MPBr60_coat026_02_qxrd_eiger'],
            ['FPI40_MPBr60_coat026_03_qxrd_eiger'],
            ['FPI60_MPBr40_coat027_01_qxrd_eiger'],
            ['FPI60_MPBr40_coat027_02_qxrd_eiger'],
            ['FPI60_MPBr40_coat027_03_qxrd_eiger'],
        ]

        # there is also a single xas measurement for each coat above.

        time_intevals = [20] * len(name_ls01)
        time_intevals[28] = 100

        # one xas, two xrd
        name_ls03 = [ # ['LaB6_ref_scan_01'],
                     ['MFPI_coat002_01'], # 70 C, not formed
                     ['MFPI_coat003_02'], # 90 C, formed
                     ['MFPI_coat004_02'], # 100 C,formed
                     ['MFPI_coat005_04'], # rt dried, not formed
                     ['MFPI_2MACl_coat006'], # not formed
                     ['MFPI_2MACl_coat007_1'], # 120 C
                     ['MFPI_2MACl_coat008'], # 130 C, not fully formed
                     ['MFPI_2MACl_coat009_01'], # 140 C
                     ['MFPI_2MACl_coat10_01'],  # 112 C to 140 C, not formed
                     ['MFPI_2MACl_coat11_01'],  # r.t dry part one
                     ['MFPI_2MACl_coat11_02'],  # r.t dry part two
                     ['MFPI_1MACl_coat12'], # redundant, same as coat14
                     ['MFPI_MACl_coat13'], # 0.25 M MACl
                     ['MFPI_MACl_coat14'], # 0.5 M MACl
                     ['MFPI_2ME_NMP_coat15_03'],
                     ['MFPI_MACl_2ME_NMP_coat16_01'],
                     ['MFPI_MACl_2ME_NMP_coat17_01'],
                     ]

        for index in range(len(name_ls03)):
            name_ls03[index].append(name_ls03[index][0] + '_eiger')
            name_ls03[index].append(name_ls03[index][0] + '_eiger')

        # two xas, two xrd
        name_ls04 = [['example'],
                     ]

        for index in range(len(name_ls04)):
            name_ls04[index].append(name_ls04[index][0])
            name_ls04[index].append(name_ls04[index][0] + '_eiger')
            name_ls04[index].append(name_ls04[index][0] + '_eiger')

        name_list_ref1 = ['MPBr_coat018',
                          'MPBr_DMSO_coat19',
                          'MPBr_DMSO_coat20',
                          'MPBr05I05_DMSO_coat21',
                          'MPI_DMSO_coat22',
                          'MPBr06I04_DMSO_coat23',
                          'MPBr06I04_DMSO_coat23_beamDamage',
                          'MPBr04I06_DMSO_coat24',
                          'FPI50_MPBr50_coat25',
                          'FPI80_MPBr20_coat26',
                          'FPI40_MPBr60_coat26', # ?
                          'FPI60_MPBr40_coat27',
                          ]

        name_list_ref3 = ['MFPI_coat002',
                          'MFPI_coat003',
                          'MFPI_coat004',
                          'MFPI_coat005',
                          'MFPI_2MACl_coat006',
                          'MFPI_2MACl_coat007',
                          'MFPI_2MACl_coat008',
                          'MFPI_2MACl_coat009',
                          'MFPI_coat010',
                          'MFPI_2MACl_coat011',
                          'MFPI_1MACl_coat012',
                          'MFPI_1MACl_coat013',
                          'MFPI_MACl_coat014', # 15 may be the same
                          'MFPI_2ME_NMP_MACl_coat016',
                          'MFPI_2ME_NMP_MACl_coat017',
                          ]

        name_list_ref4 = ['example',]

        name_list_xrf1 = ['coat18_MPBr_DMSO',
                          'coat19_MPBr_DMSO', # merged with 20
                          'coat21_MPBr0.5I0.5_DMSO',
                          'coat22_MPI_DMSO',
                          'coat23_MPBr60I40_DMSO',
                          'coat23_MPBr60I40_DMSO_degrad',
                          'coat24_MPBr40I60_DMSO',
                          'coat25_FPI40_MPBr60', # merged with 26, 27
                          ]

        name_list_xrf3 = ['coat2_MFPI_DMF',
                          'coat3_MFPI_DMF',
                          'coat4_MFPI_DMF',
                          'coat5_MFPI_DMF',
                          'coat6_MFPI_2MACl_DMF',
                          'coat7_MFPI_2MACl_DMF',
                          'coat8_MFPI_2MACl_DMF',
                          'coat9_MFPI_2MACl_DMF',
                          'coat10_MFPI_2MACl_DMF',
                          'coat11_MFPI_2MACl_DMF',
                          'coat12_MFPI_1MACl_DMF',
                          'coat13_MFPI_MACl_DMF',
                          'coat14_MFPI_MACl_DMF',
                          'coat15_MFPI_2ME_NMP',
                          'coat16_MFPI_2ME_NMP_MACl',
                          'coat17_MFPI_2ME_NMP_MACl',
                          ]

        # to change: name_list, poni_file, directory
        # name_list = name_ls01 # one xrd
        name_list = name_ls03  # one xas, two xrd

        # name_list_refl = name_list_ref1.copy()
        name_list_refl = name_list_ref3.copy()

        # name_list_pl = name_list_ref1.copy()
        name_list_pl = name_list_ref3.copy()

        self.cboxes.setMinimumHeight(int(self.screen_height * 2.7)) # for one xas, two xrd
        # self.cboxes.setMinimumHeight(int(self.screen_height * 2))  # for only one xrd

        for index in range(len(name_list_refl)):
            name_list_refl[index] = name_list_refl[index] + '_Refl'
            name_list_pl[index] = name_list_pl[index] + '_PL'

        # name_list_xrf = name_list_xrf1.copy()
        name_list_xrf = name_list_xrf3.copy()

        poni_file1 = 'LaB6_p7keV_x2rol_adjusted.poni'
        poni_file2 = 'LaB6_13keV_x2rol_adjusted.poni'
        file_directory = r"Y:\balder\20221478\2023030108"
        refl_directory = file_directory # r"Y:\balder\20221478\2023030108"
        xrf_directory = file_directory

        if name_list == name_ls01:
            poni_file1 = poni_file2

        self.fill_pnd(name_list, name_list_refl, name_list_pl, name_list_xrf,
                      poni_file1, poni_file2, time_intevals, file_directory,
                      refl_directory, xrf_directory)

    # general
    def fill_path_name_dict_general(self):
        # xrd only, name and time
        name_ls01 = [['example'],
                     [],
                     [],
                     ]

        # there is also a single xas measurement for each coat above.

        time_intevals = [20] * len(name_ls01)

        # one xas, two xrd
        name_ls03 = [['HDC11_Sulf_scan_T_ramp_4'], # 19800-21300
                     ['example2'],
                     ['example3'],
                     ]

        for index in range(len(name_ls03)):
            name_ls03[index].append(name_ls03[index][0] + '_eiger')
            name_ls03[index].append(name_ls03[index][0] + '_eiger')

        # two xas, two xrd
        name_ls04 = [['example1'],
                     ['example2'],
                     ['example3'],
                     ]

        for index in range(len(name_ls04)):
            name_ls04[index].append(name_ls04[index][0])
            name_ls04[index].append(name_ls04[index][0] + '_eiger')
            name_ls04[index].append(name_ls04[index][0] + '_eiger')

        name_list_ref1 = []

        name_list_ref3 = []

        name_list_ref4 = []

        name_list_xrf1 = []

        name_list_xrf3 = []

        # to change: name_list, poni_file, directory
        # name_list = name_ls01 # one xrd
        name_list = name_ls03  # one xas, two xrd

        name_list_refl = name_list_ref1.copy()
        # name_list_refl = name_list_ref3.copy()

        name_list_pl = name_list_ref1.copy()
        # name_list_pl = name_list_ref3.copy()

        # self.cboxes.setMinimumHeight(int(self.screen_height * 2.7)) # for one xas, two xrd
        self.cboxes.setMinimumHeight(int(self.screen_height))  # for only one xrd

        for index in range(len(name_list_refl)):
            name_list_refl[index] = name_list_refl[index] + '_Refl'
            name_list_pl[index] = name_list_pl[index] + '_PL'

        # name_list_xrf = name_list_xrf1.copy()
        name_list_xrf = name_list_xrf3.copy()

        poni_file1 = 'LaB6_19800eV.poni'
        poni_file2 = ''
        file_directory = r'W:\balder\20221329\2023042508'
        refl_directory = file_directory
        xrf_directory = file_directory

        if name_list == name_ls01:
            poni_file1 = poni_file2

        self.fill_pnd(name_list, name_list_refl, name_list_pl, name_list_xrf,
                      poni_file1, poni_file2, time_intevals, file_directory,
                      refl_directory, xrf_directory)


    def choose_data_read_mode(self, action):
        for index in range(self.repeat):
            for key in self.path_name_dict:
                if key[0:3] == 'xrd': self.methodclassdict[key] = self.data_read_dict[action.text()]['xrd']
                if key[0:3] == 'xas': self.methodclassdict[key] = self.data_read_dict[action.text()]['xas']
                if key[0:3] == 'ref': self.methodclassdict[key] = Refl
                if key[0:2] == 'pl': self.methodclassdict[key] = PL
                if key[0:3] == 'xrf': self.methodclassdict[key] = XRF

        if not self.initialized:
            self.ini_methods_cboxes()
            self.initialized = True

    def file_load(self):
        name = QFileDialog.getOpenFileName(self, 'Load Setup')
        # settings = QSettings(name[0], QSettings.IniFormat)
        # restore(settings)
        if name[0][-3::] == '.h5':
            fullname = name[0]
        else:
            fullname = name[0] + '.h5'

        try:
            f = h5py.File(fullname, 'r')
        except:
            print('wrong type of file')
        else:
            # read parameters and checks info and execute them
            for key in list(f['all methods'].keys()): # xas, xrd,...
                if key in self.checkmethods:
                    self.checkmethods[key].setChecked(True) # first creat a method obj
                    for para in list(f['all methods'][key]['parameters'].keys()): # post-edge, chi(k)
                        for subpara in list(f['all methods'][key]['parameters'][para].keys()): # dk, kmax,...
                            if subpara in self.methodict[key].parameters[para]:
                                if f['all methods'][key]['parameters'][para][subpara].dtype == 'string':
                                    self.methodict[key].parameters[para][subpara].choice = \
                                        f['all methods'][key]['parameters'][para][subpara][()].decode()
                                else:
                                    self.methodict[key].parameters[para][subpara].setvalue = \
                                        f['all methods'][key]['parameters'][para][subpara][()]

                    for subkey in list(f['all methods'][key]['pipelines'].keys()): # raw, norm,...
                        self.methodict[key].checksdict[subkey].setChecked(True)
                        for entry in f['all methods'][key]['pipelines'][subkey][()]: # I0, I1,...
                            if entry.decode() in self.methodict[key].curvedict[subkey]:
                                self.methodict[key].curvedict[subkey][entry.decode()].setChecked(True)

            if not self.slideradded:
                self.setslider()
                self.slideradded = True
                # this function is only to memorize the save relative positions for different data
                if list(f['slider']['memorized']) is not []:
                    for mem in list(f['slider']['memorized']):
                        memrange = f['slider']['max'][()] - f['slider']['min'][()]
                        range = self.slider.maximum() - self.slider.minimum()
                        memrelative = mem - f['slider']['max'][()]
                        self.slider.setValue(int(memrelative / memrange * range + self.slider.maximum()))
                        self.memorize_curve()

            f.close()

    def file_save(self):
        name = QFileDialog.getSaveFileName(self, 'Save Setup')
        # settings = QSettings(name[0], QSettings.IniFormat)
        # save(settings)
        if name[0][-3::] == '.h5':
            fullname = name[0]
        else:
            fullname = name[0] + '.h5'

        with h5py.File(fullname, 'w') as f:
            # save slider info
            if self.slideradded:
                slider = f.create_group('slider')
                slider.create_dataset('memorized', data=self.slidervalues)
                slider.create_dataset('max', data=self.slider.maximum())
                slider.create_dataset('min', data=self.slider.minimum())
            # save parameters and checks info
            allmethods = f.create_group('all methods')
            for key in self.methodict: # xas, xrd,...
                methods = allmethods.create_group(key)
                if self.methodict[key].parameters is not {}:
                    parameters = methods.create_group('parameters')
                    for para in self.methodict[key].parameters:
                        subparameters = parameters.create_group(para)
                        for subpara in self.methodict[key].parameters[para]:
                            tempobj = self.methodict[key].parameters[para][subpara]
                            if tempobj.identity == 'values':
                                subparameters.create_dataset(subpara, data=tempobj.setvalue)
                            else:
                                subparameters.create_dataset(subpara, data=tempobj.choice)

                    pipelines = methods.create_group('pipelines')
                    for subkey in self.methodict[key].checksdict: # raw, norm,...
                        if self.methodict[key].checksdict[subkey].isChecked():
                            tempstr = []
                            for entry in self.methodict[key].curvedict[subkey]: # I0, I1,...
                                if self.methodict[key].curvedict[subkey][entry].isChecked():
                                    tempstr.append(entry)

                            pipelines.create_dataset(subkey, data=tempstr)

    def ini_methods_cboxes(self):

        for key in self.methodclassdict:
            self.checkmethods[key] = QCheckBox(key)
            self.checkmethods[key].stateChanged.connect(self.graphdock)
            self.cboxwidget[key] = QWidget()
            self.cboxwidget[key].setObjectName(key)  # important for .parent recognition
            cboxverti = QVBoxLayout()
            cboxfiles = QFormLayout()
            self.path_name_widget[key] = {}
            try:
                for subkey in self.path_name_dict[key]:
                    self.path_name_widget[key][subkey] = QLineEdit(self.path_name_dict[key][subkey])
                    self.path_name_widget[key][subkey].setPlaceholderText(subkey)
                    cboxfiles.addRow(subkey, self.path_name_widget[key][subkey])
            except: print('check your choice of reading mode')
            else:
                cboxverti.addLayout(cboxfiles)
                cboxverti.addWidget(self.checkmethods[key])
                self.subcboxverti[key] = QVBoxLayout()  # where all tab graph checkboxes reside, e.g. raw, norm, ...
                cboxhori = QHBoxLayout()
                cboxhori.addSpacing(10)
                cboxhori.addLayout(self.subcboxverti[key])
                cboxverti.addLayout(cboxhori)
                cboxverti.addStretch(1)
                self.cboxwidget[key].setLayout(cboxverti)
                self.cboxes.addItem(self.cboxwidget[key], key)

    def setslider(self):
        timerange = []
        for key in self.methodict:
            try:
                timerange.append(self.methodict[key].time_range(self))
            except Exception as e:
                print('check your file name, or reselect your import mode, or your file is not complete or if you are unlucky')
                print(str(e))

        if timerange:
            self.timerangearray = np.array(timerange) * 1000 # in milliseconds ! search for [:-3]

            if not self.slideradded:
                self.slider = QSlider(Qt.Horizontal)
                self.slider.setObjectName('theSlider')
                self.slider.setRange(0, np.max(self.timerangearray[:,1]) - np.min(self.timerangearray[:,0]))
                self.slider.setPageStep(500)
                self.slider.setSingleStep(50)
                # self.slider.setRange(min(timerange, key=lambda x:x[0]), max(timerange, key=lambda x:x[1]))
                # self.slider.setTickPosition(QSlider.TicksAbove)
                # self.slider.setTickInterval(5)
                # self.slider.setSingleStep(1)
                self.slider.valueChanged.connect(self.update_timepoints)

                self.horilayout.addWidget(self.slider)
                self.sliderlabel = QLabel()
                self.horilayout.addWidget(self.sliderlabel)
                self.mem = QPushButton("Memorize")
                self.mem.setShortcut('Ctrl+M')
                self.mem.clicked.connect(self.memorize_curve)
                self.horilayout.addWidget(self.mem)
                self.clr = QPushButton("Clear")
                self.clr.setShortcut('Ctrl+C')
                self.clr.clicked.connect(self.clear_curve)
                self.horilayout.addWidget(self.clr)
                self.slideradded = True
            else:
                self.slider.setRange(0, np.max(self.timerangearray[:, 1]) - np.min(self.timerangearray[:, 0]))

            self.slider.setValue(self.slider.minimum() + 1)

    def delslider(self):
        if self.slideradded:
            self.slider.setParent(None)
            self.sliderlabel.setParent(None)
            self.mem.setParent(None)
            self.clr.setParent(None)
            self.horilayout.removeWidget(self.slider)
            self.horilayout.removeWidget(self.sliderlabel)
            self.horilayout.removeWidget(self.mem)
            self.horilayout.removeWidget(self.clr)
            self.slideradded = False

    def memorize_curve(self):
        # memorize the slider value for re-load
        self.slidervalues.append(self.slider.value()) # need to divided by 1000 ???
        for key in self.gdockdict: # key as xas, xrd,...
            # prepare the data, curve
            data_mem, curve_mem = self.methodict[key].data_curve_copy(self.methodict[key].data_timelist[0])
            self.methodict[key].data_timelist.append(data_mem)
            self.methodict[key].curve_timelist.append(curve_mem)
            # plot the curve onto corresponding tabgraph
            for subkey in self.methodict[key].curve_timelist[0]: # subkey as raw, norm,...
                if subkey != 'refinement single':
                    for entry in self.methodict[key].curve_timelist[0][subkey]: # entry as I0, I1,...
                        # this is for occasions when there is only .image
                        # and not including truncated, peaks in xrd
                        if data_mem[subkey][entry].data is not None and \
                                entry not in ['find peaks', 'truncated']:
                            curve_mem[subkey][entry] = \
                                self.gdockdict[key].tabdict[subkey].tabplot.plot(name=self.methodict[key].dynamictitle + ' ' + entry)
                            # set data for each curve
                            curve_mem[subkey][entry].setData(data_mem[subkey][entry].data)
                            if data_mem[subkey][entry].pen: curve_mem[subkey][entry].setPen(data_mem[subkey][entry].pen)
                            if data_mem[subkey][entry].symbol: curve_mem[subkey][entry].setSymbol(data_mem[subkey][entry].symbol)
                            if data_mem[subkey][entry].symbol:
                                curve_mem[subkey][entry].setSymbolSize(data_mem[subkey][entry].symbolsize)
                            if data_mem[subkey][entry].symbol:
                                curve_mem[subkey][entry].setSymbolBrush(data_mem[subkey][entry].symbolbrush)

    def clear_curve(self):
        # clear the slider value for re-load
        if self.slidervalues != []:
            self.slidervalues.pop()
        # proof to clear the first obj ?
        for key in self.gdockdict:
            if len(self.methodict[key].data_timelist) > 1:
                # tune back the colorhue
                for subkey in self.methodict[key].availablemethods:
                    for entry in self.methodict[key].availablecurves[subkey]:
                        self.methodict[key].colorhue[subkey] -= self.methodict[key].huestep
                # del individual curves
                for subkey in self.methodict[key].curve_timelist[-1]:  # subkey as raw, norm,...
                    if len(self.methodict[key].curve_timelist[-1][subkey]) > 0:
                        for entry in self.methodict[key].curve_timelist[-1][subkey]:  # entry as I0, I1,...
                            self.gdockdict[key].tabdict[subkey].tabplot.removeItem(
                                self.methodict[key].curve_timelist[-1][subkey][entry]
                            )
                # del curves
                del self.methodict[key].curve_timelist[-1]
                print('del curve in method ' + key)
                # del data
                del self.methodict[key].data_timelist[-1]
                print('del data in method ' + key)


    def graphdock(self, state):
        checkbox = self.sender() # e.g. xas, xrd,...
        if state == Qt.Checked:
            self.gdockdict[checkbox.text()] = DockGraph(checkbox.text())
            self.gdockdict[checkbox.text()].gendock(self)
            self.gdockdict[checkbox.text()].gencontroltab(self)
            # create a method object, e.g. an XAS object
            self.methodict[checkbox.text()] = self.methodclassdict[checkbox.text()](self.path_name_widget[checkbox.text()])
            self.methodict[checkbox.text()].tabchecks(self, checkbox.text()) # show all checkboxes, e.g. raw, norm,...
            # self.slider.valueChanged.connect(self.methodict[checkbox.text()].update_timepoints)
        else:
            # add someting to prevent the problem mentioned at the end of Classes
            if self.gdockdict[checkbox.text()]:
                self.methodict[checkbox.text()].deltabchecks(self, checkbox.text())
                self.gdockdict[checkbox.text()].deldock(self)
                self.gdockdict[checkbox.text()].delcontroltab(self)
                if checkbox.text()[0:3] == 'xrd' and hasattr(self.methodict[checkbox.text()], 'index_win'): # for index window deletion
                    self.methodict[checkbox.text()].index_win.deldock(self)
                # self.slider.valueChanged.disconnect(self.methodict[checkbox.text()].update_timepoints)
                del self.methodict[checkbox.text()] # del the method object
                print('del method ' + checkbox.text())
                del self.gdockdict[checkbox.text()] # del the dock object, here can do some expansion to prevent a problem mentioned in Classes
                print('del dock ' + checkbox.text())

    def graph_tab(self, state):
        checkbox = self.sender() # e.g. raw, norm,...
        tooltabname = checkbox.parent().objectName() # e.g. xas, xrd,...
        if state == Qt.Checked:
            self.gdockdict[tooltabname].tabdict[checkbox.text()] = TabGraph(checkbox.text())
            tabgraphobj = self.gdockdict[tooltabname].tabdict[checkbox.text()]
            tabgraphobj.gentab(self.gdockdict[tooltabname], self.methodict[tooltabname])
            tabgraphobj.gencontrolitem(self.gdockdict[tooltabname])
            tabgraphobj.curvechecks(checkbox.text(), self.methodict[tooltabname], self)
        else:
            if self.gdockdict[tooltabname].tabdict[checkbox.text()]:
                tabgraphobj = self.gdockdict[tooltabname].tabdict[checkbox.text()]
                tabgraphobj.delcurvechecks(checkbox.text(), self.methodict[tooltabname])
                print('del 1 ' + checkbox.text())
                tabgraphobj.deltab(self.gdockdict[tooltabname])
                print('del 2 ' + tooltabname)
                tabgraphobj.delcontrolitem(self.gdockdict[tooltabname])
                print('del 3 ' + tooltabname)
                del self.gdockdict[tooltabname].tabdict[checkbox.text()] # access violation?
                print('del 4 ' + checkbox.text())
    
    def graphcurve(self, state):
        checkbox = self.sender() # e.g. I0, I1,...
        toolitemname = checkbox.parent().objectName() # e.g. raw, norm,...
        tooltabname = checkbox.parent().accessibleName() # e.g. xas, xrd,...
        if state == Qt.Checked:
            for timelist in range(len(self.methodict[tooltabname].curve_timelist)):  # timelist: 0, 1,...
                # a curve obj is assigned to the curve dict
                self.methodict[tooltabname].curve_timelist[timelist][toolitemname][checkbox.text()] = \
                    self.gdockdict[tooltabname].tabdict[toolitemname].tabplot.plot(name=checkbox.text())
        else:
            for timelist in range(len(self.methodict[tooltabname].curve_timelist)):  # timelist: 0, 1,...
                if checkbox.text() in self.methodict[tooltabname].curve_timelist[timelist][toolitemname]:
                    self.gdockdict[tooltabname].tabdict[toolitemname].tabplot.removeItem(
                        self.methodict[tooltabname].curve_timelist[timelist][toolitemname][checkbox.text()]
                    )
                    del self.methodict[tooltabname].curve_timelist[timelist][toolitemname][checkbox.text()]
                    print('del curve in method ' + tooltabname + ' ' + checkbox.text())
                if checkbox.text() in self.methodict[tooltabname].data_timelist[0][toolitemname]:
                    if self.methodict[tooltabname].data_timelist[0][toolitemname][checkbox.text()].image is not None:
                        self.gdockdict[tooltabname].tabdict[toolitemname].tabplot.clear()

    def update_timepoints(self, slidervalue): # slidervalue in ms !
        slidervalue = (slidervalue + np.min(self.timerangearray[:, 0])) / 1000
        slidertime = datetime.fromtimestamp(slidervalue).strftime('%Y%m%d-%H:%M:%S.%f')[:-3] # trim to ms !
        self.sliderlabel.setText(slidertime)
        # maybe put a legend for each curve displayed, just show the slider time
        for key in self.gdockdict:  # xas, xrd,...
            try:
                self.methodict[key].data_update(slidervalue)
            except Exception as exc:
                print(exc)
                print('maybe you missed some process before this step?')

            if self.methodict[key].update == True:
                self.update_curves(key)

    def update_parameters(self, widgetvalue):
        if self.slideradded == False: # a way to prevent crash; may not the best way
            self.setslider()
            self.slider.setValue(self.slider.minimum() + 1) # this works!!!
            self.slideradded = True

        parawidget = self.sender() # rbkg, kmin,...
        toolitemname = parawidget.parent().objectName() # post edge, chi(k),...
        tooltabname = parawidget.parent().accessibleName() # xas, xrd,...
        temppara = self.methodict[tooltabname].parameters[toolitemname][parawidget.objectName()]
        if type(widgetvalue) == str:
            temppara.choice = widgetvalue
        else:
            nominal_value = widgetvalue * temppara.step + temppara.lower
            self.methodict[tooltabname].paralabel[toolitemname][parawidget.objectName()].setText(
                parawidget.objectName() + ':{:.1f}'.format(nominal_value))
            temppara.setvalue = nominal_value

        if tooltabname[0:3] == 'xrd' and toolitemname == 'time series':

            if parawidget.objectName() == 'scale':
                self.methodict[tooltabname].plot_from_load(self)

            elif parawidget.objectName() == 'normalization':
                self.methodict[tooltabname].plot_from_load(self)

            # elif parawidget.objectName() in ['gap y tol.', 'gap x tol.', 'min time span', '1st deriv control']:
                # self.methodict[tooltabname].catalog_peaks(self)

            # elif parawidget.objectName() in ['max diff start time', 'max diff time span']:
            #     self.methodict[tooltabname].assign_phases(self)

            elif parawidget.objectName() == 'single peak int. width':
                self.methodict[tooltabname].single_peak_width(self)

            elif parawidget.objectName() == 'symbol size':
                self.methodict[tooltabname].peak_map.setSymbolSize(int(nominal_value))
                if hasattr(self.methodict[tooltabname],'peaks_catalog') and \
                        self.methodict[tooltabname].peaks_catalog_map is not []:
                    for index in range(len(self.methodict[tooltabname].peaks_catalog_map)):
                        self.methodict[tooltabname].peaks_catalog_map[index].setSymbolSize(int(nominal_value))

                if hasattr(self.methodict[tooltabname],'phases') and \
                        self.methodict[tooltabname].phases_map is not []:
                    for index in range(len(self.methodict[tooltabname].phases_map)):
                        self.methodict[tooltabname].phases_map[index].setSymbolSize(int(nominal_value))

            elif parawidget.objectName() == 'phases':
                self.methodict[tooltabname].show_phase(self)

        else:
            # if tooltabname[0:3] == 'xas' and (toolitemname in ['normalizing', 'normalized']) and \
            #         parawidget.objectName()[0:8] == 'Savitzky':
            #     self.methodict[tooltabname].filtered = True

            self.methodict[tooltabname].data_process(True) # True means the parameters have changed
            self.update_curves(tooltabname)

    def update_curves(self, key):
        for subkey in self.gdockdict[key].tabdict: # raw, norm,...
            self.gdockdict[key].tabdict[subkey].tabplot.setTitle(self.methodict[key].dynamictitle)
            if subkey == 'refinement single':
                self.gdockdict[key].tabdict[subkey].tabplot.setTitle(
                    self.methodict[key].refinedata[self.methodict[key].parameters[subkey]['data number'].setvalue])
            for timelist in range(len(self.methodict[key].curve_timelist)): # timelist: 0, 1,...
                for entry in self.methodict[key].curve_timelist[timelist][subkey]: # I0, I1,...
                    curveobj = self.methodict[key].curve_timelist[timelist][subkey][entry]
                    if entry in self.methodict[key].data_timelist[timelist][subkey]:
                        dataobj = self.methodict[key].data_timelist[timelist][subkey][entry]
                    # if dataobj.data is not None: curveobj.setData(dataobj.data, pen=None)
                    # if dataobj.pen is not None: curveobj.setPen(dataobj.pen)
                    # if dataobj.symbol is not None: curveobj.setSymbol(dataobj.symbol)
                    # if dataobj.symbolsize is not None: curveobj.setSymbolSize(dataobj.symbolsize)
                    # if dataobj.symbolbrush is not None: curveobj.setSymbolBrush(dataobj.symbolbrush)
                        if dataobj.data is not None:
                            curveobj.setData(dataobj.data, pen=dataobj.pen, symbol=dataobj.symbol,
                                             symbolSize=dataobj.symbolsize, symbolBrush=dataobj.symbolbrush)
                        if dataobj.image is not None: # for xrd raw image
                            if hasattr(self.methodict[key],'color_bar_raw'):
                                self.gdockdict[key].tabdict[subkey].tabplot.clear() # may not be the best if you want to process raw onsite
                                self.methodict[key].color_bar_raw.close()

                            self.gdockdict[key].tabdict[subkey].tabplot.setAspectLocked()
                            self.gdockdict[key].tabdict[subkey].tabplot.addItem(dataobj.image)
                            self.methodict[key].color_bar_raw = pg.ColorBarItem(values=(0, self.methodict[key].colormax),
                                                                                cmap=pg.colormap.get('CET-L3'))
                            self.methodict[key].color_bar_raw.setImageItem(dataobj.image,
                                                                       insert_in=self.gdockdict[key].tabdict[subkey].tabplot)


if __name__ == '__main__':
    # solve the dpi issue
    # QtWidgets.QApplication.setAttribute(QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    faulthandler.enable() #start @ the beginning
    make_dpi_aware()
    app = QApplication(sys.argv)
    w = ShowData()
    w.show()
    sys.exit(app.exec_())
