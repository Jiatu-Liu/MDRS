# Multi_Modal
notes:

1, install gsas2 in your user path (by default) and install larch in your python environment

  according to xraylarch website, you need to do:
  conda create -y --name xraylarch python=>3.9.10
  conda activate xraylarch
  conda install -y "numpy=>1.20" "scipy=>1.6" "matplotlib=>3.0" scikit-learn pandas
  conda install -y -c conda-forge wxpython pymatgen tomopy pycifrw
  pip install xraylarch
  
  attention:
  do not install gsas2 in your python environment, instead using your default user path

2, put Main.py, Classes.py and draggablewidget_new.py (modified from Akihito Takeuchi's draggabletabwidget.py) in the same folder
3, install azint by conda install -c maxiv azint (this step is not necessary if you do not integrate locally)
4, run Main.py

acknowledgement:
Mahesh Ramakrishnan's code on XRD image integration
