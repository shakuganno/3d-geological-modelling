# 3d-geological-modelling
Program to build 3D geological moedlling from borehole data

PDNN depends on several standard Python packages that should be shipped with any standard distribution (and are easy to install):
numpy
torch
scipy

before run this program you need to change file path in the code，The drill data is stored in the mdb file，In the borehole data, the following parameters are used for processing and saved in the txt file：ZKX，ZKY，TCHD，TCCDSD，ZKBG，TCZCBH，ZKID，GCSY，TCYCBH，TCCYCBH. These are respectively borehole coordinates x, borehole coordinates y, soil layer thickness, soil layer bottom depth, borehole elevation, main layer label, borehole id, engineering index, sublayer label, sublayer label of sublayer.

Use PDNN to train the model, and get .pt file.

Finally, after training，you will get the pt file，Training accuracy and test accuracy.


You can predict the results based on the modeling range of your data, and the gird data will save in txt file by using Pridict.py. 
HUll3D.py is from https://github.com/rgmyr/pyConvexHull3D, it is used by Predict.py.
If you have any questions, please contact me.


