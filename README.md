# 3d-geological-modelling
Program to build 3D geological moedlling from borehole data

PDNN depends on several standard Python packages that should be shipped with any standard distribution (and are easy to install):
numpy
torch
scipy

before run this program you need to change file path in the code，The drill data is stored in the mdb file，In the borehole data, the following parameters are used for processing and saved in the txt file：ZKX，ZKY，TCHD，TCCDSD，ZKBG，TCZCBH，ZKID，GCSY，TCYCBH，TCCYCBH. These are respectively borehole coordinates x, borehole coordinates y, soil layer thickness, soil layer bottom depth, borehole elevation, main layer label, borehole id, engineering index, sublayer label, sublayer label of sublayer.

Finally, after training，you will get the pt file，Training accuracy and test accuracy.


You can predict the results based on the modeling range of your data, and visualize the grid of the predicted results to produce 3D modeling results. This part is not yet available. 
If you have any questions, please contact me.


