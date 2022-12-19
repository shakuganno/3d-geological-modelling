import pyodbc
from pyodbc import connect
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pymssql #引入pymssql模块
import time
import math
from scipy.spatial import Delaunay
import copy
import itertools
from scipy import  interp
from itertools import cycle
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc



class Drill():
    def __init__(self,id,x,y,ZKBG,ZCBH,CDSD):
        self.id=id
        self.x=x
        self.y=y
        self.ZKBG=ZKBG
        self.ZCBH=ZCBH
        self.CDSD=CDSD

class Point2d():
    def __init__(self,x,y):
        self.x=x
        self.y=y


class Point():
    def __init__(self,x,y,z):
        self.x=x
        self.y=y
        self.z=z

class Plane():
    def __init__(self,id,A,B,C,D):
        self.id=id
        self.A=A
        self.B=B
        self.C=C
        self.D=D


class mlPoint():
    def __init__(self, id, x, y, z, straid):
        self.id = id
        self.x = x
        self.y = y
        self.z = z
        self.straid = straid

class mlCube():
    def __init__(self, id, lenth, width, height, straid, point):
        self.id = id
        self.lenth = lenth
        self.width = width
        self.height = height
        self.straid = straid
        self.point = point

def getplane(P1,P2,P3):
    A=(P2.y-P1.y)*(P3.z-P1.z)-(P2.z-P1.z)*(P3.y-P1.y)
    B=(P2.z-P1.z)*(P3.x-P1.x)-(P2.x-P1.x)*(P3.z-P1.z)
    C=(P2.x-P1.x)*(P3.y-P1.y)-(P2.y-P1.y)*(P3.x-P1.x)
    D=-(A*P1.x+B*P1.y+C*P1.z)
    return A,B,C,D

class TriPlane():
    def __init__(self, drill1, drill2, drill3,planeset,straids):
        self.drill1=drill1
        self.drill2=drill2
        self.drill3=drill3
        self.planeset=planeset
        self.straids=straids



def isPointinPolygon(point, rangelist):
    lnglist = []
    latlist = []
    for i in range(len(rangelist) - 1):
        lnglist.append(rangelist[i][0])
        latlist.append(rangelist[i][1])
    maxlng = max(lnglist)
    minlng = min(lnglist)
    maxlat = max(latlist)
    minlat = min(latlist)
    if (point[0] > maxlng or point[0] < minlng or
            point[1] > maxlat or point[1] < minlat):
        return False
    count = 0
    point1 = rangelist[0]
    for i in range(1, len(rangelist)):
        point2 = rangelist[i]
        if (point[0] == point1[0] and point[1] == point1[1]) or (point[0] == point2[0] and point[1] == point2[1]):
            print("top")
            return False

        if (point1[1] < point[1] and point2[1] >= point[1]) or (point1[1] >= point[1] and point2[1] < point[1]):

            point12lng = point2[0] - (point2[1] - point[1]) * (point2[0] - point1[0]) / (point2[1] - point1[1])

            if (point12lng == point[0]):
                print("edge")
                return True
            if (point12lng < point[0]):
                count += 1
        point1 = point2
    if count % 2 == 0:
        return False
    else:
        return True








stepnum=0      #Project number


DBfile = os.path.join(r'C:\Users\DELL\Desktop', str(stepnum) + '.mdb')
conn = connect(r"Driver={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=" + DBfile + ";Uid=;Pwd=;")
cursor = conn.cursor()
Table1 = "z_g_TuCeng"
Table2 = "z_ZuanKong"
Data_FileName = os.path.join(r'E:\pyfile\try', 'GAXC2Data' + str(stepnum) + '.txt')


def Clear(FileName):
    with open(FileName, "w") as f:
        f.close()
        print("dataclear")

Clear(Data_FileName)
IDlist = []
ZKID = -1
num = 0
SQL = "Select z_ZuanKong.GCSY,z_ZuanKong.ZKBH,z_ZuanKong.ZKBG,z_g_TuCeng.TCZCBH,z_g_TuCeng.TCYCBH,z_g_TuCeng.TCCYCBH,z_g_TuCeng.TCHD,z_g_TuCeng.TCCDSD," \
      "z_ZuanKong.ZKX,z_ZuanKong.ZKY,z_ZuanKong.GCSY FROM z_ZuanKong INNER JOIN z_g_TuCeng ON z_g_TuCeng.ZKBH = z_ZuanKong.ZKBH ORDER BY z_g_TuCeng.ZKBH,z_g_TuCeng.TCZCBH,z_g_TuCeng.TCCDSD "
cursor.execute(SQL)
for row in cursor:
    exID = False
    zknum = 0
    for ZK_ID in IDlist:
        if (ZK_ID == row.ZKBH):
            exID = True
            ZKID = zknum
        else:
            zknum += 1
    if (exID == False):
        IDlist.append(row.ZKBH)
        ZKID = zknum
    with open(Data_FileName, "a") as f:
        if (row.TCZCBH != None):
            f.write(str(row.ZKX) + ",," + str(row.ZKY) + ",," + str(row.TCHD) + ",," + str(row.TCCDSD) + ",," + str(row.ZKBG) + ",," + str(int(row.TCZCBH)) + ",," + str(int(ZKID)) + ",," + str(row.GCSY) + ",," + str(row.TCYCBH) + ",," + str(row.TCCYCBH) + "\n")
    if (row.TCZCBH != None):
        num += 1
cursor.commit()
print('datanum:',num)
print('drillnum:',len(IDlist))
IDlist2 = []
drilllist = []

Data_FileName = os.path.join(r'E:\pyfile\try', 'GAXC2Data' + str(stepnum) + '.txt')
Drill_Data = np.loadtxt(Data_FileName, delimiter=',,', usecols=[0, 1, 2, 3, 4, 5, 6])


xmove = min(Drill_Data[:,0])
ymove = min(Drill_Data[:,1])
maxBG = max(Drill_Data[:, 4])


cursor.execute(SQL)
for row in cursor:
    if row.ZKBH in IDlist2:
        zkid = len(IDlist2)
        ZKX = row.ZKX
        ZKY = row.ZKY
        ZKBG = row.ZKBG
        if len(ZCBHlist)!=0 and row.TCZCBH==ZCBHlist[-1]:
            ZCBHlist.pop()
            CDSDlist.pop()
        ZCBHlist.append(row.TCZCBH)
        CDSDlist.append(row.TCCDSD)

    else:
        if len(IDlist2) != 0:

            print(len(ZCBHlist),len(CDSDlist))
            drill = Drill(zkid, ZKX - xmove, ZKY - ymove, ZKBG, ZCBHlist, CDSDlist)
            print(zkid, ZKX - xmove, ZKY - ymove, ZKBG, ZCBHlist, CDSDlist)
            drilllist.append(drill)
        ZCBHlist = []
        CDSDlist = []
        IDlist2.append(row.ZKBH)
        zkid = len(IDlist2)
        ZKX = row.ZKX
        ZKY = row.ZKY
        ZKBG = row.ZKBG
        ZCBHlist.append(row.TCZCBH)
        CDSDlist.append(row.TCCDSD)
print(zkid, ZKX, ZKY, ZKBG, ZCBHlist, CDSDlist)



cursor.close()


conn.close()

drill = Drill(zkid, ZKX - xmove, ZKY - ymove, ZKBG, ZCBHlist, CDSDlist)
drilllist.append(drill)
print(zkid, ZKX - xmove, ZKY - ymove, ZKBG, ZCBHlist, CDSDlist)

point2d = []
newp2d=[]
lastnewp2d=[]
allp2d=[]

for p in drilllist:
    point2d.append([p.x, p.y])

lastnewp2d=copy.deepcopy(point2d)
allp2d=copy.deepcopy(lastnewp2d)
tri = Delaunay(point2d)
triplaneset = []
lineset=[]

for t in tri.simplices:

    a   =  np.sqrt(np.square(point2d[t[0]][0]-point2d[t[1]][0])+np.square(point2d[t[0]][1]-point2d[t[1]][1]))
    b = np.sqrt(np.square(point2d[t[1]][0] - point2d[t[2]][0]) + np.square(point2d[t[1]][1] - point2d[t[2]][1]))
    c = np.sqrt(np.square(point2d[t[2]][0] - point2d[t[0]][0]) + np.square(point2d[t[2]][1] - point2d[t[0]][1]))

    A = math.degrees(math.acos((a * a - b * b - c * c) / (-2 * b * c)))
    B = math.degrees(math.acos((b * b - a * a - c * c) / (-2 * a * c)))
    C = math.degrees(math.acos((c * c - a * a - b * b) / (-2 * a * b)))

    minaggle=min(A,B,C)
    if minaggle>20:
        tsort=sorted(t)

        lineset.append((tsort[0],tsort[1]))
        lineset.append((tsort[1],tsort[2]))
        lineset.append((tsort[0], tsort[2]))
    else:
        print(minaggle)

lineset = list(set(lineset))
print(lineset)


for t in tri.simplices:
    a = np.sqrt(np.square(point2d[t[0]][0] - point2d[t[1]][0]) + np.square(point2d[t[0]][1] - point2d[t[1]][1]))
    b = np.sqrt(np.square(point2d[t[1]][0] - point2d[t[2]][0]) + np.square(point2d[t[1]][1] - point2d[t[2]][1]))
    c = np.sqrt(np.square(point2d[t[2]][0] - point2d[t[0]][0]) + np.square(point2d[t[2]][1] - point2d[t[0]][1]))

    A = math.degrees(math.acos((a * a - b * b - c * c) / (-2 * b * c)))
    B = math.degrees(math.acos((b * b - a * a - c * c) / (-2 * a * c)))
    C = math.degrees(math.acos((c * c - a * a - b * b) / (-2 * a * b)))

    minaggle = min(A, B, C)
    if minaggle > 20:


        planeset = []
        straids = []
        straids.extend(drilllist[t[0]].ZCBH)
        straids.extend(drilllist[t[1]].ZCBH)
        straids.extend(drilllist[t[2]].ZCBH)
        straids = list(set(straids))
        straids.sort()

        cdsd1 = 0
        cdsd2 = 0
        cdsd3 = 0
        P1 = Point(drilllist[t[0]].x, drilllist[t[0]].y, -(maxBG - drilllist[t[0]].ZKBG + cdsd1))
        P2 = Point(drilllist[t[1]].x, drilllist[t[1]].y, -(maxBG - drilllist[t[1]].ZKBG + cdsd2))
        P3 = Point(drilllist[t[2]].x, drilllist[t[2]].y, -(maxBG - drilllist[t[2]].ZKBG + cdsd3))
        A, B, C, D = getplane(P1, P2, P3)
        plane = Plane(-1, A, B, C, D)
        planeset.append(plane)
        for i in range(len(straids)):
            if straids[i] in drilllist[t[0]].ZCBH:
                cdsd1 = drilllist[t[0]].CDSD[drilllist[t[0]].ZCBH.index(straids[i])]

            if straids[i] in drilllist[t[1]].ZCBH:
                cdsd2 = drilllist[t[1]].CDSD[drilllist[t[1]].ZCBH.index(straids[i])]

            if straids[i] in drilllist[t[2]].ZCBH:
                cdsd3 = drilllist[t[2]].CDSD[drilllist[t[2]].ZCBH.index(straids[i])]

            P1 = Point(drilllist[t[0]].x, drilllist[t[0]].y, -(maxBG - drilllist[t[0]].ZKBG + cdsd1))

            P2 = Point(drilllist[t[1]].x, drilllist[t[1]].y, -(maxBG - drilllist[t[1]].ZKBG + cdsd2))
            P3 = Point(drilllist[t[2]].x, drilllist[t[2]].y, -(maxBG - drilllist[t[2]].ZKBG + cdsd3))

            A, B, C, D = getplane(P1, P2, P3)
            plane = Plane(i, A, B, C, D)
            planeset.append(plane)
        print(len(straids),len(planeset))
        print(point2d[t[0]], point2d[t[1]], point2d[t[2]], planeset, straids)
        triplane = TriPlane(drilllist[t[0]], drilllist[t[1]], drilllist[t[2]], planeset, straids)
        triplaneset.append(triplane)
    else:
        print(minaggle)





def ReadData2Tensor(FileName, col):
    Data = np.loadtxt(FileName, delimiter=',,', usecols=col)
    Data = torch.from_numpy(Data)
    return Data




Data_FileName2 = os.path.join(r'E:\pyfile\try', 'add_DD' + str(stepnum) + '.txt')
Clear(Data_FileName2)
maxZKBG = max(Drill_Data[:, 4])
print('maxZKBG:',maxZKBG)
maxstraid=int(max(Drill_Data[:, 5]))
midxyz = (0, 0, 0, 0, 0, 0)
for ii in range(0, len(IDlist)):
    midzkbg = 0
    for zkbh in Drill_Data:
        if (zkbh[6] == ii and zkbh[4] > midzkbg):
            midzkbg = zkbh[4]
            midxyz = zkbh

    for zkbh in Drill_Data:
        if (zkbh[6] == ii):
            with open(Data_FileName2, "a") as f:
                f.write(str(zkbh[0]) + ",," + str(zkbh[1]) + ",," + str(zkbh[2]) + ",," +
                        str(maxZKBG - midzkbg + zkbh[3]) + ",,"
                        + str(zkbh[4]) + ",," + str(int(zkbh[5])) + ",," +
                        str(zkbh[6]) + "\n")
Data_FileName2 = os.path.join(r'E:\pyfile\try', 'add_DD' + str(stepnum) + '.txt')
Drill_Data = np.loadtxt(Data_FileName2, delimiter=',,', usecols=[0, 1, 2, 3, 4, 5, 6])
maxZKSD = max(Drill_Data[:, 3])
print('maxZKSD',maxZKSD)
midxyz = (0, 0, 0, 0, 0, 0)
for ii in range(0, len(IDlist)):
    maxzksd = 0
    for zkbh in Drill_Data:
        if (zkbh[6] == ii and zkbh[3] > maxzksd):
            maxzksd = zkbh[3]
            midxyz = zkbh

Drill_Data = np.loadtxt(Data_FileName2, delimiter=',,', usecols=[0, 1, 2, 3, 4, 5, 6])
print(len(Drill_Data))

MiniX = min(Drill_Data[:,0])
MiniY = min(Drill_Data[:,1])

MaxiX = max(Drill_Data[:, 0])
MaxiY = max(Drill_Data[:, 1])

MinX = 0
MinY = 0
MinZ = 0

MaxX = MaxiX - MiniX
MaxY = MaxiY - MiniY
MaxZ = maxZKSD

intervalx=MaxX/100
intervaly=MaxY/100
intervalv=MaxZ/60


with open(r"E:\pyfile\try\inter"+str(stepnum)+".txt", "w") as f:
    f.write(str((MaxX)) + "," + str((MinX)) + "," + str((MaxY)) + "," + str((MinY)) + "," +
            str((MaxZ)) + "," + str((MinZ)) + "," + str((MiniX)) + "," + str((MiniY)) +"," +str(maxstraid+1)+ "," + str(maxZKBG)+"\n")





straid = [k for k in range(0, maxstraid+1)]
stranum = [0] * (maxstraid+1)
Data_FileName3 = os.path.join(r'E:\pyfile\try', 'pretrain' + str(stepnum) + '.txt')
Clear(Data_FileName3)
with open(Data_FileName3, "a") as f:
    for i in Drill_Data:
        for num in straid:
            if (int(i[5]) == num):
                for j in range(int(i[3] * 4000), int(i[3] * 4000) - int(i[2] * 4000), -int(i[2] * 4000 / 10)):

                    f.write(str((i[0] - MiniX)/(MaxX-MinX)) + ",," + str((i[1] - MiniY)/(MaxY-MinY)) + ",," +
                            str((float(j / 4000))/(MaxZ-MinZ) ) + ",," + str(int(i[5])) + ",," +str(i[6])+"\n")


Drill_Data = np.loadtxt(Data_FileName3, delimiter=',,', usecols=[0, 1, 2])
Drill_Target_ZC = np.loadtxt(Data_FileName3, delimiter=',,', usecols=3)



class Drill_Datas(torch.utils.data.Dataset):
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label

    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    def __len__(self):
        return len(self.data)

ATRDrill = Drill_Datas(torch.from_numpy(Drill_Data), torch.from_numpy(Drill_Target_ZC))
print(len(ATRDrill))
newtrdata=Drill_Data.copy()
newtrlabel=Drill_Target_ZC.copy()



for _l in lineset:

    straids = []
    samstraids =[]
    difstraids=[]
    straids.extend(drilllist[_l[0]].ZCBH)
    straids.extend(drilllist[_l[1]].ZCBH)
    straids = list(set(straids))
    straids.sort()
    samstraids=list(set(drilllist[_l[0]].ZCBH)&set(drilllist[_l[1]].ZCBH))
    difstraids = list(set(drilllist[_l[0]].ZCBH) ^ set(drilllist[_l[1]].ZCBH))

    addhnum=5
    xinc = (drilllist[_l[1]].x - drilllist[_l[0]].x)/addhnum
    yinc = (drilllist[_l[1]].y - drilllist[_l[0]].y)/addhnum

    for i in range(len(samstraids)):
        if drilllist[_l[0]].ZCBH.index(samstraids[i])!=0:
            cdsd11 = drilllist[_l[0]].CDSD[drilllist[_l[0]].ZCBH.index(samstraids[i])-1]
            cdsd12 = drilllist[_l[0]].CDSD[drilllist[_l[0]].ZCBH.index(samstraids[i])]

        else:
            cdsd11=0
            cdsd12 = drilllist[_l[0]].CDSD[drilllist[_l[0]].ZCBH.index(samstraids[i])]

        if drilllist[_l[1]].ZCBH.index(samstraids[i])!=0:
            cdsd21 = drilllist[_l[1]].CDSD[drilllist[_l[1]].ZCBH.index(samstraids[i])-1]
            cdsd22 = drilllist[_l[1]].CDSD[drilllist[_l[1]].ZCBH.index(samstraids[i])]

        else:
            cdsd21 = 0
            cdsd22 = drilllist[_l[1]].CDSD[drilllist[_l[1]].ZCBH.index(samstraids[i])]

        z11=maxBG - drilllist[_l[0]].ZKBG + cdsd11
        z12=maxBG - drilllist[_l[0]].ZKBG + cdsd12
        z21=maxBG - drilllist[_l[1]].ZKBG + cdsd21
        z22=maxBG - drilllist[_l[1]].ZKBG + cdsd22


        zupinc=(z21-z11)/addhnum
        zdoinc=(z22-z12)/addhnum


        for j in range(1,addhnum):

            zup=z11+j*zupinc
            zdown=z12+j*zdoinc
            addvnum=5
            zinc=(zdown-zup)/addvnum
            for k in range(1,addvnum):

                newtrdata1 = np.array([[(drilllist[_l[0]].x+j*xinc)/ MaxX, (drilllist[_l[0]].y+j*yinc) / MaxY, (zup+k*zinc) / MaxZ]])

                newtrdata = np.concatenate((newtrdata, newtrdata1), axis=0)
                newtrlabel1 = np.array([samstraids[i]])
                newtrlabel = np.concatenate((newtrlabel, newtrlabel1), axis=0)

newATRDrill = Drill_Datas(torch.from_numpy(newtrdata), torch.from_numpy(newtrlabel))
print(len(newATRDrill))


addtraindatafile = os.path.join(r'E:\pyfile\try', 'add_traindata' + str(stepnum) + '.txt')
Clear(addtraindatafile)
with open(addtraindatafile, "a") as f:
    for i in range(len(newtrdata)):
        f.write(str((newtrdata[i][0])*(MaxX)) + "," + str((newtrdata[i][1])*(MaxY)) + "," +str((newtrdata[i][2])*(MaxZ)) + "," + str(int(newtrlabel[i])) +"\n")

pridata=np.array([[]]*3).T
prilabel=np.array([])
print(pridata)
print(prilabel)


for i in range(0,100):
    print('x',i)
    for j in range(0,100):
        for _tri in triplaneset:

            if (isPointinPolygon([(i+0.5)*intervalx, (j+0.5)*intervaly],[(_tri.drill1.x, _tri.drill1.y), (_tri.drill2.x, _tri.drill2.y), (_tri.drill3.x, _tri.drill3.y),(_tri.drill1.x, _tri.drill1.y)])):

                for k in range(0,60):

                    if (_tri.planeset[0].A * ((i+0.5)*intervalx) + _tri.planeset[0].B * ((j+0.5)*intervaly) + _tri.planeset[0].C * (-(k+0.5)*intervalv) + _tri.planeset[0].D) > 0:
                        continue

                    elif (_tri.planeset[-1].A*((i+0.5)*intervalx)+_tri.planeset[-1].B*((j+0.5)*intervaly)+_tri.planeset[-1].C*(-(k+0.5)*intervalv)+_tri.planeset[-1].D)<0:
                        continue

                    else :
                        for _x, _y in zip(range(0, len(_tri.planeset)), range(1, len(_tri.planeset))):
                            if ((_tri.planeset[_x].A * ((i+0.5)*intervalx) + _tri.planeset[_x].B * ((j+0.5)*intervaly) + _tri.planeset[_x].C * (-(k+0.5)*intervalv) + _tri.planeset[_x].D) < 0 and (_tri.planeset[_y].A * ((i+0.5)*intervalx) + _tri.planeset[_y].B * ((j+0.5)*intervaly) + _tri.planeset[_y].C * (-(k+0.5)*intervalv) + _tri.planeset[_y].D) > 0):
                                newpridata=np.array([[((i+0.5)*intervalx)/ MaxX,((j+0.5)*intervaly)/ MaxY,((k+0.5)*intervalv)/ MaxZ]])
                                pridata = np.concatenate((pridata, newpridata), axis=0)
                                newprilabel=np.array([_tri.straids[_x]])

                                prilabel = np.concatenate((prilabel, newprilabel), axis=0)
print(len(pridata))
addpridatafile = os.path.join(r'E:\pyfile\try', 'add_pridata' + str(stepnum) + '.txt')
Clear(addpridatafile)
with open(addpridatafile, "a") as f:
    for i in range(0,len(pridata)):

        f.write(str((pridata[i][0])*(MaxX)) + "," + str((pridata[i][1])*(MaxY)) + "," +str((pridata[i][2])*(MaxZ)) + "," + str(int(prilabel[i])) +"\n")

print(prilabel)


train_size = int(0.8*len(newATRDrill))
print(train_size)
test_size = len(newATRDrill) - train_size
print(test_size)
Train, test_datas = torch.utils.data.random_split(newATRDrill, [train_size, test_size])
train_size = int(0.8 * len(Train))
print(train_size)
test_size = len(Train) - train_size
print(test_size)
train_datas, eval_datas = torch.utils.data.random_split(Train, [train_size, test_size])


Train_datas = torch.utils.data.DataLoader(train_datas, batch_size=10000, shuffle=True)
Test_datas = torch.utils.data.DataLoader(test_datas, batch_size=4096, shuffle=True)
Eval_datas = torch.utils.data.DataLoader(eval_datas, batch_size=4096, shuffle=True)
batch, batch_label = next(iter(train_datas))

print(batch.shape)
print(batch_label.shape)
for i, data in enumerate(Train_datas):

    print(" {}th Batch \n{}".format(i, data))
print("Train divide finish")






class Net(nn.Module):
    def __init__(self, input__size, hidden_size, output_size):
        super(Net, self).__init__()

        self.layer1 = nn.Linear(input__size, hidden_size)

        self.layer2 = nn.Linear(hidden_size, 2 * hidden_size )
        self.layer3 = nn.Linear(2*hidden_size, 4 * hidden_size)
        self.layer4 = nn.Linear(4*hidden_size, output_size)


    def forward(self, x):
        out = self.layer1(x)
        out1 = F.prelu(out,torch.tensor([0.05]).cuda())

        out = self.layer2(out1)
        out2 = F.prelu(out,torch.tensor([0.05]).cuda())


        out = self.layer3(out2)
        out3 = F.prelu(out,torch.tensor([0.05]).cuda())
        out = self.layer4(out3)
        return out,out1,out2,out3


net = Net(3, 128, (maxstraid+1))
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adamax(net.parameters(), lr=0.003)

print("init lr：", optimizer.defaults['lr'])

print(torch.cuda.is_available())
net = net.cuda()

train_losses = []
train_acces = []

eval_losses = []
eval_acces = []


def AccuarcyCompute(out, label):
    out = out.detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    num_correct = (np.argmax(out, 1) == label).sum().item()
    return num_correct
epochs_num=2001
n=0
plabelflag=1
for e in range(epochs_num):

    start_0 = time.clock()

    train_loss = 0
    train_acc = 0
    net.train()
    for inputs, labels in Train_datas:

        input = torch.autograd.Variable(inputs).cuda()
        label = torch.autograd.Variable(labels).cuda()

        out,out1,out2,out3 = net(input.float())

        loss1 = criterion(out, label.long())

        loss = 1*loss1

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        train_loss += loss.item()

        num_correct=AccuarcyCompute(out, label.long())

        acc = num_correct / input.shape[0]
        train_acc += acc

    train_losses.append(train_loss / len(Train_datas))
    train_acces.append(train_acc / len(Train_datas))

    if (e % 500 == 0):
        modelpath = os.path.join(r'E:\pyfile\try', 'modelueq' + str(stepnum) + str(e) + '.pt')
        modeldatapath = os.path.join(r'E:\pyfile\try', 'model_dataueq' + str(stepnum) + str(e) + '.pt')
        torch.save(net, modelpath)
        torch.save(net.state_dict(), modeldatapath)



    eval_loss = 0
    eval_acc = 0
    net.eval()


    for input, label in Eval_datas:
        input = torch.autograd.Variable(input).cuda()
        label = torch.autograd.Variable(label).cuda()

        out, out1, out2, out3 = net(input.float())

        loss = criterion(out, label.long())

        eval_loss += loss.item()

        out = out.detach().cpu().numpy()
        label = label.long().detach().cpu().numpy()
        num_correct = (np.argmax(out, 1) == label).sum().item()

        acc = num_correct / input.shape[0]
        eval_acc += acc
    end_0 = time.clock()
    eval_losses.append(eval_loss / len(Eval_datas))
    eval_acces.append(eval_acc / len(Eval_datas))
    print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}, epoch time: {:.6f}, lr:{}'.format(e, train_loss / len(Train_datas), train_acc / len(Train_datas),eval_loss / len(Eval_datas), eval_acc / len(Eval_datas),  end_0-start_0, optimizer.param_groups[0]['lr']))


    if (e==500+100*n and train_acc > 0.9 and plabelflag ):
        n=n+1
        net.eval()
        delnum=0
        plabeladdnum=0
        plabelnum=len(pridata)
        for pdata in range(0,len(pridata)):
            input = torch.autograd.Variable(torch.from_numpy(pridata[pdata-delnum])).cuda()
            out, out1, out2, out3 = net(input.float())
            pred = torch.argsort(out, 0, True)
            plabel = pred[0].item()

            if plabel == prilabel[pdata-delnum]:
                plabeladdnum=plabeladdnum+1
                newtrdata=np.concatenate((newtrdata,[pridata[pdata-delnum]]),axis=0)
                newtrlabel = np.concatenate((newtrlabel, [prilabel[pdata-delnum]]), axis=0)
                pridata=np.delete(pridata,pdata-delnum,axis=0)
                prilabel=np.delete(prilabel,pdata-delnum,axis=0)
                delnum =delnum+1
        if (plabeladdnum/plabelnum)<0.1:
            plabelflag=0


        newATRDrill = Drill_Datas(torch.from_numpy(newtrdata), torch.from_numpy(newtrlabel))
        print(len(newATRDrill))

        train_size = int(len(newATRDrill))

        test_size = 0

        train_datas, eval_datas = torch.utils.data.random_split(newATRDrill, [train_size, test_size])
        Train_datas = torch.utils.data.DataLoader(train_datas, batch_size=10000, shuffle=False)

score_list = []
label_list = []
prob_all = []
lable_all = []
conf_matrix = torch.zeros(maxstraid + 1, maxstraid + 1)
test_acc = 0
net.eval()
for inputs, labels in Test_datas:
    input = torch.autograd.Variable(inputs).cuda()
    label = torch.autograd.Variable(labels).cuda()
    out, out1, out2, out3 = net(input.float())

    prob = out.cpu().detach().numpy()
    prob_all.extend(np.argmax(prob, axis=1))
    lable_all.extend(label.cpu().numpy())

    prediction = torch.max(out, 1)[1]
    score_list.extend(out.detach().cpu().numpy())
    label_list.extend(labels.long().cpu().numpy())
    out = out.detach().cpu().numpy()
    label = label.long().detach().cpu().numpy()
    num_correct = (np.argmax(out, 1) == label).sum().item()
    acc = num_correct / input.shape[0]
    test_acc += acc
print('Test Acc: {:.6f}'.format(test_acc / len(Test_datas)))
plt.figure()
plt.plot(np.arange(len(train_losses)), train_losses,label='loss')
plt.xlabel('epochs',fontdict={'family':'Times New Roman','fontsize':18})
plt.ylabel('loss',fontdict={'family':'Times New Roman','fontsize':18})
plt.title('loss', fontdict={'family': 'Times New Roman','fontsize':18})
plt.legend(loc="upper right", prop={'family':'Times New Roman','size':18})
plt.savefig(str(stepnum) + "loss.png",dpi=400,bbox_inches='tight')
plt.figure()
plt.plot(np.arange(len(train_acces)), train_acces,label='acc')
plt.xlabel('epochs',fontdict={'family':'Times New Roman','fontsize':18})
plt.ylabel('accurary',fontdict={'family':'Times New Roman','fontsize':18})
plt.title('accurary',fontdict={'family':'Times New Roman','fontsize':18})
plt.legend(loc="lower right", prop={'family':'Times New Roman','size':18})
plt.savefig(str(stepnum) + "acc.png",dpi=400,bbox_inches='tight')
plt.show()


