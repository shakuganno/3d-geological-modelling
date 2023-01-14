import os
import pyodbc
from pyodbc import connect
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim  # 实现各种优化算法的包
from torch.utils.data import Dataset  # 子类化数据
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt  # 画图

#from myoctree_tri_poly_LWH import Octree, Net
from scipy.spatial import Delaunay
import time
from hull3D import ConvexHull3D
from scipy import spatial
from mpl_toolkits.mplot3d import Axes3D


class Net(nn.Module):
    def __init__(self, input__size, hidden_size, output_size):
        super(Net, self).__init__()

        self.layer1 = nn.Linear(input__size, hidden_size)


        self.layer2 = nn.Linear(hidden_size, 2 * hidden_size)
        self.layer3 = nn.Linear(2 * hidden_size, 4 * hidden_size)
        self.layer4 = nn.Linear(4 * hidden_size, output_size)

    def forward(self, x):
        out = self.layer1(x)
        out1 = F.prelu(out, torch.tensor([0.05]).cuda())

        out = self.layer2(out1)
        out2 = F.prelu(out, torch.tensor([0.05]).cuda())


        out = self.layer3(out2)
        out3 = F.prelu(out, torch.tensor([0.05]).cuda())
        out = self.layer4(out3)

        return out, out1, out2, out3



def Clear(FileName):
    with open(FileName, "w") as f:
        f.close()
        print("已清空数据文件")



class Point():
    def __init__(self,id,x,y,z):
        self.id=id
        self.x=x
        self.y=y
        self.z=z

class Triangle():
    def __init__(self, point1, point2, point3):
        self.p1=point1
        self.p2=point2
        self.p3=point3

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
            print("在顶点上")
            return False

        if (point1[1] < point[1] and point2[1] >= point[1]) or (point1[1] >= point[1] and point2[1] < point[1]):

            point12lng = point2[0] - (point2[1] - point[1]) * (point2[0] - point1[0]) / (point2[1] - point1[1])

            if (point12lng == point[0]):
                print("点在多边形边上")
                return True
            if (point12lng < point[0]):
                count += 1
        point1 = point2
    if count % 2 == 0:
        return False
    else:
        return True



def getplane(P1,P2,P3):
    A=(P2.y-P1.y)*(P3.z-P1.z)-(P2.z-P1.z)*(P3.y-P1.y)
    B=(P2.z-P1.z)*(P3.x-P1.x)-(P2.x-P1.x)*(P3.z-P1.z)
    C=(P2.x-P1.x)*(P3.y-P1.y)-(P2.y-P1.y)*(P3.x-P1.x)
    D=-(A*P1.x+B*P1.y+C*P1.z)
    return A,B,C,D


Geo=[]

xcubenum=200
ycubenum=200
zcubenum=120


for i in range(0,1):
    path = os.path.join(r'E:\pyfile\ruanjian\Drill_MLP\Data\min_max', 'ZKZB' + str(i) + '.txt')
    prdict_path=os.path.join(r'E:\pyfile\ruanjian\Drill_MLP\Data\yspredict', 'cxpredictorg' + str(i) + '.txt')
    Clear(prdict_path)
    boundary = np.loadtxt(path, delimiter=',', usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8,9])
    print("boundary:", boundary)
    L = boundary[0] / xcubenum
    W = boundary[2] / ycubenum
    H = boundary[4] / zcubenum
    depth = 7
    maxstraid = boundary[8]
    print('L,W,H',L,W,H)

    Data_FileName = os.path.join(r'E:\pyfile\try', 'GAXC2Data' + str(13) + '.txt')
    Drill_Data = np.loadtxt(Data_FileName, delimiter=',,', usecols=[0, 1, 2, 3, 4, 5, 6])
    uppoint = []  #

    for ii in range(0, 208):      #borehole num
        for data in Drill_Data:
            if data[6] == ii:
                uppoint.append(Point(ii, data[0] - boundary[6], data[1] - boundary[7], boundary[9] - data[4]))
                print(boundary[9] - data[4])
                break


    downpoint = []
    downpoint=np.array([[]]*3).T
    print(downpoint)


    glodata = [0, 0, 0, 0, 0, 0]
    # 13556
    for ii in range(0, 208):
        for data in Drill_Data:
            if data[6] == ii:
                glodata = data
            if data[6] == ii + 1:
                newdownpoint=np.array([[glodata[0] - boundary[6], glodata[1] - boundary[7],boundary[9] - glodata[4] + glodata[3]]])
                downpoint = np.concatenate((downpoint, newdownpoint), axis=0)

                break
    newdownpoint = np.array([[glodata[0] - boundary[6], glodata[1] - boundary[7], boundary[9] - glodata[4] + glodata[3]]])
    downpoint = np.concatenate((downpoint, newdownpoint), axis=0)

    newdownpoint = np.array([[ 0,0,0]])
    downpoint = np.concatenate((downpoint, newdownpoint), axis=0)
    newdownpoint = np.array([[ boundary[0],0,0]])
    downpoint = np.concatenate((downpoint, newdownpoint), axis=0)
    newdownpoint = np.array([[ 0,boundary[2],0]])
    downpoint = np.concatenate((downpoint, newdownpoint), axis=0)
    newdownpoint = np.array([[ boundary[0],boundary[2],0]])
    downpoint = np.concatenate((downpoint, newdownpoint), axis=0)


    Hull = ConvexHull3D(downpoint, run=True, preproc=False, make_frames=False, frames_dir='./frames/')
    vertices0 = Hull.DCEL.vertexDict.values()

    downpointlist=[]
    for value in vertices0:

        if not((value.x ==0 and value.y==0 and value.z==0)or(value.x == boundary[0] and value.y==0 and value.z==0)or (value.x == 0 and value.y==boundary[2] and value.z==0)or (value.x == boundary[0] and value.y==boundary[2] and value.z==0)):

            downpointlist.append([value.x,value.y,value.z])
            print(len(downpointlist))

    downpoint2d = []
    for p in downpointlist:
        downpoint2d.append([p[0], p[1]])
    print(downpoint2d)
    downtri = Delaunay(downpoint2d)
    downtriangle = []
    for t in downtri.simplices:
        print(downpointlist[t[0]], downpointlist[t[1]], downpointlist[t[2]])

        downtriangle.append(Triangle(Point(0,downpointlist[t[0]][0],downpointlist[t[0]][1],downpointlist[t[0]][2]), Point(0,downpointlist[t[1]][0],downpointlist[t[1]][1],downpointlist[t[1]][2]), Point(0,downpointlist[t[2]][0],downpointlist[t[2]][1],downpointlist[t[2]][2])))





    uppoint2d = []
    for p in uppoint:
        uppoint2d.append([p.x, p.y])


    tri = Delaunay(uppoint2d)
    triangle = []
    for t in tri.simplices:
        triangle.append(Triangle(uppoint[t[0]], uppoint[t[1]], uppoint[t[2]]))





    with open(os.path.join(r'E:\pyfile\ruanjian\Drill_MLP\Data\yspredict', 'cxpredictorg' + str(i) + '.txt'), "a") as f:
        f.write(str(L) + ","+str(W) + ","+str(H) + ","+str(depth)+","+str(boundary[0])+ ","+str(boundary[1])+ ","+str(boundary[2])+ ","+str(boundary[3])+ ","+str(boundary[4])+ ","+str(boundary[5])+","+str(boundary[6])+","+str(boundary[7])+"\n")
    net = Net(3, 128, int(maxstraid ))

    net = net.cuda()
    net = torch.load(os.path.join(r'E:\pyfile\ruanjian\Drill_MLP\ysmodel', 'modeltest' + str(i) + '.pt'))
    start=time.clock()
    with open(prdict_path , "a") as f:
        for x in range(1,2*xcubenum+1,2):
            for y in range(1,2*ycubenum+1,2):
                _straid=-1
                _sum=0
                for _tri in triangle:
                    if (isPointinPolygon([(x*L/2),(y*W/2)],[(_tri.p1.x, _tri.p1.y), (_tri.p2.x, _tri.p2.y), (_tri.p3.x, _tri.p3.y),(_tri.p1.x, _tri.p1.y)])):
                        upA,upB,upC,upD=getplane(_tri.p1,_tri.p2,_tri.p3)

                        for _downtri in downtriangle:
                            if (isPointinPolygon([(x * L / 2), (y * W / 2)],[(_downtri.p1.x, _downtri.p1.y), (_downtri.p2.x, _downtri.p2.y),(_downtri.p3.x, _downtri.p3.y), (_downtri.p1.x, _downtri.p1.y)])):
                                downA, downB, downC, downD = getplane(_downtri.p1, _downtri.p2, _downtri.p3)
                                for z in range(1, 2*zcubenum+1, 2):

                                    if((upA*(x*L/2)+upB*(y*W/2)+upC*(z*H/2)+upD)>0):
                                        if((downA*(x*L/2)+downB*(y*W/2)+downC*(z*H/2)+downD)>0):
                                            straid = int(boundary[8]+1)
                                        else:
                                            midnp = (x * L / 2/boundary[0], y * W / 2/boundary[2] , z * H / 2/boundary[4])
                                            inputs = torch.autograd.Variable(torch.from_numpy(np.array(midnp))).cuda()
                                            net.eval()
                                            outputs,out1,out2,out3 = net(inputs.float())

                                            pred=torch.argsort(outputs,0,True)

                                            straid = pred[0].item()

                                        if _straid==-1:
                                            _straid=straid
                                        if _straid==straid:
                                            _sum=_sum+1
                                        else:
                                            with open(prdict_path, "a") as f:
                                                f.write(str(x) + ',' + str(_straid) + ',' + str(x * L / 2) + ',' + str(    y * W / 2) + ',' + str((z-2 - _sum+1 )  * H / 2) + ',' + str(
                                                    L) + ',' + str(W) + ',' + str((_sum) * H) + '\n')
                                            _straid = straid
                                            _sum=1
                                        if (z ==2*zcubenum-1):
                                            with open(prdict_path, "a") as f:
                                                f.write(str(x) + ',' + str(_straid) + ',' + str(x * L / 2) + ',' + str(
                                                    y * W / 2) + ',' + str((z - _sum+1 )  * H / 2) + ',' + str(
                                                    L) + ',' + str(W) + ',' + str((_sum) * H) + '\n')
                                break
                        print(x,y)
                        break

    end=time.clock()
    print((end-start)/60)









