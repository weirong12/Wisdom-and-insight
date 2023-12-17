 # -- coding: utf-8 --
# 从testvideo文件夹中提取图像，检测图像变化的时刻，调用线程处理识别任务
#import sys
import threading
#import msvcrt
#测试线程------------------------
import numpy as np
#from ctypes import *
import cv2
import time
import tkinter as tk
#from operator import itemgetter
import tkinter.font as tkFont
import my_module_1 as md
#sys.path.append("./MvImport")
#from MvCameraControl_class import *

HEIGHT = 700
WIDTH = 1500

root = tk.Tk()
root.title('仪表检测')

bg_col = 'white'
digit_col = 'white'
meter_col = 'gray'
label_col = 'black'
error = np.zeros(6)   #存放检测出现错误的表盘标号，1-6
#dig_code = np.zeros((6,9),dtype=int) # 识别结果，存文件，主程序读取，显示到菜单中
#coor = np.zeros((8),dtype=int) #保存到文件中，以后的线程使用
#all_coor =np.zeros((6,28),dtype = int)  #保存到文件中面板4个，up12个，dr12个，dl4个，以后的线程使用
numofpanel = 6  #一次检测6个仪表
numofstate = 6   #被测仪表的状态数，目前暂定6个，适用NGG5000
numofcode = 9  # 上排3个数码，下排3个数码，小灯3个状态
#names = locals()  #将字符串转变成变量名,这个语句有问题
names = {}
for i in range(1,numofpanel+1):    
    names ['panel'+str(i)] = tk.StringVar()   #定义出错仪表的名字
    names ['panel'+str(i)].set('')
    for j in range(1,numofstate+1):
        names ['up_num' + str(i)+str(j)] = tk.StringVar()    #上排数字变量
        names ['up_num' + str(i)+str(j)].set('')  
        
        names ['down_num' + str(i)+str(j)] = tk.StringVar()   #下排数字变量
        names ['down_num' + str(i)+str(j)].set('')
        
        names ['lamb_num' + str(i)+str(j)] = tk.StringVar()     #小灯状态变量
        names ['lamb_num' + str(i)+str(j)].set('')
meter ={}
for i in range(1,numofpanel+1):  #定义的字符串变量名用于界面显示，后面将字符串转变成变量
    nameA = []        
    for j in range(1,numofstate+1):  #for j in range(2,numofstate+1):
        nameA += ['up_num'+str(i)+str(j)]+['down_num'+str(i)+str(j)]+['lamb_num'+str(i)+str(j)]           
    key = 'meter'+str(i)
    meter[key] = tuple(nameA)
# =============================================================================
# meter ={}
# for i in range(1,numofpanel+1):  #定义的字符串变量名用于界面显示，后面将字符串转变成变量
#     nameA = []        
#     for j in range(1,numofstate+1):  #for j in range(2,numofstate+1):
#         nameA += ['up_num'+str(i)+str(j)]+['down_num'+str(i)+str(j)]+['lamb_num'+str(i)+str(j)]           
#     key = 'meter'+str(i)
#     meter[key] = tuple(nameA)        
# =============================================================================

#将仪表的标准数据存放在standard.txt文件中，用于核对检测结果是否正确
# =============================================================================
# array = [''  for i in range(numofcode)]
# standard = [array]*(numofstate-1)   #目前仅考虑实际状态数，与standard.txt文件一致，4行9列，字符格式
# i = 0
# f = open('standard.txt','r')   #提取仪表的正确数字和小灯状态，这些数据事先保存在电脑
# for each_line in f:
#     standard[i]  = each_line.split()
#     i += 1
# f.close()
# =============================================================================
#standard = np.load('standard.npy')
#all_coor = np.load('all_coor_v1.npy')

def other(img,seq,all_coor):  #seq表示目前检测的是第几个状态，从1-6，meter没有用
    #time.sleep(3)   #为了使4张图像的结果有序显示，后面3张图像的操作需要延时
    global s1_1        
    global s1_2
    global s1_3
    
    # global error

#    all_coor = np.load('all_coor_v1.npy')  #需要检测要素的坐标
    so_1 =  np.zeros(6)   #存放其他状态小灯的状态值
    so_2 =  np.zeros(6)
    so_3 =  np.zeros(6)
    up_1,up_2,up_3,down_1,down_2,down_3 = md.one2six(img)
    images=[up_1,up_2,up_3,down_1,down_2,down_3] 
    for i in range(6):
        
        img = images[i]
#        print('仪表'+str(i)+'检测结果-----------------:')
        y1,x1,y2,x2 = all_coor[i][0:4]
        up = img[y1:y2,x1:x2]
        # cv2.imshow('panelimg',up)
         # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        binup = md.gray2bin(up)
        
        leftu = binup[:,0:all_coor[i][4]]
        midu =  binup[:,all_coor[i][5]:all_coor[i][6]]
        rightu =  binup[:,all_coor[i][7]:x2]
        
        a = md.recog_code(leftu)  #得到字符 a
        b = md.recog_code(midu)
        c = md.recog_code(rightu)
        output = []
        output = a
        output  += b
        output += c
#        print('上上上',a,b,c)
        pos_val = names[meter['meter' + str(i + 1)][(seq-1)*3]] #names将字符串转变成变量名
        pos_val.set(output)   

        y1,x1,y2,x2 = all_coor[i][8:12]
        dr = img[y1:y2,x1:x2]
        bindr = md.gray2bin(dr)
        leftd = bindr[:,0:all_coor[i][12]]
        midd=  bindr[:,all_coor[i][13]:all_coor[i][14]]
        rightd =  bindr[:,all_coor[i][15]:x2]
        a = md.recog_code(leftd)  #得到字符量 a
        b = md.recog_code(midd)
        c = md.recog_code(rightd)
        output = []
        output = a
        output  += b
        output += c
#        print('下下',a,b,c)
        pos_val = names[meter['meter' + str(i + 1)][(seq-1)*3+1]] #names将字符串转变成变量名
        pos_val.set(output)           
        
        y1,x1,y2,x2 = all_coor[i][16:20]
        lu= img[y1:y2,x1:x2]   #小灯信息
        _,W = lu.shape        
        lu1 = lu[:,0:W//2]     #上排小灯左边
        lu2 = lu[:,W//2:W]     #上排小灯右边  
        y1,x1,y2,x2 = all_coor[i][28:32]
        ld = img[y1:y2,x1:x2]
        _,W = ld.shape        
        ld1 = ld[:,0:W//2]     #下排小灯左边
              
        # 设置输出数码管
                
        # pos_val = names[meter['meter' + str(i + 1)][(seq-1)*3]] #names将字符串转变成变量名
        # pos_val.set(output)   
            
        
        # 设置输出数码管
        
        # pos_val = names[meter['meter' + str(i + 1)][(seq-2)*3+1]]#names将字符串转变成变量名
        # pos_val = names[meter['meter' + str(i + 1)][(seq-1)*3+1]]#names将字符串转变成变量名    
        # pos_val.set(output1)   
        
        if (seq == 1):
            
            binlu1 = md.gray2bin_lamb(lu1)
            
            binlu2 = md.gray2bin_lamb(lu2)
            binlu1 = 255 - np.int16(binlu1)
            binlu2 = 255 - np.int16(binlu2)
            y1,x1,y2,x2 = all_coor[i,20],all_coor[i,21],all_coor[i,22],all_coor[i,23]
            _,w1 =binlu1.shape
            
            xmin = x1-5 if x1-5 > 0 else 0
            xmax = x2+5 if x2+5 < w1 else w1  
            lamb_1 = binlu1[y1:y2,xmin:xmax]  #上排左
            y1,x1,y2,x2 = all_coor[i,24],all_coor[i,25],all_coor[i,26],all_coor[i,27]
            _,w1 =binlu1.shape
            
            xmin = x1-5 if x1-5 > 0 else 0
            xmax = x2+5 if x2+5 < w1 else w1  
            lamb_2 = binlu2[y1:y2,xmin:xmax]    #上排右
            
            #lamb_2 = np.uint8(lamb_2)  
            
            
            binld1 = md.gray2bin_lamb(ld1)            
            binld1 = 255 - np.int16(binld1)
            y1,x1,y2,x2 = all_coor[i,32],all_coor[i,33],all_coor[i,34],all_coor[i,35]
            _,w1 =binld1.shape
            
            xmin = x1-5 if x1-5 > 0 else 0
            xmax = x2+5 if x2+5 < w1 else w1  
            lamb_3= binld1[y1:y2,xmin:xmax]  #下排左
            # lamb_3 = np.uint8(lamb_3) 
            # cv2.imshow('panelimg',lamb_3)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()  
            
            area1 = sum(sum(lamb_1)/255)
            area2 = sum(sum(lamb_2)/255)
            area3 = sum(sum(lamb_3 )/255)
        
            h1,w1 = lamb_1.shape
            h2,w2 = lamb_2.shape
            h3,w3 = lamb_3.shape
            
            lamb1 = '1' if area1 > (h1*w1)/4 else '0'
            lamb2 = '1' if area2 > (h2*w2)/4 else '0'   
            lamb3 = '1' if area3 > (h3*w3)/4 else '0'

#            print ('小灯状态=',lamb1,lamb2,lamb3)

            
            if lamb1 == '1':                
                _ , s1_1[i] = cv2.meanStdDev(lamb_1)       #计算小灯区域的方差
            else:
                print('状态1小灯检测出错')
            if lamb2 == '1':
                _ , s1_2[i] = cv2.meanStdDev(lamb_2)
            else:
                print('状态1小灯检测出错')
            if lamb3 == '1':
                _ , s1_3[i] = cv2.meanStdDev(lamb_3)
            else:
                print('状态1小灯检测出错')

            output = []
            output = lamb1
            output  += lamb2
            output += lamb3   
#            print ('全8状态=',seq,s1_1[i],s1_2[i],s1_3[i])
            #print('output',output)
            pos_val = names[meter['meter' + str(i + 1)][(seq-1)*3+2]]#names将字符串转变成变量名
            pos_val.set(output)   
        else:  
            
            y1,x1,y2,x2 = all_coor[i,20],all_coor[i,21],all_coor[i,22],all_coor[i,23]
            _,w1 = lu1.shape
            
            xmin = x1-5 if x1-5 > 0 else 0
            xmax = x2+5 if x2+5 < w1 else w1
#            print(xmin,xmax)   
            lamb_1 = lu1[y1:y2,xmin:xmax]            
            _ , so_1[i] = cv2.meanStdDev(lamb_1)
            
            
            y1,x1,y2,x2 = all_coor[i,24],all_coor[i,25],all_coor[i,26],all_coor[i,27] 
            
            _,w1 = lu2.shape
           
            xmin = x1-5 if x1-5 > 0 else 0
            xmax = x2+5 if x2+5 < w1 else w1
#            print(xmin,xmax)   
            lamb_2= lu2[y1:y2,xmin:xmax]
            _ , so_2[i] = cv2.meanStdDev(lamb_2)
             
            
            y1,x1,y2,x2 = all_coor[i,32],all_coor[i,33],all_coor[i,34],all_coor[i,35]
            
            _,w = ld1.shape
            xmin = x1-5 if x1-5 > 0 else 0
            xmax = x2+5 if x2+5 < w else w

            lamb_3 = ld1[y1:y2,xmin:xmax]
            _ , so_3[i] = cv2.meanStdDev(lamb_3)
            # cv2.imshow('panelimg',lamb_3)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows() 
            if so_1[i] < s1_1[i]//4 :
                lamb1 = '0'
            else:
                lamb1 = '1'
            if so_2[i] < s1_2[i]//4 :
                lamb2 = '0'
            else:
                lamb2 = '1'   
            if so_3[i] < s1_3[i]//4 :
                lamb3 = '0'
            else:
                lamb3 = '1'
#            print ('其他状态=',seq,so_1[i],so_2[i],so_3[i],s1_1[i],s1_2[i],s1_3[i])
            output = []
            output = lamb1
            output  += lamb2
            output += lamb3   
#            print('小灯output=',output)
            pos_val = names[meter['meter' + str(i + 1)][(seq-1)*3+2]] #names将字符串转变成变量名
            pos_val.set(output)  

        
#def work_thread(cam=0, pData=0, nDataSize=0): 
def work_thread():     
# =============================================================================
#     ret = cam.MV_CC_StartGrabbing()
#     if ret != 0:
#         print ("start grabbing fail! ret[0x%x]" % ret)
#         sys.exit()
#     stFrameInfo = MV_FRAME_OUT_INFO_EX()
#     memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
# =============================================================================
    print('-----------开始拍照----------------')
#    global all_coor
    # global meter
    # global numofpanel
    # global numofstate

#    all_coor = np.load('all_coor_v1.npy')
#    pData, nDataSize, stFrameInfo = cam, byref(data_buf), nPayloadSize
    num = 0
    i = 0
    flag = -5
    threading_idx = 1
    g_bExit = False
    k = 0
    t_start = time.time()
    while True:
        i += 1
        if i == flag:
            continue
        img_gray = cv2.imread( './panel_img/'+str(k) + '.png',0) 
 
# =============================================================================
#         ret = cam.MV_CC_GetOneFrameTimeout(pData, nDataSize, stFrameInfo, 1000)
#         if ret == 0:
#             image = np.asarray(pData._obj)
#             img_gray = image.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))    # 灰度图              
# =============================================================================
#        y1,x1,y2,x2 = all_coor[0][0:4]
#        o_u = img_gray[y1:y2,x1:x2]
        # o_u = np.uint8(o_u)   
        # cv2.imshow('pane1',o_u)
        # cv2.waitKey(0) 
        #cv2.destroyAllWindows()
        if threading_idx == 1:
              mean1=np.mean(img_gray)        
              if num == 0:
                  mean2 = mean1
                  num += 1
              dis_mean = abs(mean1 - mean2)
      #        print("dis_mean=",dis_mean)                                                                                                                                                          
              if dis_mean > 3:
                  
      #            cv2.imwrite('./Img/'+str(num)+'.png',img_gray)
                  #记录当前的图像序号，设置flag
                  flag = i
              if i - flag == 2:
#                  print(dis_mean,k)
#                  img_gray8 = img_gray.copy()
                  all_coor = md.ext_all_coor(img_gray)
#                  cv2.imwrite('./Img/'+str(k)+'.png',img_gray)
#                st = time.time()
                  sub_th = threading.Thread(target = other,args = (img_gray,threading_idx,all_coor))
                  sub_th.start()
#                  print('111111111=',k)
                  H,W=img_gray.shape
                  midH = H//2
                  up_3=img_gray[0:midH,0:W//3]   
                  up_3 = md.gray2bin(up_3)
                  imgu = up_3[all_coor[0][0]:all_coor[0][1],all_coor[0][2]:all_coor[0][3]]
                  digimg= imgu[:,all_coor[0][5]:all_coor[0][6]]
                  digcode1 = md.recog_code(digimg)
                  threading_idx +=1
#                  print('111111111=',k,digcode1)
                  root.update() 
              mean2 = mean1
        else:
#            num = 0  
#            H,W=img_gray.shape
#            midH = H//2
                   
            up_3=img_gray[0:midH,0:W//3]   
            up_3 = md.gray2bin(up_3)
            imgu = up_3[all_coor[0][0]:all_coor[0][1],all_coor[0][2]:all_coor[0][3]]
            digimg= imgu[:,all_coor[0][5]:all_coor[0][6]]
            digcode2 = md.recog_code(digimg)
    #        print("dignum=",dignum,k)
    #        k += 1 
            # if num == 0:
            #     digcode2 = digcode1            
            #     num +=1
                
    #        dis_code = digcode1[0] != digcode2[0] 
#            print("digcode2=",digcode2,digcode1,k,threading_idx)
            
            if digcode1 != digcode2 :
                 flag = i
            if i - flag == 4 :  #注意，有的状态可能需要i-flag == 3  
#                 cv2.imwrite('./Img/'+str(k)+'.png',img_gray)
                 #other(img_gray,threading_idx,all_coor)
                 sub_th = threading.Thread(target = other,args = (img_gray,threading_idx,all_coor))
                 sub_th.start()
#                 print('2222222=',threading_idx,k)
                 threading_idx +=1
                 root.update() 
            digcode1 = digcode2
            if threading_idx == 7:    
                   print('time=',time.time()- t_start)
                   break
              
        k = k+1
        

# =============================================================================
#                  for k in range(numofpanel):
#             
#                      if error[k] == 1:
#                          panelname = 'panel'+str(k+1) 
#                 
#                          Rname = names[panelname]
#                          Rname.set('发现错误')
#                  g_bExit = True
#                  if g_bExit == True:
#                      print('End.................')
#                      break                         
# =============================================================================
            
        
# =============================================================================
#     ret = cam.MV_CC_StopGrabbing()
#     if ret != 0:
#         print ("stop grabbing fail! ret[0x%x]" % ret)
#         del data_buf
#         sys.exit()
#     print('停止取图像---------------')
# =============================================================================
def close_cam():  
      
#    global data_buf
#    ret = cam.MV_CC_StopGrabbing()
#    if ret != 0:
#        print ("stop grabbing fail! ret[0x%x]" % ret)
#        del data_buf
#        sys.exit()

    # ch:关闭设备 | Close device
# =============================================================================
#     ret = cam.MV_CC_CloseDevice()
#     if ret != 0:
#         print ("close deivce fail! ret[0x%x]" % ret)
#         del data_buf
#         sys.exit()
# 
#     # ch:销毁句柄 | Destroy handle
#     ret = cam.MV_CC_DestroyHandle()
#     if ret != 0:
#         print ("destroy handle fail! ret[0x%x]" % ret)
#         del data_buf
#         sys.exit()
# 
#     del data_buf
# 
# =============================================================================
     root.destroy()
     
#     root.quit()
global cam
global data_buf
global nPayloadSize 
s1_1 =  np.zeros(6)    #存放全8图像小灯区域的方差值，用于与其他状态时刻小灯区域方差比较，确定小灯是否点亮
s1_2 =  np.zeros(6)
s1_3 =  np.zeros(6)
#st = time.time()
#deviceList = MV_CC_DEVICE_INFO_LIST()
#tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
'''
# ch:枚举设备 | en:Enum device
ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
if ret != 0:
    print ("enum devices fail! ret[0x%x]" % ret)
    sys.exit()

if deviceList.nDeviceNum == 0:
    print ("find no device!")
    sys.exit()

print ("Find %d devices!" % deviceList.nDeviceNum)


nConnectionNum =  0   #input("please input the number of the device to connect:")

# ch:创建相机实例 | en:Creat Camera Object
print('tim1e=',time.time()- st)
cam = MvCamera()

# ch:选择设备并创建句柄 | en:Select device and create handle
stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents

ret = cam.MV_CC_CreateHandle(stDeviceList)
if ret != 0:
    print ("create handle fail! ret[0x%x]" % ret)
    sys.exit()

# ch:打开设备 | en:Open device
ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
if ret != 0:
    print ("open device fail! ret[0x%x]" % ret)
    sys.exit()
print('open cam success,time2=',time.time()- st)
# ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
    nPacketSize = cam.MV_CC_GetOptimalPacketSize()
    print('nPacketSize',nPacketSize)
    if int(nPacketSize) > 0:
        ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize",nPacketSize)
        if ret != 0:
            print ("Warning: Set Packet Size fail! ret[0x%x]" % ret)
    else:
        print ("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

stBool = c_bool(False)
ret =cam.MV_CC_GetBoolValue("AcquisitionFrameRateEnable", byref(stBool))
if ret != 0:
    print ("get AcquisitionFrameRateEnable fail! ret[0x%x]" % ret)
    sys.exit()

# ch:设置触发模式为off | en:Set trigger mode as off
ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
if ret != 0:
    print ("set trigger mode fail! ret[0x%x]" % ret)
    sys.exit()

# ch:获取数据包大小 | en:Get payload size
# ch:获取数据包大小 | en:Get payload size
stParam =  MVCC_INTVALUE()
memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))

ret = cam.MV_CC_GetIntValue("PayloadSize", stParam)
if ret != 0:
    print ("get payload size fail! ret[0x%x]" % ret)
    sys.exit()
nPayloadSize = stParam.nCurValue

#    # ch:开始取流 | en:Start grab image
#    ret = cam.MV_CC_StartGrabbing()
#    if ret != 0:
#        print ("start grabbing fail! ret[0x%x]" % ret)
#        sys.exit()

data_buf = (c_ubyte * nPayloadSize)()

print('open_device time =',time.time() - st)
'''
canvas=tk.Canvas(root,height=HEIGHT,width=WIDTH).pack()#Canvas()绘图形组件，可以在其中绘制图形,pack组件,包装,设置位置属性参数

back_label=tk.Label(root,background=bg_col)#标签控件（Label）指定的窗口中显示的文本和图像
back_label.place(x=0,y=0,relwidth=1,relheight=1)



frame=tk.Frame(root,bg='brown')#Frame控件在屏幕上显示一个矩形区域，多用来作为容器
frame.place(relx=0.5,rely=0.02,relwidth=0.95,relheight=0.06,anchor='n' )



f1 = tkFont.Font(family='黑体', size=22)
b1=tk.Label(frame,text='仪表检测结果',width=2,height=2,
            foreground='black',font=f1)
b1.place(relx=0.024,rely=0.1,relwidth=0.95,relheight=0.8)
wid = WIDTH//3-30
hei = HEIGHT//3
#-----界面要素布局----------------
for i in range(numofpanel):
    Rname = 'panel'+str(i+1)
    framename = 'frame'+str(i+1)
    
    if i < 3 :   #上排仪表
        names[framename]=tk.LabelFrame(root,text = '仪表'+str(i+1),bg=meter_col,font=('Arial 14'),fg ='Darkred')
        names[framename].place(x=i*wid+wid//2+10,y=100,width=wid,height=hei,anchor='n' )

    else:      #下排仪表
        names[framename]=tk.LabelFrame(root,text = '仪表'+str(i+1),bg=meter_col,font=('Arial 14'),fg ='Darkred')#Frame控件在屏幕上显示一个矩形区域，多用来作为容器
        names[framename].place(x=(i-3)*wid+wid//2+10,y=400,width=wid,height=hei,anchor='n' )
    
    up = tk.Label(names[framename],text='上排数码',width=4,height=2,fg=label_col,bg=meter_col,
    font=('Arial 10')).place(relx=0.0001,rely=0.24,relwidth=0.10,relheight=0.10)
    down = tk.Label(names[framename],text='下排数码',width=4,height=2,fg=label_col,bg=meter_col,
    font=('Arial 10')).place(relx=0.0001,rely=0.46,relwidth=0.10,relheight=0.10)
    lamb = tk.Label(names[framename],text='小灯状态',width=4,height=2,fg=label_col,bg=meter_col,
    font=('Arial 10')).place(relx=0.0001,rely=0.68,relwidth=0.10,relheight=0.10)
    # 仪表下方显示发现错误
    result = tk.Label(names[framename],textvariable=names[Rname],width=20,height=2,fg='red',bg=meter_col,
                     font=('Arial 14')).place(relx=0.4,rely=0.84,relwidth=0.19,relheight=0.10)
    for j in range(numofstate):
        statename = 's' + str(j+1)          
        names[statename]= tk.Label(names[framename],text='S' + str(j+1),width=4,height=2,
        fg=label_col,bg=meter_col,font=('Arial 13'))
#        fg = self.dtbase_color.get(),bg=meter_col,font=('Arial 13'))
        names[statename].place(relx=(0.10+j*0.15),rely=0.12,relwidth=0.10,relheight=0.10)
        #------- 数据显示部分，只显示空白----------------------

        upname = 'up_'+str(i+1)+str(j+1)
        upnumname = 'up_num'+str(i+1)+str(j+1)
        
        names[upname]=tk.Label(names[framename],textvariable=names[upnumname],width=50,height=3,
            bg=digit_col,fg= 'black',font=('Arial 13'))
        names[upname].place(relx=0.10+j*0.15,rely=0.24,relwidth=0.10,relheight=0.12)
        
        downname = 'down_'+str(i+1)+str(j+1)
        dnumname = 'down_num'+str(i+1)+str(j+1)
        
        names[downname]=tk.Label(names[framename],textvariable=names[dnumname],width=50,height=3,bg=digit_col,
                  fg= 'black', font=('Arial 13'))
        names[downname].place(relx=0.10+j*0.15,rely=0.46,relwidth=0.10,relheight=0.12)
    
        lambname = 'lamb_'+str(i+1)+str(j+1)
        lambnum ='lamb_num'+str(i+1)+str(j+1)
        
        names[lambname]=tk.Label(names[framename],textvariable=names[lambnum],width=50,height=3,bg=digit_col,
                    fg= 'black', font=('Arial 13'))
        names[lambname].place(relx=0.10+j*0.15,rely=0.68,relwidth=0.10,relheight=0.12)


#开始和结束按钮
start = tk.Button(root, text='开始检测', width=2, height=2,bg='white',fg='DarkMagenta',font=(' 13'),relief='solid',bd=3)
#bind:将事件与按钮关联。<Button-1>：鼠标左键按下，2表示中键，3表示右键；
start.bind("<Button-1>", lambda event: work_thread())
start.place(relx=0.45,rely=0.90,relwidth=0.06,relheight=0.05)
#opencam = tk.Button(root, text='打开相机', width=2, height=2,bg='white',fg='black',font=(' 13'),relief='solid',bd=3)
##bind:将事件与按钮关联。<Button-1>：鼠标左键按下，2表示中键，3表示右键；
#opencam.bind("<Button-1>", lambda event: open_cam())
#opencam.place(relx=0.25,rely=0.90,relwidth=0.06,relheight=0.05)
end = tk.Button(root, text='结束检测', width=2, height=2,bg='white',fg='DarkMagenta',font=(' 13'),relief='solid',bd=3)
#bind:将事件与按钮关联。<Button-1>：鼠标左键按下，2表示中键，3表示右键；
end.bind("<Button-2>", lambda event: close_cam())
end.place(relx=0.65,rely=0.90,relwidth=0.06,relheight=0.05)

root.protocol("WM_DELETE_WINDOW", close_cam)
root.mainloop()

