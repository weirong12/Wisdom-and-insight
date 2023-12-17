 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 13:50:38 2022

@author: jifenghuang

仪表检测程序使用的模块，供其他程序调用
已调试好的程序：extkeyimg_p.py ; xtkeyimg888_panel_img.py ;ext_up_coor.py
"""
import numpy as np
import cv2
#import matplotlib.pyplot as plt
s1_1 = np.zeros(6)    #存放自检状态小灯的方差值,全局变量，other函数使用
s1_2 =  np.zeros(6)
s1_3 =  np.zeros(6)
def one2six(original_img):
    '''
    将仪表盘的6个仪表分成6个小仪表图像,输入是大仪表盘图像，输出6个小仪表图像
    上面3个，下面3个,输入是3通道图像
    '''
#    original_img = cv2.imread(img)
    H,W=original_img.shape
    midH = H//2
    midW1 = W//3
    midW2 = W*2//3
    original_u=original_img[0:midH,:]
    original_d=original_img[midH:H,:]
    up_1= original_u[:,0:midW1]
    up_2=original_u[:,midW1:midW2]
    up_3=original_u[:,midW2:W]
    down_1= original_d[:,0:midW1]
    down_2= original_d[:,midW1:midW2]
    down_3= original_d[:,midW2:W]
    return up_1,up_2,up_3,down_1,down_2,down_3
def thre_panel(img_gray):
#根据图像直方图，寻找最佳谷点（二值化阀值）
    IDX=2 #平滑直方图，平均直方图法
    img_gray = cv2.GaussianBlur(img_gray,(3,3),0)
    hist = cv2.calcHist([img_gray],[0],None,[256],[0,256])
#    plt.plot(hist)
    #th= (hist.max()-hist.min())/2
    Meanhist=np.ones(255)
    for i in range(IDX,256-IDX):
        Meanhist[i]=sum(hist[i-IDX:i+IDX+1])//(2*IDX+1)
#    plt.plot(Meanhist)
    peak={}
    for i in range(IDX+1,256-IDX-1):
        if Meanhist[i]>=(Meanhist[i-1] ) and (Meanhist[i]>=Meanhist[i+1]+1):
            peak[i]=Meanhist[i]
    peakval = sorted(peak.items(), key=lambda item:item[1], reverse=True)
    valley={}
    if len(peak) >= 3 :
        idx1=peakval[0][0]
        idx2=peakval[1][0]
        if abs(idx1-idx2) <20:
            idx1 = idx1 if idx1 > idx2 else idx2
            idx2=peakval[2][0] 
    elif  len(peak) == 2 :
        idx1=peakval[0][0]
        idx2=peakval[1][0]
        if abs(idx1-idx2) <20:
            idx1=peakval[0][0]
            idx2=(peakval[0][0]+255)//2 
    else :
        idx1=peakval[0][0]
        idx2=(peakval[0][0]+255)//2
    
#寻找2个峰值之间的谷值
    if idx2 <idx1 :
        
        for i in range(idx2+1,idx1-1):
            if Meanhist[i]<=Meanhist[i-1] and Meanhist[i]<=Meanhist[i+1]:
                valley[i]=Meanhist[i]
#    vallegymin=min(Meanhist[idx2+1:idx1-1])
    else:
        for i in range(idx1+1,idx2-1):
            if Meanhist[i]<=Meanhist[i-1] and Meanhist[i]<=Meanhist[i+1]:
                valley[i]=Meanhist[i]
#    vallegymin=min(Meanhist[idx1+1:idx2-1])
    
    valleyval = sorted(valley.items(), key=lambda item:item[1])
    if valleyval == []:
        val=(idx1+idx2)//2
    else:
        val =valleyval[0][0]
    return val
def gray2bin(img):#输入灰度图像，输出二值图像，图像格式，背景为白，前景为黑
    # val = thre_panel(img) 
    # print('val=',val)
    # _, binimg = cv2.threshold(img, val, 255, cv2.THRESH_BINARY)
    # cv2.imshow('pane1',img)
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows()
    _, binimg = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # cv2.imshow('pane1',binimg)
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows()
    kernel=np.ones((3,3),np.uint8)
    img2= cv2.morphologyEx(binimg,cv2.MORPH_CLOSE,kernel)
    kernel=np.ones((5,5),np.uint8)
    img2 =255 - cv2.morphologyEx(img2,cv2.MORPH_OPEN,kernel)    
    return img2
def extupcoor(or_img):   #输入灰度图像，返回精确的上排数码管绝对坐标,不能处理全黑图像
    binimg = gray2bin(or_img)
    # cv2.imshow('pane1',binimg)
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows()
    binimg = np.int16(binimg)   #方便计算
    H,W = binimg.shape
    high = 5
    dis = np.zeros((high,W-1))
    a = np.zeros((high))
    H1 = H//3
    H2 = 2*(H//3)
    for k in range(H1 , H2,):
        for k1 in range(high):
            dis[k1] = binimg[k-k1,0:W-1] - binimg[k-k1,1:W]
            tp = np.nonzero(dis[k1])
            a[k1] = len(tp[0])
        b = [12]*high
        b = np.array(b)
        if all(a < b):
            break        
    y_max = k+15
    high = 15
#    H2 = k+15  if (k+15 >H) else H   #根据经验设定的值, 确定数码的上下坐标
#    H1 = k-95  if (k-95 < 0)  else 0
    dis = np.zeros((high,W-1))
    a = np.zeros((high))
    
    for k in range(y_max,0,-1):
        for k1 in range(high):
            dis[k1] = binimg[k-k1,0:W-1] - binimg[k-k1,1:W]
            tp = np.nonzero(dis[k1])
            a[k1] = len(tp[0])
        b = [12]*high
        b = np.array(b)
        if all(a < b):
            break        
    y_min = y_max -148    
    img1 = binimg[y_min:y_max,:]  #得到上下留白的上排数码图像
    # img1 = np.uint8(img1)    
    # cv2.imshow('pane555',img1)
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows()   
    H,W = img1.shape
    width = 12
    
    a = np.zeros((width))
    for k in range(W-width):
        for k1 in range(width):
            dis = img1[0:H-1,k+k1] -  img1[1:H,k+k1]
            tp = np.nonzero(dis)
            a[k1] = len(tp[0])
        b = [6]*width
        b = np.array(b)
        if all(a >= b):
            break  
    x_min = k
    a = np.zeros((width))
    for k in range(W-1,0,-1):
        for k1 in range(width):
            dis = img1[0:H-1,k-k1] -  img1[1:H,k-k1]
            tp = np.nonzero(dis)
            a[k1] = len(tp[0])
        b = [6]*width
        b = np.array(b)
        if all(a >= b):
            break  
    x_max = k        
    binup = binimg[y_min:y_max,x_min:x_max]  #3个8，2边在8的内部，上下有留白
    # binup = np.uint8(binup)    
    # cv2.imshow('pane11111',binup)
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows()   
    high = 6
    W = x_max - x_min
    dis = np.zeros((high,W-1))
    a = np.zeros((high))
    for j in range(y_min+30,y_min,-1):
        for i in range(high):
            dis[i] = binimg[j-i,x_min:x_max-1] - binimg[j-i,x_min+1:x_max]
            tp = np.nonzero(dis[i])
            a[i] = len(tp[0])
        if all( a == 0):
#             print(j,a[:])
            break    
    y_min = j
    a = np.zeros((high))
    dis = np.zeros((high,W-1))
    for j in range(y_max-20,y_max):
        for i in range(high):
            dis[i] = binimg[j+i,x_min:x_max-1] - binimg[j+i,x_min+1:x_max]
            tp = np.nonzero(dis[i])
            a[i] = len(tp[0])
        if all( a == 0):
#             print(j,a[:])
           break    
    y_max= j                
    width = 6
    H =y_max - y_min  -1
    dis = np.zeros((width,H))
    a = np.zeros((width))
    for j in range(x_min,x_min - 30,-1):
        for i in range(width):
            dis[i] = binimg[y_min:y_max-1,j-i] - binimg[y_min+1:y_max,j-i]
            tp = np.nonzero(dis[i])
            a[i] = len(tp[0])
        if all( a == 0):
#             print(j,a[:])
            break    
    x_min = j
    a = np.zeros((width))
    for j in range(x_max,x_max +30):
        for i in range(width):
            dis[i] = binimg[y_min:y_max-1,j+i] - binimg[y_min+1:y_max,j+i]
            tp = np.nonzero(dis[i])
            a[i] = len(tp[0])
        if all( a == 0):
#             print(j,a[:])
            break   
    x_max = j
    binup = binimg[y_min:y_max,x_min:x_max]   #得到顶天立地的图像
    # binup = np.uint8(binup)    
    # cv2.imshow('pane2222---',binup)
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows()   
    _,W = binup.shape
    idx =np.zeros(4).astype(int)
    for k in range(W//3,0,-1):    #分别计算3哥8的内部纵坐标
         dis = sum(255-binup[:,k])
         if dis != 0:
             idx[0] = k+1        
             break
    for k in range(W//3,2*W//3):    
         dis = sum(255-binup[:,k])
         if dis != 0:
             idx[1] = k-1          
             break  
    for k in range(2*W//3,W//3,-1):    
         dis = sum(255-binup[:,k])
         if dis != 0:
             idx[2] = k+1          
             break
    for k in range(2*W//3,W):    
         dis = sum(255-binup[:,k])
         if dis != 0:
             idx[3] = k-1           
             break  
#    binup = binup[:,idx[1]:idx[2]]
    
    return y_min,x_min,y_max,x_max,idx  #返回上排数码管的精确绝对坐标
def extdowncoor(or_imgd):   #输入灰度图像，返回精确的下排数码管绝对坐标
    binimg = gray2bin(or_imgd)
    # cv2.imshow('pane1',binimg)
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows()
    binimg = np.int16(binimg)   #方便计算
    H,W = binimg.shape
    high = 13
    dis = np.zeros((high,W-1))
    a = np.zeros((high))
    
    for k in range(H - 1):
        for k1 in range(high):
            dis[k1] = binimg[k+k1,0:W-1] - binimg[k+k1,1:W]
            tp = np.nonzero(dis[k1])
            a[k1] = len(tp[0])
        b = [12]*high
        b = np.array(b)
        if all(a >= b):
            break        
    y_min = k
    # H1 = k-12  if (k-12 >0) else 0   #根据经验设定的值, 确定数码的上下坐标
    # H2 = k+90  if (k+90<=H)  else H 
    # dis = np.zeros((high,W-1))
    # a = np.zeros((high))
    
    for k in range(H-1, y_min,-1):
        for k1 in range(high):
            dis[k1] = binimg[k-k1,0:W-1] - binimg[k-k1,1:W]
            tp = np.nonzero(dis[k1])
            a[k1] = len(tp[0])
        b = [12]*high
        b = np.array(b)
        if all(a >= b):
            break        
    y_max = k        
    img1 = binimg[y_min:y_max,:]  
    # img1 = np.uint8(img1)    
    # cv2.imshow('pane555',img1)
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows()  
    

    width = 6
    
    a = np.zeros((width))
    for k in range(W-width):
        for k1 in range(width):
            dis = binimg[0:H-1,k+k1] -  binimg[1:H,k+k1]
            tp = np.nonzero(dis)
            a[k1] = len(tp[0])
        b = [6]*width
        b = np.array(b)
        if all(a >= b):
            break  
    x_min = k
    a = np.zeros((width))
    for k in range(W-1,0,-1):
        for k1 in range(width):
            dis = binimg[0:H-1,k-k1] -  binimg[1:H,k-k1]
            tp = np.nonzero(dis)
            a[k1] = len(tp[0])
        b = [6]*width
        b = np.array(b)
        if all(a >= b):
            break  
    x_max = k        
    binup = binimg[y_min:y_max,x_min:x_max]  #3个8内圈
    # binup = np.uint8(binup)    
    # cv2.imshow('pane11111',binup)
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows()   
    high = 5
    W = x_max - x_min
    dis = np.zeros((high,W-1))
    a = np.zeros((high))
    for j in range(y_min,y_min-20,-1):
        for i in range(high):
            dis[i] = binimg[j-i,x_min:x_max-1] - binimg[j-i,x_min+1:x_max]
            tp = np.nonzero(dis[i])
            a[i] = len(tp[0])
        if all( a == 0):
#             print(j,a[:])
            break    
    y_min = j
    a = np.zeros((high))
    dis = np.zeros((high,W-1))
    for j in range(y_max,y_max+20):
        for i in range(high):
            dis[i] = binimg[j+i,x_min:x_max-1] - binimg[j+i,x_min+1:x_max]
            tp = np.nonzero(dis[i])
            a[i] = len(tp[0])
        if all( a == 0):
#             print(j,a[:])
           break    
    y_max= j        
    width = 5
    H =y_max - y_min -1
    dis = np.zeros((width,H))
    a = np.zeros((width))
    for j in range(x_min,x_min - 30,-1):
        for i in range(width):
            dis[i] = binimg[y_min:y_max-1,j-i] - binimg[y_min+1:y_max,j-i]
            tp = np.nonzero(dis[i])
            a[i] = len(tp[0])
        if all( a == 0):
#             print(j,a[:])
            break    
    x_min = j
    a = np.zeros((width))
    for j in range(x_max,x_max +20):
        for i in range(width):
            dis[i] = binimg[y_min:y_max-1,j+i] - binimg[y_min+1:y_max,j+i]
            tp = np.nonzero(dis[i])
            a[i] = len(tp[0])
        if all( a == 0):
#             print(j,a[:])
            break   
    x_max = j
    bind = binimg[y_min:y_max,x_min:x_max]  
    _,W = bind.shape
    idx =np.zeros(4).astype(int)
    for k in range(W//3,0,-1):    
         dis = sum(255-bind[:,k])
         if dis != 0:
             idx[0] = k+1        
             break
    for k in range(W//3,2*W//3):    
         dis = sum(255-bind[:,k])
         if dis != 0:
             idx[1] = k-1          
             break  
    for k in range(2*W//3,W//3,-1):    
         dis = sum(255-bind[:,k])
         if dis != 0:
             idx[2] = k+1          
             break
    for k in range(2*W//3,W):    
         dis = sum(255-bind[:,k])
         if dis != 0:
             idx[3] = k-1           
             break  
    img1 = binimg[y_min:y_max,:]  
    
    return y_min,x_min,y_max,x_max,idx
    



def gray2bin_lamb(img):#输入灰度图像，输出二值图像，图像格式，背景为白，前景为黑
    val = thre_panel(img) 
    print('val=',val)
    _, binimg = cv2.threshold(img, val, 255, cv2.THRESH_BINARY)
#    _, binimg = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel=np.ones((3,3),np.uint8)
    img2= cv2.morphologyEx(binimg,cv2.MORPH_CLOSE,kernel)
    kernel=np.ones((5,5),np.uint8)
    img2 =255 - cv2.morphologyEx(img2,cv2.MORPH_OPEN,kernel)    
    return img2
def extlamb(lambu):  #需要修改，求最大联通区域比较好
    binu = 255 - gray2bin_lamb(lambu)  #背景是黑，前景是白，轮廓处理有效
   
    kernel=np.ones((5,5),np.uint8)
    binu =cv2.morphologyEx(binu,cv2.MORPH_OPEN,kernel) #背景是白，前景是黑
    # cv2.imshow('paned',binu)
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows()   
    con,h1 = cv2.findContours(binu, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    Num = len(con)
    if Num >1:
        area = np.zeros(Num,dtype = int)
        for k1 in range(Num):
        
            area[k1] = cv2.contourArea(con[k1])
        idx = np.argmax(area)
        x,y,w,h = cv2.boundingRect(con[idx])
    elif Num == 1 :
        x,y,w,h = cv2.boundingRect(con[0])
    else: 
        print('检测小灯出错了。。。。。')
#    print(x,y,y+h,x+w)
    return y,x,y+h,x+w 

def recog_code(img):  #数码管识别程序，还有一个程序，自动生产编码，0-9，26个英文字符
    
    info ={'48':'1','109':'2','121':'3','51':'4','91':'5',
       '95':'6','112':'7','127':'8','123':'9','126':'0',
       '6':'I','21':'N','103':'P','78':'C','47':'K','99':'O'}
    # 输入的图像有可能顶天立地，加个白色的边框，2个像素宽
    H1,W1 = img.shape
        
    dig1 = 255*np.ones((H1+4,W1+4))
    dig1[2:H1+2,2:W1+2] = img
    H1,W1 =dig1.shape
    # cv2.imshow('panelimg',dig1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    digcode=np.zeros(7).astype(int)
#    #--------------
    start_y = H1//3
    disxup = abs(dig1[start_y,0:W1-1] - dig1[start_y,1:W1])
    coor1 = np.nonzero(disxup)
    disxup_nozero = len(coor1[0][:])
    if disxup_nozero == 4:
        digcode[5]=digcode[1]=1
    elif disxup_nozero == 2:
        if coor1[0][0] > W1//2:
            digcode[1] =1
        else:
            digcode[5] =1    
    #-------------
    disxdown = abs(dig1[2*start_y,0:W1-1] - dig1[2*start_y,1:W1])
    coor2 = np.nonzero(disxdown)         
    disxdown_nozero = len(coor2[0][:])
    if disxdown_nozero == 4:
        digcode[4]=digcode[2]=1
    elif disxdown_nozero == 2:
        if coor2[0][0] > W1//2:
            digcode[2] =1
        else:
            digcode[4] =1
    #------------     
    W2 = 5*W1//12 
    disymid = abs(dig1[0:H1-1,W2] - dig1[1:H1,W2])
    coor3 = np.nonzero(disymid)
    disymid_nozero = len(coor3[0][:])
    if disymid_nozero == 6:
        digcode[0]=digcode[6]=digcode[3] = 1
    elif disymid_nozero == 4:
        if coor3[0][0] < H1//4 and coor3[0][2]<3*H1//4:
            digcode[0] = digcode[6] = 1
        elif coor3[0][0] > H1//4 and coor3[0][2] > 3*H1//4:
            digcode[6] =digcode[3] = 1
        elif coor3[0][0] < H1//4 and coor3[0][2] > 3*H1//4:
            digcode[0] =digcode[3] = 1
    elif disymid_nozero == 2:
        if coor3[0][0] < H1//4:
            digcode[0] = 1
        elif coor3[0][0] > H1//4 and coor3[0][0] < 3*H1//4:
            digcode[6] = 1
        else: digcode[3] = 1
    #---------将二进制编码转变成数字-------------
    code_str = ''.join(str(i) for i in digcode)    
    code_str = '0b' + code_str
    code_dec = str(int(code_str, 2))  
    if (code_dec  not in info.keys() ):
        a = '_'
    else:
        a =info[code_dec]

    return a
def gray2bin_lamb(img):#输入灰度图像，输出二值图像，图像格式，背景为白，前景为黑
    val = thre_panel(img) 
#    print('val=',val)
    _, binimg = cv2.threshold(img, val, 255, cv2.THRESH_BINARY)
#    _, binimg = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel=np.ones((3,3),np.uint8)
    img2= cv2.morphologyEx(binimg,cv2.MORPH_CLOSE,kernel)
    kernel=np.ones((5,5),np.uint8)
    img2 =255 - cv2.morphologyEx(img2,cv2.MORPH_OPEN,kernel)    
    return img2
def extlamb(lambu):  #需要修改，求最大联通区域比较好
    binu = 255 - gray2bin_lamb(lambu)  #背景是黑，前景是白，轮廓处理有效
    
    kernel=np.ones((5,5),np.uint8)
    binu =cv2.morphologyEx(binu,cv2.MORPH_OPEN,kernel) #背景是白，前景是黑
#    cv2.imshow('panelimg',binu)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows() 
   
    con,h1 = cv2.findContours(binu, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    Num = len(con)
    if Num >1:
        area = np.zeros(Num,dtype = int)
        for k1 in range(Num):
        
            area[k1] = cv2.contourArea(con[1])
        idx = np.argmax(area)
        x,y,w,h = cv2.boundingRect(con[idx])
    elif Num == 1 :
        x,y,w,h = cv2.boundingRect(con[0])
    else: 
        print('检测小灯出错了。。。。。')
#    print(x,y,y+h,x+w)
    return y,x,y+h,x+w 
'''
def other(img,seq,meter,all_coor):  #seq表示目前检测的是第几个状态，从1-6，meter没有用
    #time.sleep(3)   #为了使4张图像的结果有序显示，后面3张图像的操作需要延时
    global s1_1        
    global s1_2
    global s1_3
    # global error

#    all_coor = np.load('all_coor_v1.npy')  #需要检测要素的坐标
    so_1 =  np.zeros(6)   #存放其他状态小灯的状态值
    so_2 =  np.zeros(6)
    so_3 =  np.zeros(6)
    up_1,up_2,up_3,down_1,down_2,down_3 = one2six(img)
    images=[up_1,up_2,up_3,down_1,down_2,down_3] 
    for i in range(6):
        
        img = images[i]
#        print('仪表'+str(i)+'检测结果-----------------:')
        y1,x1,y2,x2 = all_coor[i][0:4]
        up = img[y1:y2,x1:x2]
        # cv2.imshow('panelimg',up)
         # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        binup = gray2bin(up)
        
        leftu = binup[:,0:all_coor[i][4]]
        midu =  binup[:,all_coor[i][5]:all_coor[i][6]]
        rightu =  binup[:,all_coor[i][7]:x2]
        
        a = recog_code(leftu)  #得到字符 a
        b = recog_code(midu)
        c = recog_code(rightu)
        output = []
        output = a
        output  += b
        output += c
        print(a,b,c)
        
        y1,x1,y2,x2 = all_coor[i][8:12]
        dr = img[y1:y2,x1:x2]
        bindr = gray2bin(dr)
        leftd = bindr[:,0:all_coor[i][12]]
        midd=  bindr[:,all_coor[i][13]:all_coor[i][14]]
        rightd =  bindr[:,all_coor[i][15]:x2]
        a = recog_code(leftd)  #得到字符量 a
        b = recog_code(midd)
        c = recog_code(rightd)
        output = []
        output = a
        output  += b
        output += c
        print(a,b,c)
        
        
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
                
        pos_val = names[meter['meter' + str(i + 1)][(seq-1)*3]] #names将字符串转变成变量名
        pos_val.set(output)   
            
        
        # 设置输出数码管
        
        pos_val = names[meter['meter' + str(i + 1)][(seq-2)*3+1]]#names将字符串转变成变量名
        pos_val = names[meter['meter' + str(i + 1)][(seq-1)*3+1]]#names将字符串转变成变量名    
        pos_val.set(output)   
        
        if (seq == 1):
            
            binlu1 = gray2bin_lamb(lu1)
            
            binlu2 = gray2bin_lamb(lu2)
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
            
            
            binld1 = gray2bin_lamb(ld1)            
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

            print ('小灯状态=',lamb1,lamb2,lamb3)

            
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
            # pos_val = names[meter['meter' + str(i + 1)][(seq-1)*3+2]]#names将字符串转变成变量名
            # pos_val.set(output)   
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
            print('小灯output=',output)
            # pos_val = names[meter['meter' + str(i + 1)][(seq-1)*3+2]]#names将字符串转变成变量名
            # pos_val.set(output)  
'''
def ext_all_coor(img):  #输入6个仪表的灰度图像
     #提取上排数码管，下排数码管，小灯的坐标，要根据仪表布局的尺寸。 
    all_coor = np.zeros((6,36),dtype = int)  #2排数字与2排小灯的绝对坐标
    up_1,up_2,up_3,down_1,down_2,down_3 = one2six(img)  #将6个仪表分成6个小图，每个小图一个仪表
    images=[up_1,up_2,up_3,down_1,down_2,down_3] 

    for i in range(6):
        
        or_img = images[i]
        # or_img[317,:] = 255
        # cv2.imshow('paned',or_img)
        # cv2.waitKey(0) 
        # cv2.destroyAllWindows()  
        # break
        upy1,upx1,upy2,upx2 ,idx =extupcoor(or_img)
        binup = or_img[upy1:upy2,upx1:upx2]
        
        all_coor[i][0] = upy1        #存储上排3个8的坐标
        all_coor[i][1] = upx1
        all_coor[i][2] = upy2
        all_coor[i][3] = upx2
        all_coor[i][4:8] = idx[0:4]
        
        H1 = upy2 -upy1
        W1 = upx2 - upx1 
        dy2 = upy2 + H1 #upy2 - upy1         #下排数码的坐标，大致范围，不是顶天立地
        dy1 = upy2    + 5
        dx1 = upx1    + 60
        dx2 = upx2   + 8
#        bind = or_img[dy1:dy2,dx1:dx2]
#        cv2.imshow('paned',bind)
#        cv2.waitKey(0) 
#        cv2.destroyAllWindows()  
        
        or_imgd = or_img[dy1:dy2,dx1:dx2]
        # cv2.imshow('down',or_imgd)
        # cv2.waitKey(0) 
        # cv2.destroyAllWindows()
        y1,x1,y2,x2 ,idx= extdowncoor(or_imgd)
        ady1 = dy1 + y1 -1         #下排数码的绝对坐标，顶天立地的坐标
        ady2 = dy1 + y2 +1
        adx1 = dx1 + x1 -1
        adx2 = dx1 + x2 +1
        all_coor[i][8] = ady1
        all_coor[i][9] = adx1
        all_coor[i][10] = ady2
        all_coor[i][11] = adx2
        all_coor[i][12:16] = idx[0:4]
        bind= or_img[ady1:ady2,adx1:adx2]  #
#        bind = np.uint8(bind)    
        # cv2.imshow('down1',bind)
        # cv2.waitKey(0) 
        # cv2.destroyAllWindows()   
        
        luy2 = ady1 + int(0.35*(y2 - y1))
        luy1 = ady1 + int(0.05*(y2 - y1))
        lux2 = adx1 - int((6/20)*(x2- x1))
        lux1 = adx1 - int((17/20)*(x2- x1))
        all_coor[i][16] = luy1  #ady1
        all_coor[i][17] = lux1   #adx1
        all_coor[i][18] = luy2  #ady2
        all_coor[i][19] = lux2  #adx2
        lu = or_img[luy1:luy2,lux1:lux2]  #上排小灯
        _,W = lu.shape
        
        lu1 = lu[:,0:W//2]
        ly1,lx1,ly2,lx2 = extlamb(lu1) 
        all_coor[i][20] = ly1
        all_coor[i][21] = lx1
        all_coor[i][22] = ly2
        all_coor[i][23] = lx2
        lu2 = lu[:,W//2:W]
        ly1,lx1,ly2,lx2 = extlamb(lu2) 
        all_coor[i][24] = ly1
        all_coor[i][25] = lx1
        all_coor[i][26] = ly2
        all_coor[i][27] = lx2
        binlu2 = gray2bin_lamb(lu2)
        # cv2.imshow('lu1',binlu2)
        # cv2.waitKey(0) 
        # cv2.destroyAllWindows()
        ldx1 = lux1                         #开始计算下排小灯坐标
        ldx2 = lux2
        ldy2 = ady1 + int(0.87*(y2 - y1))
        ldy1 = ady1 + int(0.51*(y2 - y1))
        all_coor[i][28] = ldy1 #ady1
        all_coor[i][29] = ldx1 #adx1
        all_coor[i][30] = ldy2 #ady2
        all_coor[i][31] = ldx2 #adx2
        ld = or_img[ldy1:ldy2,ldx1:ldx2] #下排小灯
        H,W = ld.shape
        
        ld1 = ld[:,0:W//2]
        # cv2.imshow('lu2',ld1)
        # cv2.waitKey(0) 
        # cv2.destroyAllWindows()
        ldy1,ldx1,ldy2,ldx2 = extlamb(ld1) 
       
        all_coor[i][32] = ldy1
        all_coor[i][33] = ldx1
        all_coor[i][34] = ldy2
        all_coor[i][35] = ldx2 
        binld= gray2bin_lamb(ld)
    return all_coor        
        
        
        