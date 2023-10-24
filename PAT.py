import numpy as np
import cv2
import torch
import torch.nn as nn
import time
from PIL import Image,ImageTk
import tkinter as TK
from tkinter import filedialog
# import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
import pandas as pd
# import sysconfig



##define the network##

class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3,stride=1,padding=(1,1))
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x

class down(nn.Module):
    def __init__(self):
        super(down, self).__init__()
        self.down = nn.MaxPool2d(2,2)
        
    def forward(self, x):
        x = self.down(x)
        
        return x

class up(nn.Module): 
    def __init__(self):
        super(up, self).__init__()       
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x1, x2): 
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        x = torch.cat((x2,x1),1)
        
        return x
        
class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3,stride=1,padding=(1,1))
        self.sig=nn.Sigmoid()

    def forward(self, x):
        x = self.sig(self.conv(x))
        return x
class Diceloss(nn.Module):
    def __init__(self):
        super(Diceloss,self).__init__()
    
    def forward(self,predict,label):
        loss=torch.sum(predict*label)+1
        loss=1-loss/(torch.sum(predict)+torch.sum(label)-loss+2)
        return loss

class MyNet_4(nn.Module):
    def __init__(self,thickness):
        super(MyNet_4, self).__init__()
        self.conv1=single_conv(3,thickness)
        self.conv2=single_conv(thickness,2*thickness)
        self.conv3=single_conv(2*thickness,4*thickness)
        self.conv4=single_conv(4*thickness,thickness)
        self.conv5=single_conv(2*thickness,thickness)
        self.conv6=single_conv(4*thickness,4*thickness)
        self.conv7=single_conv(8*thickness,2*thickness)
        self.down=down()
        self.up=up()
        self.out=outconv(thickness,1)
        

    def forward(self, x):
            x1=self.conv1(x)
            x2=self.down(x1)
            x2=self.conv2(x2)
            x3=self.down(x2)
            x3=self.conv3(x3)
            x4=self.down(x3)
            x4=self.conv6(x4)
            x3=self.up(x4,x3)
            del x4
            x3=self.conv7(x3)
            x2=self.up(x3,x2)
            del x3
            x2=self.conv4(x2)
            x1=self.up(x2,x1)
            del x2
            x1=self.conv5(x1)
            x1=self.out(x1).squeeze(1)

            return x1



##get the coordinate of the apex from the mask, the output of the network##

def get_coordinate(mask,Thre=30,select_mode='mean'):
    height,width=mask.shape
    x=np.repeat(np.arange(1,width+1)[None,:],height,axis=0)*mask
    y=np.repeat(np.arange(1,height+1)[:,None],width,axis=1)*mask
    x=x[x!=0]
    y=y[y!=0]
    mask_new=(np.abs(x-np.median(x))<Thre)*(np.abs(y-np.median(y))<Thre)
    x=x[mask_new]
    y=y[mask_new]
    if select_mode=='median':
        coordinate=[np.round(np.median(x)).astype(int)-1,np.round(np.median(y)).astype(int)-1]
    else:
        coordinate=[np.round(np.mean(x)).astype(int)-1,np.round(np.mean(y)).astype(int)-1]
    
    return coordinate



##locate the search range2. ingore the pixel out of the search range2##

def locate_search_range(mask_,search_range,coordinate):
    search_center=[coordinate[0],coordinate[1]]
    search_mask=np.zeros(mask_.shape).astype(int)
    search_mask[max(search_center[1]-search_range,0):min(search_center[1]+search_range+1,mask_.shape[0]),
                    max(search_center[0]-search_range,0):min(search_center[0]+search_range+1,mask_.shape[1])]=1
    mask_=mask_*search_mask
    return mask_



##color filter to remove pixels of non-plant##

def color_filter(image):
    eb=0.001
    mask_g=(image[:,:,1]>50)*((image[:,:,1]/(image[:,:,0]+eb))>1.1)*((image[:,:,1]/(image[:,:,2]+eb))>1.1)
    mask_w=((image[:,:,1]>180)*((image[:,:,1]/(image[:,:,0]+eb))>0.9)*((image[:,:,0]/(image[:,:,1]+eb))>0.9)
    *((image[:,:,1]/(image[:,:,2]+eb))>0.9)*((image[:,:,2]/(image[:,:,1]+eb))>0.9))
    for i in [0,1,2]:
        image[:,:,i]=image[:,:,i]*(mask_g+mask_w)
    return image


## determine the search_range1##

def deside_search_range():
    test_pic=torch.zeros(1,3,401,401).to(device)
    t=time.time()
    MyModel.forward(test_pic)
    t=time.time()-t
    if t<0.04:
        return 200
    elif t>0.17:
        return 50
    else:
        return 100
    return


## open the video to analyse and initialize some parameters##

def Select_File():
    if tracking:
        return
    elif enable_selection or enable_selection2 or enable_selection3 or enable_selection4:
        return
    global file_path,label1,selected_video,cap,height,width,image,canvas,displayed_frame,fps_of_video,coordinates,rval,frame
    global MyModel,search_range,search_range2,Threshold_pick_piexls,Threshold_outliners,pick_range,confidence_threshold,learnning_rate
    global scales,label2,frame_interval,m_standard,update_decay,time_threshold
    file_path = filedialog.askopenfilename(title='Select video')
    cap = cv2.VideoCapture(file_path)
    height,width=int(cap.get(4)),int(cap.get(3))
    rval, frame = cap.read()
    # resize the frame here -- YIXIANG

    if rval==True:
        if Enable_color_filter.get():
            image=ImageTk.PhotoImage(Image.fromarray(color_filter(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))))
        else:
            image=ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
        displayed_frame=canvas.create_image(width//2, height//2, image=image)
        selected_video=True
        label1.config(text=file_path)
        canvas.config(width=width, height=height)
        # m_standard=torch.load('MyModel_%d_%d_epoch3_^.pth'%(level,thickness))
        m_standard=torch.load(model_save_path+'MyModel_%d_%d_epoch3_^.pth'%(level,thickness))
        MyModel.load_state_dict(m_standard)
        fps_of_video=[]
        coordinates=[]
        search_range=deside_search_range()
        search_range2=40
        Threshold_pick_piexls=0.75
        pick_range=15
        Threshold_outliners=pick_range 
        confidence_threshold=0.95
        learnning_rate=0.14
        update_decay=0.1
        time_threshold=4
        scales=None
        frame_interval=None
        label2.grid_forget()
        label3.grid_forget()



## start to track the apex##

def start_track():
    global tracking,fps_of_video,coordinates,selection_finished,coordinate,image,rval
    global frame,displayed_frame,num,win,lastDraw,stop_tracking,selected_apex,mask,mask_
    global m_standard,update_decay,time_threshold
    if tracking or not selected_video:
        return
    elif enable_selection or enable_selection2 or enable_selection3 or enable_selection4:
        return
    tracking=True
    button4.grid_forget()
    button5.grid(row=1,column=2,sticky=TK.W,padx=20,pady=10)
    num=1
    while cap.isOpened():
        # resize the frame here -- YIXIANG
        
        if rval==True:
            frame_=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            if Enable_color_filter.get():
                frame_=color_filter(frame_)
            image=ImageTk.PhotoImage(Image.fromarray(frame_))
            # print(frame_.shape, frame_.transpose(2,0,1).shape, frame_.(2,0,1)[None,:,:,:].shape)
            frame_ = torch.tensor(frame_.transpose(2,0,1)[None,:,:,:])/255.0
            if num==1:
                frame_=frame_.to(device)
            else:
                frame_=frame_[:,:,max(coordinates[-1][1]-search_range,0):min(coordinates[-1][1]+search_range+1,height),
                                max(coordinates[-1][0]-search_range,0):min(coordinates[-1][0]+search_range+1,width)].to(device)
            with torch.no_grad():
                mask=MyModel.forward(frame_)[0]
                mask_=np.array(mask.cpu()>Threshold_pick_piexls)
                if len(coordinates)>0:
                    if num==1:
                        mask_=locate_search_range(mask_,search_range2,coordinate)
                    else:
                        mask_=locate_search_range(mask_,search_range2,
                                                    [coordinate[0]-max(coordinates[-1][0]-search_range,0),
                                                    coordinate[1]-max(coordinates[-1][1]-search_range,0)])
            if not selected_apex:
                coordinate=get_coordinate(mask_,Threshold_outliners)
            else:
                mask_=np.zeros((height,width)).astype(int)
            if coordinate[0]<0 or coordinate[1]<0 or coordinate[0]>width or coordinate[1]>height :#process if fail to detect the apex
                if len(coordinates)>0:
                    if num==1:
                        coordinate=coordinates[-1]
                    else:
                        coordinate=[coordinates[-1][0]-max(coordinates[-1][0]-search_range,0),
                                    coordinates[-1][1]-max(coordinates[-1][1]-search_range,0)]
            else:
                block=np.zeros(mask_.shape).astype(int)
                block[max(coordinate[1]-pick_range,0):min(coordinate[1]+pick_range+1,height),
                      max(coordinate[0]-pick_range,0):min(coordinate[0]+pick_range+1,width)]=1
                mask_updater=torch.tensor(block).float().to(device)
                Loss=Diceloss()
                with torch.no_grad():
                    confidence=Loss.forward(mask,mask_updater)
                if confidence>confidence_threshold:
                    print(confidence)
                    Optimizer=torch.optim.SGD(MyModel.parameters(),learnning_rate,momentum=0.9,weight_decay=0.0005)
                    t_start=time.time()
                    while (1):
                        if time.time()-t_start>time_threshold:
                            break
                        predict=MyModel.forward(frame_)[0]
                        loss=Loss(predict,mask_updater)
                        confidence=loss.item()
                        if  confidence<=confidence_threshold*0.6:
                            break
                        Optimizer.zero_grad()
                        loss.backward()
                        Optimizer.step()
                        m_now=MyModel.state_dict()
                        for key in m_now:
                            m_now[key]=(1-update_decay)*m_standard[key].to(device)+update_decay*m_now[key]
                        MyModel.load_state_dict(m_now)
                    m_standard=m_now
            t = time.time()
            if num!=1:
                coordinate=[coordinate[0]+max(coordinates[-1][0]-search_range,0),coordinate[1]+max(coordinates[-1][1]-search_range,0)]
                fps=1/(t-tp)
                fps_of_video.append(fps)
                canvas.delete(displayed_frame)
                canvas.delete(lastDraw)
                displayed_frame=canvas.create_image(width//2, height//2, image=image)
            if stop_tracking:
                tracking=False
                stop_tracking=False
                break
            if selected_apex:
                selected_apex=False
                canvas.delete(lastDraw)
            tp=t
            lastDraw = canvas.create_rectangle(max(coordinate[0]-pick_range,0), max(coordinate[1]-pick_range,0)
                                    ,min(coordinate[0]+pick_range,width), min(coordinate[1]+pick_range,height), outline='red')
            win.update_idletasks()
            win.update()
            coordinates.append(coordinate)
            num+=1
        else:
            cap.release()
            break
        rval, frame = cap.read()
    tracking=False
    button5.grid_forget()
    button4.grid(row=1,column=2,sticky=TK.W,padx=20,pady=10)

## stop tracking##
def stop_track():
    global stop_tracking
    if tracking:
        stop_tracking=True
        button5.grid_forget()
        button4.grid(row=1,column=2,sticky=TK.W,padx=20,pady=10)
    else:
        return




## These functions is used to draw a box in the canvas to select the apex ##


def Select_apex1():
    if tracking or not selected_video:
        return
    elif enable_selection2 or enable_selection3 or enable_selection4:
        return
    global enable_selection
    enable_selection=True
    button2.grid_forget()
    button3.grid(row=1,column=1,sticky=TK.W,padx=20,pady=10)
    
def Select_apex2():
    if tracking or not selected_video:
        return
    elif enable_selection2 or enable_selection3 or enable_selection4:
        return
    global enable_selection,coordinate,selected_apex,selection_finished,pick_range
    enable_selection=False
    button3.grid_forget()
    button2.grid(row=1,column=1,sticky=TK.W,padx=20,pady=10)
    if selection_finished:
        coordinate=[(selected_area[0]+selected_area[1])//2,(selected_area[2]+selected_area[3])//2]
        if selected_area[1]>selected_area[0] and selected_area[3]>selected_area[2]:
            pick_range=(selected_area[1]-selected_area[0]+selected_area[3]-selected_area[2])//4
            Threshold_outliners=pick_range
        selected_apex=True
        selection_finished=False



## These functions is used to draw a box in the canvas to decide the search range 1 ##

def search_range_1_1():
    if tracking or not selected_video:
        return
    elif enable_selection or enable_selection3 or enable_selection4:
        return
    global enable_selection2
    enable_selection2=True
    button6.grid_forget()
    button7.grid(row=1,column=1,sticky=TK.W,padx=20,pady=10)
    
def search_range_1_2():
    if tracking or not selected_video:
        return
    elif enable_selection or enable_selection3 or enable_selection4:
        return
    global enable_selection2,selection_finished,search_range
    enable_selection2=False
    button7.grid_forget()
    button6.grid(row=1,column=1,sticky=TK.W,padx=20,pady=10)
    if selection_finished:
        if selected_area[1]>selected_area[0] and selected_area[3]>selected_area[2]:
            search_range=(selected_area[1]-selected_area[0]+selected_area[3]-selected_area[2])//4
        canvas.delete(lastDraw)
        selection_finished=False




## These functions is used to draw a box in the canvas to decide the search range 2 ##

def search_range_2_1():
    if tracking or not selected_video:
        return
    elif enable_selection or enable_selection2 or enable_selection4:
        return
    global enable_selection3
    enable_selection3=True
    button8.grid_forget()
    button9.grid(row=2,column=1,sticky=TK.W,padx=20,pady=10)
    
def search_range_2_2():
    if tracking or not selected_video:
        return
    elif enable_selection or enable_selection2 or enable_selection4:
        return
    global enable_selection3,selection_finished,search_range2
    enable_selection3=False
    button9.grid_forget()
    button8.grid(row=2,column=1,sticky=TK.W,padx=20,pady=10)
    if selection_finished:
        if selected_area[1]>selected_area[0] and selected_area[3]>selected_area[2]:
            search_range2=(selected_area[1]-selected_area[0]+selected_area[3]-selected_area[2])//4
        canvas.delete(lastDraw)
        selection_finished=False


## These functions is used to map the scales in frames to in the real world ##

def Scale_1():
    if tracking or not selected_video:
        return
    elif enable_selection or enable_selection2 or enable_selection3:
        return
    global enable_selection4
    enable_selection4=True
    button10.grid_forget()
    button11.grid(row=1,column=1,sticky=TK.W,padx=20,pady=10)
    entry1.delete(0,"end")
    entry1.grid(row=2,column=1,sticky=TK.W,padx=20)
    label2.grid_forget()   
def Scale_2():
    if tracking or not selected_video:
        return
    elif enable_selection or enable_selection2 or enable_selection3:
        return
    global enable_selection4,selection_finished,scales
    enable_selection4=False
    button11.grid_forget()
    entry1.grid_forget()
    button10.grid(row=1,column=1,sticky=TK.W,padx=20,pady=10)
    if selection_finished:
        try:
            scales=float(entry1.get())/np.sqrt((selected_area[1]-selected_area[0])**2+(selected_area[3]-selected_area[2])**2)           
        except:
            pass 
        canvas.delete(lastDraw)
        selection_finished=False
    if scales!=None:
        label2.config(text="Scales = %.3f mm/pixel"%(scales))
        label2.grid(row=2,column=1,sticky=TK.W,padx=20)


## These functions is used to map the frame interval to the time in the real world ##

def Frame_interval_1():
    if tracking or not selected_video:
        return
    elif enable_selection or enable_selection2 or enable_selection3 or enable_selection4:
        return
    button12.grid_forget()
    button13.grid(row=1,column=1,sticky=TK.W,padx=20,pady=10)
    entry2.delete(0,"end")
    entry2.grid(row=2,column=1,sticky=TK.W,padx=20)
    label3.grid_forget()   
def Frame_interval_2():
    if tracking or not selected_video:
        return
    elif enable_selection or enable_selection2 or enable_selection3 or enable_selection4:
        return
    global frame_interval
    button13.grid_forget()
    entry2.grid_forget()
    button12.grid(row=1,column=1,sticky=TK.W,padx=20,pady=10)
    try:
        frame_interval=float(entry2.get())         
    except:
        pass 
    if frame_interval!=None:
        label3.config(text="Frame interval = %.1f s/frame"%(frame_interval))
        label3.grid(row=2,column=1,sticky=TK.W,padx=20)



## These functions help some functions above draw the figure in the canvas ##

def onLeftButtonDown(event):
    if tracking:
        return
    elif not enable_selection and not enable_selection2 and not enable_selection3 and not enable_selection4:
        return
    global s_X,s_Y,selecting
    s_X = TK.IntVar(value=0)
    s_Y = TK.IntVar(value=0)
    s_X.set(event.x)
    s_Y.set(event.y)
    selecting = True
def onLeftButtonMove(event):
    global lastDraw
    if (not selecting) or tracking:
        return
    if enable_selection:
        color='red'
    elif enable_selection2:
        color='yellow'
    elif enable_selection3:
        color='green'
    elif enable_selection4:
        color='red'
    else:
        return
    try:
        canvas.delete(lastDraw)
    except:
        pass
    if enable_selection4:
        lastDraw = canvas.create_line(s_X.get(), s_Y.get(), event.x, event.y, fill=color)
    else:
        lastDraw = canvas.create_rectangle(s_X.get(), s_Y.get(), event.x, event.y, outline=color)
def onLeftButtonUp(event):
    global selecting,selected_area,enable_selection,selection_finished
    if tracking:
        return
    elif not enable_selection and not enable_selection2 and not enable_selection3 and not enable_selection4:
        return
    selecting = False
    selection_finished= True
    myleft, myright = sorted([s_X.get(), event.x])
    mytop, mybottom = sorted([s_Y.get(), event.y])
    selected_area=(myleft,myright,mytop,mybottom)



##plot the graphs of the result ##

def create_fig(fig):
    global graph_canvas,graph_toolbar
    graph_canvas = FigureCanvasTkAgg(fig, master=graph)
    graph_toolbar = NavigationToolbar2Tk(graph_canvas, graph)
    graph_toolbar.update()
    graph_canvas.get_tk_widget().pack(side=TK.TOP, fill=TK.BOTH, expand=1)
    graph_toolbar.pack(side=TK.BOTTOM, fill=TK.X)
    return
    
def X_vs_T():
    global graph_canvas,graph_toolbar
    try:
        graph_canvas.get_tk_widget().destroy()
        graph_toolbar.destroy()
    except:
        pass
    create_fig(fig1)
    return

def Y_vs_T():
    global graph_canvas,graph_toolbar
    try:
        graph_canvas.get_tk_widget().destroy()
        graph_toolbar.destroy()
    except:
        pass
    create_fig(fig2)
    return

def Y_vs_X():
    global graph_canvas,graph_toolbar
    try:
        graph_canvas.get_tk_widget().destroy()
        graph_toolbar.destroy()
    except:
        pass
    create_fig(fig3)
    return



##plot the graphs of the result ##

def Plot_the_graphs():
    if tracking or not selected_video or len(coordinates)==0:
        return
    elif enable_selection or enable_selection2 or enable_selection3 or enable_selection4:
        return
    global graph_canvas,graph_toolbar,graph,fig1,fig2,fig3
    positions=np.array(coordinates)
    positions[:,1]=height-positions[:,1]
    Times=np.arange(0, len(coordinates))
    unit_s="pixel"
    unit_t="frame"
    if scales!=None:
        positions=positions*scales
        unit_s="mm"
    if frame_interval!=None:
        Times=Times*frame_interval
        unit_t="second"
    fig1 = Figure(figsize=(8, 4), dpi=100)
    fig1.add_subplot(111,xlabel="Time (%s)"%unit_t,ylabel="X (%s)"%unit_s).plot(Times, positions[:,0])
    fig2 = Figure(figsize=(8, 4), dpi=100)
    fig2.add_subplot(111,xlabel="Time (%s)"%unit_t,ylabel="Y (%s)"%unit_s).plot(Times, positions[:,1])
    fig3 = Figure(figsize=(6, 6), dpi=100)
    fig3.add_subplot(111,xlabel="X (%s)"%unit_s,ylabel="Y (%s)"%unit_s).plot(positions[:,0], positions[:,1],'o')
    graph = TK.Toplevel(win)
    graph.title("The movement graph of the apex")
    graph_boxframe1 = TK.Frame(graph)
    graph_button1 = TK.Button(graph_boxframe1 ,text="X vs T",command=X_vs_T)
    graph_button2 = TK.Button(graph_boxframe1 ,text="Y vs T",command=Y_vs_T)
    graph_button3 = TK.Button(graph_boxframe1 ,text="Y vs X",command=Y_vs_X)
    graph_button1.grid(row=1,column=1)
    graph_button2.grid(row=1,column=2)
    graph_button3.grid(row=1,column=3)
    graph_boxframe1.pack(side=TK.TOP, fill=TK.X)
    graph.grab_set()
    return



##save the result of tracking##

def Save_the_result():
    if tracking or not selected_video or len(coordinates)==0:
        return
    elif enable_selection or enable_selection2 or enable_selection3 or enable_selection4:
        return
    global graph_canvas,graph_toolbar,graph
    positions=np.array(coordinates)
    positions[:,1]=height-positions[:,1]
    Times=np.arange(0, len(coordinates))
    unit_s="pixel"
    unit_t="frame"
    if scales!=None:
        positions=positions*scales
        unit_s="mm"
    if frame_interval!=None:
        Times=Times*frame_interval
        unit_t="second"
    data=np.hstack((Times[:,None],positions))
    csv_file=pd.DataFrame(data,columns=["Time (%s)"%unit_t,"X (%s)"%unit_s,"Y (%s)"%unit_s])
    save_path=filedialog.asksaveasfilename(title='Save the result',filetypes=[('CSV', '*.csv')])
    csv_file.to_csv(save_path, sep=",",index=False)
    return



#create the network and load parameters#

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_save_path='/Users/Shared/'  #'/Users/yixiangmao/Documents/PlantTracerProject/Hejian/src/model/' './model/'
thickness=32
level=4
try:
    MyModel=MyNet_4(32).to(device)
except:
    device = torch.device("cpu")
    MyModel=MyNet_4(32).to(device)
MyModel.load_state_dict(torch.load(model_save_path+'MyModel_%d_%d_epoch3_^.pth'%(level,thickness))) 
# MyModel.load_state_dict(torch.load('MyModel_%d_%d_epoch3_^.pth'%(level,thickness))) 




#create the UI#

win=TK.Tk()        
win.title('Plant Apex Track')
fps_of_video=[]
coordinates=[]
search_range=100
search_range2=40
Threshold_pick_piexls=0.75
pick_range=15
Threshold_outliners=pick_range 
confidence_threshold=0.95
learnning_rate=0.14
update_decay=0.1
time_threshold=4
scales=None
frame_interval=None
selected_video=False
enable_selection=False
enable_selection2=False
enable_selection3=False
enable_selection4=False
selected_apex=False
selection_finished=False
selecting=False
tracking=False
stop_tracking=False
Enable_color_filter=TK.IntVar()
file_path="Please select a video to track"
boxframe1 = TK.Frame(win, relief="sunken")
boxframe2 = TK.Frame(win, relief="sunken",borderwidth=1)
boxframe3 = TK.Frame(win, relief="sunken",borderwidth=1)
boxframe4 = TK.Frame(win, relief="sunken",borderwidth=1)
boxframe5 = TK.Frame(boxframe3)
boxframe6 = TK.Frame(boxframe3)
label1 = TK.Label(boxframe1, text = file_path)
label2= TK.Label(boxframe5)
label3= TK.Label(boxframe6)
button1 = TK.Button(boxframe1 ,text="Select video",command=Select_File)
button2 = TK.Button(boxframe4 ,text="Select apex",command=Select_apex1)
button3 = TK.Button(boxframe4 ,text="Confirm",command=Select_apex2)
button4 = TK.Button(boxframe4 ,text="Track",command=start_track)
button5=TK.Button(boxframe4 ,text="Stop",command=stop_track)
button6=TK.Button(boxframe3 ,text="Search range 1",command=search_range_1_1)
button7=TK.Button(boxframe3 ,text="Confirm",command=search_range_1_2)
button8=TK.Button(boxframe3 ,text="Search range 2",command=search_range_2_1)
button9=TK.Button(boxframe3 ,text="Confirm",command=search_range_2_2)
button10=TK.Button(boxframe5 ,text="Scale(draw a line)",command=Scale_1)
button11=TK.Button(boxframe5 ,text="Enter the length (mm)",command=Scale_2)
button12=TK.Button(boxframe6 ,text="Frame interval",command=Frame_interval_1)
button13=TK.Button(boxframe6 ,text="Enter the interval (second)",command=Frame_interval_2)
button14=TK.Button(boxframe4 ,text="Plot the graphs",command=Plot_the_graphs)
button15=TK.Button(boxframe4 ,text="Save the result",command=Save_the_result)
Checkbutton1 = TK.Checkbutton(boxframe3, text='Enable color filter', variable=Enable_color_filter, onvalue=1, offvalue=0,)
entry1=TK.Entry(boxframe5,width=14)
entry2=TK.Entry(boxframe6,width=14)
canvas = TK.Canvas(boxframe2)
canvas.bind('<Button-1>', onLeftButtonDown)
canvas.bind('<B1-Motion>', onLeftButtonMove)
canvas.bind('<ButtonRelease-1>', onLeftButtonUp)
win.grid_columnconfigure(0,weight=0)
win.grid_rowconfigure(0,weight=0)
canvas.grid(row=1,column=1,padx=20,pady=10)
label1.grid(row=1,column=2,sticky=TK.W,padx=20,pady=10)
button1.grid(row=1,column=1,sticky=TK.W,padx=20,pady=10)
button2.grid(row=1,column=1,sticky=TK.W,padx=20,pady=10)
button4.grid(row=1,column=2,sticky=TK.W,padx=20,pady=10)
button6.grid(row=1,column=1,sticky=TK.W,padx=20,pady=10)
button8.grid(row=2,column=1,sticky=TK.W,padx=20,pady=10)
button10.grid(row=1,column=1,sticky=TK.W,padx=20,pady=10)
button12.grid(row=1,column=1,sticky=TK.W,padx=20,pady=10)
button14.grid(row=1,column=3,sticky=TK.W,padx=20,pady=10)
button15.grid(row=1,column=4,sticky=TK.W,padx=20,pady=10)
Checkbutton1.grid(row=5,column=1,sticky=TK.W,padx=20,pady=10)
boxframe1.grid(row=1,column=1,columnspan=2,sticky=TK.EW)
boxframe2.grid(row=2,column=1,sticky=TK.W)
boxframe3.grid(row=2,column=2,sticky=TK.NS)
boxframe4.grid(row=3,column=1,columnspan=2,sticky=TK.EW)
boxframe5.grid(row=3,column=1,sticky=TK.EW)
boxframe6.grid(row=4,column=1,sticky=TK.EW)
win.mainloop()