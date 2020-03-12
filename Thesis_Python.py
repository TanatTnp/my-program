from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xlrd import open_workbook
from time import sleep
from myo import init, Hub, Feed, StreamEmg, Arm, VibrationType 
from pyqtgraph.Qt import QtCore, QtGui
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import xlrd
import pyqtgraph as pg
import numpy as np
import math
import time,sys
import xlsxwriter
import xlrd
import serial

win = pg.GraphicsWindow()
win.setWindowTitle('Receive: EMG signals')
win.setBackground('k')

######### create timer ##############
timer = pg.QtCore.QTimer()
timer1 = pg.QtCore.QTimer()
timer2 = pg.QtCore.QTimer()


################ create Gui ######################
proxy1 = QtGui.QGraphicsProxyWidget()
proxy2 = QtGui.QGraphicsProxyWidget()
proxy3 = QtGui.QGraphicsProxyWidget()


################ Group Box  ######################

groupBox1 = QtGui.QGroupBox("Record Traindata")

Label1 = QtGui.QLabel("Filename")
Label2 = QtGui.QLabel("non save")
Label3 = QtGui.QLabel("")
Label4 = QtGui.QLabel("Sampling")
Label5 = QtGui.QLabel("Overlap")
Label6 = QtGui.QLabel("K (the value that determine friction)")
Label7 = QtGui.QLabel("Filename of Kbase")

text1 = QtGui.QLineEdit('')
text2 = QtGui.QLineEdit('')
text3 = QtGui.QLineEdit('')
text4 = QtGui.QLineEdit('')
text5 = QtGui.QLineEdit('')

btn0 = QtGui.QPushButton('start_TrainRecording')
btn1 = QtGui.QPushButton('STOP')
btn2 = QtGui.QPushButton('EXIT')
btn3 = QtGui.QPushButton('makebase for classify K')
btn4 = QtGui.QPushButton('classify K')


groupBox2 = QtGui.QGroupBox("Train Setting")

Label21 = QtGui.QLabel("FileDatabase")
Label22 = QtGui.QLabel("FileDatatrain")
Label23 = QtGui.QLabel("")
Label24 = QtGui.QLabel("Name of new database file")

text21 = QtGui.QLineEdit('')
text22 = QtGui.QLineEdit('')
text23 = QtGui.QLineEdit('')

btn20 = QtGui.QPushButton('Train combine')
btn21 = QtGui.QPushButton('Take Train to Database')
btn22 = QtGui.QPushButton('Train Database')
btn23 = QtGui.QPushButton('Train FiledataTrain')
btn24 = QtGui.QPushButton('build train to new database')


groupBox3 = QtGui.QGroupBox("Test Setting")

Label31 = QtGui.QLabel("Name Test File")
Label32 = QtGui.QLabel("")
Label33 = QtGui.QLabel("")

text31 = QtGui.QLineEdit('')

btn30 = QtGui.QPushButton('Test NN')
btn31 = QtGui.QPushButton('STOP Test')

################ Add layout  ######################

layout1 = QtGui.QFormLayout()
layout1.addRow(Label1)
layout1.addRow(text1)
layout1.addRow(Label4)
layout1.addRow(text2)
layout1.addRow(Label5)
layout1.addRow(text3)
layout1.addRow(btn0)
layout1.addRow(btn1)
layout1.addRow(Label2)
layout1.addRow(Label3)
layout1.addRow(Label7)
layout1.addRow(text5)
layout1.addRow(Label6)
layout1.addRow(text4)
layout1.addRow(Label3)
layout1.addRow(btn3)
layout1.addRow(Label3)
layout1.addRow(btn4)
layout1.addRow(Label3)
layout1.addRow(btn2)

layout2 = QtGui.QFormLayout()
layout2.addRow(Label21)
layout2.addRow(text21)
layout2.addRow(Label22)
layout2.addRow(text22)
layout2.addRow(Label23)
layout2.addRow(btn23)
layout2.addRow(Label23)
layout2.addRow(btn20)
layout2.addRow(Label23)
layout2.addRow(btn21)
layout2.addRow(Label23)
layout2.addRow(btn22)
layout2.addRow(Label23)
layout2.addRow(Label24)
layout2.addRow(Label23)
layout2.addRow(btn24)
layout2.addRow(Label23)
layout2.addRow(text23)

layout3= QtGui.QFormLayout()
layout3.addRow(Label31)
layout3.addRow(text31)
layout3.addRow(Label32)
layout3.addRow(btn30)
layout3.addRow(Label32)
layout3.addRow(btn31)
layout3.addRow(Label32)
layout3.addRow(Label33)

groupBox1.setLayout(layout1)
proxy1.setWidget(groupBox1)
l2 = win.addLayout(row = 2, col=2)
l2.addItem(proxy1, row = 2, col=2)

groupBox2.setLayout(layout2)
proxy2.setWidget(groupBox2)
l3 = win.addLayout(row = 2, col=3)
l3.addItem(proxy2, row = 2, col=3)


groupBox3.setLayout(layout3)
proxy3.setWidget(groupBox3)
l4 = win.addLayout(row = 3, col=3)
l4.addItem(proxy3, row = 3, col=3)


################ Add plot  ######################

p13 = win.addPlot(title="Force", row = 2 , col = 1)
p13.setRange(yRange=[-30, 30])

p23 = win.addPlot(title="TestForce", row = 3 , col = 1)
p23.setRange(yRange=[-30, 30])

p33 = win.addPlot(title="loss", row = 3 , col = 2)
p33.setRange(yRange=[-15, 15])


curve13 = p13.plot(pen='w')
curve23 = p23.plot(pen='w')
curve33 = p23.plot(pen='b')
curve43 = p33.plot(pen='g')
curve53 = p33.plot(pen='m')

data13 = np.zeros(1000)
data23 = np.zeros(1000)
data33 = np.zeros(1000)
data43 = np.zeros(1000)
data53 = np.zeros(1000)

################   init MYO ARMBAND ######################################
init()
feed = Feed()
hub = Hub()
hub.run(1000, feed)
myo = feed.wait_for_single_device(timeout=2.0)
myo.set_stream_emg(StreamEmg.enabled)

print ('init myo')

################   init Arduino ######################################

ser = serial.Serial('COM3', 38400)

print ('init arduino')

################ Another variable  ######################

row = 1
col = 0
count = 0
count2 = 0
stage = 1
sheet=1


emgmeanrate=np.zeros(8)
emgrmsrate=np.zeros(8)
g = "round"
g1 = 9.81
stage = 1


nn = MLPRegressor(
        hidden_layer_sizes=(20,),  activation='identity', solver='lbfgs', alpha=0.001, batch_size='auto',
        learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
        random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
        early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)


numsheet=0


text2.setText(str("150"))
text3.setText(str("1"))

text5.setText(str("K_Database2"))

def start_TrainRecording():
    # classifyk()

    global workbook,bold,sheet,worksheet,emgfile,readfile,overlap,sampling,k,curve13,data13
    global arduinodata,emgdata,data,ForceFSRdata,SUMFSRdata,start_time,t,count

    Label2.setText("saving" )

    curve13.clear()
    curve13 = p13.plot(pen='w')
    data13 = np.zeros(1000)
    curve13.setData(data13) # 1,2,3,...

    name = text1.text()
    sampling = int(text2.text())
    overlap = int(text3.text())

    k = int(text4.text())

    sheet=1

    arduinodata = np.zeros(7)
    emgdata = np.zeros(8)
    data = np.zeros((sampling,8))
    ForceFSRdata = np.zeros((sampling,7))
    SUMFSRdata = np.zeros((sampling,1))
    start_time=np.zeros(1)
    t=np.zeros(1)

    readfile='%s_Traindata' %name
    emgfile = '%s_Traindata.xlsx' %name


    ########## Create a workbook and add a worksheet. ##################

    workbook = xlsxwriter.Workbook(emgfile)
    worksheet = workbook.add_worksheet("round%d"% (sheet))

    ###################Add a bold format to use to highlight cells.#######
    bold = workbook.add_format({'bold': 1})

    #################Write some data headers.################
    worksheet.write('A1', 'Distance', bold)
    worksheet.write('B1', 'Velocity', bold)
    worksheet.write('C1', 'Force', bold)
    worksheet.write('D1', 'Chanel 1', bold)
    worksheet.write('E1', 'Chanel 2', bold)
    worksheet.write('F1', 'Chanel 3', bold)
    worksheet.write('G1', 'Chanel 4', bold)
    worksheet.write('H1', 'Chanel 5', bold)
    worksheet.write('I1', 'Chanel 6', bold)
    worksheet.write('J1', 'Chanel 7', bold)
    worksheet.write('K1', 'Chanel 8', bold)
    worksheet.write('L1', 'FSR1', bold)
    worksheet.write('M1', 'FSR2', bold)
    worksheet.write('N1', 'FSR3', bold)
    worksheet.write('O1', 'FSR4', bold)
    worksheet.write('P1', 'k', bold)


    ################# Start Timer.################
    timer.timeout.connect(savedata)
    timer.start()

def savedata():
    global count

    if count == 0:
        
        global workbook,worksheet,bold,sheet,stage,row,col
        global ForceFSRdata,SUMFSRdata,data,start_time,t,data13,arduinodata,emgdata,told,g1

    ################# Receive value from arduino using serial comunication.################
    # print (stage)

    ser.flush()
    arduinodata = ser.readline()
    arduinodata = arduinodata.split(",")
    arduinodata = arduinodata[1:]

    if stage == 2 :

    	t=time.time() - start_time

        row = 1
        col = 0
        stage = 1
        count = 0

    ################# If lenght of value is right.################

    if len(arduinodata) == 7 and count < sampling:

        stage = 1

        if count == 0 :
            start_time = time.time()
            told = 0

        if hub.running and myo.connected:
            emgdata = myo.emg

        for i in range (0,7):
            ForceFSRdata[count][i] = float(arduinodata[i])
            SUMFSRdata[count][0]=ForceFSRdata[count][0]+ForceFSRdata[count][1]+ForceFSRdata[count][2]+ForceFSRdata[count][3]

        ForceFSRdata[count][4]=ForceFSRdata[count][4]*g1



        if type(emgdata) == tuple:
            for i in range (0,8):
                data[count][i] = emgdata[i]
                data[count][i] = data[count][i]/128


        if float(SUMFSRdata[count][0]) > 3  :

                stage = 2

        t=time.time() - start_time

        ################# Write value from arduino into excel file.################

        worksheet.write  (row, col , ForceFSRdata[count][6] )
        worksheet.write  (row, col + 1, ForceFSRdata[count][5] )
        worksheet.write  (row, col + 2, ForceFSRdata[count][4] )
        worksheet.write  (row, col + 3, data[count][0] )
        worksheet.write  (row, col + 4, data[count][1] )
        worksheet.write  (row, col + 5, data[count][2] )
        worksheet.write  (row, col + 6, data[count][3] )
        worksheet.write  (row, col + 7, data[count][4] )
        worksheet.write  (row, col + 8, data[count][5] )
        worksheet.write  (row, col + 9, data[count][6] )
        worksheet.write  (row, col + 10, data[count][7] )
        worksheet.write  (row, col + 11, ForceFSRdata[count][0] )
        worksheet.write  (row, col + 12, ForceFSRdata[count][1] )
        worksheet.write  (row, col + 13, ForceFSRdata[count][2] )
        worksheet.write  (row, col + 14, ForceFSRdata[count][3] )
        worksheet.write  (row, col + 15, k )

        ################# Set value for plot curve13.################

        data13[:-1] = data13[1:]  # shift data in the array one sample left # (see also: np.roll)
        data13[-1]  = ForceFSRdata[count][4] # degree

        curve13.setData(data13) # 1,2,3,...


        row += 1
        count += 1


    ################# If lenght of value is wrong.################

    if len(arduinodata) != 7 :

        # ################# Set zero for plot curve13.################

        # data13[:-1] = data13[1:]  # shift data in the array one sample left # (see also: np.roll)
        # data13[-1] = 0 # degree

        # curve13.setData(data13)# 1,2,3,...

        ################# If worksheet end add new worksheet.################

        if stage == 1 and count > 1 and row > 300:

            print(t-told)

            worksheet.write('Q1', t-told , bold)

            sheet += 1
            stage = 0

            worksheet = workbook.add_worksheet("round%d"% (sheet))
            print ("round%d"% (sheet))

            ################# Write title for new worksheet.################

            worksheet.write('A1', 'Distance', bold)
            worksheet.write('B1', 'Velocity', bold)
            worksheet.write('C1', 'Force', bold)
            worksheet.write('D1', 'Chanel 1', bold)
            worksheet.write('E1', 'Chanel 2', bold)
            worksheet.write('F1', 'Chanel 3', bold)
            worksheet.write('G1', 'Chanel 4', bold)
            worksheet.write('H1', 'Chanel 5', bold)
            worksheet.write('I1', 'Chanel 6', bold)
            worksheet.write('J1', 'Chanel 7', bold)
            worksheet.write('K1', 'Chanel 8', bold)
            worksheet.write('L1', 'FSR1', bold)
            worksheet.write('M1', 'FSR2', bold)
            worksheet.write('N1', 'FSR3', bold)
            worksheet.write('O1', 'FSR4', bold)
            worksheet.write('P1', 'k', bold)

            row = 1
            col = 0
            count = 0

        else:

            row = 1
            col = 0

        told = t


    ################# Shift array of variable when count equal sampling.################

    if count == sampling:

        count = sampling - overlap

        data = data[overlap:]
        ForceFSRdata = ForceFSRdata[overlap:]
        SUMFSRdata = SUMFSRdata[overlap:]
        
        data = np.pad(data, [(0,overlap),(0,0)], mode='constant', constant_values=0)
        ForceFSRdata = np.pad(ForceFSRdata, [(0,overlap),(0,0)], mode='constant', constant_values=0)
        SUMFSRdata = np.pad(SUMFSRdata, [(0,overlap),(0,0)], mode='constant', constant_values=0)

def stop():
    
    global workbook,worksheet,bold,sheet,stage,count,row,col
    global ForceFSRdata,SUMFSRdata,data,start_time,t,data13,arduinodata,emgdata,numsheet,sampling

    timer.stop()
    Label2.setText("finish" )
    workbook.close()

    sampling = int(text2.text())

    ########### Reset value in variable ##############

    arduinodata = np.zeros(7)
    emgdata = np.zeros(8)
    data = np.zeros((sampling,8))
    ForceFSRdata = np.zeros((sampling,7))
    SUMFSRdata = np.zeros((sampling,1))
    start_time=np.zeros(1)
    t=np.zeros(1)
    row = 1
    col = 0
    count = 0
    count2 = 0
    stage = 1
    sheet=1

    numsheet=0

    CutForce()

    row = 1
    col = 0
    count = 0
    count2 = 0
    stage = 1
    sheet=1
    numsheet=0

    print ("Finish record data")

def CutForce():

    global numsheet,row,col,readfile

    filer = readfile
    filew = readfile

    fileread = '%s.xlsx' %filer
    filewrite = '%s_cutForce.xlsx' %filew

    workbook = xlrd.open_workbook(fileread)
    workbook1 = xlsxwriter.Workbook(filewrite)

    bold = workbook1.add_format({'bold': 1})

    readfile ='%s_cutForce' %filew

    while numsheet <= (workbook.nsheets-2):

        sheet = workbook.sheet_by_index(numsheet)
        worksheet = workbook1.add_worksheet("round%d"% (numsheet+1))

        numsheet=numsheet+1
        row = 1
        col = 0
        print(numsheet)

        deltatime = sheet.cell_value(0,16)

        data=[[sheet.cell_value(r,c) for c in range(sheet.ncols)] for r in range(1,sheet.nrows)]

        worksheet.write('A1', 'Distance', bold)
        worksheet.write('B1', 'Velocity', bold)
        worksheet.write('C1', 'Force', bold)
        worksheet.write('Q1', 'DeltaForce', bold)
        worksheet.write('D1', 'Chanel 1', bold)
        worksheet.write('E1', 'Chanel 2', bold)
        worksheet.write('F1', 'Chanel 3', bold)
        worksheet.write('G1', 'Chanel 4', bold)
        worksheet.write('H1', 'Chanel 5', bold)
        worksheet.write('I1', 'Chanel 6', bold)
        worksheet.write('J1', 'Chanel 7', bold)
        worksheet.write('K1', 'Chanel 8', bold)
        worksheet.write('L1', 'FSR1', bold)
        worksheet.write('M1', 'FSR2', bold)
        worksheet.write('N1', 'FSR3', bold)
        worksheet.write('O1', 'FSR4', bold)
        worksheet.write('P1', 'k', bold)

        meanforce = [data[r][2] for r in range(0,len(data))]
        meanforce=abs(np.mean(meanforce))
        
        for c in xrange(0,(sheet.nrows-1)):
            if c<(sheet.nrows-2) :
                DeltaForce=data[c+1][2]-data[c][2]
                if  data[c][1] <=0.15:
                    worksheet.write  (row, col , data[c][0] )
                    worksheet.write  (row, col + 1, data[c][1] )
                    worksheet.write  (row, col + 2, data[c][2] )
                    worksheet.write  (row, col + 3, data[c][3] )
                    worksheet.write  (row, col + 4, data[c][4] )
                    worksheet.write  (row, col + 5, data[c][5] )
                    worksheet.write  (row, col + 6, data[c][6] )
                    worksheet.write  (row, col + 7, data[c][7] )
                    worksheet.write  (row, col + 8, data[c][8] )
                    worksheet.write  (row, col + 9, data[c][9] )
                    worksheet.write  (row, col + 10, data[c][10] )
                    worksheet.write  (row, col + 11, data[c][11] )
                    worksheet.write  (row, col + 12, data[c][12] )
                    worksheet.write  (row, col + 13, data[c][13] )
                    worksheet.write  (row, col + 14, data[c][14] )
                    worksheet.write  (row, col + 15, data[c][15] )
                    worksheet.write  (row, col + 16, DeltaForce )
                    
                    row += 1

        worksheet.write('R1', deltatime , bold)

        chart1 = workbook1.add_chart({"type" : "line"})
        chart2 = workbook1.add_chart({"type" : "line"})
        chart3 = workbook1.add_chart({"type" : "line"})
        chart4 = workbook1.add_chart({"type" : "line"})
        chart5 = workbook1.add_chart({"type" : "line"})
        chart6 = workbook1.add_chart({"type" : "line"})
        chart7 = workbook1.add_chart({"type" : "line"})
        chart8 = workbook1.add_chart({"type" : "line"})
        chart9 = workbook1.add_chart({"type" : "line"})
        chart10 = workbook1.add_chart({"type" : "line"})
        chart11 = workbook1.add_chart({"type" : "line"})

        chart1.set_x_axis({'name': 'Samples'})
        chart1.set_y_axis({'name': 'EMG'})
        chart2.set_x_axis({'name': 'Samples'})
        chart2.set_y_axis({'name': 'EMG'})
        chart3.set_x_axis({'name': 'Samples'})
        chart3.set_y_axis({'name': 'EMG'})
        chart4.set_x_axis({'name': 'Samples'})
        chart4.set_y_axis({'name': 'EMG'})
        chart5.set_x_axis({'name': 'Samples'})
        chart5.set_y_axis({'name': 'EMG'})
        chart6.set_x_axis({'name': 'Samples'})
        chart6.set_y_axis({'name': 'EMG'})
        chart7.set_x_axis({'name': 'Samples'})
        chart7.set_y_axis({'name': 'EMG'})
        chart8.set_x_axis({'name': 'Samples'})
        chart8.set_y_axis({'name': 'EMG'})
        chart9.set_x_axis({'name': 'Samples'})
        chart9.set_y_axis({'name': 'Force(N)'})
        chart10.set_x_axis({'name': 'Samples'})
        chart10.set_y_axis({'name': 'Velocity(m/s)'})
        chart11.set_x_axis({'name': 'Samples'})
        chart11.set_y_axis({'name': 'Delta(N)'})


        chart1.add_series({"values" : "round%d!$D$1:$D$%d"% (numsheet,row),
                "line" : {"color": "blue"},
                "name" : "Ch1"})
        chart2.add_series({"values" : "round%d!$E$1:$E$%d"% (numsheet,row),
                "line" : {"color": "red"},
                "name" : "Ch2"})
        chart3.add_series({"values" : "round%d!$F$1:$F$%d"% (numsheet,row),
                "line" : {"color": "magenta"},
                "name" : "Ch3"})
        chart4.add_series({"values" : "round%d!$G$1:$G$%d"% (numsheet,row),
                "line" : {"color": "navy"},
                "name" : "Ch4"})
        chart5.add_series({"values" : "round%d!$H$1:$H$%d"% (numsheet,row),
                "line" : {"color": "orange"},
                "name" : "Ch5"})
        chart6.add_series({"values" : "round%d!$I$1:$I$%d"% (numsheet,row),
                "line" : {"color": "lime"},
                "name" : "Ch6"})
        chart7.add_series({"values" : "round%d!$J$1:$J$%d"% (numsheet,row),
                "line" : {"color": "brown"},
                "name" : "Ch7"})
        chart8.add_series({"values" : "round%d!$K$1:$K$%d"% (numsheet,row),
                "line" : {"color": "purple"},
                "name" : "Ch8"})
        chart9.add_series({"values" : "round%d!$C$1:$C$%d"% (numsheet,row),
                "line" : {"color": "black"},
                "name" : "Force"})
        chart10.add_series({"values" : "round%d!$B$1:$B$%d"% (numsheet,row),
                "line" : {"color": "yellow"},
                "name" : "Velocity"})
        chart11.add_series({"values" : "round%d!$Q$1:$Q$%d"% (numsheet,row),
                "line" : {"color": "orange"},
                "name" : "DELTAFORCE"})

        worksheet.insert_chart("R4",chart1)
        worksheet.insert_chart("R19",chart2)
        worksheet.insert_chart("R34",chart3)
        worksheet.insert_chart("R49",chart4)
        worksheet.insert_chart("Z4",chart5)
        worksheet.insert_chart("Z19",chart6)
        worksheet.insert_chart("Z34",chart7)
        worksheet.insert_chart("Z49",chart8)
        worksheet.insert_chart("AH4",chart9)
        worksheet.insert_chart("AH34",chart10)
        worksheet.insert_chart("AH19",chart11)

    workbook1.close()
    numsheet=0

    rmscal()

def rmscal():

    global numsheet,row,col,readfile

    filer = readfile
    filew = readfile

    fileread = '%s.xlsx' %filer
    filewrite = '%s.xlsx' %filew

    workbook = xlrd.open_workbook(fileread)
    workbook1 = xlsxwriter.Workbook(filewrite)

    bold = workbook1.add_format({'bold': 1})

    while numsheet <= (workbook.nsheets-1):

        sheet = workbook.sheet_by_index(numsheet)
        worksheet = workbook1.add_worksheet("round%d"% (numsheet+1))

        numsheet = numsheet+1
        row = 1
        col = 0
        print(numsheet)

        data = [[sheet.cell_value(r,c) for c in range(sheet.ncols)] for r in range(1,sheet.nrows)]

        deltatime = sheet.cell_value(0,17)

        datacal = [[data[r][c] for c in range (3,11)] for r in range(0,len(data))]
        datacal = np.asarray(datacal)
        datacalsquare = np.power(datacal, 2)
        rms = np.sqrt(np.mean(datacalsquare, axis=0))

        worksheet.write('A1', 'Distance', bold)
        worksheet.write('B1', 'Velocity', bold)
        worksheet.write('C1', 'Force', bold)
        worksheet.write('D1', 'Chanel 1', bold)
        worksheet.write('E1', 'Chanel 2', bold)
        worksheet.write('F1', 'Chanel 3', bold)
        worksheet.write('G1', 'Chanel 4', bold)
        worksheet.write('H1', 'Chanel 5', bold)
        worksheet.write('I1', 'Chanel 6', bold)
        worksheet.write('J1', 'Chanel 7', bold)
        worksheet.write('K1', 'Chanel 8', bold)
        worksheet.write('L1', 'FSR1', bold)
        worksheet.write('M1', 'FSR2', bold)
        worksheet.write('N1', 'FSR3', bold)
        worksheet.write('O1', 'FSR4', bold)
        worksheet.write('P1', 'k', bold)
        worksheet.write('Q1', 'DeltaForce', bold)

        worksheet.write('R2', 'RMSch1', bold)
        worksheet.write('R3', 'RMSch2', bold)
        worksheet.write('R4', 'RMSch3', bold)
        worksheet.write('R5', 'RMSch4', bold)
        worksheet.write('R6', 'RMSch5', bold)
        worksheet.write('R7', 'RMSch6', bold)
        worksheet.write('R8', 'RMSch7', bold)
        worksheet.write('R9', 'RMSch8', bold)

        worksheet.write('S2', rms[0], bold)
        worksheet.write('S3', rms[1], bold)
        worksheet.write('S4', rms[2], bold)
        worksheet.write('S5', rms[3], bold)
        worksheet.write('S6', rms[4], bold)
        worksheet.write('S7', rms[5], bold)
        worksheet.write('S8', rms[6], bold)
        worksheet.write('S9', rms[7], bold)
        
        for c in xrange(0,(sheet.nrows-1)):
            worksheet.write  (row, col , data[c][0] )
            worksheet.write  (row, col + 1, data[c][1] )
            worksheet.write  (row, col + 2, data[c][2] )
            worksheet.write  (row, col + 3, data[c][3] )
            worksheet.write  (row, col + 4, data[c][4] )
            worksheet.write  (row, col + 5, data[c][5] )
            worksheet.write  (row, col + 6, data[c][6] )
            worksheet.write  (row, col + 7, data[c][7] )
            worksheet.write  (row, col + 8, data[c][8] )
            worksheet.write  (row, col + 9, data[c][9] )
            worksheet.write  (row, col + 10, data[c][10] )
            worksheet.write  (row, col + 11, data[c][11] )
            worksheet.write  (row, col + 12, data[c][12] )
            worksheet.write  (row, col + 13, data[c][13] )
            worksheet.write  (row, col + 14, data[c][14] )
            worksheet.write  (row, col + 15, data[c][15] )
            worksheet.write  (row, col + 16, data[c][16] )
            
            row += 1


        worksheet.write('R1', deltatime , bold)

        chart1 = workbook1.add_chart({"type" : "line"})
        chart2 = workbook1.add_chart({"type" : "line"})
        chart3 = workbook1.add_chart({"type" : "line"})
        chart4 = workbook1.add_chart({"type" : "line"})
        chart5 = workbook1.add_chart({"type" : "line"})
        chart6 = workbook1.add_chart({"type" : "line"})
        chart7 = workbook1.add_chart({"type" : "line"})
        chart8 = workbook1.add_chart({"type" : "line"})
        chart9 = workbook1.add_chart({"type" : "line"})
        chart10 = workbook1.add_chart({"type" : "line"})
        chart11 = workbook1.add_chart({"type" : "line"})
        chart12 = workbook1.add_chart({"type" : "column"})

        chart1.set_x_axis({'name': 'Samples'})
        chart1.set_y_axis({'name': 'EMG'})
        chart2.set_x_axis({'name': 'Samples'})
        chart2.set_y_axis({'name': 'EMG'})
        chart3.set_x_axis({'name': 'Samples'})
        chart3.set_y_axis({'name': 'EMG'})
        chart4.set_x_axis({'name': 'Samples'})
        chart4.set_y_axis({'name': 'EMG'})
        chart5.set_x_axis({'name': 'Samples'})
        chart5.set_y_axis({'name': 'EMG'})
        chart6.set_x_axis({'name': 'Samples'})
        chart6.set_y_axis({'name': 'EMG'})
        chart7.set_x_axis({'name': 'Samples'})
        chart7.set_y_axis({'name': 'EMG'})
        chart8.set_x_axis({'name': 'Samples'})
        chart8.set_y_axis({'name': 'EMG'})
        chart9.set_x_axis({'name': 'Samples'})
        chart9.set_y_axis({'name': 'Force(N)'})
        chart10.set_x_axis({'name': 'Samples'})
        chart10.set_y_axis({'name': 'Velocity(m/s)'})
        chart11.set_x_axis({'name': 'Samples'})
        chart11.set_y_axis({'name': 'Delta(N)'})
        chart12.set_x_axis({'name': 'Channels'})
        chart12.set_y_axis({'name': 'RMS of EMG'})


        chart1.add_series({"values" : "round%d!$D$1:$D$%d"% (numsheet,row),
                "line" : {"color": "blue"},
                "name" : "Ch1"})
        chart2.add_series({"values" : "round%d!$E$1:$E$%d"% (numsheet,row),
                "line" : {"color": "red"},
                "name" : "Ch2"})
        chart3.add_series({"values" : "round%d!$F$1:$F$%d"% (numsheet,row),
                "line" : {"color": "magenta"},
                "name" : "Ch3"})
        chart4.add_series({"values" : "round%d!$G$1:$G$%d"% (numsheet,row),
                "line" : {"color": "navy"},
                "name" : "Ch4"})
        chart5.add_series({"values" : "round%d!$H$1:$H$%d"% (numsheet,row),
                "line" : {"color": "orange"},
                "name" : "Ch5"})
        chart6.add_series({"values" : "round%d!$I$1:$I$%d"% (numsheet,row),
                "line" : {"color": "lime"},
                "name" : "Ch6"})
        chart7.add_series({"values" : "round%d!$J$1:$J$%d"% (numsheet,row),
                "line" : {"color": "brown"},
                "name" : "Ch7"})
        chart8.add_series({"values" : "round%d!$K$1:$K$%d"% (numsheet,row),
                "line" : {"color": "purple"},
                "name" : "Ch8"})
        chart9.add_series({"values" : "round%d!$C$1:$C$%d"% (numsheet,row),
                "line" : {"color": "black"},
                "name" : "Force"})
        chart10.add_series({"values" : "round%d!$B$1:$B$%d"% (numsheet,row),
                "line" : {"color": "yellow"},
                "name" : "Velocity"})
        chart11.add_series({"values" : "round%d!$Q$1:$Q$%d"% (numsheet,row),
                "line" : {"color": "orange"},
                "name" : "DELTAFORCE"})
        chart12.add_series({"values" : "round%d!$S$2:$S$9"% numsheet,
            "line" : {"color": "orange"},
            "name" : "RMS"})

        worksheet.insert_chart("T4",chart1)
        worksheet.insert_chart("T19",chart2)
        worksheet.insert_chart("T34",chart3)
        worksheet.insert_chart("T49",chart4)
        worksheet.insert_chart("AB4",chart5)
        worksheet.insert_chart("AB19",chart6)
        worksheet.insert_chart("AB34",chart7)
        worksheet.insert_chart("AB49",chart8)
        worksheet.insert_chart("AJ4",chart9)
        worksheet.insert_chart("AJ19",chart11)
        worksheet.insert_chart("AJ34",chart10)
        worksheet.insert_chart("AJ49",chart12)

    workbook1.close()

    numsheet=0

    FeatureCal()

def FeatureCal():

    global numsheet,row,col,readfile,sampling

    rowcal = sampling

    filer = readfile
    filew = readfile

    fileread = '%s.xlsx' %filer
    filewrite = '%s_FeatureS%dO%d.xlsx' %(filew,sampling,overlap)

    readfile ='%s_FeatureS%dO%d' %(filew,sampling,overlap)

    workbook = xlrd.open_workbook(fileread)
    workbook1 = xlsxwriter.Workbook(filewrite)

    bold = workbook1.add_format({'bold': 1})

    while numsheet <= (workbook.nsheets-1):

        sheet = workbook.sheet_by_index(numsheet)
        worksheet = workbook1.add_worksheet("round%d"% (numsheet+1))
        numsheet=numsheet+1

        row = 1
        col = 0
        start=0
        sampling = int(text2.text())

        emgmeanrate=np.zeros(8)
        emgrmsrate=np.zeros(8)
        meanfsr=np.zeros(5)

        data=[[sheet.cell_value(r,c) for c in range(sheet.ncols)] for r in range(1,sheet.nrows)]

        worksheet.write('A1', 'Distance', bold)
        worksheet.write('B1', 'Velocity', bold)
        worksheet.write('C1', 'Force', bold)
        worksheet.write('D1', 'Chanel 1', bold)
        worksheet.write('E1', 'Chanel 2', bold)
        worksheet.write('F1', 'Chanel 3', bold)
        worksheet.write('G1', 'Chanel 4', bold)
        worksheet.write('H1', 'Chanel 5', bold)
        worksheet.write('I1', 'Chanel 6', bold)
        worksheet.write('J1', 'Chanel 7', bold)
        worksheet.write('K1', 'Chanel 8', bold)
        worksheet.write('L1', 'FSR1', bold)
        worksheet.write('M1', 'FSR2', bold)
        worksheet.write('N1', 'FSR3', bold)
        worksheet.write('O1', 'FSR4', bold)
        worksheet.write('P1', 'K', bold)

        # worksheet.write('Q%d'%rowcal, 'ratemean1', bold)
        # worksheet.write('R%d'%rowcal, 'ratemean2', bold)
        # worksheet.write('S%d'%rowcal, 'ratemean3', bold)
        # worksheet.write('T%d'%rowcal, 'ratemean4', bold)
        # worksheet.write('U%d'%rowcal, 'ratemean5', bold)
        # worksheet.write('V%d'%rowcal, 'ratemean6', bold)
        # worksheet.write('W%d'%rowcal, 'ratemean7', bold)
        # worksheet.write('X%d'%rowcal, 'ratemean8', bold)
        # worksheet.write('Y%d'%rowcal, 'raterms1', bold)
        # worksheet.write('Z%d'%rowcal, 'raterms2', bold)
        # worksheet.write('AA%d'%rowcal, 'raterms3', bold)
        # worksheet.write('AB%d'%rowcal, 'raterms4', bold)
        # worksheet.write('AC%d'%rowcal, 'raterms5', bold)
        # worksheet.write('AD%d'%rowcal, 'raterms6', bold)
        # worksheet.write('AE%d'%rowcal, 'raterms7', bold)
        # worksheet.write('AF%d'%rowcal, 'raterms8', bold)

        # worksheet.write('AG%d'%rowcal, 'mean1', bold)
        # worksheet.write('AH%d'%rowcal, 'mean2', bold)
        # worksheet.write('AI%d'%rowcal, 'mean3', bold)
        # worksheet.write('AJ%d'%rowcal, 'mean4', bold)
        # worksheet.write('AK%d'%rowcal, 'mean5', bold)
        # worksheet.write('AL%d'%rowcal, 'mean6', bold)
        # worksheet.write('AM%d'%rowcal, 'mean7', bold)
        # worksheet.write('AN%d'%rowcal, 'mean8', bold)
        # worksheet.write('AO%d'%rowcal, 'rms1', bold)
        # worksheet.write('AP%d'%rowcal, 'rms2', bold)
        # worksheet.write('AQ%d'%rowcal, 'rms3', bold)
        # worksheet.write('AR%d'%rowcal, 'rms4', bold)
        # worksheet.write('AS%d'%rowcal, 'rms5', bold)
        # worksheet.write('AT%d'%rowcal, 'rms6', bold)
        # worksheet.write('AU%d'%rowcal, 'rms7', bold)
        # worksheet.write('AV%d'%rowcal, 'rms8', bold)
        # worksheet.write('AW%d'%rowcal, 'meanFSR1', bold)
        # worksheet.write('AX%d'%rowcal, 'meanFSR2', bold)
        # worksheet.write('AY%d'%rowcal, 'meanFSR3', bold)
        # worksheet.write('AZ%d'%rowcal, 'meanFSR4', bold)
        # worksheet.write('BA%d'%rowcal, 'meanvelocity', bold)
        # worksheet.write('BB%d'%rowcal, 'meanK', bold)
        # worksheet.write('BC%d'%rowcal, 'meanDistance', bold)
        # worksheet.write('BD%d'%rowcal, 'meanFORCE', bold)

        worksheet.write('Q%d'%rowcal, 'mean1', bold)
        worksheet.write('R%d'%rowcal, 'mean2', bold)
        worksheet.write('S%d'%rowcal, 'mean3', bold)
        worksheet.write('T%d'%rowcal, 'mean4', bold)
        worksheet.write('U%d'%rowcal, 'mean5', bold)
        worksheet.write('V%d'%rowcal, 'mean6', bold)
        worksheet.write('W%d'%rowcal, 'mean7', bold)
        worksheet.write('X%d'%rowcal, 'mean8', bold)
        worksheet.write('Y%d'%rowcal, 'rms1', bold)
        worksheet.write('Z%d'%rowcal, 'rms2', bold)
        worksheet.write('AA%d'%rowcal, 'rms3', bold)
        worksheet.write('AB%d'%rowcal, 'rms4', bold)
        worksheet.write('AC%d'%rowcal, 'rms5', bold)
        worksheet.write('AD%d'%rowcal, 'rms6', bold)
        worksheet.write('AE%d'%rowcal, 'rms7', bold)
        worksheet.write('AF%d'%rowcal, 'rms8', bold)
        worksheet.write('AG%d'%rowcal, 'meanFSR1', bold)
        worksheet.write('AH%d'%rowcal, 'meanFSR2', bold)
        worksheet.write('AI%d'%rowcal, 'meanFSR3', bold)
        worksheet.write('AJ%d'%rowcal, 'meanFSR4', bold)
        worksheet.write('AK%d'%rowcal, 'meanvelocity', bold)
        worksheet.write('AL%d'%rowcal, 'meanK', bold)
        worksheet.write('AM%d'%rowcal, 'meanDistance', bold)
        worksheet.write('AN%d'%rowcal, 'meanFORCE', bold)


        for c in xrange(0,(sheet.nrows-1)):

            worksheet.write  (row, col , data[c][0] )
            worksheet.write  (row, col + 1, data[c][1] )
            worksheet.write  (row, col + 2, data[c][2] )
            worksheet.write  (row, col + 3, data[c][3] )
            worksheet.write  (row, col + 4, data[c][4] )
            worksheet.write  (row, col + 5, data[c][5] )
            worksheet.write  (row, col + 6, data[c][6] )
            worksheet.write  (row, col + 7, data[c][7] )
            worksheet.write  (row, col + 8, data[c][8] )
            worksheet.write  (row, col + 9, data[c][9] )
            worksheet.write  (row, col + 10, data[c][10] )
            worksheet.write  (row, col + 11, data[c][11] )
            worksheet.write  (row, col + 12, data[c][12] )
            worksheet.write  (row, col + 13, data[c][13] )
            worksheet.write  (row, col + 14, data[c][14] )
            worksheet.write  (row, col + 15, data[c][15] )
            
            row += 1

        row=rowcal


        while (sampling <= len(data)):

            datacal = [[data[r][c] for c in range (3,11)] for r in range(start,sampling)]
            force = [[data[r][c] for c in range (2,3)] for r in range(start,sampling)]
            fsr = [[data[r][c] for c in range (11,16)] for r in range(start,sampling)]
            velocity = [[data[r][c] for c in range (1,2)] for r in range(start,sampling)]
            distance = [[data[r][c] for c in range (0,1)] for r in range(start,sampling)]

            datacal = np.asarray(datacal)
            force = np.asarray(force)
            fsr = np.asarray(fsr)
            velocity = np.asarray(velocity)
            distance = np.asarray(distance)

            mean=np.mean(abs(datacal), axis=0)

            datacalsquare=np.power(datacal, 2)
            rms=np.sqrt(np.mean(datacalsquare, axis=0))

            summean=np.sum(mean)
            sumrms=np.sum(rms)

            meanforce=np.mean(force)
            meanfsr=np.mean(fsr, axis=0)
            meanvelocity=np.mean(velocity)
            meandistance=np.mean(distance)

            if summean != 0 and sumrms!=0:

                for c in xrange(0,8):

                    # emgmeanrate[c] = mean[c]/summean
                    # emgrmsrate[c] = rms[c]/sumrms

                    # worksheet.write  (row, col + 16, emgmeanrate[0] )
                    # worksheet.write  (row, col + 17 , emgmeanrate[1] )
                    # worksheet.write  (row, col + 18 , emgmeanrate[2] )
                    # worksheet.write  (row, col + 19 , emgmeanrate[3] )
                    # worksheet.write  (row, col + 20 , emgmeanrate[4] )
                    # worksheet.write  (row, col + 21 , emgmeanrate[5] )
                    # worksheet.write  (row, col + 22 , emgmeanrate[6] )
                    # worksheet.write  (row, col + 23 , emgmeanrate[7] )
                    # worksheet.write  (row, col + 24 , emgrmsrate[0] )
                    # worksheet.write  (row, col + 25 , emgrmsrate[1] )
                    # worksheet.write  (row, col + 26 , emgrmsrate[2] )
                    # worksheet.write  (row, col + 27 , emgrmsrate[3] )
                    # worksheet.write  (row, col + 28 , emgrmsrate[4] )
                    # worksheet.write  (row, col + 29 , emgrmsrate[5] )
                    # worksheet.write  (row, col + 30 , emgrmsrate[6] )
                    # worksheet.write  (row, col + 31 , emgrmsrate[7] )
                    # worksheet.write  (row, col + 32 , mean[0] )
                    # worksheet.write  (row, col + 33 , mean[1] )
                    # worksheet.write  (row, col + 34 , mean[2] )
                    # worksheet.write  (row, col + 35 , mean[3] )
                    # worksheet.write  (row, col + 36 , mean[4] )
                    # worksheet.write  (row, col + 37 , mean[5] )
                    # worksheet.write  (row, col + 38 , mean[6] )
                    # worksheet.write  (row, col + 39 , mean[7] )
                    # worksheet.write  (row, col + 40 , rms[0] )
                    # worksheet.write  (row, col + 41 , rms[1] )
                    # worksheet.write  (row, col + 42 , rms[2] )
                    # worksheet.write  (row, col + 43 , rms[3] )
                    # worksheet.write  (row, col + 44 , rms[4] )
                    # worksheet.write  (row, col + 45 , rms[5] )
                    # worksheet.write  (row, col + 46 , rms[6] )
                    # worksheet.write  (row, col + 47 , rms[7] )

                    worksheet.write  (row, col + 16, mean[0] )
                    worksheet.write  (row, col + 17 , mean[1] )
                    worksheet.write  (row, col + 18 , mean[2] )
                    worksheet.write  (row, col + 19 , mean[3] )
                    worksheet.write  (row, col + 20 , mean[4] )
                    worksheet.write  (row, col + 21 , mean[5] )
                    worksheet.write  (row, col + 22 , mean[6] )
                    worksheet.write  (row, col + 23 , mean[7] )
                    worksheet.write  (row, col + 24 , rms[0] )
                    worksheet.write  (row, col + 25 , rms[1] )
                    worksheet.write  (row, col + 26 , rms[2] )
                    worksheet.write  (row, col + 27 , rms[3] )
                    worksheet.write  (row, col + 28 , rms[4] )
                    worksheet.write  (row, col + 29 , rms[5] )
                    worksheet.write  (row, col + 30 , rms[6] )
                    worksheet.write  (row, col + 31 , rms[7] )
                


                # worksheet.write  (row, col + 48 , meanfsr[0] )
                # worksheet.write  (row, col + 49 , meanfsr[1] )
                # worksheet.write  (row, col + 50 , meanfsr[2] )
                # worksheet.write  (row, col + 51 , meanfsr[3] )
                # worksheet.write  (row, col + 52 , meanvelocity )
                # worksheet.write  (row, col + 53 , meanfsr[4] )
                # worksheet.write  (row, col + 54 , meandistance )
                # worksheet.write  (row, col + 55 , meanforce )

                worksheet.write  (row, col + 32 , meanfsr[0] )
                worksheet.write  (row, col + 33 , meanfsr[1] )
                worksheet.write  (row, col + 34 , meanfsr[2] )
                worksheet.write  (row, col + 35 , meanfsr[3] )
                worksheet.write  (row, col + 36 , meanvelocity )
                worksheet.write  (row, col + 37 , meanfsr[4] )
                worksheet.write  (row, col + 38 , meandistance )
                worksheet.write  (row, col + 39 , meanforce )
            

                row += 1

            start=start+overlap
            sampling=sampling+overlap

    workbook1.close()
    numsheet = 0

    Combine()

def Combine():

    global row,col,readfile,sampling

    rowcal=sampling

    filer = readfile
    filew = readfile

    fileread = '%s.xlsx' %filer
    filewrite = '%s_combine.xlsx' %filew

    workbook = xlrd.open_workbook(fileread)
    workbook1 = xlsxwriter.Workbook(filewrite)

    bold = workbook1.add_format({'bold': 1})

    worksheet = workbook1.add_worksheet()

    worksheet.write('A1', 'Distance', bold)
    worksheet.write('B1', 'Velocity', bold)
    worksheet.write('C1', 'Force', bold)
    worksheet.write('D1', 'Chanel 1', bold)
    worksheet.write('E1', 'Chanel 2', bold)
    worksheet.write('F1', 'Chanel 3', bold)
    worksheet.write('G1', 'Chanel 4', bold)
    worksheet.write('H1', 'Chanel 5', bold)
    worksheet.write('I1', 'Chanel 6', bold)
    worksheet.write('J1', 'Chanel 7', bold)
    worksheet.write('K1', 'Chanel 8', bold)
    worksheet.write('L1', 'FSR1', bold)
    worksheet.write('M1', 'FSR2', bold)
    worksheet.write('N1', 'FSR3', bold)
    worksheet.write('O1', 'FSR4', bold)
    worksheet.write('P1', 'K', bold)

    numsheet = 0
    row = 1
    col = 0

    while numsheet <= (workbook.nsheets-1):

        sheet = workbook.sheet_by_index(numsheet)
        data=[[sheet.cell_value(r,c) for c in range(sheet.ncols)] for r in range(1,sheet.nrows)]

        for c in xrange(0,(sheet.nrows-1)):

            for l in xrange (16,(sheet.ncols)):

                if str(data[c][l]!=''):

                    worksheet.write  (row, col + l, data[c][l] )

            worksheet.write  (row, col , data[c][0] )
            worksheet.write  (row, col + 1, data[c][1] )
            worksheet.write  (row, col + 2, data[c][2] )        
            worksheet.write  (row, col + 3, data[c][3] )
            worksheet.write  (row, col + 4, data[c][4] )
            worksheet.write  (row, col + 5, data[c][5] )
            worksheet.write  (row, col + 6, data[c][6] )
            worksheet.write  (row, col + 7, data[c][7] )
            worksheet.write  (row, col + 8, data[c][8] )
            worksheet.write  (row, col + 9, data[c][9] )
            worksheet.write  (row, col + 10, data[c][10] )
            worksheet.write  (row, col + 11, data[c][11] )
            worksheet.write  (row, col + 12, data[c][12] )
            worksheet.write  (row, col + 13, data[c][13] )
            worksheet.write  (row, col + 14, data[c][14] )
            worksheet.write  (row, col + 15, data[c][15] )

            
            row += 1

        numsheet=numsheet+1

        print(numsheet)

    workbook1.close() 

def exit():

    hub.stop(True)
    pg.exit()

def newdatabase():

    trainfile = text22.text()
    databasefile = text23.text()

    fileread = '%s.xlsx' %str(trainfile)
    filewrite = '%s.xlsx' %str(databasefile)

    workbook = xlrd.open_workbook(fileread)
    sheet = workbook.sheet_by_index(0)

    workbook1 = xlsxwriter.Workbook(filewrite)
    worksheet = workbook1.add_worksheet()

    bold = workbook1.add_format({'bold': 1})

    row = 1
    col = 0

    title = [[sheet.cell_value(r,c) for c in range(16,sheet.ncols)] for r in range(0,1)]
    data = [[sheet.cell_value(r,c) for c in range(16,sheet.ncols)] for r in range(1,sheet.nrows)]

    k = 0

    for i in range(len(data)):

        for j in range (1):

            if isinstance(data[i][j], unicode) or isinstance(data[i][j], str):

                k=k+1

    q = np.zeros((len(data)-k, len(data[1])))
    q = q.tolist()

    n = 0
    num = 0
    for i in range(len(data)):

            if isinstance(data[i][j], unicode) or isinstance(data[i][j], str):

                num=num+1
                
            else:

                for j in range (len(data[1])):

                    q[n][j]=data[i][j]

                n=n+1
    data=q

    data=np.asarray(data)

    # xall2=[[data[r][c] for c in range(39)] for r in range(0,(len(data)))]
    # yall2=[[data[r][c] for c in range(39,40)] for r in range(0,(len(data)))]

    xall2=[[data[r][c] for c in range(23)] for r in range(0,(len(data)))]
    yall2=[[data[r][c] for c in range(23,24)] for r in range(0,(len(data)))]

    # worksheet.write('A%d'%row, 'ratemean1', bold)
    # worksheet.write('B%d'%row, 'ratemean2', bold)
    # worksheet.write('C%d'%row, 'ratemean3', bold)
    # worksheet.write('D%d'%row, 'ratemean4', bold)
    # worksheet.write('E%d'%row, 'ratemean5', bold)
    # worksheet.write('F%d'%row, 'ratemean6', bold)
    # worksheet.write('G%d'%row, 'ratemean7', bold)
    # worksheet.write('H%d'%row, 'ratemean8', bold)
    # worksheet.write('I%d'%row, 'raterms1', bold)
    # worksheet.write('J%d'%row, 'raterms2', bold)
    # worksheet.write('K%d'%row, 'raterms3', bold)
    # worksheet.write('L%d'%row, 'raterms4', bold)
    # worksheet.write('M%d'%row, 'raterms5', bold)
    # worksheet.write('N%d'%row, 'raterms6', bold)
    # worksheet.write('O%d'%row, 'raterms7', bold)
    # worksheet.write('P%d'%row, 'raterms8', bold)
    # worksheet.write('Q%d'%row, 'mean1', bold)
    # worksheet.write('R%d'%row, 'mean2', bold)
    # worksheet.write('S%d'%row, 'mean3', bold)
    # worksheet.write('T%d'%row, 'mean4', bold)
    # worksheet.write('U%d'%row, 'mean5', bold)
    # worksheet.write('V%d'%row, 'mean6', bold)
    # worksheet.write('W%d'%row, 'mean7', bold)
    # worksheet.write('X%d'%row, 'mean8', bold)
    # worksheet.write('Y%d'%row, 'rms1', bold)
    # worksheet.write('Z%d'%row, 'rms2', bold)
    # worksheet.write('AA%d'%row, 'rms3', bold)
    # worksheet.write('AB%d'%row, 'rms4', bold)
    # worksheet.write('AC%d'%row, 'rms5', bold)
    # worksheet.write('AD%d'%row, 'rms6', bold)
    # worksheet.write('AE%d'%row, 'rms7', bold)
    # worksheet.write('AF%d'%row, 'rms8', bold)
    # worksheet.write('AG%d'%row, 'meanFSR1', bold)
    # worksheet.write('AH%d'%row, 'meanFSR2', bold)
    # worksheet.write('AI%d'%row, 'meanFSR3', bold)
    # worksheet.write('AJ%d'%row, 'meanFSR4', bold)
    # worksheet.write('AK%d'%row, 'meanvelocity', bold)
    # worksheet.write('AL%d'%row, 'meanK', bold)
    # worksheet.write('AM%d'%row, 'meanDistance', bold)
    # worksheet.write('AN%d'%row, 'meanFORCE', bold)

    worksheet.write('A%d'%row, 'mean1', bold)
    worksheet.write('B%d'%row, 'mean2', bold)
    worksheet.write('C%d'%row, 'mean3', bold)
    worksheet.write('D%d'%row, 'mean4', bold)
    worksheet.write('E%d'%row, 'mean5', bold)
    worksheet.write('F%d'%row, 'mean6', bold)
    worksheet.write('G%d'%row, 'mean7', bold)
    worksheet.write('H%d'%row, 'mean8', bold)
    worksheet.write('I%d'%row, 'rms1', bold)
    worksheet.write('J%d'%row, 'rms2', bold)
    worksheet.write('K%d'%row, 'rms3', bold)
    worksheet.write('L%d'%row, 'rms4', bold)
    worksheet.write('M%d'%row, 'rms5', bold)
    worksheet.write('N%d'%row, 'rms6', bold)
    worksheet.write('O%d'%row, 'rms7', bold)
    worksheet.write('P%d'%row, 'rms8', bold)
    worksheet.write('Q%d'%row, 'meanFSR1', bold)
    worksheet.write('R%d'%row, 'meanFSR2', bold)
    worksheet.write('S%d'%row, 'meanFSR3', bold)
    worksheet.write('T%d'%row, 'meanFSR4', bold)
    worksheet.write('U%d'%row, 'meanvelocity', bold)
    worksheet.write('V%d'%row, 'meanK', bold)
    worksheet.write('W%d'%row, 'meanDistance', bold)
    worksheet.write('X%d'%row, 'meanFORCE', bold)

    numsheet = 0

    data = xall2
    target = yall2 

    for c in xrange(0,(len(data))):

        for l in xrange (0,(len(data[1]))):

            if str(data[c][l]!=''):

                worksheet.write  (row, col + l, data[c][l] )

        # worksheet.write  (row, col , data[c][0] )
        # worksheet.write  (row, col + 1, data[c][1] )
        # worksheet.write  (row, col + 2, data[c][2] )
        # worksheet.write  (row, col + 3, data[c][3] )
        # worksheet.write  (row, col + 4, data[c][4] )
        # worksheet.write  (row, col + 5, data[c][5] )
        # worksheet.write  (row, col + 6, data[c][6] )
        # worksheet.write  (row, col + 7, data[c][7] )
        # worksheet.write  (row, col + 8, data[c][8] )
        # worksheet.write  (row, col + 9, data[c][9] )
        # worksheet.write  (row, col + 10, data[c][10] )
        # worksheet.write  (row, col + 11, data[c][11] )
        # worksheet.write  (row, col + 12, data[c][12] )
        # worksheet.write  (row, col + 13, data[c][13] )
        # worksheet.write  (row, col + 14, data[c][14] )
        # worksheet.write  (row, col + 15, data[c][15] )
        # worksheet.write  (row, col + 16, data[c][16] )
        # worksheet.write  (row, col + 17, data[c][17] )
        # worksheet.write  (row, col + 18, data[c][18] )
        # worksheet.write  (row, col + 19, data[c][19] )
        # worksheet.write  (row, col + 20, data[c][20] )
        # worksheet.write  (row, col + 21, data[c][21] )
        # worksheet.write  (row, col + 22, data[c][22] )
        # worksheet.write  (row, col + 23, data[c][23] )
        # worksheet.write  (row, col + 24, data[c][24] )
        # worksheet.write  (row, col + 25, data[c][25] )
        # worksheet.write  (row, col + 26, data[c][26] )
        # worksheet.write  (row, col + 27, data[c][27] )
        # worksheet.write  (row, col + 28, data[c][28] )
        # worksheet.write  (row, col + 29, data[c][29] )
        # worksheet.write  (row, col + 30, data[c][30] )
        # worksheet.write  (row, col + 31, data[c][31] )
        # worksheet.write  (row, col + 32, data[c][32] )
        # worksheet.write  (row, col + 33, data[c][33] )
        # worksheet.write  (row, col + 34, data[c][34] )
        # worksheet.write  (row, col + 35, data[c][35] )
        # worksheet.write  (row, col + 36, data[c][36] )
        # worksheet.write  (row, col + 37, data[c][37] )
        # worksheet.write  (row, col + 38, data[c][38] )
        # worksheet.write  (row, col + 39, (np.asarray(target[c])) )


        worksheet.write  (row, col , data[c][0] )
        worksheet.write  (row, col + 1, data[c][1] )
        worksheet.write  (row, col + 2, data[c][2] )
        worksheet.write  (row, col + 3, data[c][3] )
        worksheet.write  (row, col + 4, data[c][4] )
        worksheet.write  (row, col + 5, data[c][5] )
        worksheet.write  (row, col + 6, data[c][6] )
        worksheet.write  (row, col + 7, data[c][7] )
        worksheet.write  (row, col + 8, data[c][8] )
        worksheet.write  (row, col + 9, data[c][9] )
        worksheet.write  (row, col + 10, data[c][10] )
        worksheet.write  (row, col + 11, data[c][11] )
        worksheet.write  (row, col + 12, data[c][12] )
        worksheet.write  (row, col + 13, data[c][13] )
        worksheet.write  (row, col + 14, data[c][14] )
        worksheet.write  (row, col + 15, data[c][15] )
        worksheet.write  (row, col + 16, data[c][16] )
        worksheet.write  (row, col + 17, data[c][17] )
        worksheet.write  (row, col + 18, data[c][18] )
        worksheet.write  (row, col + 19, data[c][19] )
        worksheet.write  (row, col + 20, data[c][20] )
        worksheet.write  (row, col + 21, data[c][21] )
        worksheet.write  (row, col + 22, data[c][22] )
        worksheet.write  (row, col + 23, (np.asarray(target[c])) )

        row += 1

    workbook1.close()

def Traincombinedatabase():

    databasefile = text21.text()
    trainfile = text22.text()

    fileread = '%s.xlsx' %str(databasefile)
    workbook = xlrd.open_workbook(fileread)
    sheet = workbook.sheet_by_index(0)

    title=[[sheet.cell_value(r,c) for c in range(0,sheet.ncols)] for r in range(0,1)]
    data=[[sheet.cell_value(r,c) for c in range(0,sheet.ncols)] for r in range(1,sheet.nrows)]

    data=np.asarray(data)

    # xall1=[[data[r][c] for c in range(39)] for r in range(0,(len(data)))]
    # yall1=[[data[r][c] for c in range(39,40)] for r in range(0,(len(data)))]

    xall1=[[data[r][c] for c in range(23)] for r in range(0,(len(data)))]
    yall1=[[data[r][c] for c in range(23,24)] for r in range(0,(len(data)))]

    fileread = '%s.xlsx' %str(trainfile)
    workbook = xlrd.open_workbook(fileread)
    sheet = workbook.sheet_by_index(0)

    title=[[sheet.cell_value(r,c) for c in range(16,sheet.ncols)] for r in range(0,1)]
    data=[[sheet.cell_value(r,c) for c in range(16,sheet.ncols)] for r in range(1,sheet.nrows)]

    k = 0

    for i in range(len(data)):

        for j in range (1):

            if isinstance(data[i][j], unicode) or isinstance(data[i][j], str):

                k=k+1

    q = np.zeros((len(data)-k, len(data[1])))
    q = q.tolist()

    n = 0
    num = 0
    for i in range(len(data)):

            if isinstance(data[i][j], unicode) or isinstance(data[i][j], str):

                num=num+1
                
            else:

                for j in range (len(data[1])):

                    q[n][j]=data[i][j]

                n=n+1
    data=q

    data=np.asarray(data)

    # xall2=[[data[r][c] for c in range(39)] for r in range(0,(len(data)))]
    # yall2=[[data[r][c] for c in range(39,40)] for r in range(0,(len(data)))]

    xall2=[[data[r][c] for c in range(23)] for r in range(0,(len(data)))]
    yall2=[[data[r][c] for c in range(23,24)] for r in range(0,(len(data)))]


    xall=np.concatenate((xall1, xall2), axis=0)
    yall=np.concatenate((yall1, yall2), axis=0)

    x=xall[0:int(0.8*len(xall))]
    y=yall[0:int(0.8*len(yall))]

    ####################### Neural Network ##########################

    nn = MLPRegressor(
        hidden_layer_sizes=(20,),  activation='identity', solver='lbfgs', alpha=0.001, batch_size='auto',
        learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
        random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
        early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    n = nn.fit(x, np.ravel(y))

    test_x=xall[int(0.8*len(xall)):len(xall)]
    real_y=yall[int(0.8*len(yall)):len(yall)]

    test_y = nn.predict(test_x)
    train_y = nn.predict(x)

    errortrain=[(y[c]-train_y[c]) for c in range(0,len(train_y))]
    errortest=[(real_y[c]-test_y[c]) for c in range(0,len(test_y))]

    errortestsquare=np.power(errortest,2)
    errortrainsquare=np.power(errortrain,2)

    rmstest=np.sqrt(np.mean(errortestsquare))
    rmstrain=np.sqrt(np.mean(errortrainsquare))

    plt.figure(1)

    plt.figure(1).canvas.set_window_title('NN combine Database+Train')


    plt.subplot(211)
    plt.plot(y,label="Meusure Force")
    plt.plot(train_y,label="Predict Force")
    plt.legend()
    plt.xlabel('Sample')
    plt.ylabel('DeltaForce(N)')
    plt.title('Compare Train (epoch = %d) Measure and Predict (RMSE : %f)' %(nn.n_iter_,rmstrain))
    bottom, top = plt.ylim()  # return the current ylim


    plt.subplot(212)
    plt.plot(real_y,label="Meusure Force")
    plt.plot(test_y,label="Predict Force")
    plt.legend()
    plt.xlabel('Sample')
    plt.ylabel('DeltaForce(N)')
    plt.title('Compare Test Measure and Predict (RMSE : %f)' %rmstest)
    plt.subplots_adjust(hspace=0.5)
    plt.ylim(bottom, top)

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    ####################### Support vector ##########################

    X = x
    y = y

    svr_rbf = SVR(kernel='rbf', C=100,epsilon=0.01,gamma='auto')
    svr_lin = SVR(kernel='linear', C=100,epsilon=0.01,gamma='auto')
    svr_poly = SVR(kernel='poly', C=100,epsilon=0.01, degree=2,gamma='auto')

    y_rbf = svr_rbf.fit(X, np.ravel(y)).predict(X)
    y_lin = svr_lin.fit(X, np.ravel(y)).predict(X)
    y_poly = svr_poly.fit(X, np.ravel(y)).predict(X)

    y_rbf_test = svr_rbf.fit(X, np.ravel(y)).predict(test_x)
    y_lin_test = svr_lin.fit(X, np.ravel(y)).predict(test_x)
    y_poly_test = svr_poly.fit(X, np.ravel(y)).predict(test_x)

    y_rbf_error =[(y[c]-y_rbf[c]) for c in range(0,len(y_rbf))]
    y_lin_error =[(y[c]-y_lin[c]) for c in range(0,len(y_lin))]
    y_poly_error =[(y[c]-y_poly[c]) for c in range(0,len(y_poly))]


    y_rbf_error_square=np.power(y_rbf_error,2)
    y_lin_error_square=np.power(y_lin_error,2)
    y_poly_error_square=np.power(y_poly_error,2)

    y_rbf_RMSE=np.sqrt(np.mean(y_rbf_error_square))
    y_lin_RMSE=np.sqrt(np.mean(y_lin_error_square))
    y_poly_RMSE=np.sqrt(np.mean(y_poly_error_square))

    y_rbf_test_error =[(real_y[c]-y_rbf_test[c]) for c in range(0,len(y_rbf_test))]
    y_lin_test_error =[(real_y[c]-y_lin_test[c]) for c in range(0,len(y_lin_test))]
    y_poly_test_error =[(real_y[c]-y_poly_test[c]) for c in range(0,len(y_poly_test))]

    y_rbf_test_error_square=np.power(y_rbf_test_error,2)
    y_lin_test_error_square=np.power(y_lin_test_error,2)
    y_poly_test_error_square=np.power(y_poly_test_error,2)

    y_rbf_test_RMSE=np.sqrt(np.mean(y_rbf_test_error_square))
    y_lin_test_RMSE=np.sqrt(np.mean(y_lin_test_error_square))
    y_poly_test_RMSE=np.sqrt(np.mean(y_poly_test_error_square))

    lw = 2

    plt.figure(2)

    plt.figure(2).canvas.set_window_title('SVR combine Database+Train')

    plt.subplot(321)
    plt.plot(y, color='navy', label='data')
    plt.plot(y_rbf, color='darkorange', lw=lw, label='RBF model')
    plt.xlabel('Sample')
    plt.ylabel('DeltaForce')
    plt.title('Support Vector Regression(RBF model) RMSE : %f'%y_rbf_RMSE)
    plt.subplots_adjust(hspace=0.5)
    plt.legend()
    bottom, top = plt.ylim()  # return the current ylim

    plt.subplot(322)
    plt.plot(real_y, color='navy', label='data')
    plt.plot(y_rbf_test, color='darkorange', lw=lw, label='RBF model')
    plt.xlabel('Sample')
    plt.ylabel('DeltaForce')
    plt.title('Support Vector Regression TEST (RBF model) RMSE : %f'%y_rbf_test_RMSE)
    plt.subplots_adjust(hspace=0.5)
    plt.legend()
    plt.ylim(bottom, top)

    plt.subplot(323)
    plt.plot(y, color='c', label='data')
    plt.plot(y_lin, color='darkorange', lw=lw, label='Linear model')
    plt.xlabel('Sample')
    plt.ylabel('DeltaForce')
    plt.title('Support Vector Regression(Linear model) RMSE : %f'%y_lin_RMSE)
    plt.subplots_adjust(hspace=0.5)
    plt.legend()

    plt.subplot(324)
    plt.plot(real_y, color='c', label='data')
    plt.plot(y_lin_test, color='darkorange', lw=lw, label='Linear model')
    plt.xlabel('Sample')
    plt.ylabel('DeltaForce')
    plt.title('Support Vector Regression TEST (Linear model) RMSE : %f'%y_lin_test_RMSE)
    plt.subplots_adjust(hspace=0.5)
    plt.legend()
    plt.ylim(bottom, top)

    plt.subplot(325)
    plt.plot(y, color='cornflowerblue', label='data')
    plt.plot(y_poly, color='darkorange', lw=lw, label='Polynomial model')
    plt.xlabel('Sample')
    plt.ylabel('DeltaForce')
    plt.title('Support Vector Regression(Polynomial model) RMSE : %f'%y_poly_RMSE)
    plt.subplots_adjust(hspace=0.5)
    plt.legend()

    plt.subplot(326)
    plt.plot(real_y, color='cornflowerblue', label='data')
    plt.plot(y_poly_test, color='darkorange', lw=lw, label='Polynomial model')
    plt.xlabel('Sample')
    plt.ylabel('DeltaForce')
    plt.title('Support Vector Regression TEST (Polynomial model) RMSE : %f'%y_poly_test_RMSE)
    plt.subplots_adjust(hspace=0.5)
    plt.legend()
    plt.ylim(bottom, top)

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    plt.show()

def traintobase():

    databasefile = text21.text()
    trainfile = text22.text()

    fileread = '%s.xlsx' %str(databasefile)
    workbook = xlrd.open_workbook(fileread)
    sheet = workbook.sheet_by_index(0)

    title=[[sheet.cell_value(r,c) for c in range(0,sheet.ncols)] for r in range(0,1)]
    data=[[sheet.cell_value(r,c) for c in range(0,sheet.ncols)] for r in range(1,sheet.nrows)]
    data=np.asarray(data)

    # xall1=[[data[r][c] for c in range(39)] for r in range(0,(len(data)))]
    # yall1=[[data[r][c] for c in range(39,40)] for r in range(0,(len(data)))]

    xall1=[[data[r][c] for c in range(23)] for r in range(0,(len(data)))]
    yall1=[[data[r][c] for c in range(23,24)] for r in range(0,(len(data)))]

    fileread = '%s.xlsx' %str(trainfile)
    workbook = xlrd.open_workbook(fileread)
    sheet = workbook.sheet_by_index(0)

    title=[[sheet.cell_value(r,c) for c in range(16,sheet.ncols)] for r in range(0,1)]
    data=[[sheet.cell_value(r,c) for c in range(16,sheet.ncols)] for r in range(1,sheet.nrows)]

    k = 0

    for i in range(len(data)):

        for j in range (1):

            if isinstance(data[i][j], unicode) or isinstance(data[i][j], str):

                k=k+1

    q = np.zeros((len(data)-k, len(data[1])))
    q = q.tolist()

    n = 0
    num = 0
    for i in range(len(data)):

            if isinstance(data[i][j], unicode) or isinstance(data[i][j], str):

                num=num+1
                
            else:

                for j in range (len(data[1])):

                    q[n][j]=data[i][j]

                n=n+1
    data=q

    data=np.asarray(data)

    # xall2=[[data[r][c] for c in range(39)] for r in range(0,(len(data)))]
    # yall2=[[data[r][c] for c in range(39,40)] for r in range(0,(len(data)))]

    xall2=[[data[r][c] for c in range(23)] for r in range(0,(len(data)))]
    yall2=[[data[r][c] for c in range(23,24)] for r in range(0,(len(data)))]

    xall=np.concatenate((xall1, xall2), axis=0)

    filewrite = '%s.xlsx' %str(databasefile)
    workbook1 = xlsxwriter.Workbook(filewrite)
    worksheet = workbook1.add_worksheet()
    bold = workbook1.add_format({'bold': 1})

    row = 1
    col = 0

    # worksheet.write('A%d'%row, 'ratemean1', bold)
    # worksheet.write('B%d'%row, 'ratemean2', bold)
    # worksheet.write('C%d'%row, 'ratemean3', bold)
    # worksheet.write('D%d'%row, 'ratemean4', bold)
    # worksheet.write('E%d'%row, 'ratemean5', bold)
    # worksheet.write('F%d'%row, 'ratemean6', bold)
    # worksheet.write('G%d'%row, 'ratemean7', bold)
    # worksheet.write('H%d'%row, 'ratemean8', bold)
    # worksheet.write('I%d'%row, 'raterms1', bold)
    # worksheet.write('J%d'%row, 'raterms2', bold)
    # worksheet.write('K%d'%row, 'raterms3', bold)
    # worksheet.write('L%d'%row, 'raterms4', bold)
    # worksheet.write('M%d'%row, 'raterms5', bold)
    # worksheet.write('N%d'%row, 'raterms6', bold)
    # worksheet.write('O%d'%row, 'raterms7', bold)
    # worksheet.write('P%d'%row, 'raterms8', bold)
    # worksheet.write('Q%d'%row, 'mean1', bold)
    # worksheet.write('R%d'%row, 'mean2', bold)
    # worksheet.write('S%d'%row, 'mean3', bold)
    # worksheet.write('T%d'%row, 'mean4', bold)
    # worksheet.write('U%d'%row, 'mean5', bold)
    # worksheet.write('V%d'%row, 'mean6', bold)
    # worksheet.write('W%d'%row, 'mean7', bold)
    # worksheet.write('X%d'%row, 'mean8', bold)
    # worksheet.write('Y%d'%row, 'rms1', bold)
    # worksheet.write('Z%d'%row, 'rms2', bold)
    # worksheet.write('AA%d'%row, 'rms3', bold)
    # worksheet.write('AB%d'%row, 'rms4', bold)
    # worksheet.write('AC%d'%row, 'rms5', bold)
    # worksheet.write('AD%d'%row, 'rms6', bold)
    # worksheet.write('AE%d'%row, 'rms7', bold)
    # worksheet.write('AF%d'%row, 'rms8', bold)
    # worksheet.write('AG%d'%row, 'meanFSR1', bold)
    # worksheet.write('AH%d'%row, 'meanFSR2', bold)
    # worksheet.write('AI%d'%row, 'meanFSR3', bold)
    # worksheet.write('AJ%d'%row, 'meanFSR4', bold)
    # worksheet.write('AK%d'%row, 'meanvelocity', bold)
    # worksheet.write('AL%d'%row, 'meanK', bold)
    # worksheet.write('AM%d'%row, 'meanDistance', bold)
    # worksheet.write('AN%d'%row, 'meanFORCE', bold)

    worksheet.write('A%d'%row, 'mean1', bold)
    worksheet.write('B%d'%row, 'mean2', bold)
    worksheet.write('C%d'%row, 'mean3', bold)
    worksheet.write('D%d'%row, 'mean4', bold)
    worksheet.write('E%d'%row, 'mean5', bold)
    worksheet.write('F%d'%row, 'mean6', bold)
    worksheet.write('G%d'%row, 'mean7', bold)
    worksheet.write('H%d'%row, 'mean8', bold)
    worksheet.write('I%d'%row, 'rms1', bold)
    worksheet.write('J%d'%row, 'rms2', bold)
    worksheet.write('K%d'%row, 'rms3', bold)
    worksheet.write('L%d'%row, 'rms4', bold)
    worksheet.write('M%d'%row, 'rms5', bold)
    worksheet.write('N%d'%row, 'rms6', bold)
    worksheet.write('O%d'%row, 'rms7', bold)
    worksheet.write('P%d'%row, 'rms8', bold)
    worksheet.write('Q%d'%row, 'meanFSR1', bold)
    worksheet.write('R%d'%row, 'meanFSR2', bold)
    worksheet.write('S%d'%row, 'meanFSR3', bold)
    worksheet.write('T%d'%row, 'meanFSR4', bold)
    worksheet.write('U%d'%row, 'meanvelocity', bold)
    worksheet.write('V%d'%row, 'meanK', bold)
    worksheet.write('W%d'%row, 'meanDistance', bold)
    worksheet.write('X%d'%row, 'meanFORCE', bold)

    numsheet = 0

    data = xall1
    target = yall1

    for c in xrange(0,(len(data))):

        for l in xrange (0,(len(data[1]))):

            if str(data[c][l]!=''):

                worksheet.write  (row, col + l, data[c][l] )

        # worksheet.write  (row, col , float(data[c][0] ))
        # worksheet.write  (row, col + 1, float(data[c][1] ))
        # worksheet.write  (row, col + 2, float(data[c][2] ))
        # worksheet.write  (row, col + 3, float(data[c][3] ))
        # worksheet.write  (row, col + 4, float(data[c][4] ))
        # worksheet.write  (row, col + 5, float(data[c][5] ))
        # worksheet.write  (row, col + 6, float(data[c][6] ))
        # worksheet.write  (row, col + 7, float(data[c][7] ))
        # worksheet.write  (row, col + 8, float(data[c][8] ))
        # worksheet.write  (row, col + 9, float(data[c][9] ))
        # worksheet.write  (row, col + 10, float(data[c][10] ))
        # worksheet.write  (row, col + 11, float(data[c][11] ))
        # worksheet.write  (row, col + 12, float(data[c][12] ))
        # worksheet.write  (row, col + 13, float(data[c][13] ))
        # worksheet.write  (row, col + 14, float(data[c][14] ))
        # worksheet.write  (row, col + 15, float(data[c][15] ))
        # worksheet.write  (row, col + 16, float(data[c][16] ))
        # worksheet.write  (row, col + 17, float(data[c][17] ))
        # worksheet.write  (row, col + 18, float(data[c][18] ))
        # worksheet.write  (row, col + 19, float(data[c][19] ))
        # worksheet.write  (row, col + 20, float(data[c][20] ))
        # worksheet.write  (row, col + 21, float(data[c][21] ))
        # worksheet.write  (row, col + 22, float(data[c][22] ))
        # worksheet.write  (row, col + 23, float(data[c][23] ))
        # worksheet.write  (row, col + 24, float(data[c][24] ))
        # worksheet.write  (row, col + 25, float(data[c][25] ))
        # worksheet.write  (row, col + 26, float(data[c][26] ))
        # worksheet.write  (row, col + 27, float(data[c][27] ))
        # worksheet.write  (row, col + 28, float(data[c][28] ))
        # worksheet.write  (row, col + 29, float(data[c][29] ))
        # worksheet.write  (row, col + 30, float(data[c][30] ))
        # worksheet.write  (row, col + 31, float(data[c][31] ))
        # worksheet.write  (row, col + 32, float(data[c][32] ))
        # worksheet.write  (row, col + 33, float(data[c][33] ))
        # worksheet.write  (row, col + 34, float(data[c][34] ))
        # worksheet.write  (row, col + 35, float(data[c][35] ))
        # worksheet.write  (row, col + 36, float(data[c][36] ))
        # worksheet.write  (row, col + 37, float(data[c][37] ))
        # worksheet.write  (row, col + 38, float(data[c][38] ))
        # worksheet.write  (row, col + 39, float((np.asarray(target[c])) ))

        worksheet.write  (row, col , float(data[c][0] ))
        worksheet.write  (row, col + 1, float(data[c][1] ))
        worksheet.write  (row, col + 2, float(data[c][2] ))
        worksheet.write  (row, col + 3, float(data[c][3] ))
        worksheet.write  (row, col + 4, float(data[c][4] ))
        worksheet.write  (row, col + 5, float(data[c][5] ))
        worksheet.write  (row, col + 6, float(data[c][6] ))
        worksheet.write  (row, col + 7, float(data[c][7] ))
        worksheet.write  (row, col + 8, float(data[c][8] ))
        worksheet.write  (row, col + 9, float(data[c][9] ))
        worksheet.write  (row, col + 10, float(data[c][10] ))
        worksheet.write  (row, col + 11, float(data[c][11] ))
        worksheet.write  (row, col + 12, float(data[c][12] ))
        worksheet.write  (row, col + 13, float(data[c][13] ))
        worksheet.write  (row, col + 14, float(data[c][14] ))
        worksheet.write  (row, col + 15, float(data[c][15] ))
        worksheet.write  (row, col + 16, float(data[c][16] ))
        worksheet.write  (row, col + 17, float(data[c][17] ))
        worksheet.write  (row, col + 18, float(data[c][18] ))
        worksheet.write  (row, col + 19, float(data[c][19] ))
        worksheet.write  (row, col + 20, float(data[c][20] ))
        worksheet.write  (row, col + 21, float(data[c][21] ))
        worksheet.write  (row, col + 22, float(data[c][22] ))
        worksheet.write  (row, col + 23, float((np.asarray(target[c])) ))

        row += 1

    numsheet = 0

    data = xall2
    target = yall2 

    for c in xrange(0,(len(data))):

        for l in xrange (0,(len(data[1]))):

            if str(data[c][l]!=''):

                worksheet.write  (row, col + l, data[c][l] )

        # worksheet.write  (row, col , data[c][0] )
        # worksheet.write  (row, col + 1, data[c][1] )
        # worksheet.write  (row, col + 2, data[c][2] )
        # worksheet.write  (row, col + 3, data[c][3] )
        # worksheet.write  (row, col + 4, data[c][4] )
        # worksheet.write  (row, col + 5, data[c][5] )
        # worksheet.write  (row, col + 6, data[c][6] )
        # worksheet.write  (row, col + 7, data[c][7] )
        # worksheet.write  (row, col + 8, data[c][8] )
        # worksheet.write  (row, col + 9, data[c][9] )
        # worksheet.write  (row, col + 10, data[c][10] )
        # worksheet.write  (row, col + 11, data[c][11] )
        # worksheet.write  (row, col + 12, data[c][12] )
        # worksheet.write  (row, col + 13, data[c][13] )
        # worksheet.write  (row, col + 14, data[c][14] )
        # worksheet.write  (row, col + 15, data[c][15] )
        # worksheet.write  (row, col + 16, data[c][16] )
        # worksheet.write  (row, col + 17, data[c][17] )
        # worksheet.write  (row, col + 18, data[c][18] )
        # worksheet.write  (row, col + 19, data[c][19] )
        # worksheet.write  (row, col + 20, data[c][20] )
        # worksheet.write  (row, col + 21, data[c][21] )
        # worksheet.write  (row, col + 22, data[c][22] )
        # worksheet.write  (row, col + 23, data[c][23] )
        # worksheet.write  (row, col + 24, data[c][24] )
        # worksheet.write  (row, col + 25, data[c][25] )
        # worksheet.write  (row, col + 26, data[c][26] )
        # worksheet.write  (row, col + 27, data[c][27] )
        # worksheet.write  (row, col + 28, data[c][28] )
        # worksheet.write  (row, col + 29, data[c][29] )
        # worksheet.write  (row, col + 30, data[c][30] )
        # worksheet.write  (row, col + 31, data[c][31] )
        # worksheet.write  (row, col + 32, data[c][32] )
        # worksheet.write  (row, col + 33, data[c][33] )
        # worksheet.write  (row, col + 34, data[c][34] )
        # worksheet.write  (row, col + 35, data[c][35] )
        # worksheet.write  (row, col + 36, data[c][36] )
        # worksheet.write  (row, col + 37, data[c][37] )
        # worksheet.write  (row, col + 38, data[c][38] )
        # worksheet.write  (row, col + 39, (np.asarray(target[c])) )

        worksheet.write  (row, col , data[c][0] )
        worksheet.write  (row, col + 1, data[c][1] )
        worksheet.write  (row, col + 2, data[c][2] )
        worksheet.write  (row, col + 3, data[c][3] )
        worksheet.write  (row, col + 4, data[c][4] )
        worksheet.write  (row, col + 5, data[c][5] )
        worksheet.write  (row, col + 6, data[c][6] )
        worksheet.write  (row, col + 7, data[c][7] )
        worksheet.write  (row, col + 8, data[c][8] )
        worksheet.write  (row, col + 9, data[c][9] )
        worksheet.write  (row, col + 10, data[c][10] )
        worksheet.write  (row, col + 11, data[c][11] )
        worksheet.write  (row, col + 12, data[c][12] )
        worksheet.write  (row, col + 13, data[c][13] )
        worksheet.write  (row, col + 14, data[c][14] )
        worksheet.write  (row, col + 15, data[c][15] )
        worksheet.write  (row, col + 16, data[c][16] )
        worksheet.write  (row, col + 17, data[c][17] )
        worksheet.write  (row, col + 18, data[c][18] )
        worksheet.write  (row, col + 19, data[c][19] )
        worksheet.write  (row, col + 20, data[c][20] )
        worksheet.write  (row, col + 21, data[c][21] )
        worksheet.write  (row, col + 22, data[c][22] )
        worksheet.write  (row, col + 23, (np.asarray(target[c])) )

        row += 1
        
    workbook1.close()    

def Traindatatrain():

    trainfile = text22.text()
    fileread = '%s.xlsx' %str(trainfile)

    workbook = xlrd.open_workbook(fileread)
    sheet = workbook.sheet_by_index(0)

    title=[[sheet.cell_value(r,c) for c in range(16,sheet.ncols)] for r in range(0,1)]
    data=[[sheet.cell_value(r,c) for c in range(16,sheet.ncols)] for r in range(1,sheet.nrows)]

    k = 0

    for i in range(len(data)):

        for j in range (1):

            if isinstance(data[i][j], unicode) or isinstance(data[i][j], str):

                k=k+1

    q = np.zeros((len(data)-k, len(data[1])))
    q = q.tolist()

    n = 0
    num = 0
    for i in range(len(data)):

            if isinstance(data[i][j], unicode) or isinstance(data[i][j], str):

                num=num+1
                
            else:

                for j in range (len(data[1])):

                    q[n][j]=data[i][j]

                n=n+1
    data=q

    data=np.asarray(data)

    # xall2=[[data[r][c] for c in range(39)] for r in range(0,(len(data)))]
    # yall2=[[data[r][c] for c in range(39,40)] for r in range(0,(len(data)))]

    xall2=[[data[r][c] for c in range(23)] for r in range(0,(len(data)))]
    yall2=[[data[r][c] for c in range(23,24)] for r in range(0,(len(data)))]

    xall=xall2
    yall=yall2

    x=xall[0:int(0.8*len(xall))]
    y=yall[0:int(0.8*len(yall))]

    ####################### Neural Network ##########################

    nn = MLPRegressor(
        hidden_layer_sizes=(20,),  activation='identity', solver='lbfgs', alpha=0.001, batch_size='auto',
        learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
        random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
        early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    n = nn.fit(x, np.ravel(y))

    test_x=xall[int(0.8*len(xall)):len(xall)]
    real_y=yall[int(0.8*len(yall)):len(yall)]

    test_y = nn.predict(test_x)
    train_y = nn.predict(x)

    errortrain=[(y[c]-train_y[c]) for c in range(0,len(train_y))]
    errortest=[(real_y[c]-test_y[c]) for c in range(0,len(test_y))]

    errortestsquare=np.power(errortest,2)
    errortrainsquare=np.power(errortrain,2)

    rmstest=np.sqrt(np.mean(errortestsquare))
    rmstrain=np.sqrt(np.mean(errortrainsquare))

    plt.figure(1)

    plt.figure(1).canvas.set_window_title('NN DataTrain')

    plt.subplot(211)
    plt.plot(y,label="Meusure Force")
    plt.plot(train_y,label="Predict Force")
    plt.legend()
    plt.xlabel('Sample')
    plt.ylabel('DeltaForce(N)')
    plt.title('Compare Train (epoch = %d) Measure and Predict (RMSE : %f)' %(nn.n_iter_,rmstrain))
    bottom, top = plt.ylim()  # return the current ylim

    plt.subplot(212)
    plt.plot(real_y,label="Meusure Force")
    plt.plot(test_y,label="Predict Force")
    plt.legend()
    plt.xlabel('Sample')
    plt.ylabel('DeltaForce(N)')
    plt.title('Compare Test Measure and Predict (RMSE : %f)' %rmstest)
    plt.subplots_adjust(hspace=0.5)
    plt.ylim(bottom, top)


    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    ####################### Support vector ##########################

    X = x
    y = y

    svr_rbf = SVR(kernel='rbf', C=100,epsilon=0.01,gamma='auto')
    svr_lin = SVR(kernel='linear', C=100,epsilon=0.01,gamma='auto')
    svr_poly = SVR(kernel='poly', C=100,epsilon=0.01, degree=2,gamma='auto')

    y_rbf = svr_rbf.fit(X, np.ravel(y)).predict(X)
    y_lin = svr_lin.fit(X, np.ravel(y)).predict(X)
    y_poly = svr_poly.fit(X, np.ravel(y)).predict(X)

    y_rbf_test = svr_rbf.fit(X, np.ravel(y)).predict(test_x)
    y_lin_test = svr_lin.fit(X, np.ravel(y)).predict(test_x)
    y_poly_test = svr_poly.fit(X, np.ravel(y)).predict(test_x)

    y_rbf_error =[(y[c]-y_rbf[c]) for c in range(0,len(y_rbf))]
    y_lin_error =[(y[c]-y_lin[c]) for c in range(0,len(y_lin))]
    y_poly_error =[(y[c]-y_poly[c]) for c in range(0,len(y_poly))]


    y_rbf_error_square=np.power(y_rbf_error,2)
    y_lin_error_square=np.power(y_lin_error,2)
    y_poly_error_square=np.power(y_poly_error,2)

    y_rbf_RMSE=np.sqrt(np.mean(y_rbf_error_square))
    y_lin_RMSE=np.sqrt(np.mean(y_lin_error_square))
    y_poly_RMSE=np.sqrt(np.mean(y_poly_error_square))

    y_rbf_test_error =[(real_y[c]-y_rbf_test[c]) for c in range(0,len(y_rbf_test))]
    y_lin_test_error =[(real_y[c]-y_lin_test[c]) for c in range(0,len(y_lin_test))]
    y_poly_test_error =[(real_y[c]-y_poly_test[c]) for c in range(0,len(y_poly_test))]

    y_rbf_test_error_square=np.power(y_rbf_test_error,2)
    y_lin_test_error_square=np.power(y_lin_test_error,2)
    y_poly_test_error_square=np.power(y_poly_test_error,2)

    y_rbf_test_RMSE=np.sqrt(np.mean(y_rbf_test_error_square))
    y_lin_test_RMSE=np.sqrt(np.mean(y_lin_test_error_square))
    y_poly_test_RMSE=np.sqrt(np.mean(y_poly_test_error_square))

    lw = 2

    plt.figure(2)

    plt.figure(2).canvas.set_window_title('SVR DataTrain')

    plt.subplot(321)
    plt.plot(y, color='navy', label='data')
    plt.plot(y_rbf, color='darkorange', lw=lw, label='RBF model')
    plt.xlabel('Sample')
    plt.ylabel('DeltaForce')
    plt.title('Support Vector Regression(RBF model) RMSE : %f'%y_rbf_RMSE)
    plt.subplots_adjust(hspace=0.5)
    plt.legend()
    bottom, top = plt.ylim()  # return the current ylim

    plt.subplot(322)
    plt.plot(real_y, color='navy', label='data')
    plt.plot(y_rbf_test, color='darkorange', lw=lw, label='RBF model')
    plt.xlabel('Sample')
    plt.ylabel('DeltaForce')
    plt.title('Support Vector Regression TEST (RBF model) RMSE : %f'%y_rbf_test_RMSE)
    plt.subplots_adjust(hspace=0.5)
    plt.legend()
    plt.ylim(bottom, top)

    plt.subplot(323)
    plt.plot(y, color='c', label='data')
    plt.plot(y_lin, color='darkorange', lw=lw, label='Linear model')
    plt.xlabel('Sample')
    plt.ylabel('DeltaForce')
    plt.title('Support Vector Regression(Linear model) RMSE : %f'%y_lin_RMSE)
    plt.subplots_adjust(hspace=0.5)
    plt.legend()

    plt.subplot(324)
    plt.plot(real_y, color='c', label='data')
    plt.plot(y_lin_test, color='darkorange', lw=lw, label='Linear model')
    plt.xlabel('Sample')
    plt.ylabel('DeltaForce')
    plt.title('Support Vector Regression TEST (Linear model) RMSE : %f'%y_lin_test_RMSE)
    plt.subplots_adjust(hspace=0.5)
    plt.legend()
    plt.ylim(bottom, top)

    plt.subplot(325)
    plt.plot(y, color='cornflowerblue', label='data')
    plt.plot(y_poly, color='darkorange', lw=lw, label='Polynomial model')
    plt.xlabel('Sample')
    plt.ylabel('DeltaForce')
    plt.title('Support Vector Regression(Polynomial model) RMSE : %f'%y_poly_RMSE)
    plt.subplots_adjust(hspace=0.5)
    plt.legend()

    plt.subplot(326)
    plt.plot(real_y, color='cornflowerblue', label='data')
    plt.plot(y_poly_test, color='darkorange', lw=lw, label='Polynomial model')
    plt.xlabel('Sample')
    plt.ylabel('DeltaForce')
    plt.title('Support Vector Regression TEST (Polynomial model) RMSE : %f'%y_poly_test_RMSE)
    plt.subplots_adjust(hspace=0.5)
    plt.legend()
    plt.ylim(bottom, top)

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    plt.show()

def Traindatabase():

    databasefile = text21.text()
    fileread = '%s.xlsx' %str(databasefile)

    workbook = xlrd.open_workbook(fileread)
    sheet = workbook.sheet_by_index(0)

    title=[[sheet.cell_value(r,c) for c in range(0,sheet.ncols)] for r in range(0,1)]
    data=[[sheet.cell_value(r,c) for c in range(0,sheet.ncols)] for r in range(1,sheet.nrows)]
    data=np.asarray(data)

    # xall1=[[data[r][c] for c in range(39)] for r in range(0,(len(data)))]
    # yall1=[[data[r][c] for c in range(39,40)] for r in range(0,(len(data)))]

    xall1=[[data[r][c] for c in range(23)] for r in range(0,(len(data)))]
    yall1=[[data[r][c] for c in range(23,24)] for r in range(0,(len(data)))]

    xall=xall1
    yall=yall1

    x=xall[0:int(0.8*len(xall))]
    y=yall[0:int(0.8*len(yall))]

    ####################### Neural Network ##########################

    nn = MLPRegressor(
        hidden_layer_sizes=(20,),  activation='identity', solver='lbfgs', alpha=0.001, batch_size='auto',
        learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
        random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
        early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    n = nn.fit(x, np.ravel(y))

    test_x=xall[int(0.8*len(xall)):len(xall)]
    real_y=yall[int(0.8*len(yall)):len(yall)]

    test_y = nn.predict(test_x)
    train_y = nn.predict(x)

    errortrain=[(y[c]-train_y[c]) for c in range(0,len(train_y))]
    errortest=[(real_y[c]-test_y[c]) for c in range(0,len(test_y))]

    errortestsquare=np.power(errortest,2)
    errortrainsquare=np.power(errortrain,2)

    rmstest=np.sqrt(np.mean(errortestsquare))
    rmstrain=np.sqrt(np.mean(errortrainsquare))

    plt.figure(1)

    plt.figure(1).canvas.set_window_title('NN Database')


    plt.subplot(211)
    plt.plot(y,label="Meusure Force")
    plt.plot(train_y,label="Predict Force")
    plt.legend()
    plt.xlabel('Sample')
    plt.ylabel('DeltaForce(N)')
    plt.title('Compare Train (epoch = %d) Measure and Predict (RMSE : %f)' %(nn.n_iter_,rmstrain))
    bottom, top = plt.ylim()  # return the current ylim

    plt.subplot(212)
    plt.plot(real_y,label="Meusure Force")
    plt.plot(test_y,label="Predict Force")
    plt.legend()
    plt.xlabel('Sample')
    plt.ylabel('DeltaForce(N)')
    plt.title('Compare Test Measure and Predict (RMSE : %f)' %rmstest)
    plt.subplots_adjust(hspace=0.5)
    plt.ylim(bottom, top)

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    ####################### Support vector ##########################

    X = x
    y = y

    svr_rbf = SVR(kernel='rbf', C=100,epsilon=0.01,gamma='auto')
    svr_lin = SVR(kernel='linear', C=100,epsilon=0.01,gamma='auto')
    svr_poly = SVR(kernel='poly', C=100,epsilon=0.01, degree=2,gamma='auto')

    y_rbf = svr_rbf.fit(X, np.ravel(y)).predict(X)
    y_lin = svr_lin.fit(X, np.ravel(y)).predict(X)
    y_poly = svr_poly.fit(X, np.ravel(y)).predict(X)

    y_rbf_test = svr_rbf.fit(X, np.ravel(y)).predict(test_x)
    y_lin_test = svr_lin.fit(X, np.ravel(y)).predict(test_x)
    y_poly_test = svr_poly.fit(X, np.ravel(y)).predict(test_x)

    y_rbf_error =[(y[c]-y_rbf[c]) for c in range(0,len(y_rbf))]
    y_lin_error =[(y[c]-y_lin[c]) for c in range(0,len(y_lin))]
    y_poly_error =[(y[c]-y_poly[c]) for c in range(0,len(y_poly))]


    y_rbf_error_square=np.power(y_rbf_error,2)
    y_lin_error_square=np.power(y_lin_error,2)
    y_poly_error_square=np.power(y_poly_error,2)

    y_rbf_RMSE=np.sqrt(np.mean(y_rbf_error_square))
    y_lin_RMSE=np.sqrt(np.mean(y_lin_error_square))
    y_poly_RMSE=np.sqrt(np.mean(y_poly_error_square))

    y_rbf_test_error =[(real_y[c]-y_rbf_test[c]) for c in range(0,len(y_rbf_test))]
    y_lin_test_error =[(real_y[c]-y_lin_test[c]) for c in range(0,len(y_lin_test))]
    y_poly_test_error =[(real_y[c]-y_poly_test[c]) for c in range(0,len(y_poly_test))]

    y_rbf_test_error_square=np.power(y_rbf_test_error,2)
    y_lin_test_error_square=np.power(y_lin_test_error,2)
    y_poly_test_error_square=np.power(y_poly_test_error,2)

    y_rbf_test_RMSE=np.sqrt(np.mean(y_rbf_test_error_square))
    y_lin_test_RMSE=np.sqrt(np.mean(y_lin_test_error_square))
    y_poly_test_RMSE=np.sqrt(np.mean(y_poly_test_error_square))

    lw = 2

    plt.figure(2)

    plt.figure(2).canvas.set_window_title('SVR Database')

    plt.subplot(321)
    plt.plot(y, color='navy', label='data')
    plt.plot(y_rbf, color='darkorange', lw=lw, label='RBF model')
    plt.xlabel('Sample')
    plt.ylabel('DeltaForce')
    plt.title('Support Vector Regression(RBF model) RMSE : %f'%y_rbf_RMSE)
    plt.subplots_adjust(hspace=0.5)
    plt.legend()
    bottom, top = plt.ylim()  # return the current ylim

    plt.subplot(322)
    plt.plot(real_y, color='navy', label='data')
    plt.plot(y_rbf_test, color='darkorange', lw=lw, label='RBF model')
    plt.xlabel('Sample')
    plt.ylabel('DeltaForce')
    plt.title('Support Vector Regression TEST (RBF model) RMSE : %f'%y_rbf_test_RMSE)
    plt.subplots_adjust(hspace=0.5)
    plt.legend()
    plt.ylim(bottom, top)

    plt.subplot(323)
    plt.plot(y, color='c', label='data')
    plt.plot(y_lin, color='darkorange', lw=lw, label='Linear model')
    plt.xlabel('Sample')
    plt.ylabel('DeltaForce')
    plt.title('Support Vector Regression(Linear model) RMSE : %f'%y_lin_RMSE)
    plt.subplots_adjust(hspace=0.5)
    plt.legend()

    plt.subplot(324)
    plt.plot(real_y, color='c', label='data')
    plt.plot(y_lin_test, color='darkorange', lw=lw, label='Linear model')
    plt.xlabel('Sample')
    plt.ylabel('DeltaForce')
    plt.title('Support Vector Regression TEST (Linear model) RMSE : %f'%y_lin_test_RMSE)
    plt.subplots_adjust(hspace=0.5)
    plt.legend()
    plt.ylim(bottom, top)

    plt.subplot(325)
    plt.plot(y, color='cornflowerblue', label='data')
    plt.plot(y_poly, color='darkorange', lw=lw, label='Polynomial model')
    plt.xlabel('Sample')
    plt.ylabel('DeltaForce')
    plt.title('Support Vector Regression(Polynomial model) RMSE : %f'%y_poly_RMSE)
    plt.subplots_adjust(hspace=0.5)
    plt.legend()

    plt.subplot(326)
    plt.plot(real_y, color='cornflowerblue', label='data')
    plt.plot(y_poly_test, color='darkorange', lw=lw, label='Polynomial model')
    plt.xlabel('Sample')
    plt.ylabel('DeltaForce')
    plt.title('Support Vector Regression TEST (Polynomial model) RMSE : %f'%y_poly_test_RMSE)
    plt.subplots_adjust(hspace=0.5)
    plt.legend()
    plt.ylim(bottom, top)

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    plt.show()
def TestNNCall():

    global workbook1,worksheet,bold,curve13,curve23,curve33,curve43,curve53,k,workbook2,worksheet2,time_start
    name = text31.text()
    namennfile = '%s_ANN.xlsx' %name
    namesvmfile = '%s_SVM.xlsx' %name
    Label33.setText('Start Test record in %s.xlsx'%name)
    
    curve13.clear()
    curve13 = p13.plot(pen='w')
    data13 = np.zeros(1000)
    curve13.setData(data13) # 1,2,3,...

    curve23.clear()
    curve23 = p23.plot(pen='w')
    data23 = np.zeros(1000)
    curve23.setData(data23) # 1,2,3,...

    curve33.clear()
    curve33 = p23.plot(pen='b')
    data33 = np.zeros(1000)
    curve33.setData(data33) # 1,2,3,...

    curve43.clear()
    curve43 = p33.plot(pen='g')
    data43 = np.zeros(1000)
    curve43.setData(data43) # 1,2,3,...

    curve53.clear()
    curve53 = p33.plot(pen='m')
    data53 = np.zeros(1000)
    curve53.setData(data53) # 1,2,3,...

    name = text31.text()
    emgfile = '%s_ANN.xlsx' %name
    emgfile2 = '%s_SVM.xlsx' %name
    k = int(text4.text())

    row=1
    col=0
    row2=1
    col2=0
    time_start = np.zeros(1)

    ########## Create a workbook and add a worksheet. ##################
    workbook1 = xlsxwriter.Workbook(emgfile)
    worksheet = workbook1.add_worksheet("test")

    workbook2 = xlsxwriter.Workbook(emgfile2)
    worksheet2 = workbook2.add_worksheet("test")

    bold = workbook1.add_format({'bold': 1})
    bold2 = workbook2.add_format({'bold': 1})
    timer1.timeout.connect(TestNN)
    timer1.start()

def TestNN():
    global count2
    
    if count2==0:
        global time_start,stage,count,row,col,row2,col2,ForceFSRdata,SUMFSRdata,data,start_time,t,data23,sampling,overlap,nn,data33,data43,k,workbook1,worksheet,sheet,bold,k,svr_rbf,workbook2,worksheet2
        global ANN_time,SVM_time
        sampling = int(text2.text())
        overlap = int(text3.text())
        databasefile = text21.text()

        data = np.zeros((sampling,8))
        ForceFSRdata = np.zeros((sampling,7))
        SUMFSRdata = np.zeros((sampling,1))
        arduinodata = np.zeros(7)
        emgdata = np.zeros(8)

        fileread = '%s.xlsx' %str(databasefile)

        workbook = xlrd.open_workbook(fileread)
        sheet = workbook.sheet_by_index(0)

        title=[[sheet.cell_value(r,c) for c in range(0,sheet.ncols)] for r in range(0,1)]
        datann=[[sheet.cell_value(r,c) for c in range(0,sheet.ncols)] for r in range(1,sheet.nrows)]
        datann=np.asarray(datann)

        # k=int(datann[1][37])

        # k = int(text4.text())

        # xall1=[[datann[r][c] for c in range(39)] for r in range(0,(len(datann)))]
        # yall1=[[datann[r][c] for c in range(39,40)] for r in range(0,(len(datann)))]

        xall1=[[datann[r][c] for c in range(23)] for r in range(0,(len(datann)))]
        yall1=[[datann[r][c] for c in range(23,24)] for r in range(0,(len(datann)))]

        c = 0
        xallnew = np.zeros((len(xall1),23))
        yallnew = np.zeros(len(xall1))

        for i in range (0,len(xall1)):
            if xall1[i][21] == k :
                for t in range(0,23):
                    xallnew[c][t] = xall1[i][t]
                yallnew[c] = np.asarray(yall1[i])
                c = c+1
        print(c)
        xallnew=xallnew[0:(c)]
        yallnew=yallnew[0:(c)]
        xall1 = xallnew
        yall1 = yallnew

        xall=xall1
        yall=yall1

        x=xall[0:int(0.8*len(xall))]
        y=yall[0:int(0.8*len(yall))]

        nn = MLPRegressor(
            hidden_layer_sizes=(20,),  activation='identity', solver='lbfgs', alpha=0.001, batch_size='auto',
            learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
            random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
            early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        time_start = time.time()

        n = nn.fit(x, np.ravel(y))
        ANN_time = time.time() - time_start
        print("---ANN %s seconds ---" % ANN_time)
        print(nn.loss_)
        print(nn.n_iter_)

        
        svr_rbf = SVR(kernel='rbf', C=100,epsilon=0.01,gamma='auto')
        time_start = time.time()
        y_rbf = svr_rbf.fit(x, np.ravel(y)).predict(x)
        SVM_time = time.time() - time_start
        print("---SVM %s seconds ---" % SVM_time)

        count2=1

        print("Train success")



    ser.flush()
    arduinodata = ser.readline()
    arduinodata = arduinodata.split(",")
    arduinodata = arduinodata[1:]

    if len(arduinodata) == 7 and count < sampling:

        if hub.running and myo.connected:
            emgdata = myo.emg

        for i in range (0,7):
            ForceFSRdata[count][i] = float(arduinodata[i])
            SUMFSRdata[count][0]=ForceFSRdata[count][0]+ForceFSRdata[count][1]+ForceFSRdata[count][2]+ForceFSRdata[count][3]
        ForceFSRdata[count][4]=ForceFSRdata[count][4]*g1

        if type(emgdata) == tuple:
            for i in range (0,8):
                data[count][i] = emgdata[i]
                data[count][i] = data[count][i]/128

        data13[:-1] = data13[1:]  # shift data in the array one sample left # (see also: np.roll)
        data13[-1] = ForceFSRdata[count][4] # degree

        curve13.setData(data13)# 1,2,3,...

        count = count + 1

    # if len(arduinodata) != 7 :

    #     data13[:-1] = data13[1:]  # shift data in the array one sample left # (see also: np.roll)
    #     data13[-1] = 0 # degree
    #     data23[:-1] = data23[1:]  # shift data in the array one sample left # (see also: np.roll)
    #     data23[-1] = 0 # degree
    #     data33[:-1] = data33[1:]  # shift data in the array one sample left # (see also: np.roll)
    #     data33[-1] = 0 # degree
    #     data43[:-1] = data43[1:]  # shift data in the array one sample left # (see also: np.roll)
    #     data43[-1] = 0 # degree

    #     curve13.setData(data13)# 1,2,3,...

    #     curve23.setData(data23)# 1,2,3,...

    #     curve33.setData(data33)# 1,2,3,...

    #     curve43.setData(data43)# 1,2,3,...


    if count == sampling:

        count = sampling - overlap

        datatest = data
        forcetest = [ForceFSRdata[r][4] for r in range (0,len(ForceFSRdata))]
        fsrtest = [[ForceFSRdata[r][c] for c in range (0,4)] for r in range(0,len(ForceFSRdata))]
        velocitytest = [ForceFSRdata[r][5] for r in range (0,len(ForceFSRdata))]
        distancetest = [ForceFSRdata[r][6] for r in range (0,len(ForceFSRdata))]

        meantest = np.mean(abs(datatest), axis=0)

        datasquaretest = np.power(datatest, 2)
        rmstest = np.sqrt(np.mean(datasquaretest, axis=0))

        summeantest = np.sum(meantest)
        sumrmstest = np.sum(rmstest)

        if summeantest != 0 and sumrmstest !=0 :

            # emgmeanratetest = meantest/summeantest
            # emgrmsratetest = rmstest/sumrmstest

            meanforcetest = np.mean(forcetest)
            meanvelocitytest = np.mean(velocitytest)
            meandistancetest = np.mean(distancetest)
            meanfsrtest = np.mean(fsrtest, axis=0)

            meanvelocitytest = np.reshape(meanvelocitytest, (1,))
            meandistancetest = np.reshape(meandistancetest, (1,))
            k = np.reshape(k, (1,))

            # x = np.concatenate((emgmeanratetest,emgrmsratetest,meantest,rmstest,meanfsrtest,meanvelocitytest,k,meandistancetest), axis=0)

            x = np.concatenate((meantest,rmstest,meanfsrtest,meanvelocitytest,k,meandistancetest), axis=0)
            y = meanforcetest

            x = x.reshape(1,-1)

            nntest_y = nn.predict(x)
            svmtest_y = svr_rbf.predict(x)

            errornntest=(y-nntest_y)
            errorsvmtest=(y-svmtest_y)

            data23[:-1] = data23[1:]  # shift data in the array one sample left # (see also: np.roll)
            data23[-1] = y # degree
            data33[:-1] = data33[1:]  # shift data in the array one sample left # (see also: np.roll)
            data33[-1] = nntest_y # degree
            data43[:-1] = data43[1:]  # shift data in the array one sample left # (see also: np.roll)
            data43[-1] = abs(errornntest) # degree
            data53[:-1] = data53[1:]  # shift data in the array one sample left # (see also: np.roll)
            data53[-1] = 0 # degree

            curve23.setData(data23)# 1,2,3,...
            curve33.setData(data33)# 1,2,3,...
            curve43.setData(data43)# 1,2,3,...
            curve53.setData(data53)# 1,2,3,...

            worksheet.write  (row, col , y )
            worksheet.write  (row, col + 1, nntest_y )
            worksheet.write  (row, col + 2, abs(errornntest) )

            worksheet2.write  (row, col , y )
            worksheet2.write  (row, col + 1, svmtest_y )
            worksheet2.write  (row, col + 2, abs(errorsvmtest) )

            row += 1
        data=data[overlap:]
        data=np.pad(data, [(0,overlap),(0,0)], mode='constant', constant_values=0)
        ForceFSRdata=ForceFSRdata[overlap:]
        ForceFSRdata=np.pad(ForceFSRdata, [(0,overlap),(0,0)], mode='constant', constant_values=0)
        SUMFSRdata=SUMFSRdata[overlap:]
        SUMFSRdata=np.pad(SUMFSRdata, [(0,overlap),(0,0)], mode='constant', constant_values=0)

def Stoptest():

    global workbook1,worksheet,bold,workbook2

    timer1.stop()

    Label33.setText('Stop Test')

    workbook1.close()
    workbook2.close()
    plotgraphtest()

    print('Test Finish')

def plotgraphtest():

    row=1
    col=0
    row2=1
    col2=0
    name = text31.text()
    fileread = '%s_ANN.xlsx' %name
    filewrite = '%s_ANNgraph.xlsx' %name

    fileread2 = '%s_SVM.xlsx' %name
    filewrite2 = '%s_SVMgraph.xlsx' %name

    workbook = xlrd.open_workbook(fileread)
    workbook1 = xlsxwriter.Workbook(filewrite)

    workbook2 = xlrd.open_workbook(fileread2)
    workbook3 = xlsxwriter.Workbook(filewrite2)

    bold = workbook1.add_format({'bold': 1})
    bold2 = workbook3.add_format({'bold': 1})

    sheet = workbook.sheet_by_index(0)
    worksheet = workbook1.add_worksheet("test")

    sheet2 = workbook2.sheet_by_index(0)
    worksheet2 = workbook3.add_worksheet("test")

    data=[[sheet.cell_value(r,c) for c in range(sheet.ncols)] for r in range(1,sheet.nrows)]
    data2=[[sheet2.cell_value(r,c) for c in range(sheet2.ncols)] for r in range(1,sheet2.nrows)]

    worksheet.write('A1', 'Actual', bold)
    worksheet.write('B1', 'Estimate', bold)
    worksheet.write('C1', 'Error', bold)
    worksheet2.write('A1', 'Actual', bold2)
    worksheet2.write('B1', 'Estimate', bold2)
    worksheet2.write('C1', 'Error', bold2)

    actual = [data[r][0]  for r in range(0,(len(data)))]
    predict = [data[r][1]  for r in range(0,(len(data)))]
    error = [data[r][2]  for r in range(0,(len(data)))]
    actual2 = [data2[r][0]  for r in range(0,(len(data2)))]
    predict2 = [data2[r][1]  for r in range(0,(len(data2)))]
    error2 = [data2[r][2]  for r in range(0,(len(data2)))]

    errorsquare = np.power(error, 2)
    errorsquare2 = np.power(error2, 2)

    rmserror = np.sqrt(np.mean(errorsquare, axis=0))
    rmserror2 = np.sqrt(np.mean(errorsquare2, axis=0))    
    print rmserror

    for c in xrange(0,(sheet.nrows-1)):

        worksheet.write  (row, col , data[c][0] )
        worksheet.write  (row, col + 1, data[c][1] )
        worksheet.write  (row, col + 2, data[c][2] )

        row += 1

    for c in xrange(0,(sheet2.nrows-1)):

        worksheet2.write  (row2, col2 , data2[c][0] )
        worksheet2.write  (row2, col2 + 1, data2[c][1] )
        worksheet2.write  (row2, col2 + 2, data2[c][2] )

        row2 += 1

    chart1 = workbook1.add_chart({"type" : "line"})
    chart2 = workbook1.add_chart({"type" : "line"})

    # Add a chart title and some axis labels.
    chart1.set_title ({'name': 'ANN Compare Force Estimation'})
    chart1.set_x_axis({'name': 'Sample'})
    # chart1.set_y_axis({'name': 'Force(N)','min': (int(min(actual))-1), 'max': 0})
    chart1.set_y_axis({'name': 'Force(N)'})

    chart2.set_title ({'name': 'ANN Error of Force Estimation'})
    chart2.set_x_axis({'name': 'Sample'})
    chart2.set_y_axis({'name': 'Error(N)','min': 0, 'max': abs(int(min(actual)))})

    chart1.add_series({"values" : "test!$A$1:$A$%d"% (row),
            "line" : {"color": "blue"},
            "name" : "Actual"})

    chart1.add_series({"values" : "test!$B$1:$B$%d"% (row),
        "line" : {"color": "red"},
        "name" : "Estimate"})

    chart2.add_series({"values" : "test!$C$1:$C$%d"% (row),
        "line" : {"color": "green"},
        "name" : "Error"})

    worksheet.insert_chart("H4",chart1)
    worksheet.insert_chart("H19",chart2)
    worksheet.write('D1', 'RMSE' , bold)
    worksheet.write('D2', rmserror , bold)
    worksheet.write('E1', 'ANN_train_time' , bold)
    worksheet.write('E2', ANN_time , bold)
    worksheet.write('F1', 'SVM_train_time' , bold)
    worksheet.write('F2', SVM_time , bold)
    workbook1.close()

    chart3 = workbook3.add_chart({"type" : "line"})
    chart4 = workbook3.add_chart({"type" : "line"})

    # Add a chart title and some axis labels.
    chart3.set_title ({'name': 'SVM Compare Force Estimation'})
    chart3.set_x_axis({'name': 'Sample'})
    # chart1.set_y_axis({'name': 'Force(N)','min': (int(min(actual))-1), 'max': 0})
    chart3.set_y_axis({'name': 'Force(N)'})

    chart4.set_title ({'name': 'SVM Error of Force Estimation'})
    chart4.set_x_axis({'name': 'Sample'})
    chart4.set_y_axis({'name': 'Error(N)','min': 0, 'max': abs(int(min(actual2)))})

    chart3.add_series({"values" : "test!$A$1:$A$%d"% (row2),
            "line" : {"color": "blue"},
            "name" : "Actual"})

    chart3.add_series({"values" : "test!$B$1:$B$%d"% (row2),
        "line" : {"color": "red"},
        "name" : "Estimate"})

    chart4.add_series({"values" : "test!$C$1:$C$%d"% (row2),
        "line" : {"color": "green"},
        "name" : "Error"})

    worksheet2.insert_chart("H4",chart3)
    worksheet2.insert_chart("H19",chart4)
    worksheet2.write('D1', 'RMSE' , bold2)
    worksheet2.write('D2', rmserror2 , bold2)
    worksheet2.write('E1', 'ANN_train_time' , bold)
    worksheet2.write('E2', ANN_time , bold)
    worksheet2.write('F1', 'SVM_train_time' , bold)
    worksheet2.write('F2', SVM_time , bold)
    workbook3.close()

def makebasek():

    global workbook,bold,sheet,worksheet,k,curve13,data13
    global arduinodata,data,ForceFSRdata,count,samplingK,overlapK
    
    Label2.setText("k base saving" )

    curve13.clear()
    curve13 = p13.plot(pen='w')
    data13 = np.zeros(1000)

    name = text5.text()
    samplingK = 100
    overlapK = 1
    k = int(text4.text())

    arduinodata = np.zeros(7)
    data = np.zeros((samplingK,8))
    ForceFSRdata = np.zeros((samplingK,7))
    start_time=np.zeros(1)
    t=np.zeros(1)

    kfile = '%s_kbase.xlsx' %name


    ########## Create a workbook and add a worksheet. ##################

    workbook = xlsxwriter.Workbook(kfile)
    worksheet = workbook.add_worksheet("round%d"% (sheet))

    ###################Add a bold format to use to highlight cells.#######
    bold = workbook.add_format({'bold': 1})

    #################Write some data headers.################
    worksheet.write('A1', 'Force', bold)
    worksheet.write('B1', 'Velocity', bold)
    worksheet.write('C1', 'k', bold)



    ################# Start Timer.################
    timer2.timeout.connect(Ksavedata)
    timer2.start()

def Ksavedata():
    global count

    if count == 0:
        
        global workbook,worksheet,bold,sheet,row,col
        global ForceFSRdata,data,data13,arduinodata,g1,overlapK,samplingK

    ################# Receive value from arduino using serial comunication.################

    ser.flush()
    arduinodata = ser.readline()
    arduinodata = arduinodata.split(",")
    arduinodata = arduinodata[1:]


    ################# If lenght of value is right.################

    if len(arduinodata) == 7 and count < samplingK:


        for i in range (0,7):
            ForceFSRdata[count][i] = float(arduinodata[i])

        ForceFSRdata[count][4]=ForceFSRdata[count][4]*g1


        ################# Set value for plot curve13.################

        data13[:-1] = data13[1:]  # shift data in the array one sample left # (see also: np.roll)
        data13[-1]  = ForceFSRdata[count][4] # degree

        curve13.setData(data13) # 1,2,3,...

        count += 1


    ################# If lenght of value is wrong.################

    if len(arduinodata) != 7 :

        # ################# Set zero for plot curve13.################

        # data13[:-1] = data13[1:]  # shift data in the array one sample left # (see also: np.roll)
        # data13[-1] = 0 # degree

        # curve13.setData(data13)# 1,2,3,...

        ################# If worksheet end add new worksheet.################

        if  count > 1 and row > 300:
       
            Label2.setText("finish" )
            workbook.close()

            ########### Reset value in variable ##############

            arduinodata = np.zeros(7)
            data = np.zeros((samplingK,8))
            ForceFSRdata = np.zeros((samplingK,7))
            row = 1
            col = 0
            count = 0
            count2 = 0
            sheet=1

            print ("Finish record data")

            timer2.stop()

        else:
        	
            col = 0



    ################# Shift array of variable when count equal sampling.################

    if count == samplingK:

        count = samplingK - overlapK

        force = [ForceFSRdata[r][4] for r in range (0,len(ForceFSRdata))]
        velo = [ForceFSRdata[r][5] for r in range (0,len(ForceFSRdata))]

        meanforce=np.mean(force)
        meanvelo=np.mean(velo)

        ################# Write value from arduino into excel file.################

        worksheet.write  (row, col , meanforce )
        worksheet.write  (row, col + 1, meanvelo )
        worksheet.write  (row, col + 2, int(k) )

        row += 1

        data = data[overlapK:]
        ForceFSRdata = ForceFSRdata[overlapK:]
        
        data = np.pad(data, [(0,overlapK),(0,0)], mode='constant', constant_values=0)
        ForceFSRdata = np.pad(ForceFSRdata, [(0,overlapK),(0,0)], mode='constant', constant_values=0)

def classifyk():

    global workbook,bold,sheet,worksheet,k,curve13,data13
    global arduinodata,data,ForceFSRdata,count,samplingK,overlapK,nn_K
    
    Label2.setText("push 1 cycle for k classify" )

    curve13.clear()
    curve13 = p13.plot(pen='w')

    data13 = np.zeros(1000)
    curve13.setData(data13) # 1,2,3,...

    samplingK = 100
    overlapK = 1
    # k = int(text4.text())

    arduinodata = np.zeros(7)
    data = np.zeros((samplingK,8))
    ForceFSRdata = np.zeros((samplingK,7))
    start_time=np.zeros(1)
    t=np.zeros(1)


    name = text5.text()

    kfile = '%s.xlsx' %name

    workbook = xlrd.open_workbook(kfile)
    sheet = workbook.sheet_by_index(0)

    title=[[sheet.cell_value(r,c) for c in range(0,sheet.ncols)] for r in range(0,1)]
    datann=[[sheet.cell_value(r,c) for c in range(0,sheet.ncols)] for r in range(1,sheet.nrows)]
    datann=np.asarray(datann)

    xall1=[[float(datann[r][c]) for c in range(2)] for r in range(0,(len(datann)))]
    yall1=[[float(datann[r][c]) for c in range(2,3)] for r in range(0,(len(datann)))]

    nn_K = MLPClassifier(solver='lbfgs', alpha=1e-5,
                hidden_layer_sizes=(5, 2), random_state=1)

    nn_K.fit(xall1, np.ravel(yall1))    




    ################# Start Timer.################
    timer2.timeout.connect(Krecordclassify)
    timer2.start()

def Krecordclassify():
    global count

    if count == 0:
        
        global workbook,worksheet,bold,sheet,row,col
        global ForceFSRdata,data,data13,arduinodata,g1,overlapK,samplingK,nn_K,k

        k = np.zeros(3000)
        row = 1
        col = 0

    ################# Receive value from arduino using serial comunication.################

    ser.flush()
    arduinodata = ser.readline()
    arduinodata = arduinodata.split(",")
    arduinodata = arduinodata[1:]


    ################# If lenght of value is right.################

    if len(arduinodata) == 7 and count < samplingK:


        for i in range (0,7):
            ForceFSRdata[count][i] = float(arduinodata[i])

        ForceFSRdata[count][4] = ForceFSRdata[count][4]*g1

        ################# Set value for plot curve13.################

        data13[:-1] = data13[1:]  # shift data in the array one sample left # (see also: np.roll)
        data13[-1]  = ForceFSRdata[count][4] # degree

        curve13.setData(data13) # 1,2,3,...

        count += 1


    ################# If lenght of value is wrong.################

    if len(arduinodata) != 7 :

        # ################# Set zero for plot curve13.################

        # data13[:-1] = data13[1:]  # shift data in the array one sample left # (see also: np.roll)
        # data13[-1] = 0 # degree

        # curve13.setData(data13)# 1,2,3,...

        ################# If worksheet end add new worksheet.################

        if  count > 1 and row > 200:

            # sheet += 1

            # worksheet = workbook.add_worksheet("round%d"% (sheet))
            # print ("round%d"% (sheet))

            # ################# Write title for new worksheet.################
            # worksheet.write('A1', 'Velocity', bold)
            # worksheet.write('B1', 'Force', bold)
            # worksheet.write('C1', 'k', bold)
            # force = [ForceFSRdata[r][4] for r in range (0,len(ForceFSRdata))]
            # velo = [ForceFSRdata[r][5] for r in range (0,len(ForceFSRdata))]

            # force = np.reshape(force, (1,))
            # velo = np.reshape(velo, (1,))
            # for i in range (0,len(force)):

            #     x[i] = np.concatenate((velo[i],force[i]), axis=0)
            # x = np.concatenate(np.asarray(velo),np.asarray(force))

            # x = [[velo,force]]
            # print x

            # x=[[float(ForceFSRdata[r][c]) for c in range(4,6)] for r in range(0,len(ForceFSRdata))]

            # k = nn_K.predict(x)

            # print k            
            # k = int(np.mean(k[1:199]))
            k = np.mean(k[1:199])
            print ('Predict K :' , k)
            k = int(k)
            text4.setText(str(k))

            # Label2.setText("Classify K = ",k  )

            ########### Reset value in variable ##############

            arduinodata = np.zeros(7)
            data = np.zeros((samplingK,8))
            ForceFSRdata = np.zeros((samplingK,7))
            row = 1
            col = 0
            count = 0
            count2 = 0
            sheet = 1
            data13 = np.zeros(1000)

            print ("Finish Classify")

            timer2.stop()

            # row = 1
            # col = 0
            # count = 0

        else:

            # row = 1
            col = 0



    ################# Shift array of variable when count equal sampling.################

    if count == samplingK:

        count = samplingK - overlapK

        x=[[float(ForceFSRdata[r][c]) for c in range(4,6)] for r in range(0,len(ForceFSRdata))]

        meanx = np.mean(x, axis=0)

        k[row] = nn_K.predict([meanx])
        # print k[row]

        # force = [ForceFSRdata[r][4] for r in range (0,len(ForceFSRdata))]
        # velo = [ForceFSRdata[r][5] for r in range (0,len(ForceFSRdata))]

        # meanforce[row]=np.mean(force)
        # meanvelo[row]=np.mean(velo)

        row += 1

        data = data[overlapK:]
        ForceFSRdata = ForceFSRdata[overlapK:]
        
        data = np.pad(data, [(0,overlapK),(0,0)], mode='constant', constant_values=0)
        ForceFSRdata = np.pad(ForceFSRdata, [(0,overlapK),(0,0)], mode='constant', constant_values=0)


btn0.clicked.connect(start_TrainRecording)
btn1.clicked.connect(stop)
btn2.clicked.connect(exit)
btn3.clicked.connect(makebasek)
btn4.clicked.connect(classifyk)
btn20.clicked.connect(Traincombinedatabase)
btn21.clicked.connect(traintobase)
btn22.clicked.connect(Traindatabase)
btn23.clicked.connect(Traindatatrain)
btn24.clicked.connect(newdatabase)
btn30.clicked.connect(TestNNCall)
btn31.clicked.connect(Stoptest)

if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
    hub.shutdown()






































    
    


                







        
















