import torch
from torch.autograd import Variable
from torch import nn
import numpy as np
import random
import math
import matplotlib.pyplot as plt

torch.manual_seed(1)

# 超参数
TIME_STEP = 10
INPUT_SIZE = 3
OUTPUT_SIZE1 = 18
OUTPUT_SIZE2 = 2
HIDDEN_SIZE = 128
NUM_LAYER = 1
LR = 0.0001
TOTAL_RUM = 100
ONE_SEQ = 100
MAXR = 50


def unwrap(preIn, nowIn):
    diff = nowIn - preIn
    if(diff > 180):
        res = nowIn
        while(res - preIn > 180):
            res = res - 360
    elif(diff < -180):
        res = nowIn
        while(res - preIn < -180):
            res = res + 360
    else:
        res = nowIn
    return res


def actionToSpeed(action):
    res =[] # 0位为x轴速度， 1位为y轴速度
    if(action == 1):
        res = [0,1]
    elif(action == 2):
        res = [0,-1]
    elif(action == -1):
        res = [-1,0]
    elif(action == -2):
        res = [1,0]
    else:
        res = [0,0]
    return res

def disNormalize(disIn):
    return (disIn-(math.log(MAXR)-0.5))/(2*MAXR*math.log(MAXR)/3-2*MAXR/9)

def normalizeLst(inList, mode):
    res = []
    if mode == 'MaxMin':
        dataMin, dataMax = inList[0], inList[0]
        for data in inList:
            if data < dataMin:
                dataMin = data
            if data > dataMax:
                dataMax = data
        for data in inList:
            temp = (data-dataMin)/(dataMax-dataMin)
            res.append(temp)
        return dataMin, dataMax, res
    elif mode == 'zeroMean':
        if len(inList):
            summary = 0
            for data in inList:
                summary = summary + data
            Mean = summary/len(inList)
            summary = 0
            for data in inList:
                summary = summary + (data-Mean)*(data-Mean)
            sigma = math.sqrt(summary/len(inList))
            for data in inList:
                if sigma > 0:
                    temp = (data-Mean)/sigma
                else:
                    temp = 0
                res.append(temp)
        else:
            res = []
            Mean = 0
            sigma = 0
        return res, Mean, sigma
    elif mode == 'dis':
        for data in inList:
            temp = (data-(math.log(MAXR)-0.5))/(2*MAXR*math.log(MAXR)/3-2*MAXR/9)
            res.append(temp)
        return res
    else:
        return res
                

class RNN(nn.Module):
    def __init__(self, inputSize, hiddenSize, numLayer, outputSize, learningRate):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size = inputSize,
            hidden_size = hiddenSize,
            num_layers=numLayer,
            batch_first=True,
        )
        self.out = nn.Linear(hiddenSize,outputSize)
        #self.out = nn.Softmax(dim=1)
    
    def forward(self, x, h_state):
        # x        shape (batch,    time_step,  input_size)
        # h_staten shape (n_layers,     batch, hidden_size)
        # r_out    shape (batch,    time_step, hidden_size)
        r_out, (h_staten, h_statec) = self.rnn(x, h_state)
        outs = []
        for time_step in range(r_out.size(1)): # 按照time_step将输出放入
            outs.append(self.out(r_out[:, time_step, :]))
        #outs = self.out(outs)
        outs = torch.stack(outs, dim=1)
        outs = torch.transpose(outs,1,2)
        return outs, h_staten, h_statec
    
    def learn(self, prediction, y):
        loss = self.loss_func(prediction, angle)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
        

class env(object):
    def __init__(self, targetX: int = 0,targetY: int = 0, x_drone: int = 10, y_drone: int = 10):
        self.target = {'x': targetX, 'y': targetY}
        self.bron = {'x': x_drone, 'y': y_drone}
        self.action_space = [1, 2, -1, -2] # 1 - up, 2 - down, -1 - left, -2 - right, other- no action
        self.drone = {'x': x_drone, 'y': y_drone}
        
        self.droneTotarget = {'x': x_drone - targetX, 'y': y_drone - targetY}    
        self.distance = math.sqrt(self.droneTotarget['x'] * self.droneTotarget['x'] + self.droneTotarget['y'] * self.droneTotarget['y'])
        if self.distance > 0:
            if (self.droneTotarget['y'] >= 0):
                self.angle = math.acos(self.droneTotarget['x']/self.distance) / np.pi * 180.0
            else:
                self.angle = 360.0 - math.acos(self.droneTotarget['x']/self.distance) / np.pi * 180.0
        else:
            self.angle = 0
        if self.angle > 180:
            self.angle = self.angle - 360
        if self.angle < -180:
            self.angle = self.angle + 360

    def xy_to_state(self):
        self.droneTotarget = {'x': self.drone['x'] - self.target['x'], 'y': self.drone['y'] - self.target['y']}
        self.distance = math.sqrt( self.droneTotarget['x'] * self.droneTotarget['x'] + self.droneTotarget['y'] * self.droneTotarget['y'] )
        if self.distance > 0:
            if (self.droneTotarget['y'] >= 0):
                self.angle = math.acos(self.droneTotarget['x']/self.distance) / np.pi * 180.0
            else:
                self.angle = 360.0 - math.acos(self.droneTotarget['x']/self.distance) / np.pi * 180.0
        else:
            self.angle = 0
        if self.angle > 180:
            self.angle = self.angle - 360
        if self.angle < -180:
            self.angle = self.angle + 360
    
    def step(self, action):
        if action == 1:
            self.drone['y'] = self.drone['y'] + 1
            self.xy_to_state()
        elif action == 2:
            self.drone['y'] = self.drone['y'] - 1
            self.xy_to_state()
        elif action == -1:
            self.drone['x'] = self.drone['x'] - 1
            self.xy_to_state()
        elif action == -2:
            self.drone['x'] = self.drone['x'] + 1
            self.xy_to_state()
        else:
            print('Warning: Action is not in actionspace!')
            pass
    
    def giveAction(self):
        return random.choice(self.action_space)
    
    def giveData(self):
        return float(self.distance), float(self.angle), self.droneTotarget
    
    def reset(self):
        self.drone = self.bron
        self.xy_to_state()




rnn1 = RNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYER, OUTPUT_SIZE1, LR)
rnn1 = rnn1.train()
print(rnn1)

optimizer = torch.optim.Adam(rnn1.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()



# create world
targetX, targetY = 5, 5

# 储存数据
lossRe = []
angleAll = []
angleAll2 = []
anglePre = []



actionDictionary = {'up':1, 'down':2, 'left':-1, 'right':-2, 'others':0} #1 - up, 2 - down, -1 - left, -2 - right, other- no action
'''
actionSpace = [['up', 'left', 'up', 'left', 'up'],         ['right','right','down','down','left'], 
               ['up', 'right', 'up', 'right', 'up'],       ['right','right','up','up','left'], 
               ['down', 'right', 'down', 'right', 'down'], ['left','left','down','down','right'],
               ['down', 'left', 'down', 'left', 'down'],   ['left','left','up','up','right']]
'''
actionSpace = [['up', 'up', 'up', 'left', 'left', 'left'],         ['right','right','right','down','down','down'], 
               ['up', 'up', 'up', 'right', 'right', 'right'],       ['right','right','right','up','up','up'], 
               ['down', 'down', 'down', 'right', 'right', 'right'], ['left','left','left','down','down','down'],
               ['down', 'down', 'down', 'left', 'left', 'left'],   ['left','left','left','up','up','up']]
'''
actionSpace = [['up', 'up',],  ['up', 'up'], ['right','right'], ['right','right'], 
               ['down', 'down'], ['down', 'down'], ['left','left'], ['left','left']]
'''


softMaxLayer = nn.Softmax(dim=0)

plt.figure(1, figsize=(12, 5))
plt.ion()
for batchNum in range(TOTAL_RUM):
    h_state1 = None

    nowStep = 100
    x_drone, y_drone = (random.randint(-MAXR,MAXR), random.randint(-MAXR,MAXR))
    env_test = env(targetX, targetY, x_drone, y_drone)
    actionChoose = []

    for step in range(ONE_SEQ):
        dis = []
        speedX = []
        speedY = []
        angle = []
        angleAll = []
        print("now step:{}".format(step))
        start, end = step*TIME_STEP, step*TIME_STEP + TIME_STEP - 1
        steps = np.linspace(start, end, TIME_STEP, dtype = np.float32, endpoint = True)
        for _ in range(TIME_STEP):
            if nowStep >= len(actionChoose)-1:
                nowStep = 0
                n = random.randint(0,7)
                actionChoose = actionSpace[n]
            actionTemp = actionChoose[nowStep]
            actionTemp = actionDictionary[actionTemp]
            nowStep = nowStep + 1

            env_test.step(actionTemp)
            disTemp, angleTemp, droneTemp = env_test.giveData()
            #disTemp = disNormalize(math.log(disTemp+1))
            dis.append(disTemp)
            speedTemp = actionToSpeed(actionTemp)
            speedX.append(speedTemp[0])
            speedY.append(speedTemp[1])
            if(angleTemp < 0):
                angleTemp = angleTemp+360
            angle.append(angleTemp*(OUTPUT_SIZE1/360))
            angleAll.append( int(angleTemp*(OUTPUT_SIZE1/360)) )

        x_np = np.array([dis, speedX,speedY])
        x_np = np.transpose(x_np)
        y1_np = np.array(angle)

        x = Variable(torch.from_numpy(x_np[np.newaxis, :, :]))
        y1 = Variable(torch.from_numpy(y1_np[np.newaxis,:]))

        prediction1, h_staten1, h_statec1 = rnn1(x.float(), None)
        h_state1 = (Variable(h_staten1.data), Variable(h_statec1.data))
        loss = loss_func(prediction1, y1.long())
    
        lossRe.append(loss.data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        anglePre = []
        for time_step in range(prediction1.size(2)):
            temp = 0
            nowPt = 0
            startData = prediction1[0,0,time_step]
            for data in prediction1[0,:,time_step]:
                if data > startData:
                    startData = start
                    temp = nowPt
                nowPt = nowPt + 1
            anglePre.append(temp)
    

        #angleForDraw = np.array(angleAll)
        #anglePreForDraw = np.array(anglePre)
        angleForDraw = []
        print("target",y1.long())
        #print("prediction",prediction1[0,:,-1].data)
        anglePreForDraw = softMaxLayer(prediction1[0,:,-1])
        print("softmax",anglePreForDraw.data)
        for _ in range(OUTPUT_SIZE1):
            angleForDraw.append(0)
        angleForDraw[angleAll[-1]] = 1
        #plt.scatter(steps, angleForDraw,1, c='r') 
        #plt.scatter(steps, anglePreForDraw,1, c='b')
        plt.clf()
        plt.plot(np.linspace(0,OUTPUT_SIZE1-1,OUTPUT_SIZE1,dtype = np.float32, endpoint = True),anglePreForDraw.data ,'b-')
        plt.plot(np.linspace(0,OUTPUT_SIZE1-1,OUTPUT_SIZE1,dtype = np.float32, endpoint = True),angleForDraw ,'r-')
        plt.draw(); plt.pause(0.01)
plt.ioff()
plt.show()

plt.figure(1, figsize=(12,5))
plt.plot(lossRe)
plt.show()
plt.close()


torch.save(rnn1.state_dict(), './rnn1AngleDiscrete.pth') # 保存参数
# torch.save(rnn2.state_dict(), './rnn2Loc.pth') # 保存参数

####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################

#rnn1.eval()

# create world
targetX, targetY = 5, 5

# 储存数据
lossRe = []
angleAll = []
anglePre = []

plt.figure(1, figsize=(12, 5))
plt.ion()

nowStep = 100
x_drone, y_drone = (random.randint(-MAXR,MAXR), random.randint(-MAXR,MAXR))
env_test = env(targetX, targetY, x_drone, y_drone)
actionChoose = []

for step in range(100):
    dis = []
    speedX = []
    speedY = []
    angle = []
    angleAll = []
    

    print("now step:{}".format(step))


    start, end = step*TIME_STEP, step*TIME_STEP + TIME_STEP - 1
    steps = np.linspace(start, end, TIME_STEP, dtype = np.float32, endpoint = True)
    for _ in range(TIME_STEP):
        if nowStep >= len(actionChoose)-1:
            nowStep = 0
            n = random.randint(0,7)
            actionChoose = actionSpace[n]
        actionTemp = actionChoose[nowStep]
        actionTemp = actionDictionary[actionTemp]
        nowStep = nowStep + 1

        env_test.step(actionTemp)
        disTemp, angleTemp, droneTemp = env_test.giveData()
        #disTemp = disNormalize(math.log(disTemp+1))
        dis.append(disTemp)
        speedTemp = actionToSpeed(actionTemp)
        speedX.append(speedTemp[0])
        speedY.append(speedTemp[1])
        if(angleTemp < 0):
            angleTemp = angleTemp+360
        angle.append(angleTemp*(OUTPUT_SIZE1/360))
        angleAll.append( int(angleTemp*(OUTPUT_SIZE1/360)) )

    x_np = np.array([dis, speedX,speedY])
    x_np = np.transpose(x_np)
    y1_np = np.array(angle)

    x = Variable(torch.from_numpy(x_np[np.newaxis, :, :]))
    y1 = Variable(torch.from_numpy(y1_np[np.newaxis,:]))

    prediction1, h_staten1, h_statec1 = rnn1(x.float(), None)
    h_state1 = (Variable(h_staten1.data), Variable(h_statec1.data))
    loss = loss_func(prediction1, y1.long())


    anglePre = []
    for time_step in range(prediction1.size(2)):
        temp = 0
        nowPt = 0
        startData = prediction1[0,0,time_step]
        for data in prediction1[0,:,time_step]:
            if data > startData:
                startData = start
                temp = nowPt
            nowPt = nowPt + 1
        anglePre.append(temp)
    print(prediction1[0,:,-1].data)
    print(y1.long())

    angleForDraw = []
    print("target",y1.long())
    #print("prediction",prediction1[0,:,-1].data)
    anglePreForDraw = softMaxLayer(prediction1[0,:,-1])
    print("softmax",anglePreForDraw.data)
    for _ in range(OUTPUT_SIZE1):
        angleForDraw.append(0)
    angleForDraw[angleAll[-1]] = 1
    #plt.scatter(steps, angleForDraw,1, c='r') 
    #plt.scatter(steps, anglePreForDraw,1, c='b')
    plt.clf()
    plt.plot(np.linspace(0,OUTPUT_SIZE1-1,OUTPUT_SIZE1,dtype = np.float32, endpoint = True),anglePreForDraw.data ,'b-')
    plt.plot(np.linspace(0,OUTPUT_SIZE1-1,OUTPUT_SIZE1,dtype = np.float32, endpoint = True),angleForDraw ,'r-')
    plt.draw(); plt.pause(0.01)
plt.ioff()
plt.show()