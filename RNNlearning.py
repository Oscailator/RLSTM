import torch
from torch.autograd import Variable
from torch import nn
import numpy as np
import random
import math
import matplotlib.pyplot as plt

torch.manual_seed(1)

# 超参数
TIME_STEP = 5
UPDATE_STEP = 2
INPUT_SIZE = 3
OUTPUT_SIZE1 = 1
OUTPUT_SIZE2 = 2
HIDDEN_SIZE = 32
NUM_LAYER = 1
LR = 0.5
TOTAL_RUM = 100
MAXR = 200


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
        self.optimizer = torch.optim.Adam(self.rnn.parameters(), lr=learningRate)
        self.loss_func = nn.MSELoss()
    
    def forward(self, x, h_state):
        # x        shape (batch,    time_step,  input_size)
        # h_staten shape (n_layers,     batch, hidden_size)
        # r_out    shape (batch,    time_step, hidden_size)
        r_out, (h_staten, h_statec) = self.rnn(x, h_state)
        outs = []
        for time_step in range(r_out.size(1)): # 按照time_step将输出放入
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_staten, h_statec
    
    def learn(self, prediction, y):
        loss = self.loss_func(prediction, y.float())
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
# rnn1 = rnn1.train()
print(rnn1)
rnn2 = RNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYER, OUTPUT_SIZE2, LR)
rnn2 = rnn2.train()
print(rnn2)

# optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
# loss_func = nn.MSELoss()
h_state1 = None
h_state2 = None

# create world
targetX, targetY = 5, 5
x_drone, y_drone = (random.randint(-100,100), random.randint(-100,100))
initDis = float(math.sqrt( (x_drone - targetX)*(x_drone - targetX) + (y_drone - targetY)*(y_drone - targetY) ))

env_test = env(targetX, targetY, x_drone, y_drone)

# 储存数据
dis = []
action = []
speedX = []
speedY = []
angle = []
droneLocX = []
droneLocY = []
lossRe = []
lossRe2 = []
angleAll = []
anglePre = []
anglePre2 = []
disALL = []
h_stateRe = []

DataRecordDis = []
DataRecordAngle = []
DataRecordX = []
DataRecordY = []
DataRecordAction = []

def unwrap(preIn, nowIn):
    diff = nowIn - preIn
    if(diff > 180):
        temp = diff//360
        res = nowIn - 360*(temp+1)
    elif(diff < -180):
        temp = (preIn - nowIn)//360
        res = nowIn + 360*(temp+1)
    else:
        res = nowIn
    return res


# 用来控制无人机怎么走
def pathControl(step:int = 0):
    n = random.randint(0,3)
    if n == 0:
        n = 1
    elif n == 1:
        n = -1
    elif n == 2:
        n = 2
    else:
        n = -2
    return n
    

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



def oneRun(stepNumAction):
    # 进行一次操作，然后保存数据
    actionTemp = pathControl(stepNumAction)
    env_test.step(actionTemp)
    disTemp, angleTemp, droneTemp = env_test.giveData()
    dis.append(disTemp)
    speedTemp = actionToSpeed(actionTemp)
    speedX.append(speedTemp[0])
    speedY.append(speedTemp[1])
    action.append(actionTemp)
    if(len(angle)):
        angleTemp = unwrap(angle[-1], angleTemp)
        angle.append(angleTemp)
    else:
        angle.append(angleTemp)
    droneLocX.append( float(droneTemp['x']) )
    droneLocY.append( float(droneTemp['y']) )
    DataRecordDis.append(disTemp)
    DataRecordAngle.append(angleTemp)
    DataRecordX.append( float(droneTemp['x']) )
    DataRecordY.append( float(droneTemp['y']) )
    DataRecordAction.append(actionTemp)

# 定义step数
stepNumAction = 0

for _ in range(TIME_STEP):
    oneRun(stepNumAction)
    stepNumAction = stepNumAction + 1

plt.figure(1, figsize=(12, 5))
plt.ion()

for step in range(TOTAL_RUM):
    print("now step:{}".format(step))
    start, end = step*UPDATE_STEP, step*UPDATE_STEP + TIME_STEP - 1
    if UPDATE_STEP > 1:
        steps = np.linspace(start, end, TIME_STEP, dtype = np.float32, endpoint = True)
    else:
        steps = start
    
    for _ in range(UPDATE_STEP):
        oneRun(stepNumAction)
        stepNumAction = stepNumAction + 1
        del dis[0]
        del action[0]
        del speedX[0]
        del speedY[0]
        del droneLocX[0]
        del droneLocY[0]
        del angle[0]
    '''
    disMin,disMax = dis[0],dis[0]
    disTemp = []
    for data in dis:
        if data > disMax:
            disMax = data
        if data < disMin:
            disMin = data
    for data in dis:
        temp = (data - disMin)/(disMax - disMin)
        disTemp.append(temp)
    '''
    disTemp = []
    for data in dis:
        temp = (data-2*MAXR)/(MAXR*MAXR/2)
        disTemp.append(temp)


    x_np = np.array([disTemp, speedX,speedY])
    x_np = np.transpose(x_np)
    y1_np = np.array(angle)
    y1_np = np.transpose(y1_np)
    y2_np = np.array([droneLocX, droneLocY])
    y2_np = np.transpose(y2_np)

    x = Variable(torch.from_numpy(x_np[np.newaxis, :, :]))
    y1 = Variable(torch.from_numpy(y1_np[np.newaxis, :, np.newaxis]))
    y2 = Variable(torch.from_numpy(y2_np[np.newaxis, :, :]))

    prediction1, h_staten1, h_statec1 = rnn1(x.float(), h_state1)
    h_state1 = (Variable(h_staten1.data), Variable(h_statec1.data))

    prediction2, h_staten2, h_statec2 = rnn1(x.float(), h_state2)
    h_state2 = (Variable(h_staten2.data), Variable(h_statec2.data))
    # h_state2 = Variable(h_state2.data)

    loss1 = rnn1.learn(prediction1, y1)
    lossRe.append(loss1.data)
    loss2 = rnn2.learn(prediction2, y2)
    lossRe2.append(loss2.data)

    # plt.plot(droneLocX, droneLocY, 'r-')
    temp = prediction1.data.numpy().flatten()
    plt.plot(steps, y1_np, 'r-') 
    # plt.plot(steps, temp, 'b-')
    plt.draw(); plt.pause(0.05)

plt.ioff()
plt.show()

plt.figure(1, figsize=(12,5))
plt.plot(lossRe)
plt.show()
plt.close()

plt.figure(1, figsize=(12,5))
plt.plot(lossRe2)
plt.show()
plt.close()

torch.save(rnn1.state_dict(), './rnn1Angle.pth') # 保存参数
torch.save(rnn2.state_dict(), './rnn2Loc.pth') # 保存参数
# rnn.eval()

