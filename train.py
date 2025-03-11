from visdom import Visdom
from torchmetrics.functional import auroc
from utils import pre_dataset,suffledata
from model import GCNNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.loader import  DataLoader
import torch
from sklearn.metrics import roc_auc_score
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

BATCH_SIZE = 16
LR = 1e-3

id, smiles, label = pre_dataset("./RIPK1-class1000-1.csv")

train_dataset, test_dataset, val_dataset = suffledata(smiles, label)

trianloader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
testloader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
valloader = DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)




model = GCNNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# 训练过程
num_epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)


def train(model, train_loader, criterion, optimizer):

    model.train()
    total_loss = 0.0
    acctotal = 0
    auctotal = 0

    for data in train_loader:
        data = data[0]
        data = data.to(device)
        optimizer.zero_grad()
        outputs = model(data)



        loss = criterion(outputs, data.y.unsqueeze(1).float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        for index,i in enumerate(outputs):
            if (i >= 0.5).float() == data.y.unsqueeze(1).float()[index]:
                acctotal = acctotal+1


        auctotal = auroc(outputs, data.y.unsqueeze(1) ,task='binary') + auctotal


        # 将预测结果和真实标签添加到列表中

    avg_loss = total_loss / len(train_loader)
    acc = acctotal / len(train_loader.dataset)
    auc = auctotal / (len(train_loader.dataset)/BATCH_SIZE)



    return avg_loss, acc, auc

def test(test_loader,criterion):
    model.eval()
    total_loss = 0.0
    acctotal = 0
    auctotal = 0

    df = pd.DataFrame(["ID", "Score"])

    with torch.no_grad():

        for batch in test_loader:
            batch = batch[0]
            batch.to(device)
            outputs = model(batch)

            value = outputs.tolist()
            ids = batch.y.tolist()
            temp_list = [[x, y] for x, y in zip(ids, value)]
            new_row = pd.Series(temp_list)
            df = pd.concat([df, new_row], ignore_index=True)

            loss = criterion(outputs, batch.y.unsqueeze(1).float())
            total_loss += loss.item()

            for index, i in enumerate(outputs):
                if (i >= 0.5).float() == batch.y.unsqueeze(1).float()[index]:
                    acctotal = acctotal + 1

            auctotal = auroc(outputs, batch.y.unsqueeze(1), task='binary') + auctotal
    loss = total_loss / len(test_loader)
    accuracy = acctotal / len(test_loader.dataset)
    auc = auctotal / (len(test_loader.dataset) / BATCH_SIZE)

    df.to_csv('testoutput.csv', index=False)

    return loss,accuracy,auc

def val(test_loader,criterion):
    model.eval()
    total_loss = 0.0
    acctotal = 0
    auctotal = 0
    df = pd.DataFrame(["ID", "Score"])

    with torch.no_grad():
        for batch in test_loader:
            batch = batch[0]
            batch.to(device)
            outputs = model(batch)
            value = outputs.tolist()
            ids = batch.y.tolist()
            temp_list = [[x, y] for x, y in zip(ids, value)]
            new_row = pd.Series(temp_list)
            df = pd.concat([df,new_row], ignore_index=True)

            for index, i in enumerate(outputs):
                if (i >= 0.5).float() == batch.y.unsqueeze(1).float()[index]:
                    acctotal = acctotal + 1
            auctotal = auroc(outputs, batch.y.unsqueeze(1), task='binary') + auctotal
        accuracy = acctotal / len(test_loader.dataset)
        auc = auctotal / (len(test_loader.dataset) / BATCH_SIZE)

        # 绘制混淆矩阵图表

    df.to_csv('valoutput.csv', index=False)
    torch.save(model.state_dict(), "model.pkl")
    return accuracy, auc



loss1 = []
acc1 = []
loss2 =[]
acc2 = []
auc1 = []
auc2 = []



for i in range(num_epochs):

    avg_loss, accuracy, auc = train(model,trianloader,criterion,optimizer)
    tloss, tacc, tauc = test(testloader,criterion)

    auc = auc.cpu()
    tauc = tauc.cpu()

    loss1.append(avg_loss)
    acc1.append(accuracy)
    loss2.append(tloss)
    acc2.append(tacc)
    auc1.append(auc)
    auc2.append(tauc)

    print("loss:{} acc: {} auc {} tloss {} tacc {} tauc {} epoch: {}".format(avg_loss, accuracy, auc, tloss, tacc, tauc, i))
    #print("loss:{} acc: {}  epoch: {}".format(avg_loss, accuracy, i))


a,b = val(valloader,criterion)
print("loss:{} acc: {} tloss {} tacc {} epoch: {}".format(avg_loss, accuracy,tloss, tacc, i))
print("vacc: {} vauc:{} ".format(a,b))
