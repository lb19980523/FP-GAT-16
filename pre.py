import os
import sys
import time
import torch
import pandas as pd
from utils import CustomDataset_pre, pre_dataset_pre
from torch_geometric.loader import  DataLoader
BATCH_SIZE = 16


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('model.pkl')
model.eval()
model.to(device)

datadir = 'E:/CodeNotebook/RIPK1VS/Clean/'
files = os.listdir(datadir)
flag = 0

start_time = time.time()
for name in files:
    flag += 1
    print("predata: {} total: {}/{}".format(name,flag,len(files)))
    id, smiles = pre_dataset_pre(datadir+name)
    dataset = CustomDataset_pre(smiles, id)
    preloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    df = pd.DataFrame(["ID","Score"])
    with torch.no_grad():
        tflag = 0
        for data in preloader:
            tflag += 1
            data = data[0]
            data = data.to(device)
            outputs = model(data)
            value = outputs.tolist()
            ids = data.id.tolist()
            temp_list = [[x, y] for x, y in zip(ids, value)]
            new_row = pd.Series(temp_list)
            df = pd.concat([df,new_row], ignore_index=True)
            flush = "▓" * int(tflag/len(preloader)*100)
            need_do = "-" * (100-int(tflag/len(preloader)*100))
            progress = (tflag / len(preloader)) * 100

            print("\rwork: {}/{} {:^3.0f}%[{}->{}]".format(tflag,len(preloader),progress, flush, need_do,), end="")
            sys.stdout.flush()

        df.to_csv(name[:-4]+'output.csv', index=False)
        # 删除 DataFrame 对象
        del df

end_time = time.time()

execution_time = end_time - start_time
# # 打印程序运行时间
# print(f"程序运行时间为: {execution_time}




