import random
from torch.utils.data import random_split

import pandas as pd
import numpy as np
import torch
from rdkit import Chem,DataStructs
from rdkit.Chem import AllChem as Chem
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader, Subset
from torch_geometric import data as DATA
from torch_geometric.data import InMemoryDataset, DataLoader
def pre_dataset (path):

    "获取原始数据"
    data = pd.read_csv(path)
    ID = data["ID"]
    SMILES = data["SMILES"]
    ACT = data["class"]
    dict = {}

    smiles = []
    label = []
    id = []
    for i in SMILES:
        smiles.append(str(i))
    for j in ACT:
        label.append(int(j))
    for k in ID:
        id.append(k)

    return id,smiles,label


def pre_dataset_pre (path):

    "获取原始数据"
    data = pd.read_csv(path)
    ID = data["ID"]
    SMILES = data["SMILES"]

    smiles = []
    id = []
    for i in SMILES:
        smiles.append(str(i))
    for k in ID:
        id.append(k)

    return id,smiles





def onehot_encoding(x, allowable_set):

    "onehot编码"

    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: int(x == s), allowable_set))

def get_molecule_info(smiles):

    "输入smiles，输出分子节点数，节点特征，边图，边特征"

    m = Chem.MolFromSmiles(smiles)

    '''获取分子节点信息'''
    atomnum = m.GetNumAtoms()
    atomnode = []
    for i in range(atomnum):
        atom = m.GetAtomWithIdx(i)
        aa = np.array(onehot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                      onehot_encoding(atom.GetSymbol(), ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H']) +
                      onehot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                      onehot_encoding(atom.GetHybridization(), ['SP1', 'SP2', 'SP3']) +
                      [int(atom.GetIsAromatic())] +
                      [int(atom.IsInRing())] +
                      onehot_encoding(atom.GetTotalValence(), [0, 1, 2, 3, 4, 5])
                      )
        atomnode.append(aa)#6+10+5+3+1+1+6 = 32
    atomnode = np.array(atomnode)

    '''获取分子键信息'''
    bondstype = []
    edge = []
    nodei = []
    nodej = []

    for j in m.GetBonds():

        #获取键类型
        bb = j.GetBondTypeAsDouble()
        bondstype.append(bb)

        nodei.append(j.GetBeginAtomIdx())
        nodej.append(j.GetEndAtomIdx())

    edge.append(nodei)
    edge.append(nodej)


    fp = []

    # fp1 = AllChem.GetMACCSKeysFingerprint(m)
    fp1 = Chem.GetMorganFingerprintAsBitVect(m, 2)
    for i in fp1:
        fp.append(i)
    #返回节点数，节点特征，边图，边特征
    return atomnum,atomnode,edge,bondstype,fp

def CalFingerPrint(smiles):

    mol = Chem.MolFromSmiles(smiles)
    # 创建一个示例分子
    # 计算MACCS指纹
    maccs_fp = AllChem.GetMACCSKeysFingerprint(mol)
    # 计算Morgan指纹
    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    # 计算Topological Torsion (TT)指纹
    tt_fp = AllChem.GetHashedTopologicalTorsionFingerprint(mol, nBits=2048)
    # 计算RDKit的默认哈希指纹
    rdkit_fp = Chem.RDKFingerprint(mol)

    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(maccs_fp, features)

    return features

class CustomDataset(InMemoryDataset):
    def __init__(self, data, labels):
        self.datax = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        sample_data = self.datax[index]
        sample_labels = self.labels[index]
        sample_data = self.preprocess(sample_data,sample_labels)


        return sample_data,

    def preprocess(self, sample, sample_labels):

        data_list = []

        atomnum, atomnode, edge, bondstype,fp = get_molecule_info(sample)

        fp = torch.Tensor(fp)
        fp = fp.unsqueeze(0)

        T_node_features = torch.Tensor(atomnode)
        T_edge = torch.LongTensor(edge)

        edge_tensor = torch.Tensor(bondstype)
        T_edgefeature = edge_tensor.reshape(-1, 1)

        DTIdata = DATA.Data(x=T_node_features, edge_index=T_edge, edge_attr=T_edgefeature, fp=fp, y=sample_labels)

        data_list.append(DTIdata)

        data_list, slices = self.collate(data_list)


        return data_list




class CustomDataset_pre(InMemoryDataset):
    def __init__(self, data, id):
        self.datax = data
        self.id = id

    def __len__(self):
        return len(self.id)

    def __getitem__(self, index):

        sample_data = self.datax[index]
        sample_id = self.id[index]
        sample_data = self.preprocess(sample_data,sample_id)


        return sample_data,sample_id

    def preprocess(self, sample, sample_id):

        data_list = []

        atomnum, atomnode, edge, bondstype,fp = get_molecule_info(sample)

        fp = torch.Tensor(fp)
        fp = fp.unsqueeze(0)

        T_node_features = torch.Tensor(atomnode)
        T_edge = torch.LongTensor(edge)

        edge_tensor = torch.Tensor(bondstype)
        T_edgefeature = edge_tensor.reshape(-1, 1)

        DTIdata = DATA.Data(x=T_node_features, edge_index=T_edge, edge_attr=T_edgefeature, fp=fp, id=sample_id)

        data_list.append(DTIdata)

        data_list, slices = self.collate(data_list)


        return data_list




def suffledata(smiles, label):


    # 创建自定义数据集实例
    dataset = CustomDataset(smiles, label)


    train_ratio = 0.8
    test_ratio = 0.1
    val_ratio = 0.1

    # 计算划分的样本数量
    train_size = int(train_ratio * len(dataset))
    test_size = int(test_ratio * len(dataset))
    val_size = len(dataset) - train_size -test_size

    # 使用 random_split 函数进行数据集分割
    train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, test_size, val_size])

    return train_dataset,test_dataset,val_dataset



# id, smiles, label = pre_dataset("./RIPK1-class1000-1.csv")
#
# train_dataset, test_dataset, val_dataset = suffledata(smiles, label)
#
# t = DataLoader(train_dataset,batch_size=32,shuffle=True,drop_last=True)

# for i in t:
#     for j in i[0]:
#         print(j)