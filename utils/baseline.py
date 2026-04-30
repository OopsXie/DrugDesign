import json
import re
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from scipy import stats
from easydict import EasyDict
import os

# -----------------------------------------------------------------------------
# PyTorch Dataset
# -----------------------------------------------------------------------------
class MyDataset(Dataset):
    def __init__(self, data):
        proIndices, smiIndices, labelIndices, proMask, smiMask = data
        self._len = len(proIndices)
        self.x = proIndices
        self.y = smiIndices
        self.label = labelIndices
        self.proMask = proMask
        self.smiMask = smiMask

    def __getitem__(self, idx):
        # Protein mask
        pro_len = self.proMask[idx]
        max_pro_len = len(self.x[idx])
        proMask = [1] * pro_len + [0] * (max_pro_len - pro_len)

        # SMILES mask
        smi_len = self.smiMask[idx]
        max_smi_len = len(self.label[idx])
        smiMask = [1] * smi_len + [0] * (max_smi_len - smi_len)

        return (
            np.array(self.x[idx], dtype=np.int64),
            np.array(self.y[idx], dtype=np.int64),
            np.array(self.label[idx], dtype=np.int64),
            np.array(proMask, dtype=np.float32),
            np.array(smiMask, dtype=np.float32)
        )

    def __len__(self):
        return self._len

# -----------------------------------------------------------------------------
# DataLoader 入口
# -----------------------------------------------------------------------------
def prepareDataset(config):
    train = prepareData(config, 'train')
    valid = prepareData(config, 'valid')

    trainLoader = DataLoader(
        MyDataset(train),
        shuffle=True,
        batch_size=config.batchSize,
        drop_last=False,
        num_workers=0
    )
    validLoader = DataLoader(
        MyDataset(valid),
        shuffle=False,
        batch_size=config.batchSize,
        drop_last=False,
        num_workers=0
    )
    return trainLoader, validLoader

# -----------------------------------------------------------------------------
# 坐标填充（备用功能）
# -----------------------------------------------------------------------------
def padPocCoords(Coords, MaxLen):
    return [[0.0, 0.0, 0.0]] + Coords + (MaxLen - 1 - len(Coords)) * [[0.0, 0.0, 0.0]]

def padLabelPocCoords(Coords, MaxLen):
    return Coords + (MaxLen - len(Coords)) * [[0.0, 0.0, 0.0]]

def smilesCoordsMask(mask, MaxLen):
    return mask + (MaxLen - len(mask)) * [0]

# -----------------------------------------------------------------------------
# BindingDB 读取（废弃/备用）
# -----------------------------------------------------------------------------
def readBindingDB(PATH):
    logger.warning("readBindingDB 仅用于兼容，建议使用标准 tsv 文件")
    pdbidArr, pocSeqArr, smiArr, affinityArr = [], [], [], []
    with open(PATH, 'r') as f:
        lines = f.readlines()

    for lines in tqdm(lines):
        arr = lines.strip().split()
        if len(arr) < 2:
            continue
        pocSeq, smi = arr[0], arr[1]

        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            mol = Chem.RemoveHs(mol, sanitize=False)
            smi = Chem.MolToSmiles(mol)
            if '%' in smi or '.' in smi:
                continue
        except:
            continue

        pdbidArr.append('xxxx')
        pocSeqArr.append(pocSeq)
        smiArr.append(smi)
        affinityArr.append(0.0)

    data = pd.DataFrame({
        'pdbid': pdbidArr,
        'protein': pocSeqArr,
        'smile': smiArr,
        'affinity': affinityArr,
    })
    return data

# -----------------------------------------------------------------------------
# 核心：数据预处理 → 索引化
# -----------------------------------------------------------------------------
def prepareData(config, origin, data_dir='./data'):
    logger.info(f'Preparing {origin} data...')

    with open(os.path.join(data_dir, 'train-val-split.json'), 'r') as f:
        split = json.load(f)
    slices = split[origin]

    data = pd.read_csv(os.path.join(data_dir, 'train-val-data.tsv'), sep='\t')
    data = data.loc[slices].reset_index(drop=True)

    smiArr = data['smiles'].apply(splitSmi).tolist()
    proArr = data['protein'].apply(list).tolist()

    smiIndices, labelIndices, smiMask = fetchIndices(smiArr, config.smiVoc, config.smiMaxLen)
    proIndices, _, proMask = fetchIndices(proArr, config.proVoc, config.proMaxLen)

    return proIndices, smiIndices, labelIndices, proMask, smiMask

# -----------------------------------------------------------------------------
# 构建配置：词汇表 + 最大长度
# -----------------------------------------------------------------------------
def loadConfig(args, data_dir='./data'):
    logger.info('Building vocabulary & max lengths...')
    data = pd.read_csv(os.path.join(data_dir, 'train-val-data.tsv'), sep='\t')

    # 长度
    proMaxLen = data['protein'].str.len().max() + 2
    smiMaxLen = data['smiles'].apply(splitSmi).str.len().max() + 2

    # 蛋白词汇
    pros = data['protein'].apply(list)
    pro_chars = sorted({c for seq in pros for c in seq})
    proVoc = ['^', '&', '$'] + pro_chars

    # SMILES 词汇
    smiles = data['smiles'].apply(splitSmi)
    smi_tokens = sorted({t for seq in smiles for t in seq})
    smiVoc = ['^', '&', '$'] + smi_tokens

    return EasyDict({
        'proMaxLen': proMaxLen,
        'smiMaxLen': smiMaxLen,
        'proVoc': proVoc,
        'smiVoc': smiVoc,
        'args': args,
        'batchSize': args.batchSize if hasattr(args, 'batchSize') else 32
    })

# -----------------------------------------------------------------------------
# 皮尔逊相关系数 + 置信区间
# -----------------------------------------------------------------------------
def pearsonr_ci(x, y, alpha=0.05):
    x = np.array(x)
    y = np.array(y)
    r, p = stats.pearsonr(x, y)
    r_z = np.arctanh(r)
    se = 1 / np.sqrt(x.size - 3)
    z = stats.norm.ppf(1 - alpha/2)
    lo_z, hi_z = r_z - z*se, r_z + z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi

# -----------------------------------------------------------------------------
# SMILES 分词（最关键！）
# -----------------------------------------------------------------------------
def splitSmi(smi):
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = regex.findall(smi)
    assert smi == ''.join(tokens), f"SMILES 解析失败：{smi}"
    return tokens

# -----------------------------------------------------------------------------
# 序列 → 索引
# -----------------------------------------------------------------------------
def fetchIndices(seqArr, vocab, maxLen):
    indices = []
    labels = []
    masks = []

    for seq in tqdm(seqArr, desc="Indexing sequences"):
        # 输入：& + seq + $
        inp = ['&'] + seq + ['$']
        # 标签：inp[1:]
        lab = inp[1:]
        masks.append(len(inp))

        # 填充
        inp += ['^'] * (maxLen - len(inp))
        lab += ['^'] * (maxLen - len(lab))

        indices.append([vocab.index(t) for t in inp])
        labels.append([vocab.index(t) for t in lab])

    return np.array(indices), np.array(labels), np.array(masks)


if __name__ == '__main__':
    pass