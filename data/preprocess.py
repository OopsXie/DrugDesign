"""
数据预处理模块

封装论文3.5.1节定义的5条数据过滤规则：
1. 仅保留有明确生物活性数据的分子
2. SMILES长度34-89
3. 剔除盐类、无机物、无法RDKit解析的分子
4. 过滤断开离子/孤立碎片
5. 原子类型限制 + 重原子数[10,35]
"""

import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from loguru import logger

# 论文定义的允许原子类型集合
ALLOWED_ATOMS = {'C', 'c', 'N', 'n', 'S', 's', 'P', 'O', 'o', 'B', 'F', 'I', 'Cl', '[nH]', 'Br'}
ALLOWED_SYMBOLS = {'C', 'c', 'N', 'n', 'S', 's', 'P', 'O', 'o', 'B', 'F', 'I', 'Cl', '[nH]', 'Br',
                   '1', '2', '3', '4', '5', '6', '#', '=', '-', '(', ')'}


def _has_valid_atoms(mol: Chem.Mol) -> bool:
    """检查分子是否仅包含允许的原子类型"""
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol not in ALLOWED_ATOMS:
            return False
    return True


def _has_carbon(mol: Chem.Mol) -> bool:
    """检查分子是否包含碳原子"""
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in ('C', 'c'):
            return True
    return False


def _is_single_fragment(mol: Chem.Mol) -> bool:
    """检查分子是否为单一片段（无断开离子/孤立碎片）"""
    frags = Chem.GetMolFrags(mol, asMols=True)
    return len(frags) == 1


def _has_valid_smiles_chars(smiles: str) -> bool:
    """检查SMILES字符串是否仅包含允许的符号"""
    i = 0
    while i < len(smiles):
        # 检查多字符token如 [nH], Cl, Br
        if i + 3 <= len(smiles) and smiles[i:i+3] == '[nH]':
            i += 3
            continue
        if i + 2 <= len(smiles) and smiles[i:i+2] in ('Cl', 'Br'):
            i += 2
            continue
        if smiles[i] in ALLOWED_SYMBOLS or smiles[i].isalpha():
            i += 1
            continue
        # 数字和特殊符号
        if smiles[i] in '0123456789':
            i += 1
            continue
        return False
    return True


def filter_molecules(
    df: pd.DataFrame,
    smiles_col: str = 'smiles',
    min_smiles_len: int = 34,
    max_smiles_len: int = 89,
    min_heavy_atoms: int = 10,
    max_heavy_atoms: int = 35,
) -> pd.DataFrame:
    """
    对分子数据执行论文定义的5条过滤规则。

    Args:
        df: 输入DataFrame，需包含SMILES列
        smiles_col: SMILES列名
        min_smiles_len: SMILES最小长度
        max_smiles_len: SMILES最大长度
        min_heavy_atoms: 最少重原子数
        max_heavy_atoms: 最多重原子数

    Returns:
        过滤后的DataFrame
    """
    initial_count = len(df)
    logger.info(f"开始数据过滤，初始样本数: {initial_count}")

    # 规则1: SMILES非空
    df = df.dropna(subset=[smiles_col])
    df = df[df[smiles_col].str.strip() != '']
    logger.info(f"  规则1(非空): 剩余 {len(df)}")

    # 规则2: SMILES长度约束
    df = df[df[smiles_col].str.len().between(min_smiles_len, max_smiles_len)]
    logger.info(f"  规则2(长度{min_smiles_len}-{max_smiles_len}): 剩余 {len(df)}")

    # 规则3-5: RDKit解析 + 化学规则检查
    valid_indices = []
    for idx, row in df.iterrows():
        smi = row[smiles_col]
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue

            # 规则3: 剔除盐类/无机物/无法解析
            if not _has_carbon(mol):
                continue

            # 规则4: 过滤断开离子/孤立碎片
            if not _is_single_fragment(mol):
                continue

            # 规则5a: 原子类型限制
            if not _has_valid_atoms(mol):
                continue

            # 规则5b: 重原子数限制
            num_heavy = mol.GetNumHeavyAtoms()
            if num_heavy < min_heavy_atoms or num_heavy > max_heavy_atoms:
                continue

            valid_indices.append(idx)
        except Exception:
            continue

    df = df.loc[valid_indices]
    logger.info(f"  规则3-5(化学规则): 剩余 {len(df)}")
    logger.info(f"过滤完成: {initial_count} -> {len(df)} ({len(df)/initial_count*100:.1f}%)")

    return df.reset_index(drop=True)


def preprocess_chembl(
    input_path: str,
    output_path: str,
    smiles_col: str = 'smiles',
    **filter_kwargs,
) -> pd.DataFrame:
    """
    从ChEMBL原始数据预处理并保存。

    Args:
        input_path: 输入TSV/CSV文件路径
        output_path: 输出文件路径
        smiles_col: SMILES列名
        **filter_kwargs: 传递给filter_molecules的额外参数

    Returns:
        过滤后的DataFrame
    """
    logger.info(f"读取数据: {input_path}")
    if input_path.endswith('.tsv'):
        df = pd.read_csv(input_path, sep='\t')
    else:
        df = pd.read_csv(input_path)

    logger.info(f"原始列: {list(df.columns)}")

    df = filter_molecules(df, smiles_col=smiles_col, **filter_kwargs)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    if output_path.endswith('.tsv'):
        df.to_csv(output_path, sep='\t', index=False)
    else:
        df.to_csv(output_path, index=False)

    logger.info(f"保存到: {output_path}")
    return df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='数据预处理')
    parser.add_argument('--input', type=str, required=True, help='输入文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出文件路径')
    parser.add_argument('--smiles-col', type=str, default='smiles', help='SMILES列名')
    parser.add_argument('--min-len', type=int, default=34, help='SMILES最小长度')
    parser.add_argument('--max-len', type=int, default=89, help='SMILES最大长度')
    args = parser.parse_args()

    preprocess_chembl(
        args.input,
        args.output,
        smiles_col=args.smiles_col,
        min_smiles_len=args.min_len,
        max_smiles_len=args.max_len,
    )
