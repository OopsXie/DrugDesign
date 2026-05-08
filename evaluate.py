"""
统一评估入口

评估生成分子的质量，包括：
- 基础指标：有效性(Validity)、唯一性(Uniqueness)、新颖性(Novelty)
- 理化性质：QED、SA、logP
- 分子对接：Docking Score（可选，需smina）

用法:
    # 评估已生成的SMILES文件
    python evaluate.py --generated_smiles results.csv --training_smiles train.csv

    # 评估并运行对接
    python evaluate.py --generated_smiles results.csv --run_docking \
                       --protein_pdb ./data/test_pdbs/1a9u/1a9u_protein.pdb
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Optional, Set

import numpy as np
import pandas as pd
from loguru import logger
from rdkit import Chem
from rdkit.Chem import QED, Descriptors

from model.train_utils import compute_all_metrics
from utils.sascorer import calculateScore as compute_sa_score


def load_smiles_from_file(filepath: str, smiles_col: str = 'smiles') -> List[str]:
    """从CSV/TSV文件加载SMILES列表"""
    if filepath.endswith('.tsv'):
        df = pd.read_csv(filepath, sep='\t')
    else:
        df = pd.read_csv(filepath)

    if smiles_col not in df.columns:
        # 尝试第一列
        smiles_col = df.columns[0]
        logger.warning(f"未找到'{smiles_col}'列，使用第一列: {smiles_col}")

    return df[smiles_col].dropna().tolist()


def load_training_smiles(filepath: str, smiles_col: str = 'smiles') -> set:
    """加载训练集SMILES用于新颖性计算"""
    smiles = load_smiles_from_file(filepath, smiles_col)
    canonical_set = set()
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            canonical_set.add(Chem.MolToSmiles(mol, canonical=True))
    return canonical_set


def compute_docking_scores(
    smiles_list: List[str],
    protein_pdb: str,
    ligand_ref: Optional[str] = None,
    out_dir: str = './docking_results',
) -> List[float]:
    """
    对生成分子进行分子对接评估。

    Args:
        smiles_list: 有效SMILES列表
        protein_pdb: 蛋白质PDB文件路径
        ligand_ref: 参考配体文件路径（用于确定对接盒子）
        out_dir: 输出目录

    Returns:
        对接分数列表（kcal/mol，越低越好）
    """
    try:
        from utils.docking import CaculateAffinity
    except ImportError:
        logger.warning("无法导入docking模块，跳过对接评估")
        return [0.0] * len(smiles_list)

    os.makedirs(out_dir, exist_ok=True)
    scores = []

    logger.info(f"开始对接评估，共 {len(smiles_list)} 个分子...")
    for i, smi in enumerate(smiles_list):
        try:
            score = CaculateAffinity(
                smi,
                file_protein=protein_pdb,
                file_lig_ref=ligand_ref,
                out_path=out_dir,
            )
            scores.append(score)
        except Exception as e:
            logger.debug(f"对接失败 [{i}]: {smi[:50]}... 错误: {e}")
            scores.append(0.0)

        if (i + 1) % 100 == 0:
            logger.info(f"  已完成 {i+1}/{len(smiles_list)}")

    return scores


def print_evaluation_report(metrics: Dict, docking_scores: Optional[List[float]] = None):
    """打印评估报告"""
    print("\n" + "=" * 60)
    print("  生成分子质量评估报告")
    print("=" * 60)

    print(f"\n--- 基础指标 ---")
    print(f"  总生成数:     {metrics['num_total']}")
    print(f"  有效分子数:   {metrics['num_valid']}")
    print(f"  唯一分子数:   {metrics['num_unique']}")
    print(f"  有效性 (↑):   {metrics['validity']:.4f}")
    print(f"  唯一性 (↑):   {metrics['uniqueness']:.4f}")
    if metrics['novelty'] >= 0:
        print(f"  新颖性 (↑):   {metrics['novelty']:.4f}")

    print(f"\n--- 理化性质 (有效分子均值) ---")
    print(f"  QED (↑):      {metrics['mean_qed']:.4f}")
    print(f"  SA  (↓):      {metrics['mean_sa']:.4f}")
    print(f"  logP:         {metrics['mean_logp']:.4f}")

    if docking_scores and len(docking_scores) > 0:
        valid_scores = [s for s in docking_scores if s != 0.0]
        if valid_scores:
            print(f"\n--- 分子对接 ---")
            print(f"  对接分子数:   {len(valid_scores)}")
            print(f"  平均Score:    {np.mean(valid_scores):.4f} kcal/mol")
            print(f"  最优Score:    {np.min(valid_scores):.4f} kcal/mol")
            print(f"  中位Score:    {np.median(valid_scores):.4f} kcal/mol")

            # Top-10
            sorted_idx = np.argsort(valid_scores)[:10]
            print(f"\n  Top-10 对接分数:")
            for rank, idx in enumerate(sorted_idx, 1):
                print(f"    #{rank}: {valid_scores[idx]:.4f} kcal/mol")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description='生成分子质量评估')
    parser.add_argument('--generated_smiles', type=str, required=True,
                        help='生成的SMILES文件路径(CSV/TSV)')
    parser.add_argument('--smiles_col', type=str, default='smiles', help='SMILES列名')
    parser.add_argument('--training_smiles', type=str, default='',
                        help='训练集SMILES文件路径(用于新颖性计算)')
    parser.add_argument('--run_docking', action='store_true', help='是否运行分子对接')
    parser.add_argument('--protein_pdb', type=str, default='', help='蛋白质PDB文件路径')
    parser.add_argument('--ligand_ref', type=str, default='', help='参考配体文件路径')
    parser.add_argument('--output', type=str, default='', help='评估报告输出路径(JSON)')
    parser.add_argument('--docking_dir', type=str, default='./docking_results',
                        help='对接结果输出目录')
    args = parser.parse_args()

    # 加载生成的SMILES
    logger.info(f"加载生成SMILES: {args.generated_smiles}")
    generated_smiles = load_smiles_from_file(args.generated_smiles, args.smiles_col)
    logger.info(f"  共 {len(generated_smiles)} 个分子")

    # 加载训练集SMILES（可选）
    training_smiles = None
    if args.training_smiles and os.path.exists(args.training_smiles):
        logger.info(f"加载训练集SMILES: {args.training_smiles}")
        training_smiles = load_training_smiles(args.training_smiles, args.smiles_col)
        logger.info(f"  训练集唯一分子数: {len(training_smiles)}")

    # 计算基础指标
    logger.info("计算基础指标...")
    metrics = compute_all_metrics(generated_smiles, training_smiles)

    # 对接评估（可选）
    docking_scores = None
    if args.run_docking and args.protein_pdb:
        if not os.path.exists(args.protein_pdb):
            logger.error(f"蛋白质PDB文件不存在: {args.protein_pdb}")
        else:
            # 只对有效唯一分子做对接
            valid_smiles = []
            for smi in generated_smiles:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    valid_smiles.append(Chem.MolToSmiles(mol, canonical=True))
            valid_smiles = list(set(valid_smiles))

            docking_scores = compute_docking_scores(
                valid_smiles[:500],  # 限制对接数量避免耗时过长
                args.protein_pdb,
                args.ligand_ref or None,
                args.docking_dir,
            )

    # 打印报告
    print_evaluation_report(metrics, docking_scores)

    # 保存JSON报告
    if args.output:
        report = {**metrics}
        if docking_scores:
            valid_scores = [s for s in docking_scores if s != 0.0]
            if valid_scores:
                report['docking_mean'] = float(np.mean(valid_scores))
                report['docking_min'] = float(np.min(valid_scores))
                report['docking_median'] = float(np.median(valid_scores))

        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"评估报告保存到: {args.output}")


if __name__ == '__main__':
    main()
