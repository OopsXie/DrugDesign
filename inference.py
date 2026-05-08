"""
统一推理入口

给定蛋白质氨基酸序列，使用训练好的DrugGAN-MSM或KG-DrugGAN-MSM模型
批量生成候选小分子SMILES序列，并计算基本性质。

用法:
    # DrugGAN-MSM推理
    python inference.py --checkpoint ./experiments/druggan_msm/best.pt \
                        --protein_seq "MSQERPTFYRQELNKTIWEV..." \
                        --num_samples 1000 --output results.csv

    # KG-DrugGAN-MSM推理
    python inference.py --checkpoint ./experiments/kg_druggan_msm/best.pt \
                        --model_type kg_drug_gan_msm \
                        --protein_seq "MSQERPTFYRQELNKTIWEV..." \
                        --kg_entity_idx 42 \
                        --num_samples 1000 --output results.csv
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Optional

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from loguru import logger
from rdkit import Chem

from model.DrugGAN_MSM import ProteinEncoder, Generator
from model.train_utils import compute_all_metrics, set_seed


def load_checkpoint(checkpoint_path: str, device: str = 'cpu') -> Dict:
    """加载模型checkpoint"""
    logger.info(f"加载checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    logger.info(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    return checkpoint


def build_model_from_checkpoint(checkpoint: Dict, device: str = 'cpu'):
    """从checkpoint构建模型"""
    args = checkpoint.get('args', {})

    # 从checkpoint参数或默认值构建模型
    d_model = args.get('d_model', 512)
    nhead = args.get('nhead', 8)
    num_layers = args.get('num_layers', 12)
    num_decoder_layers = args.get('num_decoder_layers', 12)
    noise_dim = args.get('noise_dim', 756)
    mask_rate = args.get('mask_rate', 0.15)

    # 从checkpoint获取词表信息
    smi_voc = checkpoint.get('smi_voc', None)
    pro_voc = checkpoint.get('pro_voc', None)

    if smi_voc is None or pro_voc is None:
        raise ValueError("Checkpoint中缺少词表信息(smi_voc/pro_voc)")

    # 构建模型
    protein_encoder = ProteinEncoder(
        pro_voc_len=len(pro_voc),
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        mask_rate=mask_rate,
    )

    generator = Generator(
        smi_voc_len=len(smi_voc),
        d_model=d_model,
        noise_dim=noise_dim,
        num_layers=num_decoder_layers,
    )

    # 加载权重
    protein_encoder.load_state_dict(checkpoint['protein_encoder_state_dict'])
    generator.load_state_dict(checkpoint['generator_state_dict'])

    protein_encoder = protein_encoder.to(device).eval()
    generator = generator.to(device).eval()

    return protein_encoder, generator, smi_voc, pro_voc


def encode_protein_sequence(
    protein_seq: str,
    pro_voc: List[str],
    max_len: int = 1200,
) -> tuple:
    """
    将蛋白质序列编码为token索引和mask。

    Args:
        protein_seq: 氨基酸序列字符串
        pro_voc: 蛋白质词表
        max_len: 最大序列长度

    Returns:
        (token_indices, mask_length)
    """
    # 添加起止符
    protein = '&' + protein_seq + '$'
    pro_list = list(protein)

    # 截断
    if len(pro_list) > max_len:
        pro_list = pro_list[:max_len]

    pro_len = len(pro_list)

    # 填充
    pro_list.extend(['^'] * (max_len - pro_len))

    # 转索引
    indices = []
    for ch in pro_list:
        if ch in pro_voc:
            indices.append(pro_voc.index(ch))
        else:
            indices.append(pro_voc.index('^'))  # 未知字符用padding

    return indices, pro_len


@torch.no_grad()
def generate_molecules(
    protein_encoder: ProteinEncoder,
    generator: Generator,
    protein_seq: str,
    smi_voc: List[str],
    pro_voc: List[str],
    num_samples: int = 1000,
    max_smi_len: int = 100,
    temperature: float = 1.0,
    device: str = 'cpu',
    batch_size: int = 64,
) -> List[str]:
    """
    批量生成候选分子SMILES。

    Args:
        protein_encoder: 蛋白质编码器
        generator: 生成器
        protein_seq: 蛋白质序列
        smi_voc: SMILES词表
        pro_voc: 蛋白质词表
        num_samples: 生成样本数
        max_smi_len: 最大SMILES长度
        temperature: 采样温度
        device: 设备
        batch_size: 批大小

    Returns:
        生成的SMILES列表
    """
    protein_encoder.eval()
    generator.eval()

    start_token = smi_voc.index('&') if '&' in smi_voc else 0
    end_token = smi_voc.index('$') if '$' in smi_voc else 2

    all_smiles = []

    # 编码蛋白质序列（只需一次）
    pro_indices, pro_len = encode_protein_sequence(protein_seq, pro_voc)
    pro_tensor = torch.tensor([pro_indices], dtype=torch.long, device=device)

    protein_features = protein_encoder(pro_tensor)  # (1, 756)

    # 分批生成
    num_batches = (num_samples + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        current_batch = min(batch_size, num_samples - batch_idx * batch_size)

        # 扩展蛋白质特征到当前batch大小
        pro_feat_batch = protein_features.expand(current_batch, -1)  # (batch, 756)

        # 采样噪声
        noise = torch.randn(current_batch, generator.noise_dim, device=device)

        # 自回归生成
        generated_tokens = torch.full((current_batch, 1), start_token, dtype=torch.long, device=device)

        for step in range(max_smi_len - 1):
            logits = generator(generated_tokens, pro_feat_batch, noise)  # (batch, seq_len, vocab)
            next_logits = logits[:, -1, :] / temperature  # (batch, vocab)

            # 采样
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)

            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

            # 检查是否所有序列都生成了结束符
            if (next_token == end_token).all():
                break

        # 解码为SMILES
        for i in range(current_batch):
            tokens = generated_tokens[i].cpu().tolist()
            smi_chars = []
            for t in tokens:
                if t == end_token:
                    break
                if t == start_token:
                    continue
                if t < len(smi_voc):
                    char = smi_voc[t]
                    if char not in ('^', '&', '$'):
                        smi_chars.append(char)
            smiles = ''.join(smi_chars)
            all_smiles.append(smiles)

    return all_smiles


def filter_and_deduplicate(smiles_list: List[str]) -> List[str]:
    """过滤无效SMILES并去重"""
    valid_canonical = []
    seen = set()

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            canonical = Chem.MolToSmiles(mol, canonical=True)
            if canonical not in seen:
                seen.add(canonical)
                valid_canonical.append(canonical)

    return valid_canonical


def main():
    parser = argparse.ArgumentParser(description='DrugGAN-MSM 分子生成推理')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型checkpoint路径')
    parser.add_argument('--protein_seq', type=str, required=True, help='蛋白质氨基酸序列')
    parser.add_argument('--num_samples', type=int, default=1000, help='生成样本数')
    parser.add_argument('--max_smi_len', type=int, default=100, help='最大SMILES长度')
    parser.add_argument('--temperature', type=float, default=1.0, help='采样温度')
    parser.add_argument('--batch_size', type=int, default=64, help='批大小')
    parser.add_argument('--device', type=str, default='0', help='GPU设备ID')
    parser.add_argument('--output', type=str, default='generated_molecules.csv', help='输出文件路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()

    set_seed(args.seed)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")

    # 加载模型
    checkpoint = load_checkpoint(args.checkpoint, device)
    protein_encoder, generator, smi_voc, pro_voc = build_model_from_checkpoint(checkpoint, device)

    logger.info(f"蛋白质序列长度: {len(args.protein_seq)}")
    logger.info(f"生成 {args.num_samples} 个分子...")

    # 生成
    raw_smiles = generate_molecules(
        protein_encoder, generator,
        args.protein_seq, smi_voc, pro_voc,
        num_samples=args.num_samples,
        max_smi_len=args.max_smi_len,
        temperature=args.temperature,
        device=device,
        batch_size=args.batch_size,
    )

    logger.info(f"原始生成数: {len(raw_smiles)}")

    # 过滤和去重
    valid_smiles = filter_and_deduplicate(raw_smiles)
    logger.info(f"有效去重后: {len(valid_smiles)}")

    # 计算指标
    metrics = compute_all_metrics(raw_smiles)
    logger.info(f"有效性: {metrics['validity']:.4f}")
    logger.info(f"唯一性: {metrics['uniqueness']:.4f}")
    logger.info(f"平均QED: {metrics['mean_qed']:.4f}")
    logger.info(f"平均SA: {metrics['mean_sa']:.4f}")
    logger.info(f"平均logP: {metrics['mean_logp']:.4f}")

    # 保存结果
    df = pd.DataFrame({'smiles': valid_smiles})
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info(f"结果保存到: {args.output}")

    # 同时保存完整指标
    metrics_path = args.output.replace('.csv', '_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"指标保存到: {metrics_path}")


if __name__ == '__main__':
    main()
