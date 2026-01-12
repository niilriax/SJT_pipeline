# -*- codeing =utf-8 -*-
# @Time :2026/1/10 14:37:15
# @Author : Scientist
# @File :Ana.py
# @Software :PyCharm
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免显示错误
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def create_virtual_subjects(
    n_subjects: int,
    driving_facet: str,
    mean: float = 50.0,
    std: float = 10.0,
    noise_std: Optional[float] = None,
    seed: Optional[int] = None
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # 只生成单个特质的分数
    driving_scores = rng.normal(mean, std, n_subjects)

    return pd.DataFrame(
        {driving_facet: driving_scores},
        index=pd.RangeIndex(1, n_subjects + 1, name="被试ID")
    )

def run_virtual_subject_simulation(
    n_subjects: int = 500,
    driving_facet: str = "N",
    mean: float = 50.0,
    std: float = 10.0,
    noise_std: Optional[float] = None,
    seed: Optional[int] = 1
) -> pd.DataFrame:
    return create_virtual_subjects(
        n_subjects=n_subjects,
        driving_facet=driving_facet,
        mean=mean,
        std=std,
        noise_std=noise_std,
        seed=seed
    )


def plot_distributions(output_dir: Path, traits: list):
    """绘制每个特质的分布图"""
    trait_names = {
        "N": "神经质",
        "E": "外向性",
        "O": "开放性",
        "A": "宜人性",
        "C": "尽责性"
    }
    
    # 创建子图：2行3列，共5个图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, trait in enumerate(traits):
        # 读取数据
        filename = f"virtual_subjects_{trait}.csv"
        filepath = output_dir / filename
        
        if not filepath.exists():
            print(f"警告: 文件 {filename} 不存在，跳过绘图")
            axes[idx].axis('off')
            continue
        
        df = pd.read_csv(filepath, index_col=0)
        data = df[trait].values
        
        # 绘制直方图和密度曲线
        ax = axes[idx]
        n, bins, patches = ax.hist(data, bins=30, density=True, alpha=0.7, 
                                   color='steelblue', edgecolor='black', linewidth=0.5)
        
        # 添加理论正态分布曲线
        mu = data.mean()
        sigma = data.std()
        x = np.linspace(data.min(), data.max(), 100)
        y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        ax.plot(x, y, 'r-', linewidth=2, label=f'理论分布 (μ={mu:.1f}, σ={sigma:.1f})')
        
        # 添加统计信息
        ax.axvline(mu, color='green', linestyle='--', linewidth=2, label=f'均值={mu:.2f}')
        ax.axvline(mu + sigma, color='orange', linestyle='--', linewidth=1, alpha=0.7)
        ax.axvline(mu - sigma, color='orange', linestyle='--', linewidth=1, alpha=0.7)
        
        ax.set_title(f'{trait} - {trait_names.get(trait, trait)}', fontsize=14, fontweight='bold')
        ax.set_xlabel('分数', fontsize=12)
        ax.set_ylabel('密度', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # 隐藏最后一个空白子图
    axes[5].axis('off')
    
    plt.tight_layout()
    
    # 保存图片以便查看
    plot_path = output_dir / "trait_distributions.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n分布图已保存到: {plot_path}")
    
    # 关闭图形释放内存
    plt.close('all')


def main():
    """主函数：为每个特质生成300个被试"""
    # 定义特质列表
    traits = ["N", "E", "O", "A", "C"]
    n_subjects_per_trait = 300
    
    # 获取当前文件所在目录
    current_dir = Path(__file__).parent
    output_dir = current_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"开始生成虚拟被试数据...")
    print(f"特质列表: {traits}")
    print(f"每个特质被试数: {n_subjects_per_trait}")
    print(f"保存目录: {output_dir}\n")
    
    # 为每个特质生成一个文件
    for iteration, trait in enumerate(traits, 1):
        print(f"=== 处理特质 {trait} ({iteration}/{len(traits)}) ===")
        
        # 使用不同的seed确保每个特质生成的数据不同
        seed = iteration * 1000 + ord(trait)
        
        # 生成虚拟被试数据
        df = run_virtual_subject_simulation(
            n_subjects=n_subjects_per_trait,
            driving_facet=trait,
            mean=50.0,
            std=10.0,
            seed=seed
        )
        
        # 保存文件
        filename = f"virtual_subjects_{trait}.csv"
        filepath = output_dir / filename
        df.to_csv(filepath, index=True, encoding='utf-8-sig')
        
        print(f"  特质 {trait}: 已保存 {len(df)} 个被试 -> {filename}\n")
    
    print(f"所有数据已保存到: {output_dir}")
    print(f"共生成 {len(traits)} 个文件（每个特质一个文件，每个文件300个被试）")
    
    # 绘制分布图
    print("\n开始绘制分布图...")
    plot_distributions(output_dir, traits)


if __name__ == "__main__":
    main()
