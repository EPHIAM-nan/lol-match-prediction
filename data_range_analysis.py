#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

def analyze_data_ranges():
    """
    分析每个特征的数据范围，帮助理解离散化效果
    """
    # 读取数据
    csv_data = './data/high_diamond_ranked_10min.csv'
    data_df = pd.read_csv(csv_data, sep=',')
    data_df = data_df.drop(columns='gameId')  # 舍去对局标号列
    
    # 特征处理（与hw1.py保持一致）
    drop_features = ['blueGoldDiff', 'redGoldDiff', 
                     'blueExperienceDiff', 'redExperienceDiff', 
                     'blueCSPerMin', 'redCSPerMin', 
                     'blueGoldPerMin', 'redGoldPerMin']
    df = data_df.drop(columns=drop_features)
    
    print("=" * 80)
    print("英雄联盟数据集特征范围分析")
    print("=" * 80)
    
    # 配置参数
    NUM_BINS = 5
    
    # 分析每个特征
    analysis_results = []
    
    for c in df.columns[1:]:  # 跳过标签列
        feature_data = df[c]
        unique_count = feature_data.nunique()
        min_val = feature_data.min()
        max_val = feature_data.max()
        mean_val = feature_data.mean()
        std_val = feature_data.std()
        
        # 判断是否需要离散化
        need_discretization = unique_count > NUM_BINS
        
        if need_discretization:
            bin_width = (max_val - min_val) / NUM_BINS
            bin_edges = [min_val + i * bin_width for i in range(NUM_BINS + 1)]
        else:
            bin_width = None
            bin_edges = None
        
        analysis_results.append({
            'feature': c,
            'unique_count': unique_count,
            'min': min_val,
            'max': max_val,
            'mean': mean_val,
            'std': std_val,
            'range': max_val - min_val,
            'need_discretization': need_discretization,
            'bin_width': bin_width,
            'bin_edges': bin_edges
        })
    
    # 按是否需要离散化分组显示
    print(f"\n📊 离散化配置: 区间个数 = {NUM_BINS}")
    print(f"📈 总共 {len(analysis_results)} 个特征")
    
    # 需要离散化的特征
    need_disc = [r for r in analysis_results if r['need_discretization']]
    skip_disc = [r for r in analysis_results if not r['need_discretization']]
    
    print(f"\n✅ 需要离散化的特征 ({len(need_disc)} 个):")
    print("-" * 80)
    print(f"{'特征名称':<25} {'唯一值':<8} {'最小值':<10} {'最大值':<10} {'均值':<10} {'标准差':<10} {'区间宽度':<10}")
    print("-" * 80)
    
    for r in need_disc:
        print(f"{r['feature']:<25} {r['unique_count']:<8} {r['min']:<10.2f} {r['max']:<10.2f} "
              f"{r['mean']:<10.2f} {r['std']:<10.2f} {r['bin_width']:<10.2f}")
    
    print(f"\n⏭️  跳过离散化的特征 ({len(skip_disc)} 个):")
    print("-" * 80)
    print(f"{'特征名称':<25} {'唯一值':<8} {'最小值':<10} {'最大值':<10} {'均值':<10} {'标准差':<10}")
    print("-" * 80)
    
    for r in skip_disc:
        print(f"{r['feature']:<25} {r['unique_count']:<8} {r['min']:<10.2f} {r['max']:<10.2f} "
              f"{r['mean']:<10.2f} {r['std']:<10.2f}")
    
    # 显示详细的区间划分示例
    print(f"\n🔍 区间划分示例 (前5个需要离散化的特征):")
    print("-" * 80)
    
    for i, r in enumerate(need_disc[:5]):
        print(f"\n{i+1}. {r['feature']}:")
        print(f"   范围: [{r['min']:.2f}, {r['max']:.2f}]")
        print(f"   区间宽度: {r['bin_width']:.2f}")
        print(f"   区间边界: {[f'{edge:.2f}' for edge in r['bin_edges']]}")
        print(f"   离散化标签: 0, 1, 2, 3, 4")
    
    # 提供调试建议
    print(f"\n💡 调试建议:")
    print(f"1. 当前区间个数: {NUM_BINS}")
    print(f"2. 如需调整，修改 hw1.py 中的 NUM_BINS 参数")
    print(f"3. 建议的区间个数范围: 3-10")
    print(f"4. 对于范围很大的特征，可以考虑增加区间个数")
    print(f"5. 对于范围很小的特征，可以减少区间个数或跳过离散化")
    
    return analysis_results

if __name__ == "__main__":
    analyze_data_ranges() 