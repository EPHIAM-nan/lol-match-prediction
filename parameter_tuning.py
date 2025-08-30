#!/usr/bin/env python
# coding: utf-8

"""
参数调优程序
通过网格搜索找到最优的NUM_BINS、MAX_DEPTH、MIN_SPLIT参数组合
"""

from collections import Counter
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score
import time
import itertools

# 导入决策树类
from decision_tree import DecisionTree

RANDOM_SEED = 2020

def load_and_preprocess_data():
    """加载和预处理数据"""
    print("正在加载和预处理数据...")
    
    # 读入数据
    csv_data = './data/high_diamond_ranked_10min.csv'
    data_df = pd.read_csv(csv_data, sep=',')
    data_df = data_df.drop(columns='gameId')
    
    # 增删特征
    drop_features = ['blueGoldDiff', 'redGoldDiff', 
                     'blueExperienceDiff', 'redExperienceDiff', 
                     'blueCSPerMin', 'redCSPerMin', 
                     'blueGoldPerMin', 'redGoldPerMin']
    df = data_df.drop(columns=drop_features)
    
    info_names = [c[3:] for c in df.columns if c.startswith('red')]
    for info in info_names:
        df['br' + info] = df['blue' + info] - df['red' + info]
    df = df.drop(columns=['blueFirstBlood', 'redFirstBlood'])
    
    return df

def discretize_features(df, num_bins):
    """特征离散化"""
    discrete_df = df.copy()
    
    for c in df.columns[1:]:
        feature_data = df[c]
        unique_count = feature_data.nunique()
        
        if unique_count <= num_bins:
            continue
        
        min_val = feature_data.min()
        max_val = feature_data.max()
        bin_width = (max_val - min_val) / num_bins
        bin_edges = [min_val + i * bin_width for i in range(num_bins + 1)]
        
        discrete_df[c] = pd.cut(feature_data, bins=bin_edges, labels=False, include_lowest=True)
    
    return discrete_df

def evaluate_parameters(df, num_bins, max_depth, min_split, impurity_t='gini'):
    """评估特定参数组合的性能"""
    try:
        # 离散化特征
        discrete_df = discretize_features(df, num_bins)
        
        # 准备数据
        all_y = discrete_df['blueWins'].values
        feature_names = discrete_df.columns[1:]
        all_x = discrete_df[feature_names].values
        
        # 划分训练集和测试集
        x_train, x_test, y_train, y_test = train_test_split(
            all_x, all_y, test_size=0.2, random_state=RANDOM_SEED
        )
        
        # 创建和训练模型
        dt = DecisionTree(
            classes=[0, 1], 
            features=feature_names, 
            max_depth=max_depth, 
            min_samples_split=min_split, 
            impurity_t=impurity_t
        )
        
        dt.fit(x_train, y_train)
        
        # 预测和评估
        y_pred = dt.predict(x_test)
        accuracy = accuracy_score(y_pred, y_test)
        
        return accuracy, len(feature_names)
        
    except Exception as e:
        print(f"参数评估出错: {e}")
        return 0.0, 0

def grid_search_parameters():
    """网格搜索最优参数"""
    print("开始参数调优...")
    
    # 加载数据
    df = load_and_preprocess_data()
    
    # 定义参数搜索范围
    param_ranges = {
        'num_bins': [8, 10, 12, 15, 20],
        'max_depth': [5, 8, 10, 12, 15],
        'min_split': [20, 30, 50, 80, 100]
    }
    
    # 生成所有参数组合
    param_combinations = list(itertools.product(
        param_ranges['num_bins'],
        param_ranges['max_depth'],
        param_ranges['min_split']
    ))
    
    print(f"总共需要测试 {len(param_combinations)} 种参数组合")
    
    # 存储结果
    results = []
    best_accuracy = 0
    best_params = None
    
    start_time = time.time()
    
    for i, (num_bins, max_depth, min_split) in enumerate(param_combinations):
        print(f"\n测试组合 {i+1}/{len(param_combinations)}: "
              f"NUM_BINS={num_bins}, MAX_DEPTH={max_depth}, MIN_SPLIT={min_split}")
        
        # 评估参数组合
        accuracy, feature_count = evaluate_parameters(df, num_bins, max_depth, min_split)
        
        # 记录结果
        result = {
            'num_bins': num_bins,
            'max_depth': max_depth,
            'min_split': min_split,
            'accuracy': accuracy,
            'feature_count': feature_count
        }
        results.append(result)
        
        print(f"准确率: {accuracy:.4f}, 特征数量: {feature_count}")
        
        # 更新最佳参数
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = (num_bins, max_depth, min_split)
            print(f"*** 新的最佳准确率: {best_accuracy:.4f} ***")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 输出结果
    print(f"\n{'='*60}")
    print("参数调优完成!")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"测试参数组合数: {len(param_combinations)}")
    
    # 按准确率排序
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print(f"\n{'='*60}")
    print("TOP 10 最佳参数组合:")
    print(f"{'排名':<4} {'NUM_BINS':<10} {'MAX_DEPTH':<10} {'MIN_SPLIT':<10} {'准确率':<10} {'特征数':<8}")
    print("-" * 60)
    
    for i, result in enumerate(results[:10]):
        print(f"{i+1:<4} {result['num_bins']:<10} {result['max_depth']:<10} "
              f"{result['min_split']:<10} {result['accuracy']:<10.4f} {result['feature_count']:<8}")
    
    print(f"\n{'='*60}")
    print("最优参数组合:")
    print(f"NUM_BINS = {best_params[0]}")
    print(f"MAX_DEPTH = {best_params[1]}")
    print(f"MIN_SPLIT = {best_params[2]}")
    print(f"最佳准确率: {best_accuracy:.4f}")
    
    # 保存结果到文件
    save_results_to_file(results, best_params, best_accuracy, total_time)
    
    return best_params, best_accuracy, results

def save_results_to_file(results, best_params, best_accuracy, total_time):
    """保存结果到文件"""
    filename = f"parameter_tuning_results_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("决策树参数调优结果\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"调优时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总耗时: {total_time:.2f} 秒\n")
        f.write(f"测试参数组合数: {len(results)}\n\n")
        
        f.write("最优参数组合:\n")
        f.write(f"NUM_BINS = {best_params[0]}\n")
        f.write(f"MAX_DEPTH = {best_params[1]}\n")
        f.write(f"MIN_SPLIT = {best_params[2]}\n")
        f.write(f"最佳准确率: {best_accuracy:.4f}\n\n")
        
        f.write("所有参数组合结果 (按准确率排序):\n")
        f.write(f"{'NUM_BINS':<10} {'MAX_DEPTH':<10} {'MIN_SPLIT':<10} {'准确率':<10} {'特征数':<8}\n")
        f.write("-" * 50 + "\n")
        
        for result in results:
            f.write(f"{result['num_bins']:<10} {result['max_depth']:<10} "
                   f"{result['min_split']:<10} {result['accuracy']:<10.4f} {result['feature_count']:<8}\n")
    
    print(f"\n结果已保存到文件: {filename}")

def quick_test():
    """快速测试几个参数组合"""
    print("快速测试模式...")
    
    df = load_and_preprocess_data()
    
    # 测试几个关键参数组合
    test_params = [
        (10, 8, 30),
        (12, 10, 50),
        (15, 12, 80),
        (8, 5, 20),
        (20, 15, 100)
    ]
    
    results = []
    for num_bins, max_depth, min_split in test_params:
        print(f"\n测试: NUM_BINS={num_bins}, MAX_DEPTH={max_depth}, MIN_SPLIT={min_split}")
        accuracy, feature_count = evaluate_parameters(df, num_bins, max_depth, min_split)
        results.append({
            'num_bins': num_bins,
            'max_depth': max_depth,
            'min_split': min_split,
            'accuracy': accuracy,
            'feature_count': feature_count
        })
        print(f"准确率: {accuracy:.4f}")
    
    # 找到最佳组合
    best_result = max(results, key=lambda x: x['accuracy'])
    print(f"\n最佳组合: NUM_BINS={best_result['num_bins']}, "
          f"MAX_DEPTH={best_result['max_depth']}, MIN_SPLIT={best_result['min_split']}")
    print(f"准确率: {best_result['accuracy']:.4f}")

if __name__ == "__main__":
    print("决策树参数调优程序")
    print("="*50)
    
    # 询问用户选择模式
    print("请选择运行模式:")
    print("1. 完整网格搜索 (推荐)")
    print("2. 快速测试")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == "1":
        best_params, best_accuracy, results = grid_search_parameters()
    elif choice == "2":
        quick_test()
    else:
        print("无效选择，运行完整网格搜索...")
        best_params, best_accuracy, results = grid_search_parameters() 