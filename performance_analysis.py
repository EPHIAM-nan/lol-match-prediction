#!/usr/bin/env python
# coding: utf-8

"""
性能分析程序
分析决策树算法的性能极限和可能的改进方向
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from decision_tree import DecisionTree
import time

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

def discretize_features(df, num_bins=8):
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

def compare_algorithms():
    """比较不同算法的性能"""
    print("="*60)
    print("算法性能对比分析")
    print("="*60)
    
    # 加载数据
    df = load_and_preprocess_data()
    discrete_df = discretize_features(df, num_bins=8)
    
    # 准备数据
    all_y = discrete_df['blueWins'].values
    feature_names = discrete_df.columns[1:]
    all_x = discrete_df[feature_names].values
    
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(
        all_x, all_y, test_size=0.2, random_state=RANDOM_SEED
    )
    
    # 定义要测试的算法
    algorithms = {
        '我们的决策树': DecisionTree(
            classes=[0, 1], 
            features=feature_names, 
            max_depth=10, 
            min_samples_split=30, 
            impurity_t='gini'
        ),
        'sklearn决策树': DecisionTreeClassifier(
            max_depth=10, 
            min_samples_split=30, 
            random_state=RANDOM_SEED
        ),
        '随机森林': RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            min_samples_split=30, 
            random_state=RANDOM_SEED
        ),
        '梯度提升': GradientBoostingClassifier(
            n_estimators=100, 
            max_depth=5, 
            learning_rate=0.1, 
            random_state=RANDOM_SEED
        ),
        'SVM': SVC(
            kernel='rbf', 
            C=1.0, 
            random_state=RANDOM_SEED
        ),
        '神经网络': MLPClassifier(
            hidden_layer_sizes=(100, 50), 
            max_iter=500, 
            random_state=RANDOM_SEED
        )
    }
    
    results = []
    
    for name, model in algorithms.items():
        print(f"\n测试算法: {name}")
        
        try:
            start_time = time.time()
            
            # 训练模型
            model.fit(x_train, y_train)
            train_time = time.time() - start_time
            
            # 预测
            start_time = time.time()
            y_pred = model.predict(x_test)
            predict_time = time.time() - start_time
            
            # 评估
            accuracy = accuracy_score(y_test, y_pred)
            
            # 交叉验证（跳过我们的决策树，因为它不支持sklearn的交叉验证）
            if name == '我们的决策树':
                cv_mean = accuracy  # 使用测试集准确率作为估计
                cv_std = 0.0
            else:
                cv_scores = cross_val_score(model, all_x, all_y, cv=5, scoring='accuracy')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
            
            results.append({
                'algorithm': name,
                'accuracy': accuracy,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'train_time': train_time,
                'predict_time': predict_time
            })
            
            print(f"  准确率: {accuracy:.4f}")
            print(f"  交叉验证: {cv_mean:.4f} ± {cv_std:.4f}")
            print(f"  训练时间: {train_time:.2f}s")
            print(f"  预测时间: {predict_time:.2f}s")
            
        except Exception as e:
            print(f"  错误: {e}")
            results.append({
                'algorithm': name,
                'accuracy': 0,
                'cv_mean': 0,
                'cv_std': 0,
                'train_time': 0,
                'predict_time': 0
            })
    
    return results, x_test, y_test, algorithms

def analyze_feature_importance():
    """分析特征重要性"""
    print("\n" + "="*60)
    print("特征重要性分析")
    print("="*60)
    
    # 加载数据
    df = load_and_preprocess_data()
    discrete_df = discretize_features(df, num_bins=8)
    
    # 准备数据
    all_y = discrete_df['blueWins'].values
    feature_names = discrete_df.columns[1:]
    all_x = discrete_df[feature_names].values
    
    # 使用随机森林分析特征重要性
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    rf.fit(all_x, all_y)
    
    # 获取特征重要性
    feature_importance = rf.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("TOP 15 最重要特征:")
    print(feature_importance_df.head(15))
    
    return feature_importance_df

def analyze_data_distribution():
    """分析数据分布"""
    print("\n" + "="*60)
    print("数据分布分析")
    print("="*60)
    
    # 加载数据
    df = load_and_preprocess_data()
    
    print(f"总样本数: {len(df)}")
    print(f"特征数: {len(df.columns) - 1}")
    print(f"蓝色方获胜比例: {df['blueWins'].mean():.3f}")
    
    # 分析标签分布
    win_counts = df['blueWins'].value_counts()
    print(f"\n胜负分布:")
    print(f"  蓝色方获胜: {win_counts[1]} ({win_counts[1]/len(df)*100:.1f}%)")
    print(f"  红色方获胜: {win_counts[0]} ({win_counts[0]/len(df)*100:.1f}%)")
    
    # 分析特征分布
    print(f"\n特征统计信息:")
    print(df.describe())

def generate_performance_report(results):
    """生成性能报告"""
    print("\n" + "="*60)
    print("性能对比报告")
    print("="*60)
    
    # 按准确率排序
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print(f"{'排名':<4} {'算法':<15} {'准确率':<10} {'交叉验证':<15} {'训练时间':<10} {'预测时间':<10}")
    print("-" * 70)
    
    for i, result in enumerate(results):
        print(f"{i+1:<4} {result['algorithm']:<15} {result['accuracy']:<10.4f} "
              f"{result['cv_mean']:<8.4f}±{result['cv_std']:<5.4f} "
              f"{result['train_time']:<10.2f} {result['predict_time']:<10.2f}")
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv('algorithm_comparison_results.csv', index=False)
    print(f"\n结果已保存到: algorithm_comparison_results.csv")

def main():
    """主函数"""
    print("决策树性能极限分析")
    print("="*60)
    
    # 1. 数据分布分析
    analyze_data_distribution()
    
    # 2. 算法性能对比
    results, x_test, y_test, algorithms = compare_algorithms()
    
    # 3. 特征重要性分析
    feature_importance_df = analyze_feature_importance()
    
    # 4. 生成报告
    generate_performance_report(results)
    
    # 5. 结论和建议
    print("\n" + "="*60)
    print("结论和建议")
    print("="*60)
    
    best_accuracy = max(results, key=lambda x: x['accuracy'])['accuracy']
    best_algorithm = max(results, key=lambda x: x['accuracy'])['algorithm']
    
    print(f"最佳算法: {best_algorithm}")
    print(f"最佳准确率: {best_accuracy:.4f}")
    
    if best_accuracy > 0.73:
        print("✅ 找到了比决策树更好的算法！")
    else:
        print("⚠️  所有算法性能都在73%左右，可能已达到数据集的性能上限")
    
    print("\n可能的改进方向:")
    print("1. 收集更多数据（更多对局记录）")
    print("2. 特征工程（构造更有意义的特征）")
    print("3. 集成学习（组合多个模型）")
    print("4. 深度学习（使用神经网络）")
    print("5. 领域知识（结合游戏专家知识）")

if __name__ == "__main__":
    main()
