#!/usr/bin/env python
# coding: utf-8

"""
决策树模块
包含完整的DecisionTree类实现
"""

import numpy as np
import pandas as pd


class DecisionTree(object):
    ismain = (__name__ == "__main__")

    def __init__(self, classes, features, 
                 max_depth=10, min_samples_split=10,
                 impurity_t='entropy', verbose=False):
        '''
        classes——模型分类总共有几类
        features——每个特征的名字
        max_depth—— 整数 构建决策树时的最大深度
        min_samples_split—— 整数 如果到达该节点的样本数小于该值则不再分裂
        impurity_t—— 字符串 计算混杂度（不纯度）的计算方式，例如entropy或gini
        '''  
        self.classes = classes
        self.features = features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.impurity_t = impurity_t
        self.verbose = verbose
        self.root = None # 定义根节点，未训练时为空
        if self.verbose:
            print(f"[DecisionTree] 混杂度计算方式: {self.impurity_t}")
        
    def impurity(self, labels):
        """
        计算混杂度（不纯度）
        labels: 标签数组
        """
        if len(labels) == 0:
            return 0
        
        # 计算每个类别的概率
        unique, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        
        if self.impurity_t == 'entropy':
            # 计算熵
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            return entropy
        elif self.impurity_t == 'gini':
            # 计算基尼指数
            gini = 1 - np.sum(probabilities ** 2)
            return gini
        else:
            raise ValueError("impurity_t must be 'entropy' or 'gini'")
    
    def gain(self, feature_values, labels, feature_idx):
        """
        计算信息增益
        feature_values: 特征值数组
        labels: 标签数组
        feature_idx: 特征索引
        """
        # 计算父节点的混杂度
        parent_impurity = self.impurity(labels)
        
        # 获取特征的所有唯一值
        unique_values = np.unique(feature_values)
        
        # 如果只有一个唯一值，无法分裂
        if len(unique_values) <= 1:
            return 0
        
        # 计算子节点的加权平均混杂度
        weighted_impurity = 0
        total_samples = len(labels)
        
        for value in unique_values:
            # 找到该特征值对应的样本
            mask = feature_values == value
            subset_labels = labels[mask]
            
            if len(subset_labels) > 0:
                subset_impurity = self.impurity(subset_labels)
                weight = len(subset_labels) / total_samples
                weighted_impurity += weight * subset_impurity
            
        # 计算信息增益
        information_gain = parent_impurity - weighted_impurity
        return information_gain
    
    def split_information(self, feature_values):
        """
        计算分裂信息（Split Information）
        分裂信息 = -Σ(|Sv|/|S|) * log2(|Sv|/|S|)
        """
        total_samples = len(feature_values)
        unique_values, counts = np.unique(feature_values, return_counts=True)
        
        split_info = 0
        for count in counts:
            if count > 0:
                p = count / total_samples
                split_info -= p * np.log2(p)
        
        return split_info
    
    def gain_ratio(self, feature_values, labels, feature_idx):
        """
        计算增益率（Gain Ratio）
        增益率 = 信息增益 / 分裂信息
        """
        # 计算信息增益
        information_gain = self.gain(feature_values, labels, feature_idx)
        
        # 计算分裂信息
        split_info = self.split_information(feature_values)
        
        # 避免除零错误，如果分裂信息为0，返回0
        if split_info == 0:
            return 0
        
        # 计算增益率
        gain_ratio = information_gain / split_info
        return gain_ratio
    
    def find_best_split(self, features, labels, depth):
        """
        找到最佳分裂特征和分裂值
        使用增益率（Gain Ratio）而不是信息增益来选择最佳特征
        """
        best_gain_ratio = 0
        best_feature = None
        best_value = None
        
        n_features = features.shape[1]
        
        for feature_idx in range(n_features):
            feature_values = features[:, feature_idx]
            gain_ratio = self.gain_ratio(feature_values, labels, feature_idx)
            
            if gain_ratio > best_gain_ratio:
                best_gain_ratio = gain_ratio
                best_feature = feature_idx
                # 对于离散特征，我们使用每个唯一值作为分裂点
                unique_values = np.unique(feature_values)
                if len(unique_values) > 1:
                    best_value = unique_values[0]  # 使用第一个唯一值作为分裂点
        
        return best_feature, best_value, best_gain_ratio
    
    def expand_node(self, features, labels, depth):
        """
        递归构建决策树节点
        """
        # 创建节点
        node = {}
        
        # 计算当前节点的标签分布
        unique_labels, counts = np.unique(labels, return_counts=True)
        majority_class = unique_labels[np.argmax(counts)]
        
        # 情况1: 无需分裂 或 达到分裂阈值
        if (len(labels) < self.min_samples_split or 
            depth >= self.max_depth or 
            len(unique_labels) == 1):
            # 创建叶节点
            node['type'] = 'leaf'
            node['prediction'] = majority_class
            return node
        
        # 情况2: 寻找最佳分裂特征
        best_feature, best_value, best_gain_ratio = self.find_best_split(features, labels, depth)
        
        # 情况3: 找不到有用的分裂特征
        if best_gain_ratio <= 0:
            # 创建叶节点
            node['type'] = 'leaf'
            node['prediction'] = majority_class
            return node
        
        # 创建内部节点
        node['type'] = 'internal'
        node['feature'] = best_feature
        node['value'] = best_value
        
        # 根据最佳分裂特征和值分裂数据
        feature_values = features[:, best_feature]
        
        # 左子树：特征值 <= 分裂值
        left_mask = feature_values <= best_value
        right_mask = ~left_mask
        
        # 递归构建子树
        if np.sum(left_mask) > 0:
            node['left'] = self.expand_node(
                features[left_mask], labels[left_mask], depth + 1
            )
        
        if np.sum(right_mask) > 0:
            node['right'] = self.expand_node(
                features[right_mask], labels[right_mask], depth + 1
            )
        
        return node
    
    def traverse_node(self, node, feature_vector):
        """
        遍历决策树进行预测
        """
        if node['type'] == 'leaf':
            return node['prediction']
        
        feature_idx = node['feature']
        feature_value = feature_vector[feature_idx]
        split_value = node['value']
        
        # 根据特征值选择分支
        if feature_value <= split_value:
            if 'left' in node:
                return self.traverse_node(node['left'], feature_vector)
            else:
                # 如果左子树不存在，返回默认预测
                return 0
        else:
            if 'right' in node:
                return self.traverse_node(node['right'], feature_vector)
            else:
                # 如果右子树不存在，返回默认预测
                return 0
        
    def fit(self, feature, label):
        assert len(self.features) == len(feature[0]) # 输入数据的特征数目应该和模型定义时的特征数目相同
        '''
        训练模型
        feature为二维numpy（n*m）数组，每行表示一个样本，有m个特征
        label为一维numpy（n）数组，表示每个样本的分类标签
        
        提示：一种可能的实现方式为
        self.root = self.expand_node(feature, label, depth=1) # 从根节点开始分裂，模型记录根节点
        '''
        self.root = self.expand_node(feature, label, depth=1)
        
    
    def predict(self, feature):
        assert len(feature.shape) == 1 or len(feature.shape) == 2 # 只能是1维或2维
        '''
        预测
        输入feature可以是一个一维numpy数组也可以是一个二维numpy数组
        如果是一维numpy（m）数组则是一个样本，包含m个特征，返回一个类别值
        如果是二维numpy（n*m）数组则表示n个样本，每个样本包含m个特征，返回一个numpy一维数组
        
        提示：一种可能的实现方式为
        if len(feature.shape) == 1: # 如果是一个样本
            return self.traverse_node(self.root, feature) # 从根节点开始路由
        return np.array([self.traverse_node(self.root, f) for f in feature]) # 如果是很多个样本


def _load_and_preprocess_data_for_debug():
    """
    仅用于本文件直接运行时的快速调试：
    - 加载数据
    - 打印并应用舍弃特征
    - 构造差值特征
    """
    csv_data = './data/high_diamond_ranked_10min.csv'
    df = pd.read_csv(csv_data, sep=',')
    if 'gameId' in df.columns:
        df = df.drop(columns='gameId')

    drop_features = ['blueGoldDiff', 'redGoldDiff',
                     'blueExperienceDiff', 'redExperienceDiff',
                     'blueCSPerMin', 'redCSPerMin',
                     'blueGoldPerMin', 'redGoldPerMin']

    exist_drop = [c for c in drop_features if c in df.columns]
    if exist_drop:
        print(f"[debug] 舍弃特征: {exist_drop}")
        df = df.drop(columns=exist_drop)

    info_names = [c[3:] for c in df.columns if c.startswith('red')]
    for info in info_names:
        blue_col = 'blue' + info
        red_col = 'red' + info
        if blue_col in df.columns and red_col in df.columns:
            df['br' + info] = df[blue_col] - df[red_col]

    for c in ['blueFirstBlood', 'redFirstBlood']:
        if c in df.columns:
            df = df.drop(columns=[c])

    return df


def _discretize_with_logging(df, num_bins=8):
    """
    离散化并打印：
    - 被离散化的特征名
    - 分隔区间数量
    - 分隔区间宽度
    """
    discrete_df = df.copy()
    discretized_report = []
    for c in df.columns[1:]:
        feature_data = df[c]
        unique_count = feature_data.nunique()
        if unique_count <= num_bins:
            continue
        min_val = feature_data.min()
        max_val = feature_data.max()
        bin_width = (max_val - min_val) / num_bins if num_bins > 0 else 0
        bin_edges = [min_val + i * bin_width for i in range(num_bins + 1)]
        discrete_df[c] = pd.cut(feature_data, bins=bin_edges, labels=False, include_lowest=True)
        discretized_report.append({
            'feature': c,
            'num_bins': num_bins,
            'bin_width': float(bin_width)
        })

    if discretized_report:
        print("[debug] 离散化特征信息:")
        for item in discretized_report[:30]:
            print(f"  - {item['feature']}: 区间数={item['num_bins']}, 区间宽度={item['bin_width']:.6f}")
        if len(discretized_report) > 30:
            print(f"  ...(其余 {len(discretized_report) - 30} 个省略)")
    else:
        print("[debug] 未发现需要离散化的特征")

    return discrete_df


if __name__ == "__main__":
    print("决策树模块调试运行模式")
    print("=" * 50)

    # 1) 加载并预处理
    df = _load_and_preprocess_data_for_debug()

    # 2) 离散化与日志
    NUM_BINS = 8
    df_disc = _discretize_with_logging(df, num_bins=NUM_BINS)

    # 3) 组装训练数据
    assert 'blueWins' in df_disc.columns, "数据缺少标签列 'blueWins'"
    y = df_disc['blueWins'].values
    feature_names = df_disc.columns[1:]
    X = df_disc[feature_names].values

    # 4) 初始化并训练（打印混杂度方式）
    dt = DecisionTree(classes=[0, 1],
                      features=feature_names,
                      max_depth=6,
                      min_samples_split=30,
                      impurity_t='gini',
                      verbose=True)

    # 仅用前一小部分样本做快速训练、预测
    n = min(2000, len(X))
    dt.fit(X[:n], y[:n])
    preds = dt.predict(X[:50])
    print(f"[debug] 预测样例(前50): {preds[:10]} ... 共 {len(preds)} 条")
        '''
        if len(feature.shape) == 1: # 如果是一个样本
            return self.traverse_node(self.root, feature) # 从根节点开始路由
        return np.array([self.traverse_node(self.root, f) for f in feature]) # 如果是很多个样本
