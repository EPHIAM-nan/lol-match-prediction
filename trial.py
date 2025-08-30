
from collections import Counter
from turtle import pen
import pandas as pd 
import numpy as np # 数学运算
from sklearn.model_selection import train_test_split, cross_validate 
from sklearn.metrics import accuracy_score 


RANDOM_SEED = 2020 # 固定随机种子
NUM_BINS = 8 #区间分裂数
MAX_DEPTH = 10 #最大数深度
MIN_SPLIT = 30 #最小样本数

csv_data = './data/high_diamond_ranked_10min.csv' # 数据路径
data_df = pd.read_csv(csv_data, sep=',') # 读入csv文件为pandas的DataFrame
data_df = data_df.drop(columns='gameId') # 舍去对局标号列

print(data_df.iloc[0]) # 输出第一行数据
data_df.describe() # 每列特征的简单统计信息

drop_features = ['blueGoldDiff', 'redGoldDiff', 
                 'blueExperienceDiff', 'redExperienceDiff', 
                 'blueCSPerMin', 'redCSPerMin', 
                 'blueGoldPerMin', 'redGoldPerMin'] # 需要舍去的特征列
df = data_df.drop(columns=drop_features) # 舍去特征列
info_names = [c[3:] for c in df.columns if c.startswith('red')] # 取出要作差值的特征名字（除去red前缀）
for info in info_names: # 对于每个特征名字
    df['br' + info] = df['blue' + info] - df['red' + info] # 构造一个新的特征，由蓝色特征减去红色特征，前缀为br
# 其中FirstBlood为首次击杀最多有一只队伍能获得，brFirstBlood=1为蓝，0为没有产生，-1为红
df = df.drop(columns=['blueFirstBlood', 'redFirstBlood']) # 原有的FirstBlood可删除


discrete_df = df.copy() # 先复制一份数据

# 配置参数：可调整的区间个数
# 通过参数调优找到的最优参数组合

for c in df.columns[1:]: 
    feature_data = df[c]
    
    unique_count = feature_data.nunique()
    
    # 如果唯一值数量太少（小于等于区间个数），则跳过离散化
    if unique_count <= NUM_BINS:
        print(f"跳过特征 {c}: 唯一值数量({unique_count}) <= 区间个数({NUM_BINS})")
        continue

    min_val = feature_data.min()
    max_val = feature_data.max()
    
    bin_width = (max_val - min_val) / NUM_BINS
    

    bin_edges = [min_val + i * bin_width for i in range(NUM_BINS + 1)]

    discrete_df[c] = pd.cut(feature_data, bins=bin_edges, labels=False, include_lowest=True)
    
    print(f"离散化特征 {c}: 范围[{min_val:.2f}, {max_val:.2f}], 区间宽度{bin_width:.2f}, 生成{NUM_BINS}个区间")

print(f"\n离散化完成！使用了 {NUM_BINS} 个区间进行等区间离散化")


all_y = discrete_df['blueWins'].values  # 蓝色方是否获胜

# 特征：用于预测的输入变量
feature_names = discrete_df.columns[1:]  # 除了第一列(blueWins)的所有列
all_x = discrete_df[feature_names].values

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(all_x, all_y, test_size=0.2, random_state=RANDOM_SEED)
all_y.shape, all_x.shape, x_train.shape, x_test.shape, y_train.shape, y_test.shape # 输出数据行列信息


# ###  决策树模型的实现
# ***本小节要求实现决策树模型，请补全算法代码***

# In[ ]:


# 定义决策树类

class DecisionTree(object):
    def __init__(self, classes, features, 
                 max_depth=10, min_samples_split=10,
                 impurity_t='entropy'):
        '''
        传入一些可能用到的模型参数，也可能不会用到
        classes表示模型分类总共有几类
        features是每个特征的名字，也方便查询总的共特征数
        max_depth表示构建决策树时的最大深度
        min_samples_split表示构建决策树分裂节点时，如果到达该节点的样本数小于该值则不再分裂
        impurity_t表示计算混杂度（不纯度）的计算方式，例如entropy或gini
        '''  
        self.classes = classes
        self.features = features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.impurity_t = impurity_t
        self.root = None # 定义根节点，未训练时为空
        
    '''
    请实现决策树算法，使得fit函数和predict函数可以正常调用，跑通之后的测试代码，
    要求之后测试代码输出的准确率大于0.6。
    
    提示：
    可以定义额外一些函数，例如
    impurity()用来计算混杂度
    gain()调用impurity用来计算信息增益
    expand_node()训练时递归函数分裂节点，考虑不同情况
        1. 无需分裂 或 达到分裂阈值
        2. 调用gain()找到最佳分裂特征，递归调用expand_node
        3. 找不到有用的分裂特征
        fit函数调用该函数返回根节点
    traverse_node()预测时遍历节点，考虑不同情况
        1. 已经到达叶节点，则返回分类结果
        2. 该特征取值在训练集中未出现过
        3. 依据特征取值进入相应子节点，递归调用traverse_node
    当然也可以有其他实现方式。

    '''
    
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
        '''
        if len(feature.shape) == 1: # 如果是一个样本
            return self.traverse_node(self.root, feature) # 从根节点开始路由
        return np.array([self.traverse_node(self.root, f) for f in feature]) # 如果是很多个样本

# 定义决策树模型，传入算法参数
DT = DecisionTree(classes=[0,1], features=feature_names, max_depth=MAX_DEPTH, min_samples_split=MIN_SPLIT, impurity_t='gini')

DT.fit(x_train, y_train) # 在训练集上训练
p_test = DT.predict(x_test) # 在测试集上预测，获得预测值
print(p_test) # 输出预测值
test_acc = accuracy_score(p_test, y_test) # 将测试预测值与测试集标签对比获得准确率
print('accuracy: {:.4f}'.format(test_acc)) # 输出准确率


# ### 模型调优
# 第一次模型测试结果可能不够好，可以先检查调试代码是否有bug，再尝试调整参数或者优化计算方法。

# ### 总结
# 一个完整的机器学习任务包括：确定任务、数据分析、特征工程、数据集划分、模型设计、模型训练和效果测试、结果分析和调优等多个阶段，本案例以英雄联盟游戏胜负预测任务为例，给出了每个阶段的一些简单例子，帮助大家入门机器学习，希望大家有所收获！
