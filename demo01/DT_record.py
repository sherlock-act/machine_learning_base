from sklearn.datasets import load_wine
from sklearn import tree
from sklearn.model_selection import train_test_split
import graphviz

# 获取红酒数据
wine = load_wine()
# print(wine)

# 划分训练集与测试集
x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target)

# 创建决策树实例对象
clf = tree.DecisionTreeClassifier(criterion="entropy")

# 使用fit方法训练模型
clf.fit(x_train, y_train)

# 使用测试集验证模型准确率
print(clf.score(x_test, y_test))

# 指定特征名称,准备画一个决策树
feature_name = ['酒精', '苹果酸', '灰', '灰的碱性', '镁', '总酚', '类黄酮', '非黄烷类酚类', '花青素', '颜色强度', '色调', 'od280/od315稀释葡萄酒', '脯氨酸']

dot_data = tree.export_graphviz(
    clf,
    out_file=None,
    feature_names=feature_name, # 指定对应特征名称
    class_names=["琴酒", "雪莉", "贝尔摩德"], # 指定对应酒的名字
    filled=True,
    rounded=True,
)

graph = graphviz.Source(dot_data)
graph.view()














