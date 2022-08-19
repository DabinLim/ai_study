import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

wine = pd.read_csv('https://bit.ly/wine_csv_data')
# wine.head()
# wine.info()

data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42)

# 결정 트리는 표준화 전처리 과정이 필요없음 (클래스별 비율인 불순도를 기준으로 계산하기 때문)
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

dt = DecisionTreeClassifier(random_state=42)
# dt.fit(train_scaled, train_target)
# print(dt.score(train_scaled, train_target))
# print(dt.score(test_scaled, test_target))


dt.fit(train_input, train_target)
print(dt.score(train_input, train_target))
print(dt.score(test_input, test_target))
plt.figure(figsize=(10, 7))
plot_tree(dt, max_depth=3, filled=True,
          feature_names=['alcohol', 'sugar', 'pH'])
plt.show()

# 특성 중요도
# print(dt.feature_importances_)


# gini : 지니 불순도
# 1 - (음성클래스비율^2 + 양성클래스비율^2)
