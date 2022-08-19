import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt

fish = pd.read_csv('https://bit.ly/fish_csv_data')
fish.head()
# print(pd.unique(fish['Species']))

fish_input = fish[['Weight', 'Length',
                   'Diagonal', 'Height', 'Width']].to_numpy()

fish_target = fish['Species'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(
    fish_input, fish_target, random_state=42)

ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# loss=log : 클래스마다 이진 분류모델을 만듬 (도미를 양성 클래스로 두고 나머지를 모두 음성 클래스로)
# loss 매개변수의 기본값은 hinge (힌지 손실)
# 힌지 손실은 서포트 벡터 머신이라고 불리는 머신러닝 알고리즘을 위한 또 다른 손실 함수
# OvR (One versus Rest)
# 조기종료 : 과대적합 되기 전에 훈련을 멈추는 것
sc = SGDClassifier(loss='log', random_state=42)
# sc.fit(train_scaled, train_target)

# sc.partial_fit(train_scaled, train_target)

# 적절한 훈련횟수 찾기
train_score = []
test_score = []
classes = np.unique(train_target)

for _ in range(300):
    sc.partial_fit(train_scaled, train_target, classes=classes)
    train_score.append(sc.score(train_scaled, train_target))
    test_score.append(sc.score(test_scaled, test_target))

print(train_score)
print(test_score)

plt.plot(train_score)
plt.plot(test_score)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()
