import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from scipy.special import expit, softmax

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

kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)

# ['Bream' 'Parkki' 'Perch' 'Pike' 'Roach' 'Smelt' 'Whitefish']
# print(kn.classes_)

# 클래스별 생선의 확률
proba = kn.predict_proba(test_scaled[:5])
# print(np.round(proba, decimals=4))

# 시그모이드 함수
z = np.arange(-5, 5, 0.1)
phi = 1 / (1 + np.exp(-z))
# plt.plot(z, phi)
# plt.xlabel('z')
# plt.ylabel('phi')
# plt.show()

bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)

# print(lr.classes_)
# 예측 출력
# print(lr.predict(train_bream_smelt[:5]))

# 예측 확률 출력
# print(lr.predict_proba(train_bream_smelt[:5]))

# 위 로지스틱 회귀 모델이 학습한 방정식
#[[-0.4037798  -0.57620209 -0.66280298 -1.01290277 -0.73168947]] [-2.16155132]
# print(lr.coef_, lr.intercept_)

# z값
# [-6.02927744  3.57123907 -5.26568906 -4.24321775 -6.0607117 ]
decisions = lr.decision_function(train_bream_smelt[:5])
# 시그모이드 함수를 통해 확률 계산
# print(expit(decisions))

# C : L2 규제를 제어 (alpha와 반대로 작을 수록 규제가 커진다)
# max_iter : 반복 훈련 횟수 조절
lr_all = LogisticRegression(C=20, max_iter=1000)
lr_all.fit(train_scaled, train_target)

print(lr_all.predict(test_scaled[:5]))
print(lr_all.coef_, lr_all.intercept_)
decision_all = lr_all.decision_function(test_scaled[:5])
print(np.round(decision_all, decimals=2))
proba = softmax(decision_all, axis=1)
print(np.round(proba, decimals=3))

# 다중 분류 에서는 소프트맥스 함수를 사용하여 z 값을 확률로 변환
# 소프트맥스 함수 : 시그모이드 함수와 달리 여러 개의 선형 방정식의 출력값을 0~1 사이로 압축하고 전체 합이 1이 되도록 만든다
# 정규화된 지수 함수 라고도 부른다.
