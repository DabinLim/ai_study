import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import matplotlib.pyplot as plt

# 특성공학 : 기존의 특성을 사용해 새로운 특성을 뽑아내는 작업

# 판다스 (pandas) : 데이터 분석 라이브러리, 데이트 프레임은 판다스의 핵심 데이터 구조이다.
df = pd.read_csv('https://bit.ly/perch_csv_data')
perch_full = df.to_numpy()

perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
                         115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
                         150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
                         218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
                         556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
                         850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
                         1000.0])

train_input, test_input, train_target, test_target = train_test_split(
    perch_full, perch_weight, random_state=42)

# include_bias=False 는 자동으로 특성에 추가된 절편 항을 무시
# degree를 통해 필요한 고차항의 최대 차수를 지정
# 샘플 개수보다 특성이 많은 경우 과대적합 발생
# 규제 : 머신러닝 모델이 훈련 세트를 너무 과도하게 학습하지 못하도록 훼방하는 것을 말한다.
# poly = PolynomialFeatures(include_bias=False)
poly = PolynomialFeatures(degree=5, include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)

# get_feature_names_out() : 특성이 어떤 입력의 조합으로 만들어졌는지 출력
print(poly.get_feature_names_out())

# 훈련 세트를 기준으로 테스트 세트를 변환
test_poly = poly.transform(test_input)

lr = LinearRegression()
lr.fit(train_poly, train_target)
# print(lr.score(train_poly, train_target))

# 특성의 스케일 정규화
ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)

# 릿지 (ridge) : 계수를 제곱한 값을 기준으로 규제를 적용
# 리쏘 (iasso) : 계수의 절댓값을 기준으로 규제를 적용
# 일반적으로 릿지를 조금 더 선호, 리쏘는 아예 0으로 만들 수 있음
# alpha 값으로 규제 강도 조절, 높으면 규제 강도가 세지므로 계수값을 줄이고 과소 적합되도록 유도한다.
# alpha 값과 같이 사람이 지정해야 하는 매개변수를 하이퍼 파라미터라고 한다.
ridge = Ridge(alpha=0.1)
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))

# 적절한 alpha 값을 찾는 방법 : alpha 값에 대한 R^2 값의 그래프를 그려 훈련 세트와 테스트 세트의 점수가 가까운 지점이 최적의 alpha 값이다.
train_score = []
test_score = []
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    ridge2 = Ridge(alpha=alpha)
    # 릿지 모델 훈련
    ridge2.fit(train_scaled, train_target)
    train_score.append(ridge2.score(train_scaled, train_target))
    test_score.append(ridge2.score(test_scaled, test_target))

# 라쏘 모델 훈련
lasso = Lasso(alpha=10)
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))
print(np.sum(lasso.coef_ == 0))
