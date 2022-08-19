import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

# 앙상블 학습 : 더 좋은 예측 결과를 만들기 위해 여러 개의 모델을 훈련하는 머신러닝 알고리즘

wine = pd.read_csv('https://bit.ly/wine_csv_data')
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42)

# OOB 샘플 : 부트스트랩 샘플에 포함되지 않고 남는 샘플
print('랜덤 포레스트')
rf = RandomForestClassifier(oob_score=True, n_jobs=-1, random_state=42)
scores_rf = cross_validate(rf, train_input, train_target,
                           return_train_score=True, n_jobs=-1)
print(np.mean(scores_rf['train_score']), np.mean(scores_rf['test_score']))

rf.fit(train_input, train_target)
print(rf.feature_importances_)
print(rf.oob_score_)

# 엑스트라 트리 : 부트스트랩 샘플을 사용하지 않고 전체 훈련 세트를 사용
# splitter='random'인 결정트리를 사용
# 특성을 무작위로 분할하여 성능은 낮아지지만 많은 트리를 앙상블 하기 때문에 과대적합을 막고 검증 세트의 점수를 높임
print('엑스트라 트리')
et = ExtraTreesClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(et, train_input, train_target,
                        return_train_score=True, n_jobs=-1)

# 그레이디언트 부스팅
# depth가 3인 결정트리를 100개 사용 - 과대적합에 강하고 높은 일반화 성능
# 경사 하강법을 사용하여 트리를 앙상블에 추가
# 분류에서는 로지스틱 손실 함수를 사용하고 회귀에서는 평균 제곱 오차 함수를 사용
# 결정트리 갯수를 500개로 늘려도 과대적합을 잘 억제한다
print('그레이디언트 부스팅')
gb = GradientBoostingClassifier(
    n_estimators=500, learning_rate=0.2, random_state=42)
scores_gb = cross_validate(gb, train_input, train_target,
                           return_train_score=True, n_jobs=-1)
print(np.mean(scores_gb['train_score']), np.mean(scores_gb['test_score']))
gb.fit(train_input, train_target)
print(gb.feature_importances_)

# 히스토그램 기반 그레이디언트 부스팅
# 특성을 256개 구간으로 나눈다
# 256개 구간 중에서 하나를 떼어 놓고 누락된 값을 위해서 사용하므로 입력에 누락된 특성이 있더라도 따로 전처리할 필요가 없다.
# 트리의 개수를 지정할 때 n_estimators 대신 부스팅 반복 횟수를 지정하는 max_iter를 사용한다.
print('히스토그램 기반 그레이디언트 부스팅')
hgb = HistGradientBoostingClassifier(random_state=42)
scores_hgb = cross_validate(hgb, train_input, train_target,
                            return_train_score=True)
print(np.mean(scores_hgb['train_score']), np.mean(scores_hgb['test_score']))

hgb.fit(train_input, train_target)
result = permutation_importance(
    hgb, train_input, train_target, n_repeats=10, random_state=42, n_jobs=-1)
print(result.importances_mean)
