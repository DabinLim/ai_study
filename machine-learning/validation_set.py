import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from scipy.stats import uniform, randint

wine = pd.read_csv('https://bit.ly/wine_csv_data')

data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

# 단순히 훈련세트에서 한번 더 split하여 검증 세트를 떼어내는 방법

train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42)


# sub_input, val_input, sub_target, val_target = train_test_split(
#     train_input, train_target, test_size=0.2, random_state=42)

dt = DecisionTreeClassifier(random_state=42)
# dt.fit(sub_input, sub_target)
# print(dt.score(sub_input, sub_target))
# print(dt.score(val_input, val_target))

# k-폴드 교차 검증
# 훈련 세트를 k 부분으로 나누어 검증하는 방식
# 훈련 세트를 섞어 폴드를 나누기 위해서 분할기 사용
# 회귀모델의 경우 KFold 분할기를 사용하고 분류 모델일 경우 StratifiedKFold를 사용
splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_validate(dt, train_input, train_target)
print(np.mean(scores['test_score']))

# 그리드서치 : 하이퍼 파라미터 탐색과 교차 검증을 한번에 수행
# params = {'min_impurity_decrease': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}
# params = {'min_impurity_decrease': np.arange(0.0001, 0.001, 0.0001),
#           'max_depth': range(5, 20, 1),
#           'min_samples_split': range(2, 100, 10)}

# n_jobs: cpu 사용량
# gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
# gs.fit(train_input, train_target)

# dt2 = gs.best_estimator_
# print(gs.best_params_)

params = {'min_impurity_decrease': uniform(0.0001, 0.001),
          'max_depth': randint(20, 50),
          'min_samples_split': randint(2, 25),
          'min_samples_leaf': randint(1, 25)
          }

gs = RandomizedSearchCV(
    DecisionTreeClassifier(random_state=42),
    params, n_iter=100,
    n_jobs=-1,
    random_state=42
)
gs.fit(train_input, train_target)
print(gs.best_params_)

dt = gs.best_estimator_
print(dt.score(test_input, test_target))
