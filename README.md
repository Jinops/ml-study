# Week 01: 회귀 <sup>```~03.28```</sup> 
회귀(regression)에 대해 알아보고, sklearn를 통한 머신러닝 코드 기초를 학습했다.
## k-최근접 회귀 (Ch 03-1)
### 기본 코드
1. `train_test_split`로 train, test 데이터를 쉽게 나눌 수 있다.
2. numpy `reshape(a,b)` 함수로 array 크기(차원)를 조정할 수 있다. 파라미터로 `-1`을 넣으면 해당 위치의 크기를 자동으로 조절한다.
> sklearn에서는 train 데이터로 2차원 배열을 요구한다.
### 알고리즘
1. target 데이터(x)와 가장 근접한 k개의 데이터(x)의 y값 평균을 예측값으로 사용한다.
2. ```KNeighborsRegressor()```로 해당 모델을 만들 수 있다. (기본 k=3)
### 모델 평가
1. 모델 변수의 `.fit(x, y)`을 사용해 모델을 피팅하고, `.score(x,y)`를 통해 모델을 평가할 수 있다.
2. `.predict(x)`로 단일 값에 대한 예측값 y를 확인할 수 있다.
3. 결정계수($R^2$)로 모델을 평가한다.  
$R^2 = 1 - \frac{(타깃-예측)^2}{(타깃-평균)^2}$ 수식을 보면, 예측 결과가 정확할수록 1이 된다.
4. `mean_absolute_error(target, predict)`으로 절댓값 오차를 계산할 수도 있다.
### 과적합
1. train에서 충분한 학습이 되지 않을 경우 underfitting(과소적합), 너무 많이 반영될 경우 overfitting(과대적합)이 일어난다.
2. 과소적합이 일어날 경우, train 데이터를 충분히 준비하거나 모델을 더 복잡하게 하여 학습을 많이 시키면 개선될 수 있다.
3. 예제에서는 k-최근접 회귀의 k값을 3->5로 바꾸어 모델을 더 복잡하게 만들어 개선하였다.

## 선형 회귀 (Ch 03-2)
### 알고리즘
1. x, y의 관계를 선형 함수로 나타내도록 하는 것이다.
2. `LinearRegression()`로 해당 모델을 만들 수 있다.
3. 모델 변수의 `.coef_`로 계수(coefficient), 즉 기울기를 알 수 있고, `.intercept_`로 y절편을 알 수 있다.

### 다항회귀
1. 다항식을 사용하여 모델의 특징을 더 잘 설명할 수도 있다.
2. $b_1x^2+b_2x+c$를 보면 2차 방정식(비선형)으로 보이지만, $x^2$를 다른 변수로 치환하여 다른 설명변수 x로 사용한다면 선형 다항식이 된다.
3. `new_x = numpy.column_stack(x**2, x)` 방식으로 배열을 만들고, 모델 변수 `.fit`에서 `new_x`를 x로 사용하면 된다.

## 특성 공학과 규제 (Ch 03-3) <sup>```Week 02```</sup> 
### 변환기 (transformer)
1. 특성을 제곱하거나 서로 곱하여 새로운 특성을 만들고 전처리하는 것이다. 이러한 개념이 특성 공학이다.
2. `sklearn.preprocessing`의 `PolynomialFeatures()`를 사용한다.
3. 모델을 학습하는 것처럼 `fit()`에 들어간(훈련한) 특성을 조합하여 `transform()`을 통해 사용할 수 있다.  
> 사이킷런은 훈련(fit) - 변환(transform)의 일관된 api 구성을 갖고 있다.
4. 특성을 더욱 많이 (고차원으로) 만들고 싶다면 `degree=5`를 파라미터로 전달해 특성을 고차원으로 만들 수 있으나, 지나치게 높으면 과적합 문제가 발생한다.

### 정규화
1. `sklearn.preprocessing`의 `StandardScaler()` 클래스를 사용한다.
2. 변환기와 마찬가지로, 훈련데이터로 `fit()`한 `StandardScaler()`로 훈련 및 테스트 데이터를 모두 `transform()`한다.

### 규제(regularization)
1. 훈련을 과하게 하지 않도록 제한을 두는 것으로, 선형 회귀에서는 계수(기울기)를 작게 만드는 것이다.
2. 릿지(ridge), 라쏘(lasso) 모델은 모두 선형 모델에 규제가 적용된 것이다.
3. 위 모델의 파라미터에 `alpha`를 넣어 규제 강도를 설정할 수 있다. 낮을수록 기존 선형모델과 유사해진다.  
해당 alpha값을 적용했을 때 train 및 test 점수가 가장 비슷하다면 적절한 alpha값을 설정한 것이다.

### 릿지 회귀
1. 계수를 제곱한 값을 기준으로 규제를 적용한다. 일반적으로 더 선호된다.
2. `sklearn.linear_model`의 `Ridge()`로 모델을 만들고, 나머지는 동일하다.

### 라쏘 회귀
1. 계수의 절댓값을 기준으로 규제를 적용한다. 계수가 0이 될 수도 있다.
2. `sklearn.linear_model`의 `Lasso()`로 모델을 만들고, 나머지는 동일하다.

---
# Week 02: 분류 <sup>```~04.11```</sup> 
다중 분류에 대해 알아보고, 분류 알고리즘 몇 가지를 학습했다. 또한 Ch03-3의 규제 관련된 내용도 학습하였다. 
## 로지스틱 회귀 (Ch 04-1)
### KNeighbors
1. 다중분류에도 KNeighbors를 활용할 수 있다. 가장 가까운 이웃 클래스들의 비율을 계산한다.
2. 그러나 이를 분류에 활용하면, k의 크기 만큼 한정된 확률이 나오기 때문에 적절하지 않다.  
> numpy의 slicing 기능은 항상 2차원 배열을 반환한다.

### 로지스틱 회귀 (Logistic regression)
1. 방정식 자체만으로는 다중 회귀 선형방정식을 사용한다.
2. 선형방정식의 결과값 z를 적절한 함수를 적용해 확률로 바꿔 분류를 적용한다.

### 기본 코드
1. `sklearn.linear_model`의 `LogisticRegression`을 사용한다.
2. 파라미터 `C`(기본 1)에 큰 값을 줄 수록 규제를 완화한다. 이는 계수의 제곱을 규제하는 L2규제라고 부른다.
3. 파라미터 `max_iter`(기본 100)는 모델 훈련의 반복 횟수이다.
4. z값은 `.decision_function()`을 통해서 확인 가능하다.
5. 학습 class들은 알파벳 순으로 정렬되며, `.classes_` 어트리뷰트로 확인 가능하다.
6. `.predict_proba()`를 사용해 각 클래스 별 확률을 확인할 수 있다.

### 시그모이드 함수
1. 아래 수식을 이용하여 결과값 z가 큰 음수일 때 0, 큰 양수일 때 1이 되도록 한다.  
1/1+e<sup>-z</sup>
2. 즉, 0~1사이의 확률로 해석할 수 있다.

### 이진 분류에 적용
1. 이진 분류는 <b>시크모이드 함수</b>를 이용하여 확률을 계산한다.
1. 교재에서는 numpy 불리언 인덱싱을 이용해 테스트 데이터를 만들었다. `np.array`를 선언하고, 인덱스에 `True`,`False`를 다시 넣어 출력하면 `True`인 데이터만 반환한다.
3. 알파벳 순으로 정렬된 class 순서대로, 음수-양수 순서이다.

### 소프트맥스 함수
1. 클래스 별 z값을 지수 함수($e^z$)를 씌우고, 각각을 총합으로 눈다.
2. 결과적으로 클래스 별 확률이 계산된다.

### 다중 분류에 적용
1. 다중분류는 각 클래스마다 z값을 계산한 뒤, 가장 큰 확률을 가진 클래스로 예측한다.
2. 확률 계산 시 <b>소프트맥스</b> 함수를 사용한다.

## 확률적 경사 하강법 (Ch 04-2)
### 점진적 학습
1. 훈련 데이터를 지속해서 추가 업데이트해야 하는 문제가 있다면?
2. 매 번 새로가 아닌, 조금씩 훈련을 추가하는 방법인 점진적 학습을 한다.

### 확률적 경사 하강법
1. 모델 훈련 시 cost를 줄이는 방향으로 가는 것 (경사를 내려가는 것)
2. 훈련 세트에 샘플을 넣고 모두 사용할 때 까지 계속해서 반복 후, 최소값으로 못가면 반복
3. <b>에포크</b>: 확률적 경사 하강법에서, 훈련 세트를 모두 사용하는 단위.   에포크 반복 시에 일정 수준 이상 성능이 오르지 않으면 중도 멈춤(early stop)을 할 수 있다.

> 데이터가 많고 모델이 복잡한 Neural Network에서 대부분 사용된다.

### 미니배치 경사 하강법
1. 훈련 세트에서 무작위 샘플을 여러개 씩 선택해서 경사를 내려가는 것
2. 배치 경사 하강법: 전체 샘플을 한 번에 모두 사용하는 것

### 손실 함수 (≒비용함수)
1. 모델 알고리즘이 얼마나 잘못되었는지 측정하는 기준
2. 손실 함수는 연속적이어야 한다. (미분 가능)
3. 분류에서는 로지스틱 손실 함수, 회귀에서는 평균 제곱 오차(mean squared error)를 주로 사용한다.

### 로지스틱 손실 함수
1. =(이진) 크로스엔트로피 손실 함수
2. 양수(확률 >0.5)일 때 $-log(확률)$, 음수(확률<0.5)일 때 $-log(1-확률)$

### 코드
1. 경사하강법 모델  
(`loss` 손실함수, `max_iter` 에포크, `tol` early stop 최소 개선치)
```
from sklearn.linear_model import SGDClassifier
sc = SGDClassifier(loss='log', max_iter=10, tol=None)
```

2. 모델 피팅
```
sc.fit(train_scaled, train_target)
# 기존 모델에 점진적(추가) 학습하기
sc.partial_fit(train_scaled, train_target)
```
# Week 03: 트리 알고리즘 <sup>```~04.18```</sup> 
트리를 통한 머신러닝 문제 해결 방식을 학습했다.
## 결정트리 (Ch 05-1)
### 기본 코드
1.  모델  
  (`max_depth` 가지치기)
```
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=3)
dt.fit(train_scaled, train_target)
```
> 결정 트리는 특성값의 스케일에 영향을 받지 않기 때문에, 데이터 표준화를 할 필요가 없다.
2. 트리 그리기
```
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(10,7))
plot_tree(dt, max_depth=2, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()
```
### 알고리즘
1. 불순도에 따라 조건을 만들어 좌(True), 우(False)로 샘플을 나눠 나간다
2. 부모 노드와 자식 노드의 <b>불순도 차이</b>(=<b>정보 이득</b>)를 최대화하도록 트리를 확장시킨다.
3. 최종 노드(리프 노드)에서 더 많이 분류된 클래스를 예측 클래스로 한다.

### Gini 불순도
1. $gini = 1 - (False 클래스 비율^2 + True 클래스 비율^2)$
2. DecisionTreeClassifer의 데이터 분할 기준(`criterion`) 기본값

### 엔트로피 불순도
1. $entropy = -False 클래스 비율 * log_2(Flase 클래스 비율) - True 클래스 비율 * log_2(True 클래스 비율)$
2. `criterion='entropy'`로 지정
### 특성 중요도
1. 결정 트리에 사용된 특성이 불순도를 감소하는데 기여한 정도
2. 최상위 노드(루트 노드)와 얕은 깊이의 노드의 특성을 보거나, `.feature_importances_`로 중요도를 확인할 수 있다.
3. 이를 이용해, 결정 트리를 특성 선택에 활용할 수 있다.


## 교차 검증과 그리드 처치 (ch 05-2) <sup>```Week 04```</sup> 
### 검증(Validation)
1. 훈련 데이터에서 또다시 몇 %의 비중을 분리하여, 테스트 데이터 이전에 사용하는 것
2. 하이퍼파라미터 튜닝 등의 절차 뒤 사용하여, 테스트 데이터의 사용을 최소화

### 교차 검증(Cross Validation)
1. validation 세트를 다른 위치로 여러번 나눠, 이 평균을 최종 점수로 하는 것
2. 이를 통해 최적의 하이퍼파라미터를 찾으면, 전체 train 세트로 모델을 다시 생성하여 적용
3. k-fold 교차검증: 훈련 세트를 k부분으로 나눠 교차검증 하는 것
4. `cross_validate(모델, train_input, train_target)`로 반환되는 dictionary의 ‘test_score’에 각 검증 결과가 배열로 있음
5. 위 과정 전 데이터를 섞고 싶다면, 아래와 같은 분할기를 선언 후 `cv=`에 파라미터로 전달
```
from sklearn.model_selection import StratifiedKFold
splitter = StratifiedKFold(n_splits=10, shuffle=True)
```

### 그리드 서치 (Grid Search)
1. 하이퍼파라미터의 탐색 및 교차 검증을 수행하는 기능
2. 아래 코드를 통해, 최적의 하이퍼파라미터를 찾아 모델에 적용 가능  
`params` 파라미터 후보
```
from sklearn.model_selection import GridSearchCV
params = {'min_impurity_decrease': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}
gs = GridSearchCV(DecisionTreeClassifier(), params, n_jobs=-1)
gs.fit(train_input, train_target)

dt = gs.best_estimator_
print('params: ', gs.best_params_)
print('score: ', dt.score(train_input, train_target))
print('best validation score: ', np.max(gs.cv_results_['mean_test_score']))


```

### 랜덤 서치 (Random Search)
1. 하이퍼파라미터를, 확률분포에 따라 랜덤으로 설정하는 것
2. 다음과 같이 사용 가능
```
from scipy.stats import uniform, randint
params = {'min_impurity_decrease': uniform(0.0001, 0.001),
          'max_depth': randint(20, 50),
          'min_samples_split': randint(2,25),
          'min_samples_leaf': range(1,25), # not random
          }

from sklearn.model_selection import RandomizedSearchCV
gs = RandomizedSearchCV(DecisionTreeClassifier(), params, n_iter=100, n_jobs=-1)
gs.fit(train_input, train_target)

dt = gs.best_estimator_
```
```
print('params: ', gs.best_params_)
print('train score: ', dt.score(train_input, train_target))
print('validation score: ', gs.cv_results_['mean_test_score'])
print('test score: ', dt.score(test_input, test_target))

```
