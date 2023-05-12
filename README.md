|회차|날짜|주제|학습내용|
|----|----|----|-------|
|1|~03/28|회귀|회귀(regression)에 대해 알아보고, sklearn를 통한 머신러닝 코드 기초를 학습했다.|
|2|~04/11|분류, 규제|다중 분류에 대해 알아보고, 분류 알고리즘 몇 가지를 학습했다. 또한 Ch03-3의 규제 관련된 내용도 학습하였다.|
|3|~04/18|트리|트리를 통한 머신러닝 문제 해결 방식을 학습했다.|
|4|~05/02|앙상블, validation|트리 기반의 앙상블 모형을 학습하고, validation 세트의 의미와 활용에 대해 학습했다.|
|5|~05/04|비지도 학습|target이 없이 학습을 하는 비지도 학습과 차원축소를 학습했다.|
|6|~05/12|신경망|딥러닝과 심층 신경망 모델을 만드는 방법에 대해 학습했다.|

# 회귀 (Ch 03)
## k-최근접 회귀 (Ch 03-1) <sup>```Week 01```</sup> 
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

## 선형 회귀 (Ch 03-2) <sup>```Week 01```</sup> 
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
# 분류 (Ch 04)
## 로지스틱 회귀 (Ch 04-1) <sup>```Week 02```</sup> 
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

## 확률적 경사 하강법 (Ch 04-2) <sup>```Week 02```</sup> 
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

---
# 트리 (Ch 05)
## 결정트리 (Ch 05-1) <sup>```Week 03```</sup> 
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


## 트리의 앙상블 (ch 05-3) <sup>```Week 04```</sup> 
### 랜덤 포레스트 (Random Forest)
1. 랜덤하게 만든 decision tree의 앙상블
2. train 데이터를 샘플링함
3. 노드 분할 시, 분류에서는 특성 개수의 제곱근 만큼 <b>특성을 선택</b>하고, 회귀에서는 특성 전체를 사용
4. 다음과 같이 정의 
```
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
```

### 부트스트랩
1. 데이터 중복을 혀용하는 샘플링 기법
2. 최종적으로 총 샘플의 개수 = train 데이터 개수
3. OOB<sup>out of bag</sup> 샘플(사용되지 않은 (남는) 샘플)을 validation 세트로 사용 가능
```
rf = RandomForestClassifier(oob_score=True, n_jobs=-1)
rf.fit(train_input, train_target)
print(rf.oob_score_)
```

### 엑스트라 트리
1. 부트스트랩 샘플을 사용하지 않고, 전체 훈련 세트를 사용하는 decision tree의 앙상블
2. 노드 분할 시 특성을 무작위로 분할함. 이에 따라 성능은 낮으나 속도는 빠름
3. 다음과 같이 정의
```
from sklearn.ensemble import ExtraTreesClassifier
et = ExtraTreesClassifier(n_jobs=-1, random_state=42)
```

### 그라디언트 부스팅
1. 깊이가 얕은 결정 트리를 이용하여, 이진 트리의 오차를 보완하는 방식으로 앙상블
2. 깊이가 얕아 트리가 많아도 과적합에 강하고, 높은 일반화 성능
3. 경사 하강법에 의해 트리를 추가하며 cost가 낮은 곳으로 이동
4. 다음과 같이 정의  
`n_estimators` 트리 개수, `learning_rate` 학습률(속도)
```
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.2)
gb.fit(train_input, train_target)
```
5. 병렬 처리(`n_jobs`)가 불가

### 히스토그램 기반 그라디언트 부스팅
1. 훈련 데이터(특성)을 256개(1개는 누락 값을 위해)의 구간으로 미리 나누어 최적 분할을 찾기 빠름
2. 그라디언트 부스팅보다 다양한 특성을 골고루 평가
3. 다음과 같이 정의  
`max_iter` 부스팅 반복 횟수 (`n_estimators`  대신 사용)
```
from sklearn.ensemble import HistGradientBoostingClassifier
hgb = HistGradientBoostingClassifier()
```
4. 다른 라이브러리에서도 구현 가능
- <b>XGBoost</b>
```
from xgboost import XGBClassifier
xgb = XGBClassifier(tree_method='hist')
```
- <b>LightGBM</b>
```
from lightgbm import LGBMClassifier
lgb = LGBMClassifier()
```

---
# 비지도 학습 (Ch 06)

## 군집 알고리즘 (ch 06-1) <sup>```Week 05```</sup> 
### 비지도 학습
1. 타깃(y값)을 모를 때 (없을 때) 학습을 하는 알고리즘

### 군집 (Clustering)
1. 비슷한 샘플끼리 그룹으로 모으는 작업
2. 클러스터: 이를 통해 만들어진 그룹

> 흑백의 이미지(흰색 배경)를 numpy 배열로 변환할 경우, 값이 반전된다.  
흰색(255)보다 검정색(0)이 찾고자 하는 물체에 가깝기 때문에, 반전을 통해 더 큰 값으로 만들어준다.

> numpy에서 axis는 0:세로(행), 1:가로(열) 이다.

## k-평균 (ch 06-2) <sup>```Week 05```</sup> 
### k-평균
1. input 데이터(x)의 평균값을 자동으로 찾아주는 군집 알고리즘
2. 클러스터 중심(cluster center), 센트로이드(centroid) : 평균값
3. 다음 코드로 정의  
`.labels_`로 어느 레이블로 분류되었는지 출력
```
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3)
km.fit(fruits_2d)
print(km.labels_)
```
> 비지도 학습이므로 fitting 시 target 데이터가 필요없다.

### 절차
1. 무작위로 k개의 클러스터 중심을 지정
2. 각 샘플에서 가장 가까운 클러스터 중심을 해당 클러스터의 샘플(조합)로 지정
3. 클러스터의 중심을 각 샘플(조합)의 평균값으로 지정
4. 클러스터 중심의 변화가 없을 때 까지 2부터 반복

### 이너셔 (inertia)
1. 클러스터 중심과 샘플의 거리 제곱 합
2. 클러스터에 속한 샘플이 얼마나 가깝게 모여있는지 나타냄
3. 일반적으로, 클러스터 개수가 늘면 이너셔는 줄어듬
4. `.inertia_`로 확인


### 엘보우 (elbow)
1. 클러스터가 몇 개가 있는지 모르기 때문에, 적절한 k를 지정해야 함
2. 엘보우는 이러한 k를 찾기 위한 알고리즘
3. 클러스터의 개수를 늘려가면서, 이너셔의 변화를 관찰함
4. 이너셔-클러스터 개수 그래프를 그려보면 어느 시점부터 이너셔가 크게 줄지 않는 팔꿈치 모양이 나타나는데, 이 지점이 적절한 k값
5. 위와 같은 지점(엘보우 지점)보다 클러스터 개수가 더 많아지면 이너셔의 변화가 줄어들어 군집 효과도 감소

## 주성분 분석 (ch-06-3) <sup>```Week 05```</sup> 
### 차원 축소
1. 차원이란, 데이터의 특성(속성) 개수 (!=배열의 차원)   ex)100*100 흑백 이미지의 경우, 10,000개의 차원
2. 차원 축소를 통해, 데이터의 크기는 줄이면서 성능을 향상. 또한 복원 시 손실을 최소화
3. 3차원 이하로 줄이면, 시각화도 용이

### 주성분 분석 (PCA)
1. 대표적인 차원 축소 알고리즘
2. 분산이 가장 큰 방향인 벡터(주성분)을 찾음.
3. 샘플 데이터를 주성분에 직각으로 투영하면 한 차원이 축소되며 주성분으로 바뀐 데이터를 얻을 수 있음
4. 이 벡터에 수직이고, 분산이 가장 큰 다음 방향을 찾아 또다른 주성분을 찾음 
5. 원본 특성의 개수만큼 반복 가능하나, 일반적으로 주성분의 개수가 더 적음

5. PCA 클래스를 통해 다음과 같이 정의 가능  
`n_components=` 차원 or 설명된 분산 비율, `.transform` 차원 축소, `.inverse_transform` 복원
```
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
pca.fit(fruits_2d)
fruits_pca=pca.transform(fruits_2d)
fruits_inverse = pca.inverse_transform(fruits_pca)
```
### 설명된 분산
1. 각 주성분의 설명된 분산 비율
2. 이를 그래프로 그려, 적절한 주성분 개수를 정할 수 있음
3. PCA 클래스의 `.explained_variance_ratio_`로 확인

---
# 딥러닝 (Ch 07) 
## 인공 신경망 (Ch 07-1) <sup>```Week 06```</sup> 
### 인공 신경망
1. 이미지 분류에는 인공 신경망이 적합
### 층(layer)
1. 층 별 값(노드)의 단위인 뉴런(neuron) 또는 유닛(unit)
2. 최초 데이터인 입력층(input layer)와 최종 z 값을 만드는 출력층(output layer)
3. 가장 기본이 되는 밀집층(dense layer)과 양 쪽 뉴런이 모두 연결되어 있는 fully connected layer

> tensorflow는 직접 GPU 연산을 하는 딥러닝 라이브러리, keras는 tensorflow의 고수준 API. tensorflow는 keras의 백엔드

### one-hot encoding
1. 타깃 데이터를, 일치하는 클래스만 1, 나머지를 모두 0인 배열로 만드는 것
2. 0~3번째 중 2번째 클래스에 속하는 값은 `[0,0,1,0]`
3. 다중 분류에서도 원-핫-인코딩 사용
4. (계속) cross-entropy를 loss function으로 사용할 때, 해당 클래스에 대한 loss function값을 최대한 1에 가깝게 만들어야 함
5.  텐서플로에서 `sparse_categorical_crossentropy`를 쓰면 원핫 인코딩을 하지 않아도 됨

### 기본 코드
1. 아래와 같은 형태
2. `metrics`은 결과에 포함시킬 값 (직접적인 영향x)
3. `fit(..., match_size=?)`로 미니배치 크기 지정 가능 (기본 32개)
```
from tensorflow import keras
from sklearn.model_selection import train_test_split
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)

dense = keras.layers.Dense(10, activation='softmax', input_shape=(784,))
model = keras.Sequential(dense)

model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
model.fit(train_scaled, train_target, epochs=5)
model.evaluate(val_scaled, val_target)
```
## 심층 신경망 (Ch 07-2) <sup>```Week 06```</sup> 
### 심층 신경망
1. 2개 이상의 층을 포함한 신경망
### 은닉층 (hidden layer)
1. 입력층과 출력층 사이에 있는 dense layer
2. activation function을 통해 비선형적으로 값을 바꿔주며 역할을 함 (없이 쓸 경우 의미x)

> 회귀는 출력층에 activation function을 적용하지 않고, 선형 방정식 계산을 그대로 출력한다.

### Flatten
1. 입력 차원을 1차원으로 바꿔주는 전처리 역할
2. `keras.layers.Flatten`을 사용
3. 실제 신경망에 기여하지 않아, 깊이 계산에서 제외

> keras에서는 입력 데이터에 대한 전처리도 되도록 모델에 포함시키려 한다.

### ReLU 함수
1. max(0,z)인 함수
2. sigmoid의 경우 좌우 끝으로 갈 수록 기울기가 감소하기 때문에, 여러 층을 거칠수록 학습을 느리게 만듦.


### 기본 코드
1. `keras.Sequential`에 dense 배열을 넣거나, 다음과 같이 `.add` method를 사용
```
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28,28)))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
```
2. `model.summary()`로 모델 정보 요약
```
Model: "sequential_4"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 flatten (Flatten)           (None, 784)               0         
                                                                 
 dense_5 (Dense)             (None, 100)               78500     
                                                                 
 dense_6 (Dense)             (None, 10)                1010      
                                                                 
=================================================================
Total params: 79,510
Trainable params: 79,510
Non-trainable params: 0
_________________________________________________________________
```

### 옵티마이저(Optimizer)
1. `compile()`에서 사용하는 경사 하강법 알고리즘
2. 기본 값으로 RMSprop을 적용
3. `compile(optimizer=)`으로 사용
### SGD
1. Optimizer로 사용되는, 가장 기본적인 확률적 경사 하강법 
2. `momentum`: 그라디언트 가속도 역할의 매개변수 (기본0, 통상 0.9 이상)
3. `nseterov`: 네스테로프 모멤턴 최적화(가속 경사) 역할의 매개변수 (기본 Flase)  
모멘텀 최적화를 2회 반복하여 구현되고, 보통 기본보다 더 나은 성능

### 적응적 학습률
1. 모델이 최적 값에 가까이 갈 수록 학습률을 낮춰 안정적으로 최적 값에 수렵하도록 하는 것
2. 이를 사용하는 옵티마이저로는 RMSprop, Adagrad가 있음
3. 모멘텀 최적화와 RMSprop의 장점을 합친 Adam도 있음
4. 세 클래스 모두 `learning_rate` 기본값은 0.001

### keras 응용
1. 편의를 위해 activation function, optimizer 등을 다음과 문자열로 넣어 사용하고 있음
```
model.compile(optimzer='sgd', loss='...')
```
2. 엄연히 각 객체를 다음과 같이 만들어서 사용해야 하나, (1)과 같이 작성하여 객체를 자동 생성
```
sgd = keras.optimizers.SGD()
sgd = keras.optimizers.SGD(learning_rate=0.1)
model.compile(optimizer=sgd, loss=‘...’)
```

## 신경망 모델 훈련 (Ch 07-3) <sup>```Week 06```</sup> 
### 손실곡선
1. `fit()` 함수는 history 객체를 반환하는데, 모델 compile 시 `metric`에 작성한 값과 `loss`값을 key로 하는 배열을 반환한다.
2. `fit(..., validation_data=(val_scaled, val_target))`로 검증 데이터를 전달 할 수 있다.
3. 
4. 이를 통해, 다음과 같은 코드로 손실곡선을 그릴 수 있다.
```
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
history = model.fit(train_scaled, train_target, epochs=5, verbose=0)

import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend('train', 'val')
plt.show()
```

### 드롭아웃(dropout)
1. 훈련 과정 중 층에 있는 뉴런을 랜덤하게 비활성화(출력 0)으로 하는 것
2. 이를 통해 과적합을 방지하도록 하며, 일종의 규제
3. 이를 두 개의 다른 모델을 일종의 앙상블한 형태로도 설명할 수 있음
4. 다음과 같이 사용하며, 훈련 중에만 적용됨  
`model.add(keras.layers.Dropout(0.3))` `0.3`은 비율

### 콜백(Callback)
1. 훈련 도중 작업을 수행하도록 하는 객체
2. ModelCheckpoint 콜백은 최선의 검증 점수 모델을 저장
3. `cb=keras.callbakcs.ModelCheckpoint(‘best.h5’)`로 콜백을 만들고, `model.fit(..., callbakcs=[cb])`로 설정

### 조기종료(Early stopping)
1. 일정 횟수 이상 검증 점수가 향상되지 않으면 훈련을 미리 종료하는 방법
2. 과적합 또한 막아주기 때문에, 일종의 규제
3. 다음과 같이 사용 `cb = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)`

### 모델 저장
1. 다음 코드로 모델을 저장
```
model.save_weights('model-weights.h5') # 파라미터
model.save('model-whole.h5') #파라미터, 모델 구조
```
2. 다음 코드로 모델 또는 파라미터 불러오기
```
model.load_weight('model-weights.h5') # 이미 만들어진 모델에 파라미터 불러오기 
model = keras.models.load_model('model-whole.h5') # 새 모델
```
