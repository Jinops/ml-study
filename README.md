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
### 기본개념
1. 다중 회귀란? 여러 개의 특성을 사용한 선형회귀. 평면공간
```선형회귀는 특성이 많을수록 효과가 커진다.```
2. 특성 공학이란? 기존 특성으로 새로운 특성을 뽑아내는 작업 (특성 간의 조합 등으로)
3. pandas의 dataframe은 numpy 배열과 유사하나 더 강력하다.
### 변환기 (transformer)
1. 특성을 만들거나 전처리하는 클래스
2. `sklearn.preprocessing`의 `PolynomialFeatures()`를 사용한다.
3. `fit()`에 들어간(훈련한) 특성을 조합하여 `transform()`을 통해 사용할 수 있다.  
> 사이킷런은 훈련(fit) - 변환(transform)의 일관된 api 구성을 갖고 있다. 하나로 사용하는 `fit_transform` 메소드도 존재한다.
4. 예를 들어 `[[2,3]]`이 파라미터로 들어가면, 각 특성의 제곱, 서로 곱한 항, 절편 1이 추가되어 `[[1,2,3,4,6,9]]`가 된다. `get_feature_names_out()`로도 알 수 있다. 

### 변환기 (transformer) : 조정
6. 절편은 기본적으로 무시하나, 제거하고 싶다면 `include_bias=False`를 PolynomialFeatures`의 파라미터로 전달하면 된다.
7. 특성을 더욱 많이 (고차원으로) 만들고 싶다면 `degree=5`를 파라미터로 전달하면 된다. (예시의 경우 5제곱)  
8. 지나치게 특성을 늘리면 과적합으로 인해 test가 음수의 결과가 나올정도로 형편없어질 수 있다.


### 정규화
1. `sklearn.preprocessing`의 `StandardScaler()` 클래스를 사용한다.
2. 변환기와 마찬가지로, 훈련데이터로 `fit()`한 `StandardScaler()`로 훈련 및 테스트 데이터를 모두 `transform()`한다.

### 규제(regularization)
1. 훈련을 과하게 하지 않도록 제한을 두는 것으로, 선형 회귀에서는 계수(기울기)를 작게 만드는 것이다.
2. 릿지(ridge), 라쏘(lasso) 모델은 모두 선형 모델에 규제가 적용된 것이다.

### 릿지 회귀
1. 계수를 곱한 값을 기준으로 규제를 적용한다. 일반적으로 더 선호된다.
2. `sklearn.linear_model`의 `Ridge()`로 모델을 만들고, 나머지는 동일하다.
3. 파라미터에 `alpha=`를 넣어 규제 강도를 설정할 수 있다. 낮을수록 기존 선형모델과 유사해진다.   해당 alpha값을 적용했을 때 train 및 test 점수가 가장 비슷하다면 적절한 alpha값을 설정한 것이다.


### 라쏘 회귀
1. 계수의 절댓값을 기준으로 규제를 적용한다. 계수가 0이 될 수도 있다.
WIP
