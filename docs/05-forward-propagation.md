# 순전파 (Forward propagation) {#forward}



## 신경망의 구조

딥러닝의 시작점인 신경망(Neural network)을 공부하기 위해서, 앞으로 우리가 다룰 모델 중 가장 간단하면서, 딥러닝에서 어떤 일이 벌어지고 있는지 상상이 가능한 신경망을 먼저 학습하기로 하자. 우리가 오늘 예로 생각할 신경망은 다음과 같다. 

<div class="figure" style="text-align: center">
<img src="./image/neuralnet1.png" alt="세상에서 가장 간단하지만 있을 건 다있는 신경망" width="100%" />
<p class="caption">(\#fig:neuralnet-example)세상에서 가장 간단하지만 있을 건 다있는 신경망</p>
</div>

위의 그림과 같은 신경망을 2단 신경망이라고 부른다. 일반적으로 단수를 셀 때 제일 처음 입력하는 층은 단수에 포함하지 않는 것에 주의하자. 각 녹색, 회색, 그리고 빨간색의 노드(node)들은 신경망의 요소를 이루는데, 각각의 이름은 다음과 같다.

* 입력층(input layer) - 2개의 녹색 노드(node)
* 은닉층(hidden layer) - 3개의 회색 노드(node)
* 출력층(output layer) - 1개의 빨강색 노드(node)

자 이제부터, 녹색 노드에는 무엇이 들어가는지, 그리고, 어떤 과정을 거쳐서 빨강색의 값이 나오는지에 대하여 알아보자. 딥러닝에서 녹색이 입력값을 넣어서 빨간색의 결과값을 얻는 과정을 **순전파(Forward propagation)**라고 부른다. `propagation`의 뜻은 증식, 혹은 번식인데, 식물이나 동물이 자라나는 것을 의미하는데, 녹색의 입력값들이 어떠한 과정을 거쳐 빨간색으로 자라나는지 한번 알아보자.

## 순전파(Forward propagation)

우리가 사용할 데이터 역시 아주 간단하다.

$$
X =\left(\begin{array}{cc}
1 & 2\\
3 & 4\\
5 & 6
\end{array}\right)
$$
가로 행이 하나의 표본을 의미하고, 세로 열 각각은 변수를 의미한다. 즉, 위의 자료 행렬은 2개의 변수 정보가 들어있는 세 개의 표본들이 있는 자료을 의미한다.

### 표본 1개, 경로 1개만 생각해보기

주의할 것은, 우리가 그려놓은 신경망의 입력층의 노드는 2개이고, 자료 행렬은 3행 2열이라는 것이다. 우리가 그려놓은 신경망으로 샘플 하나 하나가 입력층에 각각 입력되어 표본별 결과값 생성되는 것이다. 따라서 신경망을 잘 이해하기 위해서 딱 하나의 표본, 그리고 딱 하나의 경로만을 생각해보자.

    > 목표: 첫번째 표본인 $(1, 2)$가 다음과 같은 경로를 타고 어떻게 자라나는지 생각해보자. 

<div class="figure" style="text-align: center">
<img src="./image/neuralnet3.png" alt="예시 경로 1" width="100%" />
<p class="caption">(\#fig:neuralnet-path)예시 경로 1</p>
</div>

그림에서 $\beta$는 노드와 노드 사이를 지나갈 때 부여되는 웨이트들을 의미하고, $\sigma()$는 다음의 시그모이드(sigmoid) 함수를 의미한다.

$$
\sigma(x) = \frac{1}{1+e^{-x}} = \frac{e^x}{e^x+1}
$$

자료 행렬을 위에 색칠된 경로로 보낸다는 의미는 다음과 같은 계산과정을 거친다는 것이다.


```r
set.seed(1234)

# 데이터 매트릭스 
# 3 by 2
X <- torch_tensor(matrix(1:2, ncol = 2, byrow = T),
                  dtype = torch_double())
X
```

```
#> torch_tensor
#>  1  2
#> [ CPUDoubleType{1,2} ]
```

```r
# beta_1 벡터 
# 2 by 1
# 1번째 레이어에 관한 웨이트 (베타) 중 
# 다음 레이어의 1번째 노드에 대한 베타 벡터에 부여
# beta_1 = (beta_11, beta_12)
beta_1 <- torch_tensor(matrix(runif(2), ncol = 1),
                       dtype = torch_double()) 
beta_1
```

```
#> torch_tensor
#>  0.1137
#>  0.6223
#> [ CPUDoubleType{2,1} ]
```

```r
# 2번째 레이어 1번째 노드
# 3 by 1
z_21 <- X$mm(beta_1)
z_21
```

```
#> torch_tensor
#>  1.3583
#> [ CPUDoubleType{1,1} ]
```

```r
# 2번째 레이어 1번째 노드에서의 시그모이드 함수 통과
# 3 by 1
library(sigmoid)
a_21 <- sigmoid(z_21)
a_21
```

```
#> torch_tensor
#>  0.7955
#> [ CPUDoubleType{1,1} ]
```

```r
# 2번째 레이어에 관한 웨이트 (감마) 중 
# 다음 레이어의 1번째 노드에 대한 베타값에 임의의 값을 부여
# beta_1 상수 1 by 1
gamma_1 <- runif(1)

# 3번째 레이어 1번째 노드
# 3 by 1
z_31 <- a_21 * gamma_1
z_31
```

```
#> torch_tensor
#>  0.4847
#> [ CPUDoubleType{1,1} ]
```

```r
# 마지막 레이어에서 시그모이드 함수 통과
# 3 by 1
y_hat <- sigmoid(z_31)
y_hat
```

```
#> torch_tensor
#>  0.6188
#> [ CPUDoubleType{1,1} ]
```

즉, 우리가 생각하는 표본은 빨간색 노드에 도착하기 위해서 두번째 은닉층의 첫번째 노드를 통과하여 올 수 있다. 하지만 빨간색 노드에는 방금 우리가 생각한 경로 뿐만아니라 두 개의 선택지가 더 존재한다.

### 1개의 표본, 경로 한꺼번에 생각하기

세가지의 경로를 모두 생각해보면, 우리의 표본은 다음의 경로를 통해서 도착한다.

    > 목표: 첫번째 표본인 $(1, 2)$가 다음과 같은 세가지 경로를 타고 어떻게 하나로 합쳐지는지 이해해보자. 


<div class="figure" style="text-align: center">
<img src="./image/neuralnet3.png" alt="3가지 경로" width="100%" />
<p class="caption">(\#fig:neuralnet-allpath)3가지 경로</p>
</div>

이 과정을 우리가 통계 시간에 배운 회귀분석에 연결지어 생각해보면, 다음의 해석이 가능하다. 두번째 은닉층의 각각의 노드들이 하나의 회귀분석 예측 모델들이라고 생각하면, 신경망은 세 개의 회귀분석을 한 대 모아놓은 거대한 회귀분석 집합체라고 생각할 수 있게 된다. 즉, 각 회귀분석 모델들이 예측한 표본에 대한 대응변수 예측값들을 은닉층에 저장한 후, 그 예측값들을 모두 모아 마지막 빨간색 노드에서 합치면서 좀 더 좋은 예측값을 만들어 내는 것이다. 이 때, $\gamma$ 벡터를 통해 가중치를 부여하는 것이라고 해석이 가능하다. 

이 과정을 `torch` 텐서를 사용하여 깔끔하게 나타내보자.


```r
# 1개 표본 
# 1 by 2
X <- torch_tensor(matrix(1:2, ncol = 2, byrow = T),
                  dtype = torch_double()) 
X
```

```
#> torch_tensor
#>  1  2
#> [ CPUDoubleType{1,2} ]
```

```r
# 베타벡터가 세 개 존재함.
# 2 by 3
beta_1 <- torch_tensor(matrix(runif(2), ncol = 1),
                       dtype = torch_double())
beta_2 <- torch_tensor(matrix(runif(2), ncol = 1),
                       dtype = torch_double()) 
beta_3 <- torch_tensor(matrix(runif(2), ncol = 1),
                       dtype = torch_double()) 

# 정의된 베타벡터를 cbind in torch
beta <- torch_cat(c(beta_1, beta_2, beta_3), 2)
beta
```

```
#> torch_tensor
#>  0.6234  0.6403  0.2326
#>  0.8609  0.0095  0.6661
#> [ CPUDoubleType{2,3} ]
```

```r
# 2번째 레이어 z_2
# 1 by 3
z_2 <- X$mm(beta)
z_2
```

```
#> torch_tensor
#>  2.3452  0.6593  1.5647
#> [ CPUDoubleType{1,3} ]
```

```r
# 2번째 레이어 sigmoid 함수 통과
# 1 by 3
a_2 <- sigmoid(z_2)

# 2번째 레이어에 관한 웨이트 (감마) 벡터 
# 다음 레이어의 1번째 노드에 대한 베타값에 임의의 값을 부여
# gamma vector 3 by 1
gamma_1 <- runif(1)
gamma_2 <- runif(1)
gamma_3 <- runif(1)
gamma <- torch_tensor(matrix(c(gamma_1,
                               gamma_2, 
                               gamma_3), ncol = 1),
                      dtype = torch_double())

# 3번째 레이어 z_3
# 1 by 1
z_3 <- a_2$mm(gamma)
z_3
```

```
#> torch_tensor
#>  1.3771
#> [ CPUDoubleType{1,1} ]
```

```r
# 마지막 레이어에서 시그모이드 함수 통과
# 1 by 1
y_hat <- sigmoid(z_3)
y_hat
```

```
#> torch_tensor
#>  0.7985
#> [ CPUDoubleType{1,1} ]
```

`R`에서 우리가 즐겨쓰던 `cbind()`와 `rbind()`는 torch에서는 `torch_cat()` 하나의 함수으로 구현이 가능하다. 함수의 두번째 입력값은 숫자 1은 행방향(rbind)에, 2는 열방향(cbind)과 대응된다.

### 전체 표본, 경로 전체 생각해보기

이제 자료 행렬 전체를 한꺼번에 넣는 방법을 생각해보자. 입력값이 자료 행렬 전체이므로, 결과값은 이에 대응하도록 행의 갯수와 같은 벡터 형식이 될 것이라는 것을 예상하고 코드를 따라오도록 하자.

    > 목표: 전체 표본이 신경망을 통해서 예측되는 구조를 이해하자. 



```r
# 데이터 텐서 
# 3 by 2
X <- torch_tensor(matrix(1:6, ncol = 2, byrow = T),
                  dtype = torch_double()) 
X
```

```
#> torch_tensor
#>  1  2
#>  3  4
#>  5  6
#> [ CPUDoubleType{3,2} ]
```

```r
# 베타벡터가 세 개 존재함.
# 2 by 3
beta <- torch_tensor(matrix(runif(6), ncol = 3),
                     dtype = torch_double())
beta
```

```
#> torch_tensor
#>  0.2827  0.2923  0.2862
#>  0.9234  0.8373  0.2668
#> [ CPUDoubleType{2,3} ]
```

```r
# 2번째 레이어 z_2
# 3 by 3
z_2 <- X$mm(beta)
z_2
```

```
#> torch_tensor
#>  2.1296  1.9669  0.8199
#>  4.5419  4.2261  1.9260
#>  6.9543  6.4854  3.0320
#> [ CPUDoubleType{3,3} ]
```

```r
# 2번째 레이어 sigmoid 함수 통과
# 3 by 3
a_2 <- sigmoid(z_2)

# 2번째 레이어에 관한 웨이트 (감마) 벡터 
# 다음 레이어의 1번째 노드에 대한 베타값에 임의의 값을 부여
# gamma vector 3 by 1
gamma <- torch_tensor(matrix(runif(3), ncol = 1),
                      dtype = torch_double())

# 3번째 레이어 z_3
# 3 by 1
z_3 <- a_2$mm(gamma)
z_3
```

```
#> torch_tensor
#>  0.5904
#>  0.6900
#>  0.7205
#> [ CPUDoubleType{3,1} ]
```

```r
# 마지막 레이어에서 시그모이드 함수 통과
# 3 by 1
y_hat <- sigmoid(z_3)
y_hat
```

```
#> torch_tensor
#>  0.6435
#>  0.6660
#>  0.6727
#> [ CPUDoubleType{3,1} ]
```
