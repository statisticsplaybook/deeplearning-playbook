# 나의 첫 신경망 학습

저번 시간 우리는 토치에서 신경망을 정의하는 방법에 대하여 알아보았다. 오늘은 정의한 신경망을 어떻게 학습하는가에 대하여 알아보도록 하자. 

## 학습 준비 - 데이터 만들기

필자는 [유튜브에 R을 사용한 통계관련 수업](https://youtube.com/playlist?list=PLKtLBdGREmMnLbQnqGEfpCBtkGj2g_d-B)들을 올려놓았다. 이 수업에서 큰 축을 이루는 것 중 하나가 바로 회귀분석이다. 회귀분석은 주어진 데이터를 모델링할 때 신경망의 가장 큰 장점은 회귀직선과 같은 선형모형들이 가지는 한계를 넘어서, 비선형 모델링을 할 수 있게 해준다는 것이다. 이러한 장점들을 잘 확인해보기 위해서 비선형 모델에서 관찰값을 뽑아 모의 데이터로 만들어 보도록 하자.


```r
library(tidyverse)

# 재현 가능을 위한 시드 고정
set.seed(2021)

# x 자리 임의 생성
x <- sort(sample(1:100, 100))

# 모델을 위한 f 함수 정의
f <- function(x){
    x + 30 * sin(0.1 * x)
}

# noise을 가미한 관찰값 생성
y <- f(x) + 5 * rnorm(100)

obs_data <- tibble(x = x, y = y)
head(obs_data)
```

```
#> # A tibble: 6 x 2
#>       x     y
#>   <int> <dbl>
#> 1     1 10.1 
#> 2     2  7.89
#> 3     3  2.88
#> 4     4 17.0 
#> 5     5 21.9 
#> 6     6 19.1
```
관찰값 $y$가 발생되는 코드를 살펴보면, $y$는 발생되는 실제 함수 $f$는 다음과 같이 비선형성을 가지고 있고, 거기에 잡음이 섞여서 관찰되는 형태를 띄고 있다.

$$
\begin{equation}
\begin{aligned} 
f(x) &= x + 30 sin(0.1 x), \\
y & = f(x) + \epsilon, \quad \epsilon \sim \mathbb{N}(0, 5^2)
\end{aligned}
(\#eq:model-sample)
\end{equation}
$$

관찰값과 모델 함수 그려보도록 하자. 모델 함수의 경우 점선으로 표시했다.


```r
library(ggthemes)
library(latex2exp)

theme_set(theme_igray())

x_true <- 1:100

model_data <- tibble(x = x_true, y = f(x_true))

# 관찰값 시각화
p <- obs_data %>% 
    ggplot(aes(x = x, y = y)) +
    geom_point(color = "#E69F00") +
    labs(x = "x", y = "f(x)",
         caption = "https://www.youtube.com/c/statisticsplaybook")
p + geom_line(data = model_data,
              aes(x = x, y = y),
              linetype = "dashed") # 모델 함수
```

<div class="figure" style="text-align: center">
<img src="08-train-mynn_files/figure-html/data-vis-1.png" alt="샘플 데이터 시각화. 비선형성이 잘 드러나있다." width="100%" />
<p class="caption">(\#fig:data-vis)샘플 데이터 시각화. 비선형성이 잘 드러나있다.</p>
</div>

위에 주어진 관찰값을 사용해서 회귀직선을 구해보면 다음과 같이 회색의 직선을 구할 수 있다.

<div class="figure" style="text-align: center">
<img src="08-train-mynn_files/figure-html/add-regression-1.png" alt="회귀직선(회색 직선)은 자료를 가장 잘 설명하는 선형모델로 볼 수 있다." width="100%" />
<p class="caption">(\#fig:add-regression)회귀직선(회색 직선)은 자료를 가장 잘 설명하는 선형모델로 볼 수 있다.</p>
</div>

추후 신경망 모델과의 비교를 위해서 회귀직선과 관찰값 사이의 잔차들의 제곱의 평균을 구해놓자.


```r
model <- lm(y ~ x, obs_data)

# Mean Squared Error for train data
mean((model$residuals)^2)
```

```
#> [1] 431.8235
```

## 신경망과 블랙박스(Black-box)

Figure \@ref(fig:add-regression) 을 보면 회귀 직선도 사실 자료의 `x`값에 따른 함수값의 변화를 아주 잘 잡아내는 것을 알 수 있다. 하지만, 우리가 자료를 발생시키는 함수의 구조가 비선형을 띈다는 것을 알고 있는 상태에서 보면(현실에서는 아무도 모른다.), 비선형성을 잡아내지 못하는 회귀 직선의 한계가 뚜렷하게 보인다. 따라서 통계학에서는 이러한 비선형성을 잡아내기 위해서 일반화선형모형(General Linear Model)이나 [일반화 가법모형](https://bookdown.org/cardiomoon/gam/)(Generalized Additive Model; GAM)^[링크는 가톨릭대 문건웅 교수님이 쓴 일반화 가법모델에 대한 내용이다. R코드와 함께 친절하게 설명이 되어있다.] 등 여러가지 기법들이 발달했다. 신경망 역시 이것들의 연장선 상에 위치하는 모델이라고 생각해도 무방하다. 

다만 모델의 해석적인 측면에서 일반화 선형모형에서는 모델을 만드는 사람들이 비선형성을 부여하는 툴들을 (예를 들어 link 함수를 자료를 보고 사용자가 선택한다.) 조절해서, 모델의 결과 해석력이 우수했다면, 일반화 가법모형과 신경망으로 갈 수록 사용자가 모델의 비선형성을 조절한다기모다 기법 자체가 비선형성을 잘 다루도록 설계가 되어있어서 해석력은 떨어지고 예측력은 증가했다. 신경망은 특히 모델 자체에 자유도를 높이고, 대량의 데이터를 사용하여 학습하면서 실제 함수를 찾아가는 방식이라서, 학습된 모델의 해석이 거의 불가능하게 되어버려서 블랙박스(Black box - 안이 어떻게 돌아가는지 모름) 모델이라는 별명이 생겼다.

모델러 입장에서는 '뭐가 뭔지는 모르겠는데, 예측은 잘한다'라는 느낌이 드는 아이인데, 실제 성능적인 측면에서 기존 모델보다 월등하게 잘 예측을 하기 때문에, 신경망의 핫하게 된 이유가 되었다.

개인적으로 필자는 책의 뒷부분에 소개할 신경망의 여러 구조들이 결국에는 신경망의 모수 학습시(실제 함수를 찾아갈 때) 데이터를 넣었을 때 발산하지 않고 실제 함수로 잘 수렴하도록 잘 이끌어주는 신경망 구조를 만들어가는 것이라고 생각하고 있다. 즉, 대략적인 구조를 잡아주고, 그 안에서 데이터를 사용하여 tuning을 하는 방식이다. 어찌보면 이러한 과정은 통계에서 특정 조건을 만족하는 함수들의 집합을 정의하고 (회귀분석의 경우는 선형 함수들만을 생각하고), 그 안에서 최적 모델을 찾아가는 방식과 유사하다.

## 신경망 학습

앞에서 만들어낸 데이터를 사용하여, 신경망을 학습하도록 하자. 
신경망을 학습한다는 이야기를 통계적으로 보면 주어진 혹은 설정한 손실 함수(loss function) 값을 최소화 시키는 신경망의 모수(weights)값을 찾는다는 이야기이다. 이런 최적 모수값 찾는 방법에는 여러가지가 있는데, `torch`에서는 이제까지 제안된 많은 방법들이 최적화 함수 (optimizer) 클래스 형식으로 제공이 된다. 당연한 것이겠지만 어떤 최적화 함수를 사용하느냐에 따라서 학습 결과가 달라진다.

앞에서 정의한 신경망 코드를 가져오자.


```r
library(torch)

torch_manual_seed(2021)

TwoLayerNet <- nn_module(
    classname = "TowLayerNet",
    initialize = function(data_in, hidden, data_out){
        
        cat("Initiation complete!")
        
        self$hidden1 <- nn_linear(data_in, hidden)
        self$hidden2 <- nn_linear(hidden, hidden)
        self$hidden3 <- nn_linear(hidden, hidden)
        self$output_layer <- nn_linear(hidden, data_out)
        self$tanh <- nn_tanh()
    },
    # 순전파 멤버함수 forward 정의 부분
    forward = function(X) {
        x <- self$tanh(self$hidden1(X))
        x <- self$tanh(self$hidden2(x))
        x <- self$hidden3(x)
        y_hat <- self$output_layer(x)
        return(y_hat)
    }
)

library(zeallot)

# GPU available
cuda_is_available()
```

```
#> [1] TRUE
```

```r
gpu <- torch_device("cuda")

x_tensor <- torch_tensor(scale(x), dtype = torch_float(),
                         requires_grad = TRUE,
                         device = gpu)$view(c(-1, 1))
y_tensor <- torch_tensor(y, dtype = torch_float(),
                         device = gpu)$view(c(-1, 1))

c(D_in, H, D_out) %<-%  c(1, 10, 1)
my_net <- TwoLayerNet(D_in, H, D_out)
```

```
#> Initiation complete!
```

```r
my_net$cuda()
my_net
```

```
#> An `nn_module` containing 251 parameters.
#> 
#> -- Modules ---------------------------------------------------------------------
#> * hidden1: <nn_linear> #20 parameters
#> * hidden2: <nn_linear> #110 parameters
#> * hidden3: <nn_linear> #110 parameters
#> * output_layer: <nn_linear> #11 parameters
#> * tanh: <nn_tanh> #0 parameters
```

### 손실함수와 최적화 방법 선택

토치에서는 많은 손실함수 (loss function)와 최적화 함수 (optimizer) 를 [모두 제공](https://torch.mlverse.org/docs/reference/index.html)하는데, 그 중 가장 기본적인 손실함수인 MSE(Mean Squared Error)와 최적화 방법 SGD(Stochastic Gradient Desent) 방법을 사용하도록 하자. 둘은 다음과 같은 방법으로 선언한다.


```r
mse_loss <- nn_mse_loss(reduction = "mean")
optimizer <- optim_sgd(my_net$parameters, lr = 1e-5) # 
```

손실함수와 최적화 방법에 대한 깊은 내용은 다른 챕터에서 다루도록 하고, 일단 간단하게 정리만 해보자. 지금은 신경망이 어떤 식의 구조를 가진 코드로 학습할 수 있는지 집중한다.

1. `nn_mse_loss()`

`nn_mse_loss` 함수의 경우, 다음의 두 가지 타입 손실함수를 제공한다. `reduction` 옵션을 `sum` 설정할 경우 손실함수는 다음과 같다.

$$
L(\hat{\boldsymbol{y}}, \boldsymbol{y}) = \sum_i^{n}(\hat{y_i}-y_i)^2
$$
혹은 `reduction` 옵션을 `mean`으로 설정할 경우 손실함수는 MES를 반환한다.

$$
L(\hat{\boldsymbol{y}}, \boldsymbol{y}) = \frac{1}{n}\sum_i^{n}(\hat{y_i}-y_i)^2
$$

참고로 `none`으로 설정시 입력한 두 벡터의 차이의 제곱값들이 벡터 형식으로 나온다.

앞에 신경망 정의에서 보았듯 히든 레이어를 지날 때, activation 함수를 통과하므로, **로스값 역시 어떤 activation 함수를 사용하느냐에 따라서 달라질 수 있다는 것을 염두해두자**. 주어진 데이터에 대한 손실 함수 값은 다음과 같이 구할 수 있다.


```r
y_hat <- my_net(x_tensor)
mse_loss(y_hat, y_tensor)
```

```
#> torch_tensor
#> 4268.52
#> [ CUDAFloatType{} ]
```

1. `optim_sgd()`

최적화 함수에 대하여는 나중에 따로 포스트로 다루겠다. 현재는 `optim_sgd`가 토치에서 제공하는 최적화 함수 중 하나이며, 입력값으로 신경망의 모수(weights)와 학습률(learning rate), `lr`,을 받는다는 것을 알아두자. 

학습률(learning rate)은 자동 미분 챕터에서 다뤘던 경사하강도 알고리즘을 설명했던 부분에서도 다뤘는데, 신경망 학습 과정에서 중요한 역할을 차지한다. 이것에 따라서 학습이 잘 될 수도, 그렇지 않을 수도 있다. 보통 신경망이 복잡해 질 수록 학습률은 좀 더 세밀한 탐색을 위해 작게 잡아준다. 하지만, 학습률이 작은 경우에는 신경망을 학습하는 시간이 길어지게 된다. 최적은 학습률을 정하는 주제는 학문적으로도 아주 중요하고 방대한 주제이다. 한가지 예만 들면, 굳이 우리가 신경망을 학습시킬때 [학습률을 동일하게 고정할 필요가 있을까?](https://www.jeremyjordan.me/nn-learning-rate/) 어떻게 보면 너무나 중용하고, 실무적인(당장 신경망 학습에 막대한 영향을 미치므로), 연구 주제같다.

### 학습 구현

경사하강법에서 모수가 점점 업데이트 되면서 최적값으로 수렴하는 것을 보았다. 이렇게 업데이트 한번 진행이 되는 단계 단계를 딥러닝에서는 epoch라고 한다. 보통 데이터가 너무 많은 경우 전체 데이터를 한꺼번에 사용하는 것이 아니라 작은 단위로 잘라서 컴퓨터 메모리에 올리게 되는데, 이렇게 작게 잘린 데이터 단위를 배치(batch)라고 하며, 배치의 크기는 배치 안에 몇 개의 데이터가 들어가 있는가를 의미한다. 이와 관련한 내용은 추후에 데이터셋(Dataset) 클래스와 데이터 로더(Data loader) 클래스를 다룰 때 다시 자세하게 이야기하도록 한다.

다음의 코드는 `mse_loss` 값을 업데이트 단계마다 저장하고, 총 1000번의 모수 업데이트를 수행하여 신경망의 모수를 학습시키는 코드이다.  


```r
store_loss <- rep(0, 50000)
for (epoch in 1:50000){
    optimizer$zero_grad()
    output <- my_net(x_tensor)
    loss <- mse_loss(output, y_tensor)
    loss$backward()
    optimizer$step()
    store_loss[epoch] <- as.numeric(loss$item())
  
    if (epoch %% 5000 == 0){
        cat(sprintf("Loss at epoch %d: %.2f\n", epoch, store_loss[epoch]))
    }
}
```

```
#> Loss at epoch 5000: 123.00
#> Loss at epoch 10000: 94.12
#> Loss at epoch 15000: 86.84
#> Loss at epoch 20000: 81.58
#> Loss at epoch 25000: 76.47
#> Loss at epoch 30000: 70.47
#> Loss at epoch 35000: 58.99
#> Loss at epoch 40000: 41.10
#> Loss at epoch 45000: 28.03
#> Loss at epoch 50000: 23.86
```


### 시각화

이전 섹션에서 우리는 신경망의 학습이 진행되면서 손실함수(loss)값이 점점 줄어드는 것을 확인할 수 있었다. 최종적으로 학습된 신경망은 어떻게 생겼을까? .

<div class="figure" style="text-align: center">
<img src="08-train-mynn_files/figure-html/vis-result-1.png" alt="학습된 신경망과 회귀직선 비교" width="100%" />
<p class="caption">(\#fig:vis-result)학습된 신경망과 회귀직선 비교</p>
</div>

학습된 신경망이 데이터가 발생되는 함수의 비선형성을 잘 반영하고 있는 것을 확인할 수 있다. 

## 과적합(overfitting)과의 싸움

이렇게 학습한 신경망은 너무나도 완벽해 보이지만, 사실 중대한 문제점이 있다. 바로 학습에 사용된 데이터에 나타난 패턴을 너무나도 잘 반영하고 있는 것이 문제이다. 사실 뭐가 문제냐 싶지만, 우리가 흔히 말하는 "이론과 현실은 달라요." 라는 말이 신경망 학습에서도 그대로 적용이 된다고 생각하면 된다.

즉, 학습 데이터를 너무나 잘 반영하는 것도 좋지만, 이렇게 학습된 신경망을 사용해서 예측을 할 때, 신경망에 입력 될 데이터는 대부분 학습 데이터와 비슷하기도 하겠지만, 비슷하지 않은 전혀 다른 입력값이 들어올 수 있다. 이런 상황에 잘 대비하기(?) 위해서 혹은 신경망이 성능을 잘 내기 위해서는 신경망을 학습을 할 때 성과측정을 신경망의 학습에 한번도 사용되지 않는 새로운 데이터로 평가를 해야만 한다.

이렇게 모델이 학습 데이터 패턴을 너무나 많이 반영하고 있는 현상을 과하게 적합이 되어 있다고 하여 모델 과적합(overfitting) 상태라고 부른다. 기계학습과 딥러닝에서는 일단 베이스라인 모델이 정해진 후에는 어떤 모수값(weights)이 최적의 모수인지를 찾아내야하는 과정을 거친다. 이 과정을 거치는 가장 큰 이유는 모델 과적합 방지에 있다. 어떻게 하면 학습 데이터의 패턴은 잘 반영하면서, 새로이 들어올 데이터에도 잘 반응할 수 있는 모델을 세울 수 있을지 앞으로의 학습을 통해서 차근차근 배워보자.
