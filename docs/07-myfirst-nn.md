# `torch_nn` 모듈로 첫 신경망 정의하기

이제까지 `torch`의 자동미분(auto grad) 기능과 순전파(forward propagation)에 대하여 알아보았다. 오늘은 드디어, `torch` 라이브러리에서 제공하는 함수들을 이용해서 챕터 \@ref(forward) 에서 정의해본 신경망을 정의해 보도록 한다.

<div class="figure" style="text-align: center">
<img src="./image/neuralnet1.png" alt="다시 두두등장! 세상 간단한 신경망" width="100%" />
<p class="caption">(\#fig:neuralnet-example2)다시 두두등장! 세상 간단한 신경망</p>
</div>

## 신경망 정의 (Custom nn Modules)

토치를 사용해서 신경망을 정의할 때 사용하는 함수가 있다. 바로 `nn_module()`이라는 함수인데, `torch`에서 신경망을 정의할 때, 이 함수를 사용해서 "클래스"를 만들어 정의한다! 왜 우리가 챕터 \@ref(r6)에서 R6관련 클래스 내용을 그렇게도 공부했었는지에 대한 답을 바로 이 챕터에서 찾을 수 있을 것이다.

### `nn_module`과 클래스 상속

`nn_module`이 어떤 역할을 하는지에 대하여 알아보기 위해 가장 간단한 신경망을 작성해보도록 하자. 바로 우리가 앞서 살펴본 2단 레이어 네트워크 예제에서 사용한 데이터를 만들어 보자.


```r
library(torch)

X <- torch_tensor(matrix(1:6, ncol = 2, byrow = T),
                  dtype = torch_float()) 
X
```

```
#> torch_tensor
#>  1  2
#>  3  4
#>  5  6
#> [ CPUFloatType{3,2} ]
```

먼저, `TwoLayerNet`이라는 이름의 신경망 클래스를 정의한다(기억하시나? 클래스의 이름은 카멜 형식이다!). `nn_module()` 함수는 클래스를 정의하는 함수인데, 이 함수를 사용해서 만들어진 클래스는 자동으로 신경망과 관련한 클래스인 `basic-nn-module` 클래스를 상속하게 만든다. 즉, `nn_module`안에는 신경망 관련 클래스들 속에는 신경망과 관련한 많은 함수가 정의되어 있을 것이고, 이것을 다 상속받아서 클래스가 만들어지는 것이다. 다음의 코드는 위의 신경망을 정의한 코드이다.


```r
TwoLayerNet <- nn_module(
    classname = "TowLayerNet",
    initialize = function(data_in, hidden, data_out){
        
        cat("Initiation complete!")
        
        self$hidden_layer <- nn_linear(data_in, hidden, bias=FALSE)
        self$output_layer <- nn_linear(hidden, data_out, bias=FALSE)
        
    }
)

myfirst_model <- TwoLayerNet(2, 3, 1)
```

```
#> Initiation complete!
```

```r
myfirst_model
```

```
#> An `nn_module` containing 9 parameters.
#> 
#> -- Modules ---------------------------------------------------------------------
#> * hidden_layer: <nn_linear> #6 parameters
#> * output_layer: <nn_linear> #3 parameters
```

결과를 살펴보면 `TwoLayerNet` 클래스에 의하여 만들어진 `myfirst_model`는 두 개의 층이 들어있는 것을 확인할 수 있다. 이 두개 층에 관련한 모수 갯수를 그림과 한번 연결 시켜보면 잘 정의가 되어있다는 것을 알 수 있다.

* hidden_layer: 그림에서 첫번째와 두번째 층을 연결하는 다리가 6개라는 것을 주목하자. 모수의 갯수는 그래서 6개!
* output_layer: 그림에서 두번째와 마지막 층을 연결하는 다리는 3개이므로, 모수의 갯수는 3개가 된다.

## `nn_linear` 클래스

`nn_linear`의 입력값은 입력변수의 갯수, 출력변수의 갯수, 그리고 bias 항의 유무를 나타내는 옵션 이렇게 세개가 된다. 예제의 경우, 데이터 텐서 $X$의 features 갯수가 2개이므로, 히든 레이어의 입력값 갯수가 2개가 되어야 한다. 또한 히든 레이어의 노드 갯수가 3개이므로 결과 행력의 features 갯수가 3개가 되어야 한다. 

### bias 없는 경우

우리가 예전에 다루었던 예제에서는 `bias` 항이 없었으므로, `bias=FALSE`를 해주어야 함에 주의하자.


```r
mat_op <- nn_linear(2, 3, bias = FALSE)
mat_op$weight
```

```
#> torch_tensor
#> -0.5329 -0.6421
#> -0.2611  0.3294
#>  0.6383 -0.4162
#> [ CPUFloatType{3,2} ]
```

`mat_op`을 nn.Linear(2, 3) 클래스로 만들어진 클래스 생성자로 이해 할 수 있다. 그리고 이것의 수학적 의미는 행렬 연산으로 이해할 수 있겠다. `mat_op`가 생성될 때 임의의 `weight` 텐서, $W$, 와 `bias`, $b$,가 생성이 되고, 입력값으로 들어오는 `X`에 대하여 다음의 연산을 수행한 후 결괏값을 내보낸다.

$$
y = X\beta = XW^T
$$

결과를 코드로 확인해보자.


```r
X$mm(mat_op$weight$t())
```

```
#> torch_tensor
#> -1.8171  0.3977 -0.1942
#> -4.1672  0.5342  0.2498
#> -6.5173  0.6708  0.6938
#> [ CPUFloatType{3,3} ]
```

```r
mat_op(X)
```

```
#> torch_tensor
#> -1.8171  0.3977 -0.1942
#> -4.1672  0.5342  0.2498
#> -6.5173  0.6708  0.6938
#> [ CPUFloatType{3,3} ]
```

### bias 있는 경우

`bias=TRUE`를 해주면 `weight` 텐서 $W$와 더불어 bias 텐서가 생성이 된다.


```r
mat_op2 <- nn_linear(2, 3, bias = TRUE)
mat_op2$weight
```

```
#> torch_tensor
#>  0.3400 -0.5988
#>  0.3999 -0.0115
#>  0.1655  0.1199
#> [ CPUFloatType{3,2} ]
```

```r
mat_op2$bias
```

```
#> torch_tensor
#>  0.5163
#>  0.0683
#> -0.5182
#> [ CPUFloatType{3} ]
```

따라서 정의된 신경망의 연산 역시 다음과 같이 바뀐다.

$$
y = X\beta + b = XW^T + b
$$


```r
X$mm(mat_op2$weight$t()) + mat_op2$bias
```

```
#> torch_tensor
#> -0.3413  0.4453 -0.1129
#> -0.8589  1.2221  0.4579
#> -1.3765  1.9990  1.0287
#> [ CPUFloatType{3,3} ]
```

```r
mat_op2(X)
```

```
#> torch_tensor
#> -0.3413  0.4453 -0.1129
#> -0.8589  1.2221  0.4579
#> -1.3765  1.9990  1.0287
#> [ CPUFloatType{3,3} ]
```

## 순전파(Forward propagation) 정의

`torch`를 공부하면서 신기한 걸 많이 배우고 있다. 그 중 한가지가 바로 객체지향 프로그래밍을 사용해서 신경망을 정의한다는 것이다. 앞선 예제를 이어가보면, 우리는 신경망의 순전파를 구현해야 한다.

순전파의 경우 다음과 같이 `forward` 멤버 함수를 정의해서 구현할 수 있다.


```r
TwoLayerNet <- nn_module(
    classname = "TowLayerNet",
    initialize = function(data_in, hidden, data_out){
        
        cat("Initiation complete!")
        
        self$hidden_layer <- nn_linear(data_in, hidden, bias=FALSE)
        self$output_layer <- nn_linear(hidden, data_out, bias=FALSE)
        self$sigmoid <- nn_sigmoid()
        
    },
    # 순전파 멤버함수 forward 정의 부분
    forward = function(X) {
        z1 <- self$hidden_layer(X)
        a1 <- self$sigmoid(z1)
        z2 <- self$output_layer(a1)
        y_hat <- self$sigmoid(z2)
        return(y_hat)
    }
)

library(zeallot)
c(D_in, H, D_out) %<-%  c(2, 3, 1)
my_net <- TwoLayerNet(D_in, H, D_out)
```

```
#> Initiation complete!
```

```r
my_net(X)
```

```
#> torch_tensor
#>  0.3694
#>  0.3493
#>  0.3458
#> [ CPUFloatType{3,1} ]
```

위의 코드를 한번 살펴보자. 먼저 `zeallot` 패키지는 `%<-%`를 포함하는 패키지인데, 여러 개의 변수에 한꺼번에 값을 부여하는 연산자이기 때문에 알아두면 편한 패키지 이다.

새로 정의된 `TwoLayerNet` 클래스에는 \@ref(fig:neuralnet-example2)의 2단 신경망의 순전파(forward propagation)가 구현된 멤버함수 `forward`가 정의되어 있다. 이 함수는 입력 텐서 `X`가 신경망으로 들어오게 되면, 은닉층(hidden_layer) $\rightarrow$ 활성함수 (activation function; 여기서는 nn_sigmoid 함수) $\rightarrow$ 출력층(output_layer) $\rightarrow$ 활성함수 순으로 내보내게 된다.


