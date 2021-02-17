# 텐서의 이동; CPU $\leftrightarrow$  GPU



딥러닝(deep learning)에서는 네트워크의 구조가 조금만 복잡해져도, 필요한 계산량이 엄청나게 늘어나기 때문에 GPU는 사실 필수적이다. `torch` 패키지에서는 텐서를 다룰때에 현재 다루는 텐서가 어디에 저장되어있는가에 대한 일종의 태그를 달아놓는다. 다음의 코드를 살펴보자.


```r
a <- torch_tensor(1:4)
a
```

```
#> torch_tensor
#>  1
#>  2
#>  3
#>  4
#> [ CPULongType{4} ]
```

`a`는 3이라는 상수가 담겨있는 텐서이다. 이 `a`를 콘솔에서 돌렸을때에 나오는 결과 `[ CPUFloatType{1} ]`를 통해서 우리는 a가 현재 CPU의 메모리를 이용하고 있으며, 모양은 `{1}`인 실수을 담은 텐서라는 것을 알 수 있다.

## GPU 사용 가능 체크

앞서 정의한 텐서 `a`를 GPU의 메모리로 옮기기 위해서는, 너무나 당연하게 GPU가 현재 시스템에서 접근 가능한지에 대하여 알아보아야한다. GPU 접근성은 `cuda_is_available()`을 사용한다.


```r
cuda_is_available()
```

```
#> [1] TRUE
```

## CPU to GPU

이미 정의된 텐서 a를 GPU로 옮기려면 다음과 같이 `cuda()` 함수를 이용하면 된다.


```r
a
```

```
#> torch_tensor
#>  1
#>  2
#>  3
#>  4
#> [ CPULongType{4} ]
```

```r
a$cuda()
```

```
#> torch_tensor
#>  1
#>  2
#>  3
#>  4
#> [ CUDALongType{4} ]
```


```r
gpu <- torch_device("cuda")
a$to(device = gpu)
```

```
#> torch_tensor
#>  1
#>  2
#>  3
#>  4
#> [ CUDALongType{4} ]
```

옮길 때에 `dtype`을 사용하여 다음과 같이 자료형을 바꿔줄 수도 있다.


```r
a$to(device = gpu, dtype = torch_double())
```

```
#> torch_tensor
#>  1
#>  2
#>  3
#>  4
#> [ CUDADoubleType{4} ]
```

## GPU to CPU

GPU 상에 직접 텐서를 만드는 방법은 다음과 같다.


```r
b <- torch_tensor(1:4, device=gpu)
b
```

```
#> torch_tensor
#>  1
#>  2
#>  3
#>  4
#> [ CUDALongType{4} ]
```

이전 섹션에서 CPU에서 GPU로 옮기는 방법과 비슷하게 다음의 코드가 작동한다.


```r
b$cpu()
```

```
#> torch_tensor
#>  1
#>  2
#>  3
#>  4
#> [ CPULongType{4} ]
```

```r
# to 함수 이용
cpu <- torch_device("cpu")
a$to(device = cpu)
```

```
#> torch_tensor
#>  1
#>  2
#>  3
#>  4
#> [ CPULongType{4} ]
```

