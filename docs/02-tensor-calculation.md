# 텐서 (tensor) 연산 {#operation}

지난 챕터에서 우리는 텐서가 행렬의 연산에 적용되는 `%*%`과 호환이 되지 않는 다는 것을 알게되었다. 이번 챕터에서는 텐서들의 연산에 대하여 알아보도록 하자.

## 토치 (torch) 불러오기 및 준비물 준비

토치 (torch) 를 불러오고, 이번 챕터에 사용될 텐서 A, B, 그리고 C를 준비하자. 지난 챕터에서 배운 난수를 이용한 텐서도 만들 예정이니 난수를 고정한다.


```r
library(torch)

# 난수 생성 시드 고정 
torch_manual_seed(2021)
```


```r
A <- torch_tensor(1:6)
B <- torch_rand(2, 3)
C <- torch_rand(2, 3, 2)
A; B; C
```

```
#> torch_tensor
#>  1
#>  2
#>  3
#>  4
#>  5
#>  6
#> [ CPULongType{6} ]
```

```
#> torch_tensor
#>  0.5134  0.7426  0.7159
#>  0.5705  0.1653  0.0443
#> [ CPUFloatType{2,3} ]
```

```
#> torch_tensor
#> (1,.,.) = 
#>   0.9628  0.2943
#>   0.0992  0.8096
#>   0.0169  0.8222
#> 
#> (2,.,.) = 
#>   0.1242  0.7489
#>   0.3608  0.5131
#>   0.2959  0.7834
#> [ CPUFloatType{2,3,2} ]
```

만들어진 세 개의 텐서 결과를 살펴보면 다음과 같다.

1.  텐서 A: 정수들로 구성이 되어있고, 6개의 원소들이 벡터를 이루고 있다.
2.  텐서 B: 실수들로 구성이 되어있고, 똑같이 6개의 원소들이 있지만, 모양이 4행 3열인 2차원 행렬의 모양을 하고 있다.
3.  텐서 C: 실수들로 구성이 되어있고, 총 원소 갯수는 12개지만, 모양은 3행 2열의 행렬이 두개가 쌓여진 꼴의 3차원 배열 (array) 이다.

## 텐서의 연산

### 형(type) 변환

먼저 주목해야 할 것은 바로 텐서 A와 B의 자료형이 다르다는 것이다. 이게 무슨뜻이냐면 A에는 정수만이 담길 수 있고, B에는 실수만이 담길 수 있도록 설계가 되어있다는 것이다. 앞에서 확인한 자료형을 좀 더 명확하게 확인하기 위해서는 `type()` 사용한다.


```r
A$dtype
```

```
#> torch_Long
```

```r
B$dtype
```

```
#> torch_Float
```

텐서 A를 실수형 텐서로 바꿔보자. 텐서의 형을 변환할 때에는 A텐서 안에 속성으로 들어가있는 to() 함수를 사용 (좀 더 어려운 관점에서는 OOP의 method를 사용) 해서 바꿔줄 수 있다.


```r
A <- A$to(dtype = torch_double())
A
```

```
#> torch_tensor
#>  1
#>  2
#>  3
#>  4
#>  5
#>  6
#> [ CPUDoubleType{6} ]
```

torch에는 정말 많은 자료형이 있는데, 그 목록은 [다음](https://torch.mlverse.org/docs/reference/torch_dtype.html)을 참고하자.

### 모양 변환

앞에서 텐서 A를 B와 같은 실수를 담을 수 있는 형으로 바꾸었다. 그렇다면 이 두 개를 더할 수 있을까? 답은 "아니올시다." 이다. 왜냐하면 모양이 다르기 때문이다.


```r
A + B
```

```
#> Error in (function (self, other, alpha) : The size of tensor a (6) must match the size of tensor b (3) at non-singleton dimension 1
#> Exception raised from infer_size at ../aten/src/ATen/ExpandUtils.cpp:24 (most recent call first):
#> frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0x69 (0x7fe76f0afb89 in /home/issac/R/x86_64-pc-linux-gnu-library/4.0/torch/deps/./libc10.so)
#> frame #1: at::infer_size(c10::ArrayRef<long>, c10::ArrayRef<long>) + 0x552 (0x7fe75ebfe382 in /home/issac/R/x86_64-pc-linux-gnu-library/4.0/torch/deps/./libtorch_cpu.so)
#> frame #2: at::TensorIterator::compute_shape(at::TensorIteratorConfig const&) + 0xde (0x7fe75f100c2e in /home/issac/R/x86_64-pc-linux-gnu-library/4.0/torch/deps/./libtorch_cpu.so)
#> frame #3: at::TensorIterator::build(at::TensorIteratorConfig&) + 0x64 (0x7fe75f1031e4 in /home/issac/R/x86_64-pc-linux-gnu-library/4.0/torch/deps/./libtorch_cpu.so)
#> frame #4: at::TensorIterator::TensorIterator(at::TensorIteratorConfig&) + 0xdd (0x7fe75f10399d in /home/issac/R/x86_64-pc-linux-gnu-library/4.0/torch/deps/./libtorch_cpu.so)
#> frame #5: at::TensorIterator::binary_op(at::Tensor&, at::Tensor const&, at::Tensor const&) + 0x130 (0x7fe75f103b30 in /home/issac/R/x86_64-pc-linux-gnu-library/4.0/torch/deps/./libtorch_cpu.so)
#> frame #6: at::native::add(at::Tensor const&, at::Tensor const&, c10::Scalar) + 0x53 (0x7fe75edb6bc3 in /home/issac/R/x86_64-pc-linux-gnu-library/4.0/torch/deps/./libtorch_cpu.so)
#> frame #7: <unknown function> + 0x13311bd (0x7fe75f41d1bd in /home/issac/R/x86_64-pc-linux-gnu-library/4.0/torch/deps/./libtorch_cpu.so)
#> frame #8: <unknown function> + 0xaf2045 (0x7fe75ebde045 in /home/issac/R/x86_64-pc-linux-gnu-library/4.0/torch/deps/./libtorch_cpu.so)
#> frame #9: at::Tensor c10::Dispatcher::callWithDispatchKey<at::Tensor, at::Tensor const&, at::Tensor const&, c10::Scalar>(c10::TypedOperatorHandle<at::Tensor (at::Tensor const&, at::Tensor const&, c10::Scalar)> const&, c10::DispatchKey, at::Tensor const&, at::Tensor const&, c10::Scalar) const + 0x27f (0x7fe75f5c881f in /home/issac/R/x86_64-pc-linux-gnu-library/4.0/torch/deps/./libtorch_cpu.so)
#> frame #10: at::add(at::Tensor const&, at::Tensor const&, c10::Scalar) + 0x123 (0x7fe75f4befd3 in /home/issac/R/x86_64-pc-linux-gnu-library/4.0/torch/deps/./libtorch_cpu.so)
#> frame #11: <unknown function> + 0x2a0f2bb (0x7fe760afb2bb in /home/issac/R/x86_64-pc-linux-gnu-library/4.0/torch/deps/./libtorch_cpu.so)
#> frame #12: <unknown function> + 0xaf2045 (0x7fe75ebde045 in /home/issac/R/x86_64-pc-linux-gnu-library/4.0/torch/deps/./libtorch_cpu.so)
#> frame #13: at::Tensor c10::Dispatcher::callWithDispatchKey<at::Tensor, at::Tensor const&, at::Tensor const&, c10::Scalar>(c10::TypedOperatorHandle<at::Tensor (at::Tensor const&, at::Tensor const&, c10::Scalar)> const&, c10::DispatchKey, at::Tensor const&, at::Tensor const&, c10::Scalar) const + 0x27f (0x7fe75f5c881f in /home/issac/R/x86_64-pc-linux-gnu-library/4.0/torch/deps/./libtorch_cpu.so)
#> frame #14: at::add(at::Tensor const&, at::Tensor const&, c10::Scalar) + 0x123 (0x7fe75f4befd3 in /home/issac/R/x86_64-pc-linux-gnu-library/4.0/torch/deps/./libtorch_cpu.so)
#> frame #15: _lantern_add_tensor_tensor_scalar + 0x64 (0x7fe76f4310e4 in /home/issac/R/x86_64-pc-linux-gnu-library/4.0/torch/deps/liblantern.so)
#> frame #16: cpp_torch_namespace_add_self_Tensor_other_Tensor(Rcpp::XPtr<XPtrTorchTensor, Rcpp::PreserveStorage, &(void Rcpp::standard_delete_finalizer<XPtrTorchTensor>(XPtrTorchTensor*)), false>, Rcpp::XPtr<XPtrTorchTensor, Rcpp::PreserveStorage, &(void Rcpp::standard_delete_finalizer<XPtrTorchTensor>(XPtrTorchTensor*)), false>, Rcpp::XPtr<XPtrTorchScalar, Rcpp::PreserveStorage, &(void Rcpp::standard_delete_finalizer<XPtrTorchScalar>(XPtrTorchScalar*)), false>) + 0x48 (0x7fe76fd76fe8 in /home/issac/R/x86_64-pc-linux-gnu-library/4.0/torch/libs/torchpkg.so)
#> frame #17: _torch_cpp_torch_namespace_add_self_Tensor_other_Tensor + 0x9c (0x7fe76fb0f00c in /home/issac/R/x86_64-pc-linux-gnu-library/4.0/torch/libs/torchpkg.so)
#> frame #18: <unknown function> + 0xf9310 (0x7fe78572a310 in /usr/lib/R/lib/libR.so)
#> frame #19: <unknown function> + 0xf9826 (0x7fe78572a826 in /usr/lib/R/lib/libR.so)
#> frame #20: <unknown function> + 0x137106 (0x7fe785768106 in /usr/lib/R/lib/libR.so)
#> frame #21: Rf_eval + 0x180 (0x7fe7857746f0 in /usr/lib/R/lib/libR.so)
#> frame #22: <unknown function> + 0x14550f (0x7fe78577650f in /usr/lib/R/lib/libR.so)
#> frame #23: Rf_applyClosure + 0x1c7 (0x7fe7857772d7 in /usr/lib/R/lib/libR.so)
#> frame #24: Rf_eval + 0x353 (0x7fe7857748c3 in /usr/lib/R/lib/libR.so)
#> frame #25: <unknown function> + 0xc650d (0x7fe7856f750d in /usr/lib/R/lib/libR.so)
#> frame #26: <unknown function> + 0x137106 (0x7fe785768106 in /usr/lib/R/lib/libR.so)
#> frame #27: Rf_eval + 0x180 (0x7fe7857746f0 in /usr/lib/R/lib/libR.so)
#> frame #28: <unknown function> + 0x14550f (0x7fe78577650f in /usr/lib/R/lib/libR.so)
#> frame #29: Rf_applyClosure + 0x1c7 (0x7fe7857772d7 in /usr/lib/R/lib/libR.so)
#> frame #30: <unknown function> + 0x13a989 (0x7fe78576b989 in /usr/lib/R/lib/libR.so)
#> frame #31: Rf_eval + 0x180 (0x7fe7857746f0 in /usr/lib/R/lib/libR.so)
#> frame #32: <unknown function> + 0x14550f (0x7fe78577650f in /usr/lib/R/lib/libR.so)
#> frame #33: Rf_applyClosure + 0x1c7 (0x7fe7857772d7 in /usr/lib/R/lib/libR.so)
#> frame #34: <unknown function> + 0x13a989 (0x7fe78576b989 in /usr/lib/R/lib/libR.so)
#> frame #35: Rf_eval + 0x180 (0x7fe7857746f0 in /usr/lib/R/lib/libR.so)
#> frame #36: <unknown function> + 0x14550f (0x7fe78577650f in /usr/lib/R/lib/libR.so)
#> frame #37: Rf_applyClosure + 0x1c7 (0x7fe7857772d7 in /usr/lib/R/lib/libR.so)
#> frame #38: <unknown function> + 0x13a989 (0x7fe78576b989 in /usr/lib/R/lib/libR.so)
#> frame #39: Rf_eval + 0x180 (0x7fe7857746f0 in /usr/lib/R/lib/libR.so)
#> frame #40: <unknown function> + 0x14550f (0x7fe78577650f in /usr/lib/R/lib/libR.so)
#> frame #41: Rf_applyClosure + 0x1c7 (0x7fe7857772d7 in /usr/lib/R/lib/libR.so)
#> frame #42: <unknown function> + 0x13a989 (0x7fe78576b989 in /usr/lib/R/lib/libR.so)
#> frame #43: Rf_eval + 0x180 (0x7fe7857746f0 in /usr/lib/R/lib/libR.so)
#> frame #44: <unknown function> + 0x14550f (0x7fe78577650f in /usr/lib/R/lib/libR.so)
#> frame #45: Rf_applyClosure + 0x1c7 (0x7fe7857772d7 in /usr/lib/R/lib/libR.so)
#> frame #46: <unknown function> + 0x12d83b (0x7fe78575e83b in /usr/lib/R/lib/libR.so)
#> frame #47: <unknown function> + 0x9021b (0x7fe7856c121b in /usr/lib/R/lib/libR.so)
#> frame #48: Rf_eval + 0x706 (0x7fe785774c76 in /usr/lib/R/lib/libR.so)
#> frame #49: <unknown function> + 0x149782 (0x7fe78577a782 in /usr/lib/R/lib/libR.so)
#> frame #50: <unknown function> + 0x137106 (0x7fe785768106 in /usr/lib/R/lib/libR.so)
#> frame #51: Rf_eval + 0x180 (0x7fe7857746f0 in /usr/lib/R/lib/libR.so)
#> frame #52: <unknown function> + 0x14550f (0x7fe78577650f in /usr/lib/R/lib/libR.so)
#> frame #53: Rf_applyClosure + 0x1c7 (0x7fe7857772d7 in /usr/lib/R/lib/libR.so)
#> frame #54: <unknown function> + 0x13a989 (0x7fe78576b989 in /usr/lib/R/lib/libR.so)
#> frame #55: Rf_eval + 0x180 (0x7fe7857746f0 in /usr/lib/R/lib/libR.so)
#> frame #56: <unknown function> + 0x1440ac (0x7fe7857750ac in /usr/lib/R/lib/libR.so)
#> frame #57: Rf_eval + 0x454 (0x7fe7857749c4 in /usr/lib/R/lib/libR.so)
#> frame #58: <unknown function> + 0x14a22c (0x7fe78577b22c in /usr/lib/R/lib/libR.so)
#> frame #59: <unknown function> + 0x1871fd (0x7fe7857b81fd in /usr/lib/R/lib/libR.so)
#> frame #60: <unknown function> + 0x1353c4 (0x7fe7857663c4 in /usr/lib/R/lib/libR.so)
#> frame #61: Rf_eval + 0x180 (0x7fe7857746f0 in /usr/lib/R/lib/libR.so)
#> frame #62: <unknown function> + 0x14550f (0x7fe78577650f in /usr/lib/R/lib/libR.so)
#> frame #63: Rf_applyClosure + 0x1c7 (0x7fe7857772d7 in /usr/lib/R/lib/libR.so)
```

모양이 다른 텐서를 더하려고 하면 R은 위에서 보듯 너무나 많은 에러를 쏟아낸다. 모양이 다른 두 텐서를 더하기 위해서는 모양을 같게 맞춰줘야 한다. A의 모양을 B의 모양과 같이 바꿔보도록 하자. 모양을 바꿀때는 `view()` 함수를 사용하고, 안에 모양의 형태를 벡터 형식으로 짚어 넣는다는 것을 기억하자.


```r
A <- A$view(c(2, 3))
A
```

```
#> torch_tensor
#>  1  2  3
#>  4  5  6
#> [ CPUDoubleType{2,3} ]
```

한가지 짚고 넘어가야하는 기능이 있는데, R에서 행렬을 정의할 때, 주어진 원소벡터를 넣고, 가로행과 세로열 중 하나만 입력을 해도 잘 정의가 되는 것을 기억할 것이다. view 함수 역시 비슷한 기능이 있는데, 바로 `-1`을 이용해서 모양을 변환시키는 방법이다. 앞선 예제에서 2행 3열이 텐서를 1행의 가로 텐서로 변환 시키려면 다음과 같이 `view()` 함수의 입력값을 조정할 수 있다.


```r
A$view(c(1, -1))
```

```
#> torch_tensor
#>  1  2  3  4  5  6
#> [ CPUDoubleType{1,6} ]
```

### 덧셈과 뺄셈

앞에서 형(type)과 모양(shape)까지 맞춰놨으니, 텐서끼리의 덧셈과 뺄셈을 할 수 있다.


```r
A + B
```

```
#> torch_tensor
#>  1.5134  2.7426  3.7159
#>  4.5705  5.1653  6.0443
#> [ CPUDoubleType{2,3} ]
```


```r
A - B
```

```
#> torch_tensor
#>  0.4866  1.2574  2.2841
#>  3.4295  4.8347  5.9557
#> [ CPUDoubleType{2,3} ]
```

사실, 텐서끼리의 연산은 **모양만 맞으면 가능**하다. 즉, 다음의 연산이 성립한다.


```r
A_ <- A$to(dtype = torch_long())
A_ + B
```

```
#> torch_tensor
#>  1.5134  2.7426  3.7159
#>  4.5705  5.1653  6.0443
#> [ CPUFloatType{2,3} ]
```

결과에서 알 수 있듯, 정수를 담을 수 있는 텐서와 실수를 담을 수 있는 텐서를 더하면, 결과는 실수를 담을 수 있는 텐서로 반환이 된다. 하지만, 필자는 이러한 코딩은 피해야 한다고 생각한다. 즉, 모든 연산을 할 경우, 명시적으로 형변환을 한 후 연산을 할 것을 권한다. 왜냐하면, 언제나 우리는 코드를 다른 사람이 보았을 때, 이해하기 쉽도록 짜는 것을 추구해야 한다. (코드는 하나의 자신의 생각을 적은 글이다.)

### 상수와의 연산

R에서와 마찬가지로, 텐서와 상수와의 사칙연산은 각 원소에 적용되는 것을 확인하자.


```r
A + 2
```

```
#> torch_tensor
#>  3  4  5
#>  6  7  8
#> [ CPUDoubleType{2,3} ]
```

```r
B^2
```

```
#> torch_tensor
#>  0.2636  0.5514  0.5125
#>  0.3254  0.0273  0.0020
#> [ CPUFloatType{2,3} ]
```

```r
A %/% 3
```

```
#> torch_tensor
#>  0  0  1
#>  1  1  2
#> [ CPUDoubleType{2,3} ]
```

```r
A %% 3
```

```
#> torch_tensor
#>  1  2  0
#>  1  2  0
#> [ CPUDoubleType{2,3} ]
```

### 제곱근과 로그

제곱근(square root)나 로그(log) 함수 역시 각 원소별 적용이 가능하다.


```r
A
```

```
#> torch_tensor
#>  1  2  3
#>  4  5  6
#> [ CPUDoubleType{2,3} ]
```

```r
torch_sqrt(A)
```

```
#> torch_tensor
#>  1.0000  1.4142  1.7321
#>  2.0000  2.2361  2.4495
#> [ CPUDoubleType{2,3} ]
```

위의 연산이 에러가 나는 이유는 A가 정수를 담는 텐서였는데, 연산을 수행한 후에 실수가 담겨져서 나오는 에러이다. R과는 사뭇다른 예민한 아이 `torch`를 위해 형을 바꿔준 후에 연산을 실행하도록 하자.


```r
torch_sqrt(A$to(dtype = torch_double()))
```

```
#> torch_tensor
#>  1.0000  1.4142  1.7321
#>  2.0000  2.2361  2.4495
#> [ CPUDoubleType{2,3} ]
```

```r
torch_log(B)
```

```
#> torch_tensor
#> -0.6667 -0.2977 -0.3342
#> -0.5613 -1.8002 -3.1166
#> [ CPUFloatType{2,3} ]
```

### 텐서의 곱셈

텐서의 곱셈 역시 모양이 맞아야 하므로, 3행 2열이 두개가 붙어있는 C에서 앞에 한장을 떼어내도록 하자.


```r
B
```

```
#> torch_tensor
#>  0.5134  0.7426  0.7159
#>  0.5705  0.1653  0.0443
#> [ CPUFloatType{2,3} ]
```

```r
D <- C[1,,]
D
```

```
#> torch_tensor
#>  0.9628  0.2943
#>  0.0992  0.8096
#>  0.0169  0.8222
#> [ CPUFloatType{3,2} ]
```

텐서의 곱셈은 `torch_matmul()` 함수를 사용한다.


```r
# 파이프 사용해도 무방하다.
# B %>% torch_matmul(D)
torch_matmul(B, D)
```

```
#> torch_tensor
#>  0.5800  1.3409
#>  0.5664  0.3381
#> [ CPUFloatType{2,2} ]
```

토치의 텐서 곱셈은 다음과 같은 방법들도 있으니 알아두자.


```r
torch_mm(B, D)
```

```
#> torch_tensor
#>  0.5800  1.3409
#>  0.5664  0.3381
#> [ CPUFloatType{2,2} ]
```

```r
B$mm(D)
```

```
#> torch_tensor
#>  0.5800  1.3409
#>  0.5664  0.3381
#> [ CPUFloatType{2,2} ]
```

```r
B$matmul(D)
```

```
#> torch_tensor
#>  0.5800  1.3409
#>  0.5664  0.3381
#> [ CPUFloatType{2,2} ]
```

### 텐서의 전치(transpose)

전치(transpose)는 주어진 텐서를 뒤집는 것인데, 다음의 문법 구조를 가지고 있다.


```r
torch_transpose(input, dim0, dim1)
```

`dim0`, `dim1`는 바꿀 차원을 의미한다. '바꿀 차원은 두 개 밖에 없지 않나?' 라고 생각할 수 있다. 2 차원 텐서의 경우에는 그렇다. 우리가 행렬을 전치하는 경우에는 transpose를 취하는 대상이 2차원이므로 지정해주는 차원이 정해져있다. 하지만, 텐서의 차원이 3차원 이상이 되면 전치를 해주는 차원을 지정해줘야한다.


```r
A
```

```
#> torch_tensor
#>  1  2  3
#>  4  5  6
#> [ CPUDoubleType{2,3} ]
```

위의 텐서 A의 차원은 행과 열, 즉, 2개이다. 다음의 코드들은 A 텐서의 첫번째 차원과 두번째 차원을 뒤집는 효과를 가져온다. 즉, 전치 텐서가 된다.


```r
torch_transpose(A, 1, 2)
```

```
#> torch_tensor
#>  1  4
#>  2  5
#>  3  6
#> [ CPUDoubleType{3,2} ]
```

```r
A$transpose(1, 2)
```

```
#> torch_tensor
#>  1  4
#>  2  5
#>  3  6
#> [ CPUDoubleType{3,2} ]
```

```r
A %>% torch_transpose(1, 2)
```

```
#> torch_tensor
#>  1  4
#>  2  5
#>  3  6
#> [ CPUDoubleType{3,2} ]
```

3차원의 텐서를 살펴보자.


```r
C
```

```
#> torch_tensor
#> (1,.,.) = 
#>   0.9628  0.2943
#>   0.0992  0.8096
#>   0.0169  0.8222
#> 
#> (2,.,.) = 
#>   0.1242  0.7489
#>   0.3608  0.5131
#>   0.2959  0.7834
#> [ CPUFloatType{2,3,2} ]
```

텐서 C는 위와 같이 2차원 텐서가 두 개 포개져 있다고 생각하면 된다. 텐서의 결과물을 잘 살펴보면, 제일 앞에 위치한 1, 2가 나타내는 것이 우리가 흔히 생각하는 2차원 텐서들의 색인(index) 역할을 한다는 것을 알 수 있다. 앞으로는 편의를 위해서 3차원 텐서의 색인 역할을 하는 차원을 깊이(depth)라고 부르도록 하자. 앞에서 주어진 텐서 C 안의 포개져있는 2차원 텐서들을 전치하기 위해서는 이들을 관할(?)하는 두번째와 세번째 차원을 바꿔줘야 한다.


```r
torch_transpose(C, 2, 3)
```

```
#> torch_tensor
#> (1,.,.) = 
#>   0.9628  0.0992  0.0169
#>   0.2943  0.8096  0.8222
#> 
#> (2,.,.) = 
#>   0.1242  0.3608  0.2959
#>   0.7489  0.5131  0.7834
#> [ CPUFloatType{2,2,3} ]
```

결과를 살펴보면, 잘 바뀌어 있음을 알 수 있다.

### R에서의 3차원 배열

앞에서 다룬 `torch`에서의 3차원 텐서 부분은 [R에서 기본적으로 제공하는 array의 문법과 차이가 난다.](https://rstudio.github.io/reticulate/articles/arrays.html) 다음의 코드를 살펴보자. 먼저 R에서 2행 3열의 행렬을 두 개 포개어 놓은 3차원 배열을 만드는 코드이다.


```r
array(1:12, c(2, 3, 2)) 
```

```
#> , , 1
#> 
#>      [,1] [,2] [,3]
#> [1,]    1    3    5
#> [2,]    2    4    6
#> 
#> , , 2
#> 
#>      [,1] [,2] [,3]
#> [1,]    7    9   11
#> [2,]    8   10   12
```

필자는 참고로 `matrix()`를 만들때에도 `byrow` 옵션을 써서 만드는 것을 좋아하는데, `array()`에서 `byrow` 옵션 효과를 적용하려면 `aperm()` 함수를 사용해야 한다. 따라서, 좀 더 직관적으로 쓰기위해서 다음의 함수를 사용하자.


```r
array_3d_byrow <- function(num_vec, nrow, ncol, ndeath){
    aperm(array(num_vec, c(ncol, nrow, ndeath)), c(2, 1, 3))    
}

E <- array_3d_byrow(1:12, 2, 3, 2)
E
```

```
#> , , 1
#> 
#>      [,1] [,2] [,3]
#> [1,]    1    2    3
#> [2,]    4    5    6
#> 
#> , , 2
#> 
#>      [,1] [,2] [,3]
#> [1,]    7    8    9
#> [2,]   10   11   12
```

이러한 코드를 앞서 배웠던 `torch_tensor()` 함수에 넣어보자.


```r
E %>% torch_tensor()
```

```
#> torch_tensor
#> (1,.,.) = 
#>   1  7
#>   2  8
#>   3  9
#> 
#> (2,.,.) = 
#>    4  10
#>    5  11
#>    6  12
#> [ CPULongType{2,3,2} ]
```

결과를 살펴보면, 우리가 예상했던 2행 3열의 텐서가 두개 겹쳐있는 텐서의 모양이 나오지 않는다는 것을 알 수 있다. 이유는 `torch`에서 정의된 3차원 텐서의 경우, 첫번째 차원이 텐서가 얼마나 겹쳐있는지를 나타내는 깊이(depth)를 나타내기 때문이다. 문제를 해결하기 위해서는 `aperm()` 사용해서 차원을 바꿔주면 된다.


```r
E %>% 
  aperm(c(3, 1, 2)) %>% # 3 번째 차원을 맨 앞으로, 나머지는 그대로
  torch_tensor()
```

```
#> torch_tensor
#> (1,.,.) = 
#>   1  2  3
#>   4  5  6
#> 
#> (2,.,.) = 
#>    7   8   9
#>   10  11  12
#> [ CPULongType{2,2,3} ]
```

위의 경우를 좀더 직관적인 함수명으로 바꿔서 사용하도록 하자.


```r
array_to_torch <- function(mat, n_dim = 3){
    torch_tensor(aperm(mat, c(n_dim:3, 1, 2)))
}
E <- array_to_torch(E)
E
```

```
#> torch_tensor
#> (1,.,.) = 
#>   1  2  3
#>   4  5  6
#> 
#> (2,.,.) = 
#>    7   8   9
#>   10  11  12
#> [ CPULongType{2,2,3} ]
```

### 다차원 텐서와 1차원 벡터 텐서의 연산

R에서 우리가 아주 애용하는 기능 중 하나가 바로 `recycling` 개념이다. 즉, 길이 혹은 모양이 맞지 않는 개체(object)들을 연산할 때, 자동으로 길이와 모양을 맞춰서 연산을 해주는 기능인데, torch에서도 이러한 기능을 제공한다. 다음의 코드를 살펴보자.


```r
A
```

```
#> torch_tensor
#>  1  2  3
#>  4  5  6
#> [ CPUDoubleType{2,3} ]
```

```r
A + torch_tensor(1:3)
```

```
#> torch_tensor
#>  2  4  6
#>  5  7  9
#> [ CPUDoubleType{2,3} ]
```


```r
A
```

```
#> torch_tensor
#>  1  2  3
#>  4  5  6
#> [ CPUDoubleType{2,3} ]
```

```r
A + torch_tensor(matrix(2:3, ncol = 1))
```

```
#> torch_tensor
#>  3  4  5
#>  7  8  9
#> [ CPUDoubleType{2,3} ]
```

### 1차원 텐서 끼리의 연산, 내적과 외적

1차원 텐서끼리의 연산도 2차원 텐서끼리의 연산과 마찬가지라고 생각하면 된다. 내적과 외적 역시 그냥 모양을 맞춰서 곱하면 된다.


```r
A_1 <- A$view(c(1, -1))
A_1
```

```
#> torch_tensor
#>  1  2  3  4  5  6
#> [ CPUDoubleType{1,6} ]
```

```r
A_2 <- A$view(c(-1, 1))
A_2
```

```
#> torch_tensor
#>  1
#>  2
#>  3
#>  4
#>  5
#>  6
#> [ CPUDoubleType{6,1} ]
```

```r
A_1$mm(A_2)
```

```
#> torch_tensor
#>  91
#> [ CPUDoubleType{1,1} ]
```

```r
A_2$mm(A_1)
```

```
#> torch_tensor
#>   1   2   3   4   5   6
#>   2   4   6   8  10  12
#>   3   6   9  12  15  18
#>   4   8  12  16  20  24
#>   5  10  15  20  25  30
#>   6  12  18  24  30  36
#> [ CPUDoubleType{6,6} ]
```

한가지 주의할 점은 1차원 텐서끼리의 연산이더라도 꼭 차원을 선언해줘서 열벡터와 행벡터를 분명히 해줘야 한다는 점이다.


```r
A_3 <- torch_tensor(1:6)
A_1$mm(A_3)
```

```
#> Error in (function (self, mat2) : mat2 must be a matrix
#> Exception raised from mm_cpu at ../aten/src/ATen/native/LinearAlgebra.cpp:399 (most recent call first):
#> frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0x69 (0x7fe76f0afb89 in /home/issac/R/x86_64-pc-linux-gnu-library/4.0/torch/deps/./libc10.so)
#> frame #1: at::native::mm_cpu(at::Tensor const&, at::Tensor const&) + 0x334 (0x7fe75eefe194 in /home/issac/R/x86_64-pc-linux-gnu-library/4.0/torch/deps/./libtorch_cpu.so)
#> frame #2: <unknown function> + 0x133236d (0x7fe75f41e36d in /home/issac/R/x86_64-pc-linux-gnu-library/4.0/torch/deps/./libtorch_cpu.so)
#> frame #3: <unknown function> + 0xaf1c34 (0x7fe75ebddc34 in /home/issac/R/x86_64-pc-linux-gnu-library/4.0/torch/deps/./libtorch_cpu.so)
#> frame #4: at::Tensor c10::Dispatcher::callWithDispatchKey<at::Tensor, at::Tensor const&, at::Tensor const&>(c10::TypedOperatorHandle<at::Tensor (at::Tensor const&, at::Tensor const&)> const&, c10::DispatchKey, at::Tensor const&, at::Tensor const&) const + 0x1ce (0x7fe75f5c624e in /home/issac/R/x86_64-pc-linux-gnu-library/4.0/torch/deps/./libtorch_cpu.so)
#> frame #5: at::mm(at::Tensor const&, at::Tensor const&) + 0xb7 (0x7fe75f4ac947 in /home/issac/R/x86_64-pc-linux-gnu-library/4.0/torch/deps/./libtorch_cpu.so)
#> frame #6: <unknown function> + 0x2a5db24 (0x7fe760b49b24 in /home/issac/R/x86_64-pc-linux-gnu-library/4.0/torch/deps/./libtorch_cpu.so)
#> frame #7: <unknown function> + 0xaf1c34 (0x7fe75ebddc34 in /home/issac/R/x86_64-pc-linux-gnu-library/4.0/torch/deps/./libtorch_cpu.so)
#> frame #8: at::Tensor c10::Dispatcher::callWithDispatchKey<at::Tensor, at::Tensor const&, at::Tensor const&>(c10::TypedOperatorHandle<at::Tensor (at::Tensor const&, at::Tensor const&)> const&, c10::DispatchKey, at::Tensor const&, at::Tensor const&) const + 0x1ce (0x7fe75f5c624e in /home/issac/R/x86_64-pc-linux-gnu-library/4.0/torch/deps/./libtorch_cpu.so)
#> frame #9: at::Tensor::mm(at::Tensor const&) const + 0xb7 (0x7fe75f72fd67 in /home/issac/R/x86_64-pc-linux-gnu-library/4.0/torch/deps/./libtorch_cpu.so)
#> frame #10: _lantern_Tensor_mm_tensor_tensor + 0x4c (0x7fe76f3ed79c in /home/issac/R/x86_64-pc-linux-gnu-library/4.0/torch/deps/liblantern.so)
#> frame #11: cpp_torch_method_mm_self_Tensor_mat2_Tensor(Rcpp::XPtr<XPtrTorchTensor, Rcpp::PreserveStorage, &(void Rcpp::standard_delete_finalizer<XPtrTorchTensor>(XPtrTorchTensor*)), false>, Rcpp::XPtr<XPtrTorchTensor, Rcpp::PreserveStorage, &(void Rcpp::standard_delete_finalizer<XPtrTorchTensor>(XPtrTorchTensor*)), false>) + 0x2c (0x7fe76fd1f4fc in /home/issac/R/x86_64-pc-linux-gnu-library/4.0/torch/libs/torchpkg.so)
#> frame #12: _torch_cpp_torch_method_mm_self_Tensor_mat2_Tensor + 0x82 (0x7fe76fac4f22 in /home/issac/R/x86_64-pc-linux-gnu-library/4.0/torch/libs/torchpkg.so)
#> frame #13: <unknown function> + 0xf932c (0x7fe78572a32c in /usr/lib/R/lib/libR.so)
#> frame #14: <unknown function> + 0xf9826 (0x7fe78572a826 in /usr/lib/R/lib/libR.so)
#> frame #15: <unknown function> + 0x137106 (0x7fe785768106 in /usr/lib/R/lib/libR.so)
#> frame #16: Rf_eval + 0x180 (0x7fe7857746f0 in /usr/lib/R/lib/libR.so)
#> frame #17: <unknown function> + 0x14550f (0x7fe78577650f in /usr/lib/R/lib/libR.so)
#> frame #18: Rf_applyClosure + 0x1c7 (0x7fe7857772d7 in /usr/lib/R/lib/libR.so)
#> frame #19: Rf_eval + 0x353 (0x7fe7857748c3 in /usr/lib/R/lib/libR.so)
#> frame #20: <unknown function> + 0xc650d (0x7fe7856f750d in /usr/lib/R/lib/libR.so)
#> frame #21: <unknown function> + 0x137106 (0x7fe785768106 in /usr/lib/R/lib/libR.so)
#> frame #22: Rf_eval + 0x180 (0x7fe7857746f0 in /usr/lib/R/lib/libR.so)
#> frame #23: <unknown function> + 0x14550f (0x7fe78577650f in /usr/lib/R/lib/libR.so)
#> frame #24: Rf_applyClosure + 0x1c7 (0x7fe7857772d7 in /usr/lib/R/lib/libR.so)
#> frame #25: <unknown function> + 0x13a989 (0x7fe78576b989 in /usr/lib/R/lib/libR.so)
#> frame #26: Rf_eval + 0x180 (0x7fe7857746f0 in /usr/lib/R/lib/libR.so)
#> frame #27: <unknown function> + 0x14550f (0x7fe78577650f in /usr/lib/R/lib/libR.so)
#> frame #28: Rf_applyClosure + 0x1c7 (0x7fe7857772d7 in /usr/lib/R/lib/libR.so)
#> frame #29: <unknown function> + 0x13a989 (0x7fe78576b989 in /usr/lib/R/lib/libR.so)
#> frame #30: Rf_eval + 0x180 (0x7fe7857746f0 in /usr/lib/R/lib/libR.so)
#> frame #31: <unknown function> + 0x14550f (0x7fe78577650f in /usr/lib/R/lib/libR.so)
#> frame #32: Rf_applyClosure + 0x1c7 (0x7fe7857772d7 in /usr/lib/R/lib/libR.so)
#> frame #33: Rf_eval + 0x353 (0x7fe7857748c3 in /usr/lib/R/lib/libR.so)
#> frame #34: <unknown function> + 0x1470a2 (0x7fe7857780a2 in /usr/lib/R/lib/libR.so)
#> frame #35: Rf_eval + 0x572 (0x7fe785774ae2 in /usr/lib/R/lib/libR.so)
#> frame #36: <unknown function> + 0x14550f (0x7fe78577650f in /usr/lib/R/lib/libR.so)
#> frame #37: Rf_applyClosure + 0x1c7 (0x7fe7857772d7 in /usr/lib/R/lib/libR.so)
#> frame #38: Rf_eval + 0x353 (0x7fe7857748c3 in /usr/lib/R/lib/libR.so)
#> frame #39: <unknown function> + 0x149782 (0x7fe78577a782 in /usr/lib/R/lib/libR.so)
#> frame #40: <unknown function> + 0x137106 (0x7fe785768106 in /usr/lib/R/lib/libR.so)
#> frame #41: Rf_eval + 0x180 (0x7fe7857746f0 in /usr/lib/R/lib/libR.so)
#> frame #42: <unknown function> + 0x14550f (0x7fe78577650f in /usr/lib/R/lib/libR.so)
#> frame #43: Rf_applyClosure + 0x1c7 (0x7fe7857772d7 in /usr/lib/R/lib/libR.so)
#> frame #44: <unknown function> + 0x13a989 (0x7fe78576b989 in /usr/lib/R/lib/libR.so)
#> frame #45: Rf_eval + 0x180 (0x7fe7857746f0 in /usr/lib/R/lib/libR.so)
#> frame #46: <unknown function> + 0x1440ac (0x7fe7857750ac in /usr/lib/R/lib/libR.so)
#> frame #47: Rf_eval + 0x454 (0x7fe7857749c4 in /usr/lib/R/lib/libR.so)
#> frame #48: <unknown function> + 0x14a22c (0x7fe78577b22c in /usr/lib/R/lib/libR.so)
#> frame #49: <unknown function> + 0x1871fd (0x7fe7857b81fd in /usr/lib/R/lib/libR.so)
#> frame #50: <unknown function> + 0x1353c4 (0x7fe7857663c4 in /usr/lib/R/lib/libR.so)
#> frame #51: Rf_eval + 0x180 (0x7fe7857746f0 in /usr/lib/R/lib/libR.so)
#> frame #52: <unknown function> + 0x14550f (0x7fe78577650f in /usr/lib/R/lib/libR.so)
#> frame #53: Rf_applyClosure + 0x1c7 (0x7fe7857772d7 in /usr/lib/R/lib/libR.so)
#> frame #54: <unknown function> + 0x13a989 (0x7fe78576b989 in /usr/lib/R/lib/libR.so)
#> frame #55: Rf_eval + 0x180 (0x7fe7857746f0 in /usr/lib/R/lib/libR.so)
#> frame #56: <unknown function> + 0x1440ac (0x7fe7857750ac in /usr/lib/R/lib/libR.so)
#> frame #57: <unknown function> + 0x1444e4 (0x7fe7857754e4 in /usr/lib/R/lib/libR.so)
#> frame #58: <unknown function> + 0x1377d4 (0x7fe7857687d4 in /usr/lib/R/lib/libR.so)
#> frame #59: Rf_eval + 0x180 (0x7fe7857746f0 in /usr/lib/R/lib/libR.so)
#> frame #60: <unknown function> + 0x14550f (0x7fe78577650f in /usr/lib/R/lib/libR.so)
#> frame #61: Rf_applyClosure + 0x1c7 (0x7fe7857772d7 in /usr/lib/R/lib/libR.so)
#> frame #62: <unknown function> + 0x13a989 (0x7fe78576b989 in /usr/lib/R/lib/libR.so)
#> frame #63: Rf_eval + 0x180 (0x7fe7857746f0 in /usr/lib/R/lib/libR.so)
```

위의 코드는 연산 에러가 나는데, 이유는 `A_3`의 모양이 `A_1`의 모양과 맞지 않기 때문이다.


```r
A_1$size()
```

```
#> [1] 1 6
```

```r
A_3$size()
```

```
#> [1] 6
```
