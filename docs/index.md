---
title: "딥러닝 공략집 with R"
author: "[슬기로운통계생활](https://www.youtube.com/c/statisticsplaybook)"
date: "2021-02-17"
site: bookdown::bookdown_site
output: bookdown::gitbook
documentclass: book
bibliography: [reference.bib]
biblio-style: apalike
link-citations: yes
image: "./image/deeplearning-playbook.png"
github-repo: statisticsplaybook/r-torch-playbook
description: "딥러닝 라이브러리 Rtorch를 사용하여 딥러닝의 끝판왕을 정복해보자. 본격 R 딥러닝 공략집"
---

# 들어가며 {.unnumbered}

![](./image/deeplearning-playbook.png)

이제까지 R에서의 딥러닝은 Python의 라이브러리들을 `reticulate` 패키지를 이용하여 빌려온 형태였지만, [torch for R](https://torch.mlverse.org/) 패키지는 C 라이브러리를 Torch를 기반으로 R을 wrapper 언어로서 사용하여 패키지를 만들었다. 즉, Torch + Python = PyTorch, Torch + R = RTorch가 되는 셈이다.

## 설치하기 {.unnumbered}

설치 역시 간단한다. 여느 R패키지와 같이 `install.packages()` 함수를 사용하면 된다. 서브 라이브러리인 `torchaudio`와 `torchvision`이 있으나, 책의 뒷부분에서 다루기로 한다.


```r
install.packages("torch")
# 혹은 개발버전을 다운 받고 싶다면 다음의 코드를 사용한다.
# devtools::install_github("mlverse/torch")
```



## 기본 패키지 {.unnumbered}

앞으로의 내용에 있어서 다음의 두 패키지는 기본으로 불러와서 사용하는 것을 약속으로 한다.


```r
library(tidyverse)
library(torch)
```
