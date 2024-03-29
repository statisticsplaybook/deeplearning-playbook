# R6와 텐서 {#r6}

![](https://r6.r-lib.org/reference/figures/logo.png)



`torch`의 코드를 살펴보면 우리가 늘상 사용하던 R의 패키지들과는 어딘가 다른점이 있다고 느껴질 것이다. 이것의 근본적인 이유는 바로 torch 패키지가 객체지향언어 (Object Oriented Programming; OOP)를 할 수 있도록 해주는 `R6` 패키지를 기반으로 하고있기 때문이다. 좀 더 직접적으로 말하면, `torch`의 텐서와 신경망들이 `R6` 패키지의 클래스들로 정의되어 있기 때문에, 일반적인 R 패키지들보다 `$`을 통한 함수(OOP에서는 method 라고 부른다.) 접근이 가능하다. 어떤 이야기인지 한번 좀 더 깊게 들어가보자.

## 시작하기

여느 패키지와 다를바가 없다. `R6` 패키지를 설치하도록 하자.


```r
# install.packages("R6")
library(R6)
```

## 클래스(Class)와 멤버함수(Method), 그리고 필드(Field)

`R6` 패키지에는 딱 하나의 함수가 존재한다. 바로 `R6Class()` 함수이다. 이 함수의 입력값은 두가지 인데, 첫번째는 클래스 이름 `clasename`이고, 두번째는 공개될 정보들을 담을 `public`이라는 입력값이다. `public`에는 우리가 만들 클래스에서 사용이 가능한 멤버함수들(methods)과 변수(fields)들을 몽땅 다 떼려넣은 리스트(list) 형태가 들어간다.


```r
ExampleClass <- R6Class(classname = "Example", public = list(
    # 변수(fields) 정의
    # 멤버함수(methods) 정의
))
ExampleClass
```

```
#> <Example> object generator
#>   Public:
#>     clone: function (deep = FALSE) 
#>   Parent env: <environment: R_GlobalEnv>
#>   Locked objects: TRUE
#>   Locked class: FALSE
#>   Portable: TRUE
```

한가지 꼭 짚고 넘어가야하는 것이 있는데, 바로 이름을 정하는 방식이다.

1. 클래스의 이름은 `UpperCamelCase` 형식으로 짓는다. 즉, 클래스의 이름을 선언할 때 띄어쓰기를 하지않고, 대신 대문자를 사용한다.
2. 두번째 리스트에 들어가는 요소들의 이름은 `snake_case`를 사용한다. 즉, 모두 소문자를 유지하고, 띄어쓰기 대신에 밑줄을 사용하여 선언한다.

이렇게 규칙을 따라서 작성하게 되면, 나중에 다른 사람이 짜놓은 코드를 보게 되더라도, 선언된 이름의 구조를 보고, 이게 클래스인지, 클래스 안에 정의된 함수 혹은 변수인지를 구분 할 수 있어서 좋다.

### 클래스는 왜 필요할까?

필자도 클래스의 개념을 처음 들었을때 대체 이게 무슨 소리인지.. 했던 기억이 있다. 심지어 필자의 경우 R밖에 모르던 터여서, OOP가 필요가 있는지에 대한 의문까지 들 정도였으니, (사실 지금도 생각이 많이 바뀌지 않았다.) 머리에 아예 들어오지를 않았다. 

그런 필자를 클래스 개념에 대하여 한방에 이해시킨 예제가 바로 학생 클래스이다. 자고로 모든 개념은 예를 들어 설명을 하는 것이 아주 효과적이라고 필자는 믿고있다.

    > 목표: OOP의 개념와 왜 사용을 하는지에 대하여 이해한다.

### 학생자료 입력 예제
    
다음의 코드를 생각해보자.


```r
student <- function(){
    list()
}
issac <- student()
bomi <- student()
issac
```

```
#> list()
```

```r
bomi
```

```
#> list()
```

student라는 함수는 빈 리스트를 반환을 하는데, 우리가 이 함수를 사용하여 `issac`과 `bomi`라는 학생의 정보를 담는 리스트를 만들 수 있다. 만약 우리가 다음과 같은 추가 정보를 저장하려고 한다고 가정해보자.

* issac
    - last name: Lee
    - first name: Issac
    - email: issac-lee@gmail.com
    - midterm: 70
    - final: 50
* bomi
    - last name: Kim
    - first name: Bomi
    - email: bomi-kim@gmail.com
    - midterm: 65
    - final: 80

위의 정보를 저장하기 위해서는 다음과 같이 `$` 기호를 통하여 저장할 수 있다.


```r
issac$first <- "Issac"
issac$last <- "Lee"
issac$email <- "issac-lee@gmail.com"
issac$midterm <- 70
issac$final <- 50

bomi$first <- "Bomi"
bomi$last <- "Kim"
bomi$email <- "bomi-kim@gmail.com"
bomi$midterm <- 65
bomi$final <- 80
issac
```

```
#> $first
#> [1] "Issac"
#> 
#> $last
#> [1] "Lee"
#> 
#> $email
#> [1] "issac-lee@gmail.com"
#> 
#> $midterm
#> [1] 70
#> 
#> $final
#> [1] 50
```

```r
bomi
```

```
#> $first
#> [1] "Bomi"
#> 
#> $last
#> [1] "Kim"
#> 
#> $email
#> [1] "bomi-kim@gmail.com"
#> 
#> $midterm
#> [1] 65
#> 
#> $final
#> [1] 80
```

위의 코드는 OOP관점에서 상당히 중복 코드가 많은 비효율적인 코드이다. 이러한 코드를 우리가 배운 `R6Class()`를 사용하여 어떻게 줄일 수 있는지 알아보자.

### 클래스(Class) 정의하기

앞에서 우리는 `issac`과 `bomi`라는 변수를 생성했는데, 둘의 공통점은 학생이라는 점이었다. 사실 앞선 코드를 작성을 한다는 것은 `issac`과 `bomi` 뿐 아니라 엄청 많은 수의 학생들에 대한 데이터를 다루고 있는 상황일 수도 있다. 우리들이 써놓은 코드를 잘 뜯어보니, 학생 데이터로 입력되는 각 개인들은 성과 이름, 이메일, 그리고, 중간, 기말고사 점수의 정보들을 가지고 있다. 즉, 학생, `Student`, 라는 클래스는 항상 성(last)과 이름(first), 중간(midterm), 기말고사(final) 성적이 저장되어 있고, 이메일의 경우 이름과 성을 이용해서 작성을 하되, 모두 소문자로 입력된 자료 형태를 가지고 있는 구조를 갖는 어떤 추상적인 개념이라는 것을 알 수 있다. 이러한 정보를 사용하여 우리는 다음과 같이 `Student` 클래스를 선언 할 수 있다.


```r
Student <- R6Class("Student", list(
    # 필요한 변수 (field) 선언
    first = NULL,
    last = NULL,
    email = NULL,
    midterm = NA,
    final = NA,
    
    # 클래스 안의 객체를 만들때 사용되는 initialize
    initialize = function(first, last, midterm, final){
        self$first = first
        self$last  = last
        self$email = glue::glue("{tolower(first)}-{tolower(last)}@gmail.com")
        self$midterm = midterm
        self$final = final
    }    
))

Student
```

```
#> <Student> object generator
#>   Public:
#>     first: NULL
#>     last: NULL
#>     email: NULL
#>     midterm: NA
#>     final: NA
#>     initialize: function (first, last, midterm, final) 
#>     clone: function (deep = FALSE) 
#>   Parent env: <environment: R_GlobalEnv>
#>   Locked objects: TRUE
#>   Locked class: FALSE
#>   Portable: TRUE
```

결과값을 유심히 살펴보면, `<Student> object generator` 라는 부분이 있는데, `Student` 라는 클래스는 객체(object)들을 만들어내는 생성자(generator)라는 것을 알 수 있다. 우리가 만들 `Student` 생성자를 통해서 도장을 찍듯, `new()` 함수를 사용하여 issac과 bomi를 다음과 같이 만들 수 있다. 


```r
issac <- Student$new("Issac", "Lee", 70, 50)
bomi <- Student$new("Bomi", "Kim", 65, 80)
issac
```

```
#> <Student>
#>   Public:
#>     clone: function (deep = FALSE) 
#>     email: issac-lee@gmail.com
#>     final: 50
#>     first: Issac
#>     initialize: function (first, last, midterm, final) 
#>     last: Lee
#>     midterm: 70
```

```r
bomi
```

```
#> <Student>
#>   Public:
#>     clone: function (deep = FALSE) 
#>     email: bomi-kim@gmail.com
#>     final: 80
#>     first: Bomi
#>     initialize: function (first, last, midterm, final) 
#>     last: Kim
#>     midterm: 65
```

즉, OOP의 장점은 공을 들여 한번 클래스를 잘 만들어놓으면, 한번 작성된 함수나 변수들의 재 사용율이 엄청 좋아지는 것이다.

### print()를 사용한 결과물 정리

정의된 클래스는 기본적으로 동작하는 함수들을 덮어서 쓸 수 있다. 예를들어 `print()`를 함수로 정의해버리면, base에 있는 `print()` 동작을 덮어서 쓸 수 있다. 즉, 기본 함수들 `print()`, `plot()` 같은 함수들을 우리가 정의한 클래스에서 나온 객체들에 적용했을때의 작동을 정해줄 수 있다는 것이다.


```r
Student <- R6Class("Student", list(
    # 필요한 변수 (field) 선언
    first = NULL,
    last = NULL,
    email = NULL,
    midterm = NA,
    final = NA,
    
    # 클래스 안의 객체를 만들때 사용되는 initialize
    initialize = function(first, last, midterm, final){
        self$first = first
        self$last  = last
        self$email = glue::glue("{tolower(first)}-{tolower(last)}@gmail.com")
        self$midterm = midterm
        self$final = final
    },
    print = function(...){
        cat("Student: \n")
        cat(glue::glue("
                Name  : {self$first} {self$last}
                E-mail: {self$email}
                Midterm Score : {self$midterm}
                Final Score: {self$final}
            "))
        invisible(self)
    }
))

soony <- Student$new("Soony", "Kim", 70, 20)
soony
```

```
#> Student: 
#>     Name  : Soony Kim
#>     E-mail: soony-kim@gmail.com
#>     Midterm Score : 70
#>     Final Score: 20
```

`print()` 멤버 함수를 추가한 후에 만들어진 `soony`의 정보는 클래스안에 정의된 `print()`를 통해서 보여진다는 것을 확인할 수 있다. 한가지 주의할 점은 `print()`가 클래스 안에 정의되어 있지 않은 채로 생성된 `issac`과 `bomi`의 경우는 `print()`가 작동하지 않는다는 것이다. 즉, 클래스에 정의된 함수들은 객체가 클래스로부터 생성될 때, 따라와서 붙는다.


```r
issac$print()
```

```
#> Error in eval(expr, envir, enclos): 함수가 아닌것에 적용하려고 합니다
```

```r
soony$print()
```

```
#> Student: 
#>     Name  : Soony Kim
#>     E-mail: soony-kim@gmail.com
#>     Midterm Score : 70
#>     Final Score: 20
```

### set을 이용한 클래스 조정

앞에서 우리는 `print()` 함수를 추가하기 위하여 전체 클래스를 다시 정의하였다. 하지만, 이렇게 클래스안에 함수를 추가하기 위해서 전체 클래스를 다시 정의하기보단, `set()`을 이용해서 변수나 함수를 추가할 수 있다.


```r
Student$set("public", "total", NA)
Student$set("public", "calculate_total", function(){
    self$total <- self$midterm + self$final
    invisible(self)
})
```

`invisible()` 함수는 결과를 반환하되, 결과물을 보여주지 않는 것인데, 클래스에서 함수를 정의할 때에 반드시 `invisible(self)`를 반환해줘야만 한다. 따라서 함수이지만, 함수와는 다른 이 클래스 안의 함수들을 멤버함수 `method()`라고하여 일반 함수와 구분을 지어서 부른다.


```r
jelly <- Student$new("Jelly", "Lee", 35, 23)
jelly
```

```
#> Student: 
#>     Name  : Jelly Lee
#>     E-mail: jelly-lee@gmail.com
#>     Midterm Score : 35
#>     Final Score: 23
```

```r
jelly$total
```

```
#> [1] NA
```

```r
jelly$calculate_total()
jelly$total
```

```
#> [1] 58
```

## 상속(Inheritance) - 클래스 물려받기

OOP가 코드의 중복을 되도록 피할 수 있도록 설계되어 있다는 것을 어렴풋이나마 앞의 예제를 통하여 알 수 있을 것이다. 이러한 OOP의 코드 재사용 관점에서 상속(Inheritance)의 개념은 꽃 중에 꽃이라 불릴 만하다. 단 한 줄의 코드로 미리 작성해놓은 함수들에 접근이 가능하기 때문이다.

상속(Inheritance)이라고 하면 뭔가 거창할 것 같지만, 그냥 미리 정의해둔 클래스의 정보(멤버함수과 필드)를 다른 클래스를 정의할 때 받아올 수 있다는 말이다. 예를 들어보자.

이제까지 사용해 온 학생 개념, `Student` 클래스를 좀 더 세분화를 한다면 학교별로 나눌 수 있을 것이다. `Student` 클래스를 상속받는 슬통대학교(University of Statistics Playbook; `USP`) 학생들을 위한 서브 클래스(`sub class`)는 다음과 같이 생성할 수 있다.


```r
UspStudent <- R6Class("UspStudent",
    inherit = Student,
    public = list(
        university_name = "University of Statistics Playbook",
        class_year = NA,
        average = NA,
        calculate_average = function(){
            self$average <- mean(c(self$midterm, self$final))
            invisible(self)
        },
        calculate_total = function(){
            cat("The total score of midterm and final exam is calculated. \n")
            super$calculate_total()
        }
    )
)

sanghoon <- UspStudent$new("Sanghoon", "Park", 80, 56)
sanghoon
```

```
#> Student: 
#>     Name  : Sanghoon Park
#>     E-mail: sanghoon-park@gmail.com
#>     Midterm Score : 80
#>     Final Score: 56
```

새로 정의된 `UspStudent` 클래스는 상위 클래스인 `Student` 클래스의 멤버함수들과 변수들을 그대로 물려받는다. 여기서 코드의 재사용성이 증가한다. 또한 상위 클래스가 가지고 있던 `calculate_total()` 멤버함수에 접근하여, 새롭게 고쳐서 사용하는 것도 가능하다. 다음은 정의된 멤버함수들을 사용하여 변수들에 계산을 해서 넣는 과정을 보여준다.


```r
sanghoon$university_name
```

```
#> [1] "University of Statistics Playbook"
```

```r
sanghoon$calculate_average()
sanghoon$average
```

```
#> [1] 68
```

```r
sanghoon$calculate_total()
```

```
#> The total score of midterm and final exam is calculated.
```

```r
sanghoon$total
```

```
#> [1] 136
```

## 공개(Public)정보와 비공개(Private) 정보의 필요성

앞에서 살펴본 `R6Class()` 함수의 두 가지 입력값은 클래스 이름(`classname`)과 공개정보(`public`) 였다. 클래스를 만들고 사용하다보면, 때로는 클래스 안의 함수들을 사용하기 위해서 만들어야하는 변수나 함수들이 있는데, 이러한 정보들은 굳이 클래스를 사용하는 사용자들에게 보여줄 필요가 없다. 우리네 인생도 그러하다. 우리는 때로는 너무 많은 정보 제공에 피로감과 불편을 겪는 경우가 많다. 따라서, 클래스에 대한 정보의 접근을 적절하게 조절할 필요가 있는데, 클래스의 정보들을 공개될 정보(public)와 비공개 정보(private)들로 분류함으로써 조절할 수 있다. 


```r
UspStudent <- R6Class("UspStudent",
    inherit = Student,
    public = list(
        university_name = "University of Statistics Playbook",
        class_year = NA,
        calculate_average = function(){
            private$.average <- mean(c(self$midterm, self$final))
            cat("Average score is", private$.average)
            invisible(self)
        },
        calculate_total = function(){
            cat("The total score of midterm and final exam is calculated. \n")
            super$calculate_total()

        }
    ),
    private = list(
        .average = NA    
    )
)

taemo <- UspStudent$new("Taemo", "Bang", 80, 56)
taemo$calculate_average()
```

```
#> Average score is 68
```

위의 `UspStudent` 클래스에는 비공개 정보가 하나들어있다. 바로 중간 기말고사 점수의 평균을 저장하는 `average` 변수인데, 클래스의 정의시 `private()`에 감싸져서 입력이 되었음에 주목하자. 

`average` 변수는 클래스 안에서의 멤버함수를 통해서 접근할 땐 `private$name` 형식으로 접근이 가능함에 반하여, 클래스를 사용하는 사용자 입장에서는 가려져서 보이지 않는 정보에 해당한다.


```r
taemo$.average
```

```
#> NULL
```

해들리 위캠의 말을 빌리면, 공개-비공개 정보의 구분은 큰 패키지나 클래스를 정의할 때 [가장 중요한 단계가 된다](https://adv-r.hadley.nz/r6.html#privacy). 왜냐하면 비공개 정보의 경우는 개발자의 입장에서 언제든지 수정할 수 있는 정보가 되지만, 공개된 멤버함수나 필드들에 대해서는 쉽게 바꿀 수가 없기 때문이다.

    > 여기서 하나 짚고 넘어가면 좋은 것이 있는데, 바로 R에서의 이름 짓기 방식이다. R의 기본 함수들 중에서 .을 사용해서 지어진 경우가 있는데, 현재는 권장하지 않고 있다. 이유는 바로 비공개 정보를 갖는 변수나 함수들을 나타내는데에 .을 찍어서 나타내기 때문이다. `.average` 역시 변수의 이름에서 이 변수는 클래스 안에서만 접근이 가능하다는 것을 변수 이름만 보고도 알 수 있도록 만들어졌다.
    
### 활성 변수(active field)를 사용한 읽기 전용 변수

[Advance R의 14장](https://adv-r.hadley.nz/r6.html#active-fields)의 내용을 보면, R6의 접근성을 다루면서 `active field`의 개념이 나온다. 자세한 내용이 궁금한 독자들은 찾아보기 바란다. `active field`의 좋은 점은 이것을 사용해서 클래스 사용자들에게 읽기 전용 정보를 제공해줄수 있기 때문이다.

앞에서의 예를 들어보면 중간, 기말고사의 평균 정보는 클래스 사용자들에게 유요한 정보가 될 수 있다. 하지만, private으로 감싸버리면 사용자들은 이 정보에 접근을 할 수 없게 된다. 사용자는 평균 정보에 접근하고 싶어하지만, 개발자의 입장에서는 쉽게 공개정보로 바꾸기가 쉽지 않다. 왜냐하면 사용자들이 마음대로 평균 변수에 접근해서 정보를 변경시켜버리면 클래스에서 평균 정보를 가져다가 쓰는 멤버함수들이 잘 작동하지 않을 수 있기 때문이다. 이럴 경우 `active field`를 사용해서 average를 읽기전용으로만 접근 가능하도록 설계할 수 있다.


```r
UspStudent <- R6Class("UspStudent",
    inherit = Student,
    ## active field
    active = list(
        average = function(value) {
            if (missing(value)) {
                private$.average
            } else {
                stop("`$average` is read only", call. = FALSE)
            }
        }
    ),
    public = list(
        university_name = "University of Statistics Playbook",
        class_year = NA,
        calculate_average = function(){
            private$.average <- mean(c(self$midterm, self$final))
            cat("Average score is", private$.average)
            invisible(self)
        },
        calculate_total = function(){
            cat("The total score of midterm and final exam is calculated. \n")
            super$calculate_total()

        }
    ),
    private = list(
        .average = NA    
    )
)

conie <- UspStudent$new("Connie", "", 78, 82)
conie$calculate_average()
```

```
#> Average score is 80
```

```r
conie$average
```

```
#> [1] 80
```
위에서 정의된 `UspStudent` 클래스에서는 사용자에게 평균값을 구하는 함수와 구한 평균값에 접근을 허용하지만, 사용자가 average값에 접근하여 바꾸려고 하면 에러를 뱉어내도록 설계가 되어있다.


```r
conie$average <- 60
```

```
#> Error: `$average` is read only
```

## 텐서와 R6의 관계



## R6 관련자료

R6에 대한 더 깊은 내용은 Hadley Wickham의 [Advanced R](https://adv-r.hadley.nz/index.html)과 [R6 패키지의 웹사이트](https://r6.r-lib.org/)를 참고하도록 하자.

