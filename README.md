# Systems and methods of decision making

# Метрические алгоритмы классификации

Метрический алгоритм классификации с обучающей выборкой ![](https://github.com/e-sobolev/Systems_and_methods_of_decision_making/blob/master/img/Img_Metric_1.PNG?raw=true) относит объект ***u*** к тому классу ***y ∈ Y*** , для которого суммарный вес ближайших обучающих объектов ![](https://github.com/e-sobolev/Systems_and_methods_of_decision_making/blob/master/img/Img_Metric_2.PNG?raw=true) максимален:
![](https://github.com/e-sobolev/Systems_and_methods_of_decision_making/blob/master/img/Img_Metric_3.png?raw=true)
где весовая функция ![](https://github.com/e-sobolev/Systems_and_methods_of_decision_making/blob/master/img/Img_Metric_4.PNG?raw=true) оценивает степень важности ***i***-го соседа для классификации объекта ***u*** . Функция ![](https://github.com/e-sobolev/Systems_and_methods_of_decision_making/blob/master/img/Img_Metric_2.PNG?raw=true) — называется оценкой близости объекта *u* к классу ***y***.

## Алгоритм k ближайших соседей
**Алгоритм k ближайших соседей** относит объект ***u*** к тому классу, элементов которого больше среди ***k*** ближайших соседей ***x***

К примеру, при ***k*** = 1, наш алгоритм будет выглядеть следующим образом:

![](https://github.com/e-sobolev/Systems_and_methods_of_decision_making/blob/master/img/1NN.png?raw=true)

А при ***k*** = 100 так:

![](https://github.com/e-sobolev/Systems_and_methods_of_decision_making/blob/master/img/100NN.png?raw=true)

**Вопрос: Как выбирать k?**
**Ответ:** На практике оптимальное k подбирается по критерию скользящего контроля LOO.

Проверим по критерию скользящего контроля алгоритм kNN для выборки Ирисов Фишера, найдем оптимальный k. Начнем проверку ***k*** от 1 до 10.
Результат видим на графике, оценка LOO достигает минимума, при ***k = 6*** и равна 0.3333333 (96% правильных ответов).
![](https://github.com/e-sobolev/Systems_and_methods_of_decision_making/blob/master/img/LOO_KNN_10.png?raw=true)
![](https://github.com/e-sobolev/Systems_and_methods_of_decision_making/blob/master/img/LOO_KNN_100.png?raw=true)

### Пример 1: 
*На основе обучающей выборки Ирисы Фишера (в качестве признаков берем длину и ширину чашелистика), с помощью алгоритма kNN проклассифицировать объект(ы) с координатами **x** и **y**.*

Выборка имеет вид:
![](https://github.com/e-sobolev/Systems_and_methods_of_decision_making/blob/master/img/iris.png?raw=true)

- В качестве объектов для классификации сгенерируем 50 точек, значения по ***x*** которых могут лежать в диапазоне от 1 од 7, а по ***y*** от 0 до 2.5.
- В качестве метрики возьмем Евклидово пространсвто.
- В качестве параметра k возьмем оптимальное равное 6.

Результат работы:
![](https://github.com/e-sobolev/Systems_and_methods_of_decision_making/blob/master/img/6NN_random_points.png?raw=true)

Также рассмотрим вариант, когда мы определяем для каждой точки её класс и обозначаем их, что можно увидеть ниже:

![](https://github.com/e-sobolev/Systems_and_methods_of_decision_making/blob/master/img/6NN_all.png?raw=true)

## Алгоритм k взвешенных ближайших соседей

Очень часто оценка весов в методе ближайших соседей оказывается некорректной, из-за того, у классифицируемого объкта одинаковое количество соседей, принадлежащих разным классам. В этом случае применяется алгоритм k взвешенных ближайших соседей. Отличие от kNN, только в том, что функция весов ![](https://github.com/e-sobolev/Systems_and_methods_of_decision_making/blob/master/img/Img_Metric_4.PNG?raw=true) домнажается на ![](https://github.com/e-sobolev/Systems_and_methods_of_decision_making/blob/master/img/Img_Metric_6.PNG?raw=true) и алгоритм классификации принимает вид: 
![](https://github.com/e-sobolev/Systems_and_methods_of_decision_making/blob/master/img/Img_Metric_7.PNG?raw=true)

![](https://github.com/e-sobolev/Systems_and_methods_of_decision_making/blob/master/imgs/Img_Metric_6.PNG?raw=true) — строго убывающая последовательность вещественных весов, задающая вклад ***i***-го соседа при классификации объекта ***u***.

Примеры весов:
- ![](https://github.com/e-sobolev/Systems_and_methods_of_decision_making/blob/master/img/Img_Metric_8.PNG?raw=true)
- ![](https://github.com/e-sobolev/Systems_and_methods_of_decision_making/blob/master/img/Img_Metric_9.PNG?raw=true) — геометрическая прогрессия со
знаменателем ***q ∈ (0, 1)***, который можно подбирать по критерию LOO.

Применим LOO для подбора ***q*** как параметра kwNN на обучающей выборке "Ирисы фишера", ***k=6***.
Результат видим на графике, оценка LOO достигает минимума, при ***k = 6 && q = 1.0*** и равна 0.3333333 (96% правильных ответов).

![](https://github.com/e-sobolev/Systems_and_methods_of_decision_making/blob/master/img/LOO_kWNN.png?raw=true)

![](https://github.com/e-sobolev/Systems_and_methods_of_decision_making/blob/master/img/kWNN.png?raw=true)

## Парзеновское окно

Алгоритм ![](https://github.com/e-sobolev/Systems_and_methods_of_decision_making/blob/master/img/Img_Metric_10.PNG?raw=true) называется алгоритмом парзеновского окна. Параметр ***h*** называется шириной окна и играет примерно ту же роль, что и число соседей ***k***. “Окно” — это сферическая окрестность объекта ***u*** радиуса ***h***, при попадании в которую ***i***-й обучающий объект "голосует" за отнесение объекта ***u*** к классу ***i-го*** объекта. . Параметр ***h*** можно задавать априори или определять по скользящему контролю.

![](https://github.com/e-sobolev/Systems_and_methods_of_decision_making/blob/master/img/PW.png?raw=true)

### Преимущества:

1.Простота реализации.

2.Все точки, попадающие в окно, расстояние между которыми одинаково, будут учитываться (в отличие от алгоритма взвешенного kNN).

3.Скорость классификации быстрее, т.к. не требуется сортировка расстояний (O(l)).

4.Окно с переменной шириной решает проблему разрешимости задач, в которых обучающие выборки распределены неравномерно по пространству X (как ирисы Фишера).

5.Выбор финитного ядра позволяет свести классификацию объекта u к поиску k его ближайших соседей, тогда как при не финитном ядре (например, гауссовском) нужно перебирать всю обучающую выборку Xl, что может приводить к неприемлемым затратам времени (при большом l).

### Недостатки:

1.Слишком узкие окна приводят к неустойчивой классификации, а слишком широкие - к вырождению алгоритма в константу.

2.Диапазон, из которого выбирается параметр h, нужно подбирать самим.

3."Скудный" набор параметров.

4.Если в окно, радиуса h, не попало ни одной точки x_i, то алгоритм не способен классифицировать объект u.

5.Если суммарные веса классов оказываются одинаковыми, то алгоритм относит классифицируемый объект u к любому из классов.

# Байесовские алгоритмы классификации

## Наивный байесовский классификатор

**Наивный байесовский классификатор** — специальный частный случай байесовского классификатора, основанный на дополнительном предположении, что объекты x описываются n статистически независимыми признаками:

![](https://github.com/e-sobolev/Systems_and_methods_of_decision_making/blob/master/img/nb1.png?raw=true)

Предположение о независимости означает, что функции правдоподобия классов представимы в виде

![](https://github.com/e-sobolev/Systems_and_methods_of_decision_making/blob/master/img/nb2.png?raw=true) где ![](https://github.com/e-sobolev/Systems_and_methods_of_decision_making/blob/master/img/nb3.png?raw=true) — плотность распределения значений j-го признака для класса y.

Предположение о независимости существенно упрощает задачу, так как оценить n одномерных плотностей гораздо легче, чем одну n-мерную плотность. К сожалению, оно крайне редко выполняется на практике, отсюда и название метода.

**Наивный байесовский классификатор** может быть как параметрическим, так и непараметрическим, в зависимости от того, каким методом восстанавливаются одномерные плотности.

Основные **преимущества** наивного байесовского классификатора:

1.Простота реализации.

2.Низкие вычислительные затраты при обучении и классификации.

3.В тех редких случаях, когда признаки действительно независимы (или почти независимы), наивный байесовский классификатор (почти) оптимален.

Основной его **недостаток** — относительно низкое качество классификации в большинстве реальных задач.

Чаще всего он используется либо как примитивный эталон для сравнения различных моделей алгоритмов, либо как элементарный строительный блок в алгоритмических композициях.

![](https://github.com/e-sobolev/Systems_and_methods_of_decision_making/blob/master/img/NBC.png?raw=true)

## Подстановочный алгортим(plug-in)

*Реализуем подстановочный байесовский алгоритм на сгенерированных данных. Рассмотрим случаи, когда разделяющая кривая является: параболой, эллипсом и гиперболой.*

Выбирая различные матрицы ковариации и центры для генерации тестовых данных, будем получать различные виды дискриминантной функции.

**1.Парабола**
Центр первого класса ***(1,0)***
Ковариационная матрица первого класса ![](https://github.com/e-sobolev/Systems_and_methods_of_decision_making/blob/master/img/Img_Bayes_1.gif?raw=true)

Центр второго класса ***(15,0)***
Ковариационная матрица второго класса ![](https://github.com/e-sobolev/Systems_and_methods_of_decision_making/blob/master/img/Img_Bayes_2.gif?raw=true)

Результат:

![](https://github.com/e-sobolev/Systems_and_methods_of_decision_making/blob/master/img/plug-in_parabola.png?raw=true)

**2.Эллипс**
Центр первого класса ***(2,2)***
Ковариационная матрица первого класса ![](https://github.com/e-sobolev/Systems_and_methods_of_decision_making/blob/master/img/Img_Bayes_3.gif?raw=true)

Центр второго класса ***(15,2)***
Ковариационная матрица второго класса ![](https://github.com/e-sobolev/Systems_and_methods_of_decision_making/blob/master/img/Img_Bayes_4.gif?raw=true)

Результат:

![](https://github.com/e-sobolev/Systems_and_methods_of_decision_making/blob/master/img/plug-in_ellips.png?raw=true)

**3.Гипербола**
Центр первого класса ***(-1,0)***
Ковариационная матрица первого класса ![](https://github.com/e-sobolev/Systems_and_methods_of_decision_making/blob/master/img/Img_Bayes_5.gif?raw=true)

Центр второго класса ***(3,0)***
Ковариационная матрица второго класса ![](https://github.com/e-sobolev/Systems_and_methods_of_decision_making/blob/master/img/Img_Bayes_5.gif?raw=true)

Результат:

![](https://github.com/e-sobolev/Systems_and_methods_of_decision_making/blob/master/img/plug-in_giperb.png?raw=true)
