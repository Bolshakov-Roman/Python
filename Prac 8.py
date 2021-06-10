from kanren import run, var, fact
from kanren.assoccomm import eq_assoccomm as eq
from kanren.assoccomm import commutative, associative

#определяем математические операции, которые мы собираемся использовать
add = 'add'#сложение
mul = 'mul'#умножение

#и сложение, и умножение являются коммуникативными процессами, указываем это
fact(commutative, mul)
fact(commutative, add)
fact(associative, mul)
fact(associative, add)

#определяем переменные
a, b = var('a'), var('b')

"""
нужно сопоставить выражение с исходным шаблоном. есть следующий
оригинальный шаблон (5 + a) * b
"""
Original_pattern = (mul, (add, 5, a), b)

"""
есть два выражения, которые соответствуют исходному шаблону - 3, 4
"""
exp1 = (mul, 2, (add, 3, 1))
exp2 = (add, 5, (mul, 8, 1))
exp3 = (mul, (add, 5, a), b)
exp4 = (mul, (add, 5, a), 2)

#вывод
print(run(0, (a,b), eq(Original_pattern, exp1)))#eq - конструктор цели указывает, что два выражения равны
print(run(0, (a,b), eq(Original_pattern, exp2))) 
print(run(0, (a,b), eq(Original_pattern, exp3)))
print(run(0, (a,b), eq(Original_pattern, exp4)))

"""
Код приведенный ниже,найдет простое число из списка чисел,
а также сгенерирует первые 10 простых чисел.
"""
from kanren import isvar, run, membero
from kanren.core import success, fail, goaleval, condeseq, eq, var
from sympy.ntheory.generate import prime, isprime
import itertools as it

#определим функцию, которая будет проверять простые числа
def prime_check(x):
    if isvar(x):
        return(condeseq([(eq, x, p)] for p in map(prime, it.count(1))))
    else:
        return(success if isprime(x) else fail)

#определяем переменную
x = var('x')

#результат
print((set (run (0, x, (membero, x, (12, 14, 15, 19, 20, 21, 22, 23, 29, 30, 41, 44, 52, 62, 65, 85)), (prime_check, x)))))
print((run (10, x, prime_check(x))))

"""
There are five houses.
The English man lives in the red house.
The Swede has a dog.
The Dane drinks tea.
The green house is immediately to the left of the white house.
They drink coffee in the green house.
The man who smokes Pall Mall has birds.
In the yellow house they smoke Dunhill.
In the middle house they drink milk.
The Norwegian lives in the first house.
The man who smokes Blend lives in the house next to the house with cats.
In a house next to the house where they have a horse, they smoke Dunhill.
The man who smokes Blue Master drinks beer.
The German smokes Prince.
The Norwegian lives next to the blue house.
They drink water in a house next to the house where they smoke Blend.
"""
from kanren import *
from kanren.core import lall
import time

#определяем две функции left () и next (), чтобы проверить, чей дом оставлен или рядом с чьим-то домом
def left(q, p, list):
    return membero((p, q),  zip(list, list[1:]))#membero (item, coll) - утверждает, что item является членом коллекции coll.
def next(q, p, list):
    return conde([left(q, p, list)], [left(p, q, list)])#conde - конструктор целей для логических "и" и "или".

#объявим переменную house
houses = var()

#определяем правила с помощью пакета lall,
rules_zebraproblem = lall(
    #5 домов
    (eq, (var(), var(), var(), var(), var()), houses),
    #англичанин в красном доме
    (membero,('Englishman', var(), var(), var(), 'red'), houses),
    #у свида есть собака
    (membero,('Swede', var(), var(), 'dog', var()), houses),
    #дани пьет чай
    (membero,('Dane', var(), 'tea', var(), var()), houses),
    #зеленый дом слева от белого
    (left,(var(), var(), var(), var(), 'green'), (var(), var(), var(), var(), 'white'), houses),
    #кофе пьют в зеленом доме
    (membero,(var(), var(), 'coffee', var(), 'green'), houses),
    #у чела, который курит пол-мол есть птицы
    (membero,(var(), 'Pall Mall', var(), 'birds', var()), houses),
    #в желтом доме - чел курящий данхил
    (membero,(var(), 'Dunhill', var(), var(), 'yellow'), houses),
    #в доме посередине - пьют молоко
    (eq,(var(), var(), (var(), var(), 'milk', var(), var()), var(), var()), houses),
    #норвежец в первом доме
    (eq,(('Norwegian', var(), var(), var(), var()), var(), var(), var(), var()), houses),
    #чел курящий бленд - рядом с домом где есть кошки 
    (next,(var(), 'Blend', var(), var(), var()), (var(), var(), var(), 'cats', var()), houses),
    #чел курящий донхел - рядом с домом где есть лошадь
    (next,(var(), 'Dunhill', var(), var(), var()), (var(), var(), var(), 'horse', var()), houses),
    #чел курящий блу мастер - пьет пиво
    (membero,(var(), 'Blue Master', 'beer', var(), var()), houses),
    #немец - курит принц
    (membero,('German', 'Prince', var(), var(), var()), houses),
    #норвежец - рядом с голубым домом
    (next,('Norwegian', var(), var(), var(), var()), (var(), var(), var(), var(), 'blue'), houses),
    #в доме рядом с челом крящим бленд - пьют воду
    (next,(var(), 'Blend', var(), var(), var()), (var(), var(), 'water', var(), var()), houses),
    #в одном из домом есть зебра
    (membero,(var(), var(), var(), 'zebra', var()), houses)
 )

#запускаем решатель с предыдущими ограничениями
solutions = run(0, houses, rules_zebraproblem)

#извлекаем вывод из решателя
output_zebra = [house for house in solutions[0] if 'zebra' in house][0][0]

#печатаем решение 
print ('\n'+ output_zebra + ' owns zebra.')
