import numpy as np
import random as rand

arr = [i for i in range(500)]

def arr_generator(array):
    for i in range(0, len(array), 50):
        yield 'i : {}'.format(i)
    for j in range(0, 100, -10):
        yield 'j : {}'.format(j)

for i in arr_generator(arr):
    print i
    print
