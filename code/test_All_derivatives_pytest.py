from All_derivatives import all_derivatives, vector_func_all_derivatives

#---------------Func for testing ------------------------------------

def func1(vars):
    return vars[0]**2 + vars[1]**2

def func2(vars):
    return vars[0]**2*vars[1]**2

#-----------------------testing-all_derivatives------------------------------------------

def test_all_derivatives_1():
    a, b = all_derivatives(func1, ['x','y'], [3,2])
    assert a == 13
    assert b == [6, 4]

def test_all_derivatives_2():
    a, b = all_derivatives(func2, ['x','y'], [3,2])
    assert a == 36
    assert b == [24, 36]

#-----------------------testing-vector_func_all_derivatives------------------------------------------

def test_vector_all_derivatives_1():
    a, b = vector_func_all_derivatives([func1, func2], ['x','y'], [3,2])
    assert a == [13, 36]
    assert b == [[13, 36], [[6, 4], [24, 36]]]
