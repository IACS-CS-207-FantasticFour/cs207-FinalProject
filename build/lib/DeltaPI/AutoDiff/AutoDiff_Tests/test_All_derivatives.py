from All_derivatives import all_derivatives, vector_func_all_derivatives
import numpy as np
#---------------Funcs for testing ------------------------------------

def func1(vars):
    return vars[0]**2 + vars[1]**2

def func2(vars):
    return vars[0]**2*vars[1]**2

def func3(vars):
    return np.sin(vars[0])**2/np.sqrt(vars[1])

def func4(vars):
    return vars[0]**2*vars[1]**2 + np.cos(vars[2])

#-----------------------testing-all_derivatives------------------------------------------

def test_all_derivatives_1():
    a, b = all_derivatives(func1, ['x','y'], [3,2])
    assert a == 13
    assert np.array_equal(b, [6, 4])

def test_all_derivatives_2():
    a, b = all_derivatives(func2, ['x','y'], [3,2])
    assert a == 36
    assert np.array_equal(b, [24, 36])

def test_all_derivatives_3():
    a, b = all_derivatives(func3, ['x','y'], [3,2])
    assert round(a,3) == .0140
    assert np.array_equal([round(b[0],3), round(b[1],3)], [-.198, -.004])

def test_all_derivatives_4():
    a, b = all_derivatives(func4, ['x','y','z'], [3,2,1])
    assert round(a,3) == 36.540
    assert np.array_equal([round(b[0],3),round(b[1],3),round(b[2],3)],  [24, 36, -0.841])

#-----------------------testing-vector_func_all_derivatives------------------------------------------


def test_vector_all_derivatives_1():
    a, b = vector_func_all_derivatives([func1, func2], ['x','y'], [3,2])
    assert np.array_equal(a, [13, 36])
    assert np.array_equal(b,  [ [6, 4], [24, 36] ] )

def test_vector_all_derivatives_2():
    a, b = vector_func_all_derivatives([func1, func2, func3], ['x','y'], [3,2])
    assert np.array_equal([a[0], a[1], round(a[2],3)], [13, 36, .014])
    assert np.array_equal( [ [ b[0][0], b[0][1] ], [ b[1][0], b[1][1] ], [ round(b[2][0], 3), round(b[2][1],3) ] ],  [ [6, 4], [24, 36], [-.198, -.004] ] )

