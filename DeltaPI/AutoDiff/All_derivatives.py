from AutoDiff import AutoDiff
import numpy as np


def all_derivatives(func, in_vars, point):
    '''
    Function that calculates the value of the input function at the input point and all its partial derivatives

    :params func: (function) function to evaluate whose input is a list of variables
            in_vars: (list)  list of variables to use when calculating the values of the input function
            point: (list)    list of values of each input variable at the input point

    :returns: out_val (scalar) the value of the input function at the input point
              out_dervs (numpy array , 1 dimensional) an array of the 1st derivative values of each input variable
                                                      at the input point

    :dependencies: requires class AutoDiff from module AutoDiff, package numpy imported ad np

    Example:

    f(x,y) = x^2 + y^2

    def func(vars):
        return vars[0]**2 + vars[1]**2

    a, b = all_derivatives(func, ['x','y'], [3,2])
    print(a, b)
    13.0, [6.0, 4.0]

    '''
    auto_diff_vars = []   # list to hold AutoDiff objects, one for each var in in_vars
    out_dervs =[]         # list to hold derivative values, one for each var in in_vars

    # Make AutoDiff objects
    for var, pt in zip(in_vars, point):
        var = AutoDiff(pt, 0)
        auto_diff_vars.append(var)

    # Calculate the value of func at the input point
    f = func(auto_diff_vars)
    out_val = f.val

    # Calculate the partial derivatives of func at the input point
    # Go through AutoDiff objects in auto_diff_vars
    # and during each pass change the derv value of 1 variable to 1, while others are held at 0
    # and calculate the partial derivative
    for i in range(len(auto_diff_vars)):
        auto_diff_vars[i].derv = 1
        f = func(auto_diff_vars)
        out_dervs.append(f.derv)
        auto_diff_vars[i].derv = 0

    return out_val, np.array(out_dervs)



def multi_func_all_derivatives(functions, in_vars, point):
    '''
    Function that calculates the value of all input functions at the input point and all their partial derivatives
    producing a numpy array of output values and a numpy 2D array (matrix) of partial derivatives, the Jacobian

    :params func: (functions) list of function to evaluate whose input is a list of variables
            in_vars: (list)   list of variables to use when calculating the values of the input functions
            point: (list)     list of values of each input variable at the input point

    :returns: out_vals (numpy array 1 dimensional) the values of the input functions at the input point
              out_dervs_matrix (numpy array 2 dimensional) a matrix of the 1st derivative values of each input variable
                        at the input point, the Jacobian.

    :dependencies: requires class AutoDiff from module AutoDiff, function all_derivatives from this module,
                            All_derivatives, and the package numpy imported as np

    Example:

    f(x,y) = x^2 + y^2

    (x,y) = x^2*y^2


    def func1(vars):
        return vars[0]**2 + vars[1]**2

    def func2(vars):
        return vars[0]**2*vars[1]**2

    a, b = multi_func_all_derivatives([func1,func2] , ['x','y'], [3,2])
    print(a, b)
    [13. 36.] [[ 6.  4.], [24. 36.]]

    '''
    out_vals = []
    out_dervs_matrix =[]

    # for each function in functions list
    # run all_derivatives and store output
    for func in functions:
        val, dervs = all_derivatives(func, in_vars, point)
        out_vals.append(val)
        out_dervs_matrix.append(dervs)

    return np.array(out_vals), np.array(out_dervs_matrix)

