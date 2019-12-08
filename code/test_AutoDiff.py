from AutoDiff import AutoDiff
import numpy as np

#---------------Testing Urnary------------------------------------
def test_pos():
    x = AutoDiff(3, 1)
    f = +x
    assert f.val == 3
    assert f.derv == 1

def test_neg():
    x = AutoDiff(3, 1)
    f = -x
    assert f.val == -3
    assert f.derv == -1

def test_pos_0():
    x = AutoDiff(3, 0)
    f = +x
    assert f.val == 3
    assert f.derv == 0

def test_neg_0():
    x = AutoDiff(3, 0)
    f = -x
    assert f.val == -3
    assert f.derv == 0

#---------------Testing Addition------------------------------------
def test_left_addition():
    x = AutoDiff(3, 1)
    f = x + 2
    assert f.val == 5
    assert f.derv == 1

def test_right_addition():
    x = AutoDiff(3, 1)
    f = 2 + x
    assert f.val == 5
    assert f.derv == 1

def test_x_y_addition_dx():
    x = AutoDiff(3, 1)
    y = AutoDiff(2, 0)
    f = y + x
    assert f.val == 5
    assert f.derv == 1

def test_x_y_addition_dy():
    x = AutoDiff(3, 0)
    y = AutoDiff(2, 1)
    f = y + x
    assert f.val == 5
    assert f.derv == 1

#---------------Testing Subtraction------------------------------------
def test_left_subtraction():
    x = AutoDiff(3, 1)
    f = x - 2
    assert f.val == 1
    assert f.derv == 1

def test_right_subtraction():
    x = AutoDiff(3, 1)
    f = 2 - x
    assert f.val == -1
    assert f.derv == -1

def test_x_y_subtraction_dx():
    x = AutoDiff(3, 1)
    y = AutoDiff(2, 0)
    f = y - x
    assert f.val == -1
    assert f.derv == -1

def test_x_y_subtraction_dy():
    x = AutoDiff(3, 0)
    y = AutoDiff(2, 1)
    f = y - x
    assert f.val == -1
    assert f.derv == 1

#---------------Testing Multiplication------------------------------------
def test_left_multiply():
    x = AutoDiff(3, 1)
    f = x*2
    assert f.val == 6
    assert f.derv == 2

def test_right_multiply():
    x = AutoDiff(3, 1)
    f = 2*x
    assert f.val == 6
    assert f.derv == 2

def test_x_y_multiple_dx():
    x = AutoDiff(3, 1)
    y = AutoDiff(2, 0)
    f = x*y
    assert f.val == 6
    assert f.derv == 2

def test_x_y_multiple_dy():
    x = AutoDiff(3, 0)
    y = AutoDiff(2, 1)
    f = x*y
    assert f.val == 6
    assert f.derv == 3

#---------------Testing Division------------------------------------
def test_left_divide():
    x = AutoDiff(4, 1)
    f = x/2
    assert f.val == 2
    assert f.derv == 0.5

def test_right_divide():
    x = AutoDiff(4, 1)
    f = 2/x
    assert f.val == 0.5
    assert f.derv == -0.125

def test_x_y_divide_dx():
    x = AutoDiff(4, 1)
    y = AutoDiff(2, 0)
    f = x/y
    assert f.val == 2
    assert f.derv == 0.5

def test_x_y_divide_dy():
    x = AutoDiff(4, 0)
    y = AutoDiff(2, 1)
    f = x/y
    assert f.val == 2
    assert f.derv == -1

#---------------Testing Powers------------------------------------
def test_x_a_power():
    x = AutoDiff(4, 1)
    f = x**2
    assert f.val == 16
    assert f.derv == 8

def test_a_x_power():
    x = AutoDiff(4, 1)
    f = 2**x
    assert f.val == 16
    assert f.derv == 16*np.log(2)

def test_x_y_power_dx():
    x = AutoDiff(4, 1)
    y = AutoDiff(2, 0)
    f = x**y
    assert f.val == 16
    assert f.derv == 8

def test_x_y_power_dy():
    x = AutoDiff(4, 0)
    y = AutoDiff(2, 1)
    f = x**y
    assert f.val == 16
    assert f.derv == 16*np.log(4)

#---------------Testing trig functions------------------------------------
def test_sinx():
    x = AutoDiff(np.pi/2, 1)
    f = np.sin(x)
    assert f.val == np.sin(np.pi/2)
    assert f.derv == np.cos(np.pi/2)

def test_cosx():
    x = AutoDiff(np.pi/2, 1)
    f = np.cos(x)
    assert f.val == np.cos(np.pi/2)
    assert f.derv == -1*np.sin(np.pi/2)

def test_tanx():
    x = AutoDiff(np.pi/4, 1)
    f = np.tan(x)
    assert f.val == np.sin(np.pi/4)/np.cos(np.pi/4)
    assert round(f.derv) == 2

#---------------Testing inverse trig functions------------------------------------
def test_arcsinx():
    x = AutoDiff(0.5, 1)
    f = np.arcsin(x)
    assert f.val == np.arcsin(0.5)
    assert round(f.derv,4) == 1.1547

def test_arccosx():
    x = AutoDiff(0.5, 1)
    f = np.arccos(x)
    assert f.val == np.arccos(0.5)
    assert round(f.derv,4) == -1.1547

def test_arctanx():
    x = AutoDiff(0.5, 1)
    f = np.arctan(x)
    assert f.val == np.arctan(0.5)
    assert round(f.derv,4) == 0.8


#---------------Testing Hyperbolic functions------------------------------------
def test_sinhx():
    x = AutoDiff(1.5, 1)
    f = np.sinh(x)
    assert f.val == np.sinh(1.5)
    assert round(f.derv,4) == 2.3524

def test_cosxh():
    x = AutoDiff(1.5, 1)
    f = np.cosh(x)
    assert f.val == np.cosh(1.5)
    assert round(f.derv,4) == 2.1293

def test_tanhx():
    x = AutoDiff(1.5, 1)
    f = np.tanh(x)
    assert f.val == np.tanh(1.5)
    assert round(f.derv,4) == 0.1807


#---------------Testing Inverse Hyperbolic functions------------------------------------
def test_arcsinhx():
    x = AutoDiff(1.5, 1)
    f = np.arcsinh(x)
    assert f.val == np.arcsinh(1.5)
    assert round(f.derv,4) == 0.5547

def test_arccosxh():
    x = AutoDiff(1.5, 1)
    f = np.arccosh(x)
    assert f.val == np.arccosh(1.5)
    assert round(f.derv,4) == 0.8944

def test_arctanhx():
    x = AutoDiff(0.5, 1)
    f = np.arctanh(x)
    assert f.val == np.arctanh(0.5)
    assert round(f.derv,4) == 1.3333


#---------------Testing exp, log, log2, log10, sqrt, functions------------------------------------
def test_sqrt():
    x = AutoDiff(25, 1)
    f = np.sqrt(x)
    assert f.val == 5
    assert f.derv == 0.1

def test_exp():
    x = AutoDiff(4, 1)
    f = np.exp(x)
    assert f.val == np.exp(4)
    assert f.derv == np.exp(4)

def test_log():
    x = AutoDiff(4, 1)
    f = np.log(x)
    assert f.val == np.log(4)
    assert f.derv == 0.25

def test_log2():
    x = AutoDiff(100, 1)
    f = np.log2(x)
    assert round(f.val,4) == 6.6439
    assert round(f.derv,4) == 0.0144

def test_log10():
    x = AutoDiff(100, 1)
    f = np.log10(x)
    assert round(f.val,4) == 2
    assert round(f.derv,4) == 0.0043

#---------------Testing comparison operators ------------------------------------

def test_less_than_greater_than1():
    x = AutoDiff(2, 1)
    y = AutoDiff(3, 1)
    assert (x < y) == True
    assert (x <= y) == True
    assert (x > y) == False
    assert (x >= y) == False
    assert (x == y) == False
    assert (y < x) == False
    assert (y > x) == True
    assert (y <= x) == False
    assert (y >= x) == True
    assert (y == x) == False

def test_less_than_greater_than2():
    x = AutoDiff(2, 1)
    y = AutoDiff(3, 1)
    a = 1
    assert (a < y) == True
    assert (x <= a) == False
    assert (x > a) == True
    assert (x >= a) == True
    assert (a == y) == False
    assert (y < a) == False
    assert (y > a) == True
    assert (a <= x) == True
    assert (a >= x) == False
    assert (y == a) == False
    a = 3
    assert (y == a) == True
    assert (a == y) == True
    assert (a < y) == False
    assert (y < a) == False
    assert(y <= a) == True
    assert (a <= y) == True
    assert (y >= a) == True
    assert (a >= y) == True
    assert (a > y) == False
    assert (a >  x) == True
    assert (x > a) == False


def test_not_equal():
    x = AutoDiff(2, 1)
    y = AutoDiff(3, 1)
    a = 2
    assert (x !=  y) ==True
    assert (y != x) == True
    assert (y == x) == False
    assert (x == y) == False
    assert (a != x) == False
    assert (a == x) == True
    assert (x == a) == True
    assert (x != a) == False
    assert (y != a) == True


#---------------Testing Complicated functions------------------------------------
def test_1_dx():
    x = AutoDiff(3, 1)
    y = AutoDiff(2, 0)
    f = x + 2*y
    assert f.val == 7
    assert f.derv == 1

def test_1_dy():
    x = AutoDiff(3, 0)
    y = AutoDiff(2, 1)
    f = x + 2*y
    assert f.val == 7
    assert f.derv == 2

def test_2_dx():
    x = AutoDiff(3, 1)
    y = AutoDiff(2, 0)
    f = x**2 + y**2 + 4
    assert f.val == 17
    assert f.derv == 6

def test_2_dy():
    x = AutoDiff(3, 0)
    y = AutoDiff(2, 1)
    f = x**2 + y**2 + 4
    assert f.val == 17
    assert f.derv == 4

def test_3_dx():
    x = AutoDiff(3, 1)
    y = AutoDiff(2, 0)
    f = y*x+np.exp(y*x)+y**2
    assert round(f.val,2) == 413.43
    assert round(f.derv,2) == 808.86

def test_3_dy():
    x = AutoDiff(3, 0)
    y = AutoDiff(2, 1)
    f = y*x+np.exp(y*x)+y**2
    assert round(f.val,2) == 413.43
    assert round(f.derv,2) == 1217.29

def test_4_dx():
    x = AutoDiff(3, 1)
    y = AutoDiff(2, 0)
    f = x*np.sin(x*y) + 2*y*x + np.cos(x)
    assert round(f.val,2) == 10.17
    assert round(f.derv,2) == 9.34

def test_4_dy():
    x = AutoDiff(3, 0)
    y = AutoDiff(2, 1)
    f = x*np.sin(x*y) + 2*y*x + np.cos(x)
    assert round(f.val,2) == 10.17
    assert round(f.derv,2) == 14.64


def test_5_dx():
    x = AutoDiff(3, 1)
    y = AutoDiff(2, 0)
    f = x*np.log(x*y)/(2*y*x) + np.tan(x) + np.sqrt(y)
    assert round(f.val,2) == 1.72
    assert round(f.derv,2) == 1.10

def test_6_dx():
    x = AutoDiff(3, 1)
    y = AutoDiff(2, 0)
    f = 3 - x**np.log(x**2) - np.tan(y)/2 - 3**y
    assert round(f.val,2) == -16.08
    assert round(f.derv,2) == -16.37

def test_6_dy():
    x = AutoDiff(3, 0)
    y = AutoDiff(2, 1)
    f = 3 - x**np.log(x**2) - np.tan(y)/2 - 3**y
    assert round(f.val,2) == -16.08
    assert round(f.derv,2) == -12.77
