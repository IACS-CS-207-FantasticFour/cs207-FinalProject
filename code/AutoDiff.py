import numpy as np

class AutoDiff():
    '''
    Class to use for automatic differentiation, forward mode

    Addresses the following need.

    1) The user has some function which maps some number of input variables, for example x and y, to a single
    output value, f(x,y).

    2) The user would like to calculate f(x,y) at a point (x,y) that they specify, as well as the partial
    derivatives df/dx and df/dy at that same point (x, y).

    By first defining their input variables as AutoDiff objects, and then using these objects in
    standard python calculations, including with numpy functions, the user is returned a AutoDiff object
    whose val attribute is the value of the function, and whose derv attribute is the derivative of the function.

    Which partial derivative is stored in the derv attribute of the final AutoDiff object depends on which
    input variable's derv attribute is initially set to 1, while the derv attributes of the other input variables
    are set to 0.

    Example:
    x = AutoDiff(3,1)
    y = AutoDiff(2,0)
    f = x**2 + y**2 + 4
    print(f.val, f.dev)
    (17, 6)
    '''

    def __init__(self, val, derv=1):
        self.val = float(val)
        self.derv = float(derv)

    # -----------Overloading Urnary functions ---------------------------------------------

    def __pos__(self):
        return AutoDiff(self.val, self.derv)

    def __neg__(self):
        return AutoDiff(-1*self.val, -1*self.derv)

    # -----------Add, Subtract, Multiple, Divide functions --------------------------------

    def __mul__(self, other):
        try:
            return AutoDiff(self.val * other.val, self.derv * other.val + other.derv*self.val)
        except:
            return AutoDiff(self.val * other, self.derv * other)

    def __rmul__(self, other):
        return AutoDiff(self.val * other, self.derv * other)

    def __truediv__(self, other):
        try:
            return AutoDiff(self.val/other.val, self.derv/other.val + ( -1*self.val/(other.val**2) ) *other.derv)
        except:
            return AutoDiff(self.val/other, self.derv/other)

    def __rtruediv__(self, other):
        return AutoDiff(other/self.val,  -1*other/(self.val**2) )

    def __add__(self, other):
        try:
            return AutoDiff(self.val + other.val, self.derv + other.derv)
        except:
            return AutoDiff(self.val + other, self.derv + 0)

    def __radd__(self, other):
        return AutoDiff(self.val + other, self.derv + 0)

    def __sub__(self, other):
        try:
            return AutoDiff(self.val - other.val, self.derv + -1*other.derv)
        except:
            return AutoDiff(self.val - other, self.derv + 0)

    def __rsub__(self, other):
        return AutoDiff(other - self.val, -1*self.derv + 0)

    # -----------Overloading Trigonometric functions ---------------------------------------------

    def sin(self):
        return AutoDiff(np.sin(self.val), np.cos(self.val)*self.derv)

    def cos(self):
        return AutoDiff(np.cos(self.val), -1*np.sin(self.val)*self.derv)

    def tan(self):
        return AutoDiff(np.tan(self.val), (1/(np.cos(self.val)**2))*self.derv)

    def arcsin(self):
        if (self.val < -1) or (self.val > 1):
            raise ValueError(' Input to arcsin() must be between -1 and 1 inclusive.')
        else:
            return AutoDiff(np.arcsin(self.val), 1/np.sqrt(1 - (self.val)**2)*self.derv)

    def arccos(self):
        if (self.val < -1) or (self.val > 1):
            raise ValueError(' Input to arccos() must be between -1 and 1 inclusive.')
        else:
            return AutoDiff(np.arccos(self.val), -1/np.sqrt(1 - (self.val)**2)*self.derv)

    def arctan(self):
        return AutoDiff(np.arctan(self.val), 1/(1 + (self.val)**2)*self.derv)

    # -----------Overloading Hyperbolic functions ---------------------------------------------

    def sinh(self):
        return AutoDiff(np.sinh(self.val), np.cosh(self.val)*self.derv)

    def cosh(self):
        return AutoDiff(np.cosh(self.val), np.sinh(self.val)*self.derv)

    def tanh(self):
        return AutoDiff(np.tanh(self.val), (1/np.cosh(self.val))**2*self.derv)

    def arcsinh(self):
        return AutoDiff(np.arcsinh(self.val), 1/np.sqrt(1 +(self.val)**2)*self.derv)

    def arccosh(self):
        return AutoDiff(np.arccosh(self.val), 1/np.sqrt((self.val)**2 - 1) *self.derv)

    def arctanh(self):
        return AutoDiff(np.arctanh(self.val), 1/(1 - (self.val)**2) *self.derv)

    # -----------Overloading Power functions ---------------------------------------------

    def __pow__(self, other):
        try:
            return AutoDiff(self.val**other.val, other.val*self.val**(other.val - 1)*self.derv  +  self.val**other.val*np.log(self.val)*other.derv )
        except:
            return AutoDiff(self.val**other, other*self.val**(other - 1)*self.derv)

    def __rpow__(self, other):
        return AutoDiff(other**self.val, other**self.val*np.log(other)*self.derv)

    def exp(self):
        return AutoDiff(np.exp(self.val), np.exp(self.val)*self.derv)

    # -----------Overloading log functions ---------------------------------------------

    def log(self):
        return AutoDiff(np.log(self.val), (1/self.val)*self.derv)

    def log10(self):
        return AutoDiff(np.log10(self.val), (1/(np.log(10)*self.val))*self.derv)

    def log2(self):
        return AutoDiff(np.log2(self.val), (1 / (np.log(2) * self.val)) * self.derv)

    # -----------Overloading the Square Root function ------------------------------------------

    def sqrt(self):
        if self.val < 0:
            print('negative number has been input to the sqaure root function')
            return 0
        else:
            return AutoDiff(np.sqrt(self.val), (1/(2*np.sqrt(self.val) )*self.derv) )

    # -----------Overloading the logistic function ---------------------------------------------

    def logist(self):
        return AutoDiff( 1/(1 + np.exp(-1*(self.val))), (1/(1 + np.exp(-1*(self.val)))) * (1 - (1/(1 + np.exp(-1*(self.val))))) *self.derv)

    #-----------Overloading the Comparison Operators >, <, >=, <=, ==. != -------------------------

    def __lt__(self, other):
        try:
            if self.val < other.val:
                return True
            else:
                return False
        except:
            if self.val < other:
                return True
            else:
                return False


    def __gt__(self, other):
        try:
            if self.val > other.val:
                return True
            else:
                return False
        except:
            if self.val > other:
                return True
            else:
                return False


    def __le__(self, other):
        try:
            if self.val <= other.val:
                return True
            else:
                return False
        except:
            if self.val <= other:
                return True
            else:
                return False

    def __ge__(self, other):
        try:
            if self.val >= other.val:
                return True
            else:
                return False
        except:
            if self.val >= other:
                return True
            else:
                return False

    def __eq__(self, other):
        try:
            if self.val == other.val:
                return True
            else:
                return False
        except:
            if self.val == other:
                return True
            else:
                return False

    def __ne__(self, other):
        try:
            if self.val != other.val:
                return True
            else:
                return False
        except:
            if self.val != other:
                return True
            else:
                return False


def logist(x):
    if isinstance(x, AutoDiff):
        return AutoDiff(1 / (1 + np.exp(-1 * (x.val))),
                        (1 / (1 + np.exp(-1 * (x.val)))) * (1 - (1 / (1 + np.exp(-1 * (x.val))))) * x.derv)
    else:
        return 1/(1+np.exp(-1*x))

def logN(x, N):
    if isinstance(x, AutoDiff):
        return AutoDiff(np.log(x.val)/np.log(N), (1/(x.val*np.log(N)) *x.derv))
    else:
        return np.log(x)/np.log(N)

