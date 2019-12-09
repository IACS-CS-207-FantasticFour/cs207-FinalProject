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

    def __pos__(self):
        return AutoDiff(self.val, self.derv)

    def __neg__(self):
        return AutoDiff(-1*self.val, -1*self.derv)

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

    def __pow__(self, other):
        try:
            return AutoDiff(self.val**other.val, other.val*self.val**(other.val - 1)*self.derv  +  self.val**other.val*np.log(self.val)*other.derv )
        except:
            return AutoDiff(self.val**other, other*self.val**(other - 1)*self.derv)

    def __rpow__(self, other):
        return AutoDiff(other**self.val, other**self.val*np.log(other)*self.derv)

    def sin(self):
        return AutoDiff(np.sin(self.val), np.cos(self.val)*self.derv)

    def cos(self):
        return AutoDiff(np.cos(self.val), -1*np.sin(self.val)*self.derv)

    def tan(self):
        return AutoDiff(np.tan(self.val), (1/(np.cos(self.val)**2))*self.derv)

    def exp(self):
        return AutoDiff(np.exp(self.val), np.exp(self.val)*self.derv)

    def log(self):
        return AutoDiff(np.log(self.val), (1/self.val)*self.derv)

    def log10(self):
        return AutoDiff(np.log10(self.val), (1/(np.log(10)*self.val))*self.derv)

    def sqrt(self):
        return AutoDiff(np.sqrt(self.val), (1/(2*np.sqrt(self.val) )*self.derv) )

