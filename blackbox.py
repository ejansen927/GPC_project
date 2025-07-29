# example simple blackbox

import numpy as np

def blackboxfunc(X): # 3 phase
    x0,x1=X[0],X[1]
    eq = np.sin(x0*x0)*np.cos(x1*x1) + 0.7*x1*x0 + 0.2*np.sin(x0*x1)
    if eq<0.2:
        return 0
    elif eq<0.7:
        return 1
    else:
        return 2
