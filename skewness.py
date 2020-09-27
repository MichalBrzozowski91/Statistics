import pandas as pd 
from scipy import stats

values = [1,1,2]
data = pd.DataFrame({'value':values})
skewness_pd = data.skew()
skewness_scipy = stats.skew(data, bias = True)
skewness_scipy_nb = stats.skew(data, bias = False)
print('Pandas skewness is equal to', skewness_pd[0])
print('SciPy skewness with bias = True is equal to', skewness_scipy[0])
print('SciPy skewness with bias = False is equal to', skewness_scipy_nb[0])

def biased_skewness(x, ddof = 1):
    moment_3 = 0
    mean = x.mean()
    moment_2 = x.std(ddof = ddof) # Square root of unbiased variance
    for i in x:
        moment_3 += (i - mean)**3
    moment_3 /= len(x)
    return moment_3/(moment_2**3)

def adjusted_skewness(x):
    n = len(x)
    moment_3 = 0
    mean = x.mean()
    moment_2 = x.std() # Square root of unbiased variance
    for i in x:
        moment_3 += (i - mean)**3
    unbiased_cummulant_3 = moment_3 * n /((n-1)*(n-2)) 
    return unbiased_cummulant_3/(moment_2**(3))

print('Moment method skewness is equal to', biased_skewness(data.value))
print('Moment method skewness with biased std estimator is equal to', biased_skewness(data.value,0))
print('Adjusted Fisher–Pearson standardized moment coefficient  is equal to', adjusted_skewness(data.value))