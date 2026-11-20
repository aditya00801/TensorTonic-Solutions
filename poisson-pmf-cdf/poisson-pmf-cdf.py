import math

def poisson_pmf_cdf(lam: float, k: int) -> dict:
    """
    Returns a dictionary with pmf and cdf.
    """
     # Calculate PMF: P(X = k)
    pmf = (math.exp(-lam)) * (lam**k) / math.factorial(k)

     # Calculate CDF: P(X <= k)
    
    cdf = 0.0
    for i in range(k +1): 
        cdf += (math.exp(-lam) * (lam**i))/math.factorial(i)

    return {
        "pmf": float(pmf),
        "cdf": float(cdf)
    
    }
    
    