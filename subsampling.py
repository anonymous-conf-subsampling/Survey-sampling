import scipy.stats as stats
import scipy.special as sp
import matplotlib.pyplot as plt
import numpy as np
import math

# Compute Neyman allocation
def neyman_alloc(total_sample_size, stratum_pop_size, stratum_pop_std, pop_size_list, pop_std_list):
    num = total_sample_size * (stratum_pop_size * stratum_pop_std)
    denom = sum([pop_size_list[i]*pop_std_list[i] for i in range(len(pop_std_list))])
    stratum_sample_size = num / denom
    return stratum_sample_size

# Find eps guarantee based on two Laplace distributions
def compute_lb_eps(eps0, m1, m1p, gs=1.0):
    # Compute Laplace density at 0 with gs/eps0 noise centered at m1 and m1p
    # for laplace, error = sqrt(2)*scale = sqrt(2)/eps -> eps = sqrt(2)/error
    l1 = stats.laplace.pdf(0, m1, gs/eps0)
    l1p = stats.laplace.pdf(0, m1p, gs/eps0)
    # Find maximum epsilon difference (since two laplaces, they should all be equal)
    eps = np.log(max(l1, l1p)/min(l1,l1p))
    return eps

def max_diff_std(data_range, stratum_pop_size):
    return math.sqrt( (data_range**2 / stratum_pop_size) - (data_range / stratum_pop_size**2))

# Find sample sizes for strata (s1, s1') using Neyman allocation
def get_sizes(total_sample_size, s1_size, s1_std, s1p_size, s1p_std, rest_pop_size_list, rest_pop_std_list, rest_popp_size_list, rest_popp_std_list):
    # Get size for s1
    pop_size_list = rest_pop_size_list.copy()
    pop_size_list.append(s1_size)
    std_list = rest_pop_std_list.copy()
    std_list.append(s1_std)
    m1 = neyman_alloc(total_sample_size, s1_size, s1_std, pop_size_list, std_list)
    # Get size for s1'
    popp_size_list = rest_popp_size_list.copy()
    popp_size_list.append(s1p_size)
    stdp_list = rest_popp_std_list.copy()
    stdp_list.append(s1p_std)
    m1p = neyman_alloc(total_sample_size, s1p_size, s1p_std, popp_size_list, stdp_list)
    return m1, m1p

# Get global sensitivity of sample size based on Neyman allocation
def compute_gs(total_sample_size, min_pop_size, max_pop_size, min_pop_std, max_pop_std, 
               rest_min_pop_size_list, rest_min_pop_std_list, rest_max_pop_size_list, rest_max_pop_std_list):
    # Set worst case s1, s1'
    s1_size = min_pop_size
    s1_std = min_pop_std
    s1p_size = max_pop_size
    s1p_std = max_pop_std
    # Get worst case sizes
    m1, m1p = get_sizes(total_sample_size, s1_size, s1_std, s1p_size, s1p_std, 
        rest_max_pop_size_list, rest_max_pop_std_list, rest_min_pop_size_list, rest_min_pop_std_list)
    # Return global sensitivity, min, and max
    gs = abs(m1-m1p)
    min_stratum_sample_size = min(m1, m1p)
    max_stratum_sample_size = max(m1, m1p)
    return gs, min_stratum_sample_size, max_stratum_sample_size

# Risk lower bound for a given privacy guarantee (see Prop. 4.1 of our paper or Thm. 1 of Asi-Duchi arXiv paper)
def compute_risk_lower_bound(n, epsilon, data_range, total_sample_size, s1_size, s1_std, 
               rest_pop_size_list, rest_pop_std_list, rest_popp_size_list, rest_popp_std_list):
    err_list = []
    for k in range(1, int(n+1)):
        s1p_size = s1_size + k
        s1p_std = s1_std + k*max_diff_std(data_range, s1_size)
        min_stratum_sample_size, max_stratum_sample_size = get_sizes(total_sample_size, s1_size, s1_std, 
            s1p_size, s1p_std, rest_pop_size_list, rest_pop_std_list, rest_popp_size_list, rest_popp_std_list)
        # Compute lower bound using squared error
        omega = max_stratum_sample_size - min_stratum_sample_size
        l_omega = (omega/2.0)**2
        err = l_omega / (np.exp(2*k*epsilon) + 1.0)  
        err_list.append(err)  
    return np.max(err_list)

# Binary search to find range of eps for given risk
def eps_binsearch_ad(max_pop_size, target_risk, num_iters, eps_upper, data_range, total_sample_size, s1_size, s1_std, 
               rest_pop_size_list, rest_pop_std_list, rest_popp_size_list, rest_popp_std_list):
    llim = 0.00001
    rlim = eps_upper
    for t in range(num_iters):
        mideps = (rlim+llim)/2
        curr_risk = compute_risk_lower_bound(max_pop_size-s1_size, mideps, data_range, total_sample_size, s1_size, s1_std, 
               rest_pop_size_list, rest_pop_std_list, rest_popp_size_list, rest_popp_std_list)
        # print("mideps:", mideps, "curr_risk:", curr_risk, "target_risk", target_risk)
        if curr_risk < target_risk:
            llim = llim
            rlim = mideps
        else:
            llim = mideps
            rlim = rlim
    return llim, rlim

# Binary search plus step search to find min eps for given risk
def eps_search_ad(max_pop_size, target_risk, num_iters, eps_upper, data_range, total_sample_size, s1_size, s1_std, 
               rest_pop_size_list, rest_pop_std_list, rest_popp_size_list, rest_popp_std_list):
    llim, rlim = eps_binsearch_ad(max_pop_size, target_risk, num_iters, eps_upper, data_range, total_sample_size, s1_size, s1_std, 
               rest_pop_size_list, rest_pop_std_list, rest_popp_size_list, rest_popp_std_list)
    # print("eps llim:", llim, "rlim:", rlim)
    num_options = 100
    eps_list = list(np.linspace(llim, rlim+1, num_options))
    risk_list = []
    for i in range(len(eps_list)):
        eps = eps_list[i]
        curr_risk = compute_risk_lower_bound(max_pop_size-s1_size, eps, data_range, total_sample_size, s1_size, s1_std, 
               rest_pop_size_list, rest_pop_std_list, rest_popp_size_list, rest_popp_std_list)
        risk_list.append(curr_risk)
        # print("eps:", eps, "curr_risk:", curr_risk, "target_risk", target_risk)
        
    indices = np.where(risk_list <= target_risk)[0]
    index = min(indices) if len(indices) >0 else len(eps_list)-1
    return eps_list[index]

# Grid search over space of neighboring strata
def worst_case_ad_grid_search(target_error, num_iters, eps_upper, data_range, total_sample_size,
    min_pop_size, max_pop_size, num_sizes, min_pop_std, max_pop_std, num_stds, 
    rest_min_pop_size_list, rest_min_pop_std_list):

    eps_list = []
    size_list = np.linspace(min_pop_size, max_pop_size, num_sizes+1)[:-1]
    std_list = np.linspace(min_pop_std, max_pop_std, num_stds+1)[:-1]
    # print("size_list:", size_list, "std_list:", std_list)

    results = []
    for i in range(len(size_list)):
        s1_size = size_list[i]
        for j in range(len(std_list)):
            s1_std = std_list[j]
            eps_ad_lb = eps_search_ad(max_pop_size, target_error, num_iters, eps_upper, data_range, total_sample_size, s1_size, s1_std, 
               rest_min_pop_size_list, rest_min_pop_std_list, rest_min_pop_size_list, rest_min_pop_std_list)
            results.append(eps_ad_lb)
    # print("Worst case results:", results, "max:", max(results))
    return max(results)

# Define exponential mechanism
def expMechProbs(n, eps, k):
    probs = []
    normalization = 0.0
    for i in range(n+1):
        p = np.exp(-eps * abs(i - k))
        normalization += p
        probs.append(p)
    probs = [probs[i] / normalization for i in range(n+1)]
    return probs

# Compute amplification
def computeAmp(n, eps, p1, p2, ftilde1, ftilde2):
    prob1 = [0]*(n+1)
    prob2 = [0]*(n+1)
    for j in range(len(ftilde1)):
        # ftilde1, p1
        m = ftilde1[j][0]
        q = ftilde1[j][1]
        for k in range(m):
            em_probs = expMechProbs(n, eps, k)
            for x in range(n+1):
                prob1[x] = prob1[x] + em_probs[x] * (sp.comb(int(p1*n), k) * sp.comb(n-int(p1*n), m-k) / sp.comb(n, m)) * q #stats.binom.pmf(k, m, p1) * q
        # ftilde2, p2
        m = ftilde2[j][0]
        q = ftilde2[j][1]
        for k in range(m):
            em_probs = expMechProbs(n, eps, k)
            for x in range(n+1):       
                prob2[x] = prob2[x] + em_probs[x] *  (sp.comb(int(p2*n), k) * sp.comb(n-int(p2*n), m-k) / sp.comb(n, m)) * q #stats.binom.pmf(k, m, p2) * q
    logratios = [np.log(prob1[i]/prob2[i]) for i in range(len(prob1))]
    new_eps = max(np.abs(logratios))
    return new_eps