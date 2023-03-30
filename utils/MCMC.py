import numpy as np
from scipy.stats import gamma

def compute_P_Lotus_given_research_effort(lotus_n_papers, gamma, delta):
    P_m = np.sum(lotus_n_papers, axis=1)
    Q_s = np.sum(lotus_n_papers, axis=0)
    clipped_exponent = np.clip(-gamma * P_m[:, None] - delta * Q_s[None, :], -50, 50)
    total_research_effort = 1 - np.exp(clipped_exponent, dtype="float32")
    #total_research_effort = 1 - np.exp(-gamma * P_m[:, None] - delta * Q_s[None, :], dtype="float32")
    
    #put proba to 1 if it has already been found
    #found_in_Lotus = np.where(lotus_n_papers != 0)
    #total_research_effort[found_in_Lotus] = 1
    
    return total_research_effort

def log_likelihood(lotus_n_papers, gamma, delta):
    prob_Lotus = compute_P_Lotus_given_research_effort(lotus_n_papers, gamma, delta)
    log_likelihood = np.sum(np.log(prob_Lotus[lotus_n_papers != 0]+ 1e-10)) + np.sum(np.log(1 - prob_Lotus[lotus_n_papers == 0]+ 1e-10))
    return log_likelihood 

#def log_prior(gamma, delta):
#    return -gamma - delta
def log_prior(gamma_param, delta, shape=2, scale=1):
    return gamma.logpdf(gamma_param, a=shape, scale=scale) + gamma.logpdf(delta, a=shape, scale=scale)

def proposal(gamma, delta, proposal_scale=0.1):
    gamma_new = gamma + np.random.normal(loc=0, scale=proposal_scale)
    delta_new = delta + np.random.normal(loc=0, scale=proposal_scale)
    
    # Apply reflection strategy by taking absolute values
    gamma_new = np.abs(gamma_new)
    delta_new = np.abs(delta_new)
    
    return gamma_new, delta_new

def metropolis_hastings_accept(lotus_n_papers, gamma, delta, gamma_new, delta_new):
    curr_log_likelihood = log_likelihood(lotus_n_papers, gamma, delta)
    new_log_likelihood = log_likelihood(lotus_n_papers, gamma_new, delta_new)
    curr_log_prior = log_prior(gamma, delta)
    new_log_prior = log_prior(gamma_new, delta_new)
    
    if not (np.isfinite(curr_log_likelihood) and np.isfinite(new_log_likelihood) and np.isfinite(curr_log_prior) and np.isfinite(new_log_prior)):
        return False
    
    log_ratio = (new_log_likelihood + new_log_prior) - (curr_log_likelihood + curr_log_prior)
    return np.random.uniform() < np.exp(min(0, log_ratio))

def run_mcmc(lotus_n_papers, n_iter, gamma_init, delta_init):
    gamma, delta = gamma_init, delta_init
    samples = np.zeros((n_iter, 2))
    
    accept_gamma = 0
    accept_delta = 0
    for i in range(n_iter):
        # Update gamma
        gamma_new, _ = proposal(gamma, delta)
        if metropolis_hastings_accept(lotus_n_papers, gamma, delta, gamma_new, delta):
            gamma = gamma_new
            accept_gamma += 1

        # Update delta
        _, delta_new = proposal(gamma, delta)
        if metropolis_hastings_accept(lotus_n_papers, gamma, delta, gamma, delta_new):
            delta = delta_new
            accept_delta += 1

        samples[i] = [gamma, delta]
    
    return samples, accept_gamma/n_iter, accept_delta/n_iter
