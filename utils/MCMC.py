import numpy as np
from scipy.stats import gamma

def compute_P_Lotus_given_research_effort(lotus_n_papers, gamma, delta):
    P_m = np.sum(lotus_n_papers, axis=1)
    Q_s = np.sum(lotus_n_papers, axis=0)
    clipped_exponent = np.clip(-gamma * P_m[:, None] - delta * Q_s[None, :], -50, 50)
    total_research_effort = 1 - np.exp(clipped_exponent * np.power(np.inf, lotus_n_papers), dtype="float32")
    #total_research_effort = 1 - np.exp(-gamma * P_m[:, None] - delta * Q_s[None, :], dtype="float32")
    
    ##put proba to 1 if it has already been found
    #found_in_Lotus = np.where(lotus_n_papers != 0)
    #total_research_effort[found_in_Lotus] = 1.0
    
    return total_research_effort

def log_likelihood(lotus_n_papers, gamma, delta):
    prob_Lotus = compute_P_Lotus_given_research_effort(lotus_n_papers, gamma, delta)
    #log_likelihood = np.sum(np.log(prob_Lotus + 1e-10))
    log_likelihood = np.sum(np.log(prob_Lotus + 1e-10)) + np.sum(np.log(1 - prob_Lotus + 1e-10))
    return log_likelihood 

#def log_prior(gamma, delta):
#    return -gamma - 0.1 * delta
#def log_prior(gamma_param, delta, shape=0.1, scale=1):
#    return gamma.logpdf(gamma_param, a=shape, scale=scale) + gamma.logpdf(delta, a=shape, scale=scale)
def log_prior(gamma, delta, gamma_min=0, gamma_max=1, delta_min=0, delta_max=1):
    if gamma_min <= gamma <= gamma_max and delta_min <= delta <= delta_max:
        return 0
    else:
        return -np.inf

def proposal(gamma, delta, proposal_scale_gamma, proposal_scale_delta):
    gamma_new = gamma + np.random.normal(loc=0, scale=proposal_scale_gamma)
    delta_new = delta + np.random.normal(loc=0, scale=proposal_scale_delta)
    
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

def run_mcmc(lotus_n_papers, n_iter, gamma_init, delta_init, target_acceptance_rate=(0.25, 0.35), check_interval=100):
    gamma, delta = gamma_init, delta_init
    samples = np.zeros((n_iter, 2))

    accept_gamma = 0
    accept_delta = 0
    proposal_scale_gamma = 0.001
    proposal_scale_delta = 0.0001

    for i in range(n_iter):
        # Update gamma
        gamma_new, _ = proposal(gamma, delta, proposal_scale_gamma, proposal_scale_delta)
        if metropolis_hastings_accept(lotus_n_papers, gamma, delta, gamma_new, delta):
            gamma = gamma_new
            accept_gamma += 1

        # Update delta
        _, delta_new = proposal(gamma, delta, proposal_scale_gamma, proposal_scale_delta)
        if metropolis_hastings_accept(lotus_n_papers, gamma, delta, gamma, delta_new):
            delta = delta_new
            accept_delta += 1

        samples[i] = [gamma, delta]

        # Check acceptance rate and adjust proposal_scale if necessary
        if (i + 1) % check_interval == 0:
            acceptance_rate_gamma = accept_gamma / check_interval
            acceptance_rate_delta = accept_delta / check_interval

            if not (target_acceptance_rate[0] <= acceptance_rate_gamma <= target_acceptance_rate[1] and
                    target_acceptance_rate[0] <= acceptance_rate_delta <= target_acceptance_rate[1]):
                # Adjust proposal_scale_gamma and proposal_scale_delta
                if acceptance_rate_gamma < target_acceptance_rate[0]:
                    proposal_scale_gamma *= 0.5
                elif acceptance_rate_gamma > target_acceptance_rate[1]:
                    proposal_scale_gamma *= 2

                if acceptance_rate_delta < target_acceptance_rate[0]:
                    proposal_scale_delta *= 0.5
                elif acceptance_rate_delta > target_acceptance_rate[1]:
                    proposal_scale_delta *= 2

                # Reset acceptance counters
                accept_gamma = 0
                accept_delta = 0

    acceptance_rate_gamma = accept_gamma / n_iter
    acceptance_rate_delta = accept_delta / n_iter

    return samples, acceptance_rate_gamma, acceptance_rate_delta
