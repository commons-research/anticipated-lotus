# cython: boundscheck=False, wraparound=False, nonecheck=False
import numpy as np
cimport numpy as np
np.import_array()

# Define a function that calculates the log likelihood of the LOTUS model
cdef np.ndarray log_likelihood(np.ndarray[np.int32_t, ndim=2] lotus_n_papers, np.ndarray[np.uint8_t, ndim=2] x, double gamma, double delta):
    cdef np.ndarray[np.float64_t, ndim=1] P_m, Q_s
    cdef np.ndarray[np.float64_t, ndim=2] likelihood = np.zeros_like(lotus_n_papers, dtype=np.float64)

    # Calculate the total number of papers published per molecule
    P_m = np.sum(lotus_n_papers, axis=1, dtype=np.float64)
    
    # Calculate the total number of papers published per species
    Q_s = np.sum(lotus_n_papers, axis=0, dtype=np.float64)
    
    # Calculate the total research effort for each (molecule, species) pair
    clipped_exponent = np.clip(-gamma * P_m[:, None] - delta * Q_s[None, :], -50, 50)
    total_research_effort = 1 - np.exp(clipped_exponent, dtype="float32")
    
    
    #add the four conditions of our model
    x_0_L_0 = np.where((lotus_n_papers == 0) & (x==0))
    x_0_L_1 = np.where((lotus_n_papers != 0) & (x==0))
    x_1_L_0 = np.where((lotus_n_papers == 0) & (x!=0))
    x_1_L_1 = np.where((lotus_n_papers != 0) & (x!=0))
    
    #fullfil the four conditions
    likelihood[x_0_L_0] = 1
    likelihood[x_0_L_1] = 0
    likelihood[x_1_L_0] = 1-total_research_effort[x_1_L_0]
    likelihood[x_1_L_1] = total_research_effort[x_1_L_1]
    
    # Calculate the likelihood of observing the data
    log_likelihood = np.sum(np.log(likelihood + 1e-12))
    return log_likelihood

# Define a function that calculates the log prior of the parameters gamma and delta
def log_prior(double gamma, double delta, double gamma_min=0, double gamma_max=10, double delta_min=0, double delta_max=10):
    # Return 0 if the parameters are within the specified bounds, and -inf otherwise
    if gamma_min <= gamma <= gamma_max and delta_min <= delta <= delta_max:
        return 0
    else:
        return -np.inf

# Define a function that proposes new values for gamma and delta based on the current values
def proposal(double gamma, double delta, double proposal_scale_gamma, double proposal_scale_delta):
    # Generate new values for gamma and delta by adding Gaussian noise with mean 0 and standard deviation proposal_scale_gamma and proposal_scale_delta, respectively
    cdef double gamma_new, delta_new
    gamma_new = gamma + np.random.normal(loc=0, scale=proposal_scale_gamma)
    delta_new = delta + np.random.normal(loc=0, scale=proposal_scale_delta)
    
    # Apply a reflection strategy by taking the absolute value of the new values
    gamma_new = np.abs(gamma_new)
    delta_new = np.abs(delta_new)
    
    return gamma_new, delta_new

# Define a function that determines whether to accept or reject the proposed values of gamma and delta
def metropolis_hastings_accept(np.ndarray[np.int32_t, ndim=2] lotus_n_papers, np.ndarray[np.uint8_t, ndim=2] x, double gamma, double delta, double gamma_new, double delta_new):
    # Calculate the log likelihood and log prior of the current and proposed values of gamma and delta
    cdef double curr_log_likelihood, new_log_likelihood

    curr_log_likelihood = log_likelihood(lotus_n_papers, x, gamma, delta)
    new_log_likelihood = log_likelihood(lotus_n_papers, x, gamma_new, delta_new)
    curr_log_prior = log_prior(gamma, delta)
    new_log_prior = log_prior(gamma_new, delta_new)
    
    # Check if the log likelihood and log prior are finite, and return False if any of them are not
    if not (np.isfinite(curr_log_likelihood) and np.isfinite(new_log_likelihood) and np.isfinite(curr_log_prior) and np.isfinite(new_log_prior)):
        return False
    
    # Calculate the log ratio of the new and current values of gamma and delta
    log_ratio = (new_log_likelihood + new_log_prior) - (curr_log_likelihood + curr_log_prior)
    
    # Accept the proposal with probability exp(min(0, log_ratio))
    return np.random.uniform() < np.exp(min(0, log_ratio))

# Define a function that runs the Metropolis-Hastings MCMC algorithm
def run_mcmc(np.ndarray[np.int32_t, ndim=2] lotus_n_papers, np.ndarray[np.uint8_t, ndim=2] x, int n_iter, double gamma_init, double delta_init, target_acceptance_rate=(0.25, 0.35), int check_interval=500):
    cdef double gamma, delta, accept_gamma, accept_delta, proposal_scale_gamma, proposal_scale_delta

    gamma, delta = gamma_init, delta_init
    samples = np.zeros((n_iter, 2))

    accept_gamma = 0
    accept_delta = 0
    proposal_scale_gamma = 0.1
    proposal_scale_delta = 0.1

    cdef int i
    for i in range(n_iter):
        # Update gamma
        gamma_new, _ = proposal(gamma, delta, proposal_scale_gamma, proposal_scale_delta)
        if metropolis_hastings_accept(lotus_n_papers, x, gamma, delta, gamma_new, delta):
            gamma = gamma_new
            accept_gamma += 1

        # Update delta
        _, delta_new = proposal(gamma, delta, proposal_scale_gamma, proposal_scale_delta)
        if metropolis_hastings_accept(lotus_n_papers, x, gamma, delta, gamma, delta_new):
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
                    proposal_scale_gamma *= 0.8
                elif acceptance_rate_gamma > target_acceptance_rate[1]:
                    proposal_scale_gamma *= 1.2

                if acceptance_rate_delta < target_acceptance_rate[0]:
                    proposal_scale_delta *= 0.8
                elif acceptance_rate_delta > target_acceptance_rate[1]:
                    proposal_scale_delta *= 1.2

            # Reset acceptance counters
            accept_gamma = 0
            accept_delta = 0

    return samples, acceptance_rate_gamma, acceptance_rate_delta