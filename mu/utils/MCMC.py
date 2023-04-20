import numpy as np
import scipy.special

def mh_accept_mu(mu, mu_new, e, x):
    # Calculate the log likelihood of the current and proposed values of the interaction matrix
    curr_log_likelihood = log_likelihood_mu(mu=mu, e=e, x=x)
    new_log_likelihood = log_likelihood_mu(mu=mu_new, e=e, x=x)

    # Check if the log likelihoods are finite, and return False if any of them are not
    if not (np.isfinite(curr_log_likelihood) and np.isfinite(new_log_likelihood)):
        return False

    # Calculate the log ratio of the new and current values of the interaction matrix
    log_ratio = new_log_likelihood - curr_log_likelihood

    # Accept the proposal with probability exp(min(0, log_ratio))
    return np.random.uniform() < np.exp(min(0, log_ratio))

def log_likelihood_mu(mu, e, x):
    p_x_1_mu = scipy.special.expit(mu + e)
    p_x_0_mu = 1-p_x_1_mu
    
    likelihood_ = np.zeros(x.shape)
    x_0 = np.where(x==0)
    x_1 = np.where(x==1)
    likelihood_[x_0] = p_x_0_mu[x_0]
    likelihood_[x_1] = p_x_1_mu[x_1]
    return np.sum(np.log(likelihood_ + 1e-12))

def proposal_mu(x, proposal_scale=0.01):
    return x + np.random.normal(loc=0, scale=proposal_scale)

def update_mu(mu_arrays):
    mu = np.sum(np.ix_(*mu_arrays), axis=0)
    return mu

def gibbs_sample_x(lotus_n_papers, mu, e, gamma, delta):
    #calculate prior
    p_x_1 = scipy.special.expit(mu + e)
    p_x_0 = 1 - p_x_1
    
    #calculate likelihood of data given x
    likelihood_x_0 = likelihood(lotus_n_papers, np.zeros_like(mu), gamma, delta)
    likelihood_x_1 = likelihood(lotus_n_papers, np.ones(shape=mu.shape), gamma, delta)
    
    #compute likelihood times prior
    numerator_0 = likelihood_x_0 * p_x_0
    numerator_1 = likelihood_x_1 * p_x_1

    denominator = numerator_0 + numerator_1
    
    conditional_probability = numerator_1/denominator
    return np.random.binomial(n=1, p=conditional_probability)

def likelihood(lotus_n_papers, x, gamma, delta):
    # Calculate the total number of papers published per molecule
    P_m = np.sum(lotus_n_papers, axis=1)
    
    # Calculate the total number of papers published per species
    Q_s = np.sum(lotus_n_papers, axis=0)
    
    # Calculate the total research effort for each (molecule, species) pair
    clipped_exponent = np.clip(-gamma * P_m[:, None] - delta * Q_s[None, :], -50, 50)
    total_research_effort = 1 - np.exp(clipped_exponent, dtype="float32")
    
    likelihood_ = np.zeros_like(total_research_effort)
    
    #add the four conditions of our model
    x_0_L_0 = (lotus_n_papers == 0) & (x==0)
    #x_0_L_1 =((lotus_n_papers != 0) & (x==0)
    x_1_L_0 = (lotus_n_papers == 0) & (x!=0)
    x_1_L_1 = (lotus_n_papers != 0) & (x!=0)
    
    #fullfil the four conditions
    likelihood_[x_0_L_0] = 1
    #likelihood[x_0_L_1] = 0
    likelihood_[x_1_L_0] = 1-total_research_effort[x_1_L_0]
    likelihood_[x_1_L_1] = total_research_effort[x_1_L_1]
    
    return likelihood_

# Define a function that calculates the log likelihood of the LOTUS model
def log_likelihood(lotus_n_papers, x, gamma, delta):
    likelihood_ = likelihood(lotus_n_papers, x, gamma, delta)
    
    # Calculate the likelihood of observing the data
    log_likelihood = np.sum(np.log(likelihood_ + 1e-12))
    return log_likelihood

# Define a function that calculates the log prior of the parameters gamma and delta
def log_prior(gamma, delta, gamma_min=0, gamma_max=10, delta_min=0, delta_max=10):
    # Return 0 if the parameters are within the specified bounds, and -inf otherwise
    if gamma_min <= gamma <= gamma_max and delta_min <= delta <= delta_max:
        return 0
    else:
        return -np.inf

# Define a function that proposes new values for gamma and delta based on the current values
def proposal(gamma, delta, proposal_scale_gamma, proposal_scale_delta):
    # Generate new values for gamma and delta by adding Gaussian noise with mean 0 and standard deviation proposal_scale_gamma and proposal_scale_delta, respectively
    gamma_new = gamma + np.random.normal(loc=0, scale=proposal_scale_gamma)
    delta_new = delta + np.random.normal(loc=0, scale=proposal_scale_delta)
    
    # Apply a reflection strategy by taking the absolute value of the new values
    gamma_new = np.abs(gamma_new)
    delta_new = np.abs(delta_new)
    
    return gamma_new, delta_new

# Define a function that determines whether to accept or reject the proposed values of gamma and delta
def metropolis_hastings_accept(lotus_n_papers, x, gamma, delta, gamma_new, delta_new):
    # Calculate the log likelihood and log prior of the current and proposed values of gamma and delta
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
def run_mcmc_with_gibbs(lotus_n_papers, x_init, n_iter, gamma_init, delta_init,
                        mu_arrays, e,
                        target_acceptance_rate=(0.25, 0.35), check_interval=500,thinning_factor=10):
    gamma, delta, x = gamma_init, delta_init, x_init
    mu = update_mu(mu_arrays)
    
    # Initialize the samples array to store gamma and delta values
    samples = np.zeros((n_iter // thinning_factor, 2))
    sample_count = 0
    
    # Initialize a list to store x values at each iteration
    x_samples = []
    mu_samples = []

    accept_gamma = 0
    accept_delta = 0
    accept_mu = np.zeros(len(mu_arrays))
    proposal_scale_gamma = 0.1
    proposal_scale_delta = 0.01
    proposal_scale_mu = 0.1

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
        
        # Update x using Gibbs sampling
        x = gibbs_sample_x(lotus_n_papers, mu, e, gamma, delta)
        
        for m in range(len(mu_arrays)):
            for j in range(len(mu_arrays[m])):
                mu_arrays_new = mu_arrays.copy()
                mu_arrays_new[m] = mu_arrays[m].copy()
                mu_arrays_new[m][j] = proposal_mu(mu_arrays[m][j], proposal_scale=proposal_scale_mu)
                
                #update mu
                mu_new = update_mu(mu_arrays_new)
                if mh_accept_mu(mu=mu, mu_new=mu_new, e=e, x=x):
                    mu = mu_new
                    mu_arrays[m][j] = mu_arrays_new[m][j]
                    accept_mu[m] += 1
        
        # Store gamma and delta values in the samples array if the current iteration is a multiple of the thinning factor
        if (i + 1) % thinning_factor == 0:
            samples[sample_count] = [gamma, delta]
            sample_count += 1

            # Append the current x value to the x_samples list
            x_samples.append(x)
            mu_samples.append(mu_arrays)

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

            for m in range(len(mu_arrays)):
                acceptance_rate_mu = accept_mu[m] / (check_interval * len(mu_arrays[m]))
                if acceptance_rate_mu < target_acceptance_rate[0]:
                    proposal_scale_mu *= 0.8
                elif acceptance_rate_mu > target_acceptance_rate[1]:
                    proposal_scale_mu *= 1.2
                accept_mu[m] = 0
    return samples, x_samples, mu_samples, acceptance_rate_gamma, acceptance_rate_delta
