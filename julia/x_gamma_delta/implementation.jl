include("utils/sigma_utils.jl")
include("utils/utils.jl")
include("utils/MCMC.jl")
using Distributions
using DataFrames

# Initialize size of simulated data
T = ["m", "s"]
n_t = [100, 10]
blocks = [4, 2]
@assert length(T) == length(n_t)
@assert length(n_t) == length(blocks)

println("Simulating priors... ")
mu, alpha, beta =  simulate_from_prior(T, n_t)
sum_mus =  compute_sum_of_mus(mu)

sigma =  simulate_from_prior_sigma(T, n_t, blocks)

sigma_blocks =  simulate_sigma_blocks(blocks)
sigma_blocks_multivar =  rand(MultivariateNormal(zeros(size(sigma_blocks, 1)), sigma_blocks))

big_cov_mat =  compute_epsilon(alpha, sigma)
epsilon_c_multivar =  [rand(MvNormal(zeros(size(i, 1)), i)) for i in big_cov_mat]

println("Generating data... ")
prob_X =  compute_prob_X(sum_mus, epsilon_c_multivar, sigma_blocks_multivar, blocks, n_t)
x =  simulate_X(prob_X)

#get probability of Lotus and simulate Lotus
prob_Lotus, n_papers, gamma, delta = compute_prob_L(x)
lotus_binary, lotus_n_papers = simulate_lotus(prob_Lotus, n_papers)

# Run the MCMC chain
n_iter = 100000
gamma_init = 1.0
delta_init = 1.0
x_init = zeros(size(lotus_binary))
#x_init = x
println("Running MCMC")
samples, x_samples, accept_gamma, accept_delta = run_mcmc_with_gibbs(lotus_n_papers, x_init, n_iter,
                                                                     gamma_init, delta_init,
                                                                     sum_mus,
                                                                     epsilon_c_multivar,
                                                                     sigma_blocks_multivar,
                                                                     blocks)

burn_in = Int64(0.5 * n_iter)  # Remove the first 50% of the samples
post_burn_in_samples = samples[burn_in:end]
x_samples_burn = x_samples[burn_in:end]

# Extract the posterior mean estimates for gamma and delta
gamma_posterior_mean = mean([sublist[1] for sublist in post_burn_in_samples])
delta_posterior_mean = mean([sublist[2] for sublist in post_burn_in_samples])

println("True gamma: ", gamma)
println("Estimated gamma: ", gamma_posterior_mean)
println("True delta: ", delta)
println("Estimated delta: ",delta_posterior_mean)
println("rate accept gamma : ", accept_gamma)
println("rate accept delta : ", accept_delta)

#print(np.corrcoef(x.flatten(), average_x.flatten()))