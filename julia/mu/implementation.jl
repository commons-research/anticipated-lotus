include("utils/sigma_utils.jl")
include("utils/utils.jl")
include("utils/MCMC.jl")
using Distributions
using DataFrames
using Statistics, LinearAlgebra, Plots

# Initialize size of simulated data
T = ["m", "s"]
n_t = [100, 26]
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
n_iter = 20000
gamma_init = 1.0
delta_init = 1.0
x_init = zeros(size(lotus_binary))
#x_init = x
mu_init = [zeros(i) for i in n_t]
println("Running MCMC")
println("N iterations : ", n_iter)
samples, x_samples, mu_samples, accept_gamma, accept_delta = run_mcmc_with_gibbs(lotus_n_papers, x_init, n_iter,
                                                                     gamma_init, delta_init,
                                                                     mu_init,
                                                                     epsilon_c_multivar,
                                                                     sigma_blocks_multivar,
                                                                     blocks)

burn_in = Int64(0.5 * size(samples)[1])  # Remove the first 50% of the samples

post_burn_in_samples = samples[burn_in+1:end, :]
x_samples_burn = x_samples[burn_in+1:end]

# Extract the posterior mean estimates for gamma and delta
mean_of_samples = mean(post_burn_in_samples, dims=1)
gamma_posterior_mean = mean_of_samples[1]
delta_posterior_mean = mean_of_samples[2]

println("True gamma: ", gamma)
println("Estimated gamma: ", gamma_posterior_mean)
println("True delta: ", delta)
println("Estimated delta: ",delta_posterior_mean)
println("rate accept gamma : ", accept_gamma)
println("rate accept delta : ", accept_delta)

# check correlation of x (should be 1 if x is fixed, check in the MCMC file)
x_samples_burn = x_samples[burn_in+1:end]
average_x_values = mean(x_samples_burn)
correlation = cor(vec(x), vec(average_x_values))
println("Correlation of X : ", correlation)

mu_samples_burn = mu_samples[burn_in+1:end]

average_estimations = Dict()
for key in keys(mu)
    key_index = findfirst(isequal(key), collect(keys(mu)))
    key_estimations = [estimation[key_index] for estimation in mu_samples_burn]
    key_avg = mean(key_estimations, dims=1)[1]
    average_estimations[key] = key_avg
    println("Correlation mu '$(key)' : ", cor(mu[key], average_estimations[key]))
end

#gr()
#display(plot(range(1, length(x_samples)), samples[:, 1], title="Gamma estimation", seriestype=:scatter))