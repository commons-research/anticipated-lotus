include("./utils.jl")

function log_likelihood_mu(mu, e, e_blocks, blocks, x)
    p_x_1_mu = compute_prob_X(Float32.(mu), e, e_blocks, blocks, collect(size(x)))
    p_x_0_mu = 1 .- p_x_1_mu
    
    likelihood_ = zeros(size(x))
    x_0 = findall(x .== 0)
    x_1 = findall(x .== 1)

    likelihood_[x_0] = p_x_0_mu[x_0]
    likelihood_[x_1] = p_x_1_mu[x_1]

    return sum(log.(likelihood_ .+ 1e-18))
end

function mh_accept_mu(mu, mu_new, e, e_blocks, blocks, x)
    # Calculate the log likelihood of the current and proposed values of the interaction matrix
    curr_log_likelihood = log_likelihood_mu(mu, e, e_blocks, blocks, x)
    new_log_likelihood = log_likelihood_mu(mu_new, e, e_blocks, blocks, x)

    # Calculate the log ratio of the new and current values of the interaction matrix
    log_ratio = new_log_likelihood - curr_log_likelihood

    # Accept the proposal with probability exp(min(0, log_ratio))
    return rand(Uniform()) < exp(min(0,log_ratio))
end

function proposal_mu(mu, proposal_scale=0.01)
    return mu + rand(Normal(0, proposal_scale))
end

function update_mu(mu_arrays)
    mu = sum.(Iterators.product(mu_arrays...))
    return mu
end

function gibbs_sample_x(lotus_n_papers, mu, e, e_blocks, gamma, delta, blocks)
    p_x_1 = compute_prob_X(Float32.(vec(mu)), e, e_blocks, blocks, collect(size(lotus_n_papers)))
    #calculate prior
    p_x_0 = 1 .- p_x_1
    
    #calculate likelihood of data given x
    likelihood_x_0 = likelihood(lotus_n_papers, zeros(size(p_x_1)), gamma, delta)
    likelihood_x_1 = likelihood(lotus_n_papers, ones(size(p_x_1)), gamma, delta)

    #compute likelihood times prior
    numerator_0 = likelihood_x_0 .* p_x_0
    numerator_1 = likelihood_x_1 .* p_x_1

    denominator = numerator_0 .+ numerator_1
    
    conditional_probability = numerator_1 ./ denominator
    return map(p -> rand(Binomial(1, p)), conditional_probability)
end

function likelihood(lotus_n_papers, x, gamma, delta)
    # Calculate the total number of papers published per molecule
    P_m = sum(lotus_n_papers, dims=2)
    
    # Calculate the total number of papers published per species
    Q_s = sum(lotus_n_papers, dims=1)
    
    # Calculate the total research effort for each (molecule, species) pair
    clipped_exponent = clamp.(-gamma .* P_m .- delta .* Q_s, -50, 50)
    total_research_effort = 1 .- exp.(clipped_exponent)
    
    likelihood_ = zeros(size(total_research_effort))
    
    # Add the four conditions of our model
    x_0_L_0 = (lotus_n_papers .== 0) .& (x .== 0)
    x_1_L_0 = (lotus_n_papers .== 0) .& (x .!= 0)
    x_1_L_1 = (lotus_n_papers .!= 0) .& (x .!= 0)
    
    # Fulfil the four conditions
    likelihood_[x_0_L_0] .= 1
    likelihood_[x_1_L_0] .= 1 .- total_research_effort[x_1_L_0]
    likelihood_[x_1_L_1] .= total_research_effort[x_1_L_1]
    
    return likelihood_
end

function log_likelihood(lotus_n_papers, x, gamma, delta)
    likelihood_ = likelihood(lotus_n_papers, x, gamma, delta)
    
    # Calculate the likelihood of observing the data
    log_likelihood = sum(log.(likelihood_ .+ 1e-12))
    return log_likelihood
end

function log_prior(gamma::Float64, delta::Float64, gamma_min::Float64 = 0.0,
                    gamma_max::Float64 = 2.0, delta_min::Float64 = 0.0, delta_max::Float64 = 2.0)
    # Return 0 if the parameters are within the specified bounds, and -Inf otherwise
    if gamma_min <= gamma <= gamma_max && delta_min <= delta <= delta_max
        return 0.0
    else
        return -Inf
    end
end

function proposal(gamma::Float64, delta::Float64, proposal_scale_gamma::Float64, proposal_scale_delta::Float64)
    # Generate new values for gamma and delta by adding Gaussian noise with mean 0 and standard deviation proposal_scale_gamma and proposal_scale_delta, respectively
    gamma_new = gamma + rand(Normal(0.0, proposal_scale_gamma))
    delta_new = delta + rand(Normal(0.0, proposal_scale_delta))
    
    # Apply a reflection strategy by taking the absolute value of the new values
    gamma_new = abs(gamma_new)
    delta_new = abs(delta_new)
    
    return gamma_new, delta_new
end

function metropolis_hastings_accept(lotus_n_papers, x, gamma, delta, gamma_new, delta_new)
    # Calculate the log likelihood and log prior of the current and proposed values of gamma and delta
    curr_log_likelihood = log_likelihood(lotus_n_papers, x, gamma, delta)
    new_log_likelihood = log_likelihood(lotus_n_papers, x, gamma_new, delta_new)
    curr_log_prior = log_prior(gamma, delta)
    new_log_prior = log_prior(gamma_new, delta_new)

    # Calculate the log ratio of the new and current values of gamma and delta
    log_ratio = (new_log_likelihood + new_log_prior) - (curr_log_likelihood + curr_log_prior)

    return rand(Uniform()) < exp(min(0,log_ratio))
end

function run_mcmc_with_gibbs(lotus_n_papers, x_init, n_iter, gamma_init, delta_init,
                        mu_arrays, e, e_blocks, blocks,
                        target_acceptance_rate=(0.25, 0.35), check_interval=500,
                        thinning_factor=5)
    
    gamma, delta, x = gamma_init, delta_init, x_init
    mu = update_mu(mu_arrays)

    total_entries = Int(n_iter // thinning_factor)
    #Initialize the samples array to store gamma and delta values
    samples = zeros((total_entries, 2))
    sample_count = 1

    # Initialize a list to store x values at each iteration
    x_samples = Vector{Any}(undef, total_entries)
    mu_samples = Vector{Any}(undef, total_entries)

    accept_gamma = 0
    accept_delta = 0
    accept_mu = zeros(length(mu_arrays))
    proposal_scale_gamma = 0.1
    proposal_scale_delta = 0.01
    proposal_scale_mu = 0.1
    print_var = n_iter/10
    acceptance_rate_gamma = 0.0
    acceptance_rate_delta = 0.0

    for i in 1:n_iter
        if mod(i,print_var) == 0
            println("Done : ", i/n_iter *100, "% of total iterations")
        end

        #update gamma
        gamma_new, _ = proposal(gamma, delta, proposal_scale_gamma, proposal_scale_delta)
        if metropolis_hastings_accept(lotus_n_papers, x, gamma, delta, gamma_new, delta)
            gamma = gamma_new
            accept_gamma += 1
        end

        # Update delta
        _, delta_new = proposal(gamma, delta, proposal_scale_gamma, proposal_scale_delta)
        if metropolis_hastings_accept(lotus_n_papers, x, gamma, delta, gamma, delta_new)
            delta = delta_new
            accept_delta += 1
        end

        # Update x using Gibbs sampling
        x = gibbs_sample_x(lotus_n_papers, mu, e, e_blocks, gamma, delta, blocks)

        for m in eachindex(mu_arrays)
            for j in eachindex(mu_arrays[m])
                mu_arrays_new = deepcopy(mu_arrays)
                mu_arrays_new[m] = deepcopy(mu_arrays[m])
                mu_arrays_new[m][j] = proposal_mu(mu_arrays[m][j], proposal_scale_mu)
        
                # update mu
                mu_new = update_mu(mu_arrays_new)
                if mh_accept_mu(vec(mu), vec(mu_new), e, e_blocks, blocks, x)
                    mu = mu_new
                    mu_arrays[m][j] = mu_arrays_new[m][j]
                    accept_mu[m] += 1
                end
            end
        end

        # Store gamma and delta values in the samples array if the current iteration is a multiple of the thinning factor
        if mod(i + 1, thinning_factor)  == 0
            samples[sample_count,:] = [gamma, delta]
            # Append the current x value to the x_samples list
            x_samples[sample_count] = x
            mu_samples[sample_count] = mu_arrays
            sample_count += 1
        end

        # Check acceptance rate and adjust proposal_scale if necessary
        if mod(i,check_interval) == 0
            acceptance_rate_gamma = accept_gamma / check_interval
            acceptance_rate_delta = accept_delta / check_interval

            if !(target_acceptance_rate[1] <= acceptance_rate_gamma <= target_acceptance_rate[2] && 
                 target_acceptance_rate[1] <= acceptance_rate_delta <= target_acceptance_rate[2])
                # Adjust proposal_scale_gamma and proposal_scale_delta
                if acceptance_rate_gamma < target_acceptance_rate[1]
                    proposal_scale_gamma *= 0.8
                elseif acceptance_rate_gamma > target_acceptance_rate[2]
                    proposal_scale_gamma *= 1.2
                end

                if acceptance_rate_delta < target_acceptance_rate[1]
                    proposal_scale_delta *= 0.8
                elseif acceptance_rate_delta > target_acceptance_rate[2]
                    proposal_scale_delta *= 1.2
                end
            end

            # Reset acceptance counters
            accept_gamma = 0
            accept_delta = 0

            for m in eachindex(mu_arrays)
                acceptance_rate_mu = accept_mu[m] / (check_interval * length(mu_arrays[m]))
                if acceptance_rate_mu < target_acceptance_rate[1]
                    proposal_scale_mu *= 0.8
                elseif acceptance_rate_mu > target_acceptance_rate[2]
                    proposal_scale_mu *= 1.2
                end
                accept_mu[m] = 0
            end            
        end
    end
    return samples, x_samples, mu_samples, acceptance_rate_gamma, acceptance_rate_delta
end