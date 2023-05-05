include("utils.jl")

function gibbs_sample_x(lotus_n_papers, mu, e, e_blocks, gamma, delta, blocks)
    p_x_1 = compute_prob_X(mu, e, e_blocks, blocks, collect(size(lotus_n_papers)))
    # Calculate prior
    p_x_0 = 1 .- p_x_1
    
    # Calculate likelihood of data given x
    likelihood_x_0 = likelihood(lotus_n_papers, zeros(size(p_x_1)), gamma, delta)
    likelihood_x_1 = likelihood(lotus_n_papers, ones(size(p_x_1)), gamma, delta)
    
    # Compute likelihood times prior
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

function run_mcmc_with_gibbs(lotus_n_papers, x_init, n_iter::Int64, gamma_init::Float64, delta_init::Float64,
                            mu, e, e_blocks, blocks,
                            target_acceptance_rate::Tuple=(0.25, 0.35), check_interval::Int64=500)
    gamma, delta, x = gamma_init, delta_init, x_init
    
    # Initialize the samples array to store gamma and delta values
    samples = Vector{Any}(undef, n_iter)
    
    # Initialize a list to store x values at each iteration
    x_samples = Vector{Any}(undef, n_iter)

    accept_gamma = 0
    accept_delta = 0
    acceptance_rate_gamma = 0.0
    acceptance_rate_delta = 0.0
    proposal_scale_gamma = 0.1
    proposal_scale_delta = 0.01
    print_var = n_iter/10

    for i in 1:n_iter
        if mod(i,print_var) == 0
            println("Done : ", i/n_iter *100, "% of total iterations")
        end
        # Update gamma
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
        
        # Store gamma and delta values in the samples array
        samples[i] = [gamma, delta]
        
        # Append the current x value to the x_samples list
        x_samples[i] = x

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
        end
    end

    return samples, x_samples, acceptance_rate_gamma, acceptance_rate_delta
end
