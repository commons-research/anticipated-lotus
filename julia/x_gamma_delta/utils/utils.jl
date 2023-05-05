import Random 
using Distributions
using LinearAlgebra
using StatsFuns: logistic
Random.seed!(156)

function simulate_from_prior(T::Vector{String}, n_t::Vector{Int64})
    mu = Dict(T[i] => Random.rand(Distributions.Normal(0, 1), n_t[i]) for i in 1:length(T))
    alpha = Dict(T[i] => Random.rand(Distributions.Exponential(2), 1) for i in 1:length(T))
    beta = Dict(T[i] => Random.rand(Distributions.Exponential(3), n_t[i]) for i in 1:length(T))
    return mu, alpha, beta
end

function create_blocks(T, n::Int, block_count::Int)
    block_size = n ÷ block_count
    blocks = Matrix{Float32}[]

    for i in 0:(block_count - 1)
        start_index = i * block_size
        end_index = i == block_count - 1 ? n : start_index + block_size
        scale = Matrix{Float32}(I, end_index - start_index, end_index - start_index) * (1 / (end_index - start_index))
        block = rand(Wishart(end_index - start_index, scale))
        push!(blocks, block)
    end

    return blocks
end

function simulate_from_prior_sigma(T::Vector{String}, n_t::Vector{Int64}, blocks::Vector{Int64})
    sigma = Dict(T[i] => Dict(string(T[i], "_", j) => create_blocks(T[i], n_t[i], blocks[i])[j + 1] for j in 0:(blocks[i] - 1)) for i in eachindex(T))
    return sigma
end

function simulate_sigma_blocks(blocks::Vector{Int64})
    total_size = prod(blocks)
    scale = Matrix{Float32}(I, total_size, total_size) * (1 / total_size)
    covariance_matrix = rand(Wishart(total_size, scale))
    return covariance_matrix
end


function compute_sum_of_mus(mu::Dict)
    arrays = collect(values(mu))

    # Create an iterator for every combination of values
    combinations = product(arrays...)

    # Sum the values in each combination and store in a list
    sum_combinations = Float32[sum(comb) for comb in combinations]

    return vec(sum_combinations)
end

function find_mu_blocks(mu_flat, blocks, original_shape)
    block_rows, block_cols = blocks
    row_size, col_size = original_shape

    rows_per_block = row_size ÷ block_rows
    cols_per_block = col_size ÷ block_cols

    mu_blocks = Int[]

    for index in eachindex(mu_flat)
        row = (index-1) ÷ col_size
        col = (index-1) % col_size

        block_row = row ÷ rows_per_block
        block_col = col ÷ cols_per_block

        block_number = block_row * block_cols + block_col
        push!(mu_blocks, block_number)
    end

    return mu_blocks .+ 1
end

function find_corresponding_sigmas(mu_flat::Vector{Float32}, blocks::Vector{Int64}, original_shape::Vector{Int64}, sigmas)
    block_rows, block_cols = blocks
    row_size, col_size = original_shape

    rows_per_block = row_size ÷ block_rows
    cols_per_block = col_size ÷ block_cols

    corresponding_sigmas = Float32[]

    for index in eachindex(mu_flat)
        row = (index - 1) ÷ col_size
        col = (index - 1) % col_size

        block_row = row ÷ rows_per_block
        block_col = col ÷ cols_per_block

        block_number = block_row * block_cols + block_col + 1

        within_block_row = row % rows_per_block
        within_block_col = col % cols_per_block
        within_block_index = within_block_row * cols_per_block + within_block_col +1

        corresponding_sigma = sigmas[block_number][within_block_index]
        push!(corresponding_sigmas, corresponding_sigma)
    end

    return corresponding_sigmas
end

function compute_prob_X(sum_mus::Vector{Float32}, sigmas::Vector{Vector{Float64}}, sigma_blocks::Vector{Float64}, blocks::Vector{Int64}, dim_size::Vector{Int64})
    in_what_block_is_mu = find_mu_blocks(sum_mus, blocks, dim_size)
    corresponding_sigmas = find_corresponding_sigmas(sum_mus, blocks, dim_size, sigmas)
    result = Array{Float32}(undef, length(sum_mus))

    for i in eachindex(sum_mus)
        result[i] = sum_mus[i] + sigma_blocks[in_what_block_is_mu[i]] + corresponding_sigmas[i]
    end

    return logistic.(reshape(result, Tuple(dim_size)))
end

function simulate_X(prob_x::Matrix{Float32})
    binomial_rand(x) = rand(Binomial(1, Float64(x)))
    return binomial_rand.(prob_x)
end

function compute_prob_L(x)
    n_papers = 1 .+ rand(Poisson(1), size(x))

    # Set presence or absence in nature
    not_present_in_nature = findall(x .== 0)
    present_in_x = findall(x .== 1)

    # Change number of papers to 0 in the n_papers array if the entry is not present in nature
    n_papers[not_present_in_nature] .= 0

    # Generate gamma and delta, and calculate the 
    gamma = rand(Exponential(0.1), 1)
    delta = rand(Exponential(0.01), 1)
    P_m = sum(n_papers, dims=2)
    Q_s = sum(n_papers, dims=1)

    R_ms = 1 .- exp.(-gamma .* P_m .- delta .* Q_s)
    prob_L = ifelse.(x .== 1, R_ms, 0)
    
    return prob_L, n_papers, gamma, delta
end

function simulate_lotus(prob_lotus, n_papers)
    Lotus_binary = map(p -> rand(Binomial(1, p)), prob_lotus)

    not_present_in_lotus = findall(Lotus_binary .== 0)
    Lotus_N_papers = n_papers
    Lotus_N_papers[not_present_in_lotus] .= 0

    return Lotus_binary, Lotus_N_papers
end
