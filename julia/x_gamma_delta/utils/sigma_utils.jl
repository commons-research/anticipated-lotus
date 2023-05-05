using Base.Threads
using .Iterators
using LinearAlgebra
using Combinatorics
using ThreadPools
using StaticArrays

function extract_nested_dict_values(alpha::Dict, sigma::Dict)
    key_list = collect(keys(sigma))
    values_combinations = Array[]

    n = length(key_list)
    for i in eachindex(key_list)
        key1 = key_list[i]
        for j in (i+1):n
            key2 = key_list[j]
            for subkey1 in keys(sigma[key1])
                for subkey2 in keys(sigma[key2])
                    push!(values_combinations, [sigma[key1][subkey1] .* alpha[key1], sigma[key2][subkey2] .* alpha[key2]])
                end
            end
        end
    end

    return values_combinations
end

function prepare_sigma(array::Vector{Matrix{Float64}})
    matrix_dims = [size(matrix, 1) for matrix in array]

    index_list = collect(product(repeat([1:dim for dim in matrix_dims], 2)...))

    cov_matrices = [Matrix{Float32}(matrix) for matrix in array]

    return cov_matrices, matrix_dims, index_list
end

function ravel_multi_index(coords::Tuple{Int64, Int64}, dims::Vector{Int64})
    idx = 0
    for i in eachindex(coords)
        idx *= dims[i]
        idx += coords[i]
    end
    return idx
end


function sigma_numba(cov_matrices::Vector{Matrix{Float32}}, matrix_dims::Vector{Int64}, index_list::Array{NTuple{4, Int64}, 4})
    summed_matrix_dim = prod(matrix_dims)

    # Initialize the summed matrix
    summed_matrix = zeros(summed_matrix_dim, summed_matrix_dim)

    @threads for i in eachindex(index_list)
        input_idx = index_list[i][1:length(matrix_dims)]
        output_idx = index_list[i][(length(matrix_dims) + 1):end]

        # Compute the product using a loop instead of a generator expression
        element_prod = 1
        for j in eachindex(cov_matrices)
            matrix = cov_matrices[j]
            element_prod *= matrix[input_idx[j], output_idx[j]]
        end
        # Compute the linear index using the custom function
        input_linear_idx = ravel_multi_index(input_idx .- 1, matrix_dims)
        output_linear_idx = ravel_multi_index(output_idx .- 1, matrix_dims)

        summed_matrix[input_linear_idx + 1, output_linear_idx + 1] = element_prod
    end
    return summed_matrix
end


function compute_epsilon(alpha::Dict, sigma::Dict)
    values_combinations = extract_nested_dict_values(alpha, sigma)

    output = Array{Any}(undef, length(values_combinations))
    for (i, array) in enumerate(values_combinations)
        cov_matrices, matrix_dims, index_list = prepare_sigma(array)
        output[i] = sigma_numba(cov_matrices, matrix_dims, index_list)
    end

    return output
end

