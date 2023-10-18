using Graphs
using Printf
using CSV
using DataFrames
using LinearAlgebra
using SpecialFunctions
using TikzGraphs   # for TikZ plot output
using TikzPictures # to save TikZ as PDF

struct Variable 
    name::String
    r::Int # number of possible values
end 

struct K2Search
    ordering::Vector{Int}
end

struct LocalDirectedGraphSearch
    G # initial graph
    k_max # number of iterations
end

"""
    write_gph(dag::DiGraph, idx2names, filename)

Takes a DiGraph, a Dict of index to names and a output filename to write the graph in `gph` format.
"""
function write_gph(dag::DiGraph, idx2names, filename)
    open(filename, "w") do io
        for edge in edges(dag)
            @printf(io, "%s,%s\n", idx2names[src(edge)], idx2names[dst(edge)])
        end
    end
end


function compute(infile, outfile)
    # WRITE YOUR CODE HERE
    # FEEL FREE TO CHANGE ANYTHING ANYWHERE IN THE CODE
    # THIS INCLUDES CHANGING THE FUNCTION NAMES, MAKING THE CODE MODULAR, BASICALLY ANYTHING

    # Read the input file
    df = DataFrame(CSV.File(infile))
    vars_names = names(df)
    mat = transpose(Matrix(df))
    var_num = maximum(mat, dims=2)
    println("var_num: ", var_num)
    vars = [Variable(vars_names[i], var_num[i]) for i in 1:length(vars_names)]
    println("vars_names: ", vars_names)
    #g = SimpleDiGraph(length(vars_names)) # create a directed graph
    #add_edge!(g, 1, 2) # add edge from parent1,child1
    #add_edge!(g, 3, 4) # add edge from parent2,child2
    #add_edge!(g, 5, 6) # add edge from parent3,child3
    #add_edge!(g, 1, 4) # add edge from parent1,child2
    #add_edge!(g, 5, 4) # add edge from parent3,child2
    #score = bayesian_score(vars, g, mat)
    @time begin
    ## K2 search
    k2 = K2Search([i for i in 1:length(vars_names)])
    g = fit(k2, vars, mat)

    # Local search
    #g = SimpleDiGraph(length(vars_names)) # create a directed graph
    #k_max = 30
    #local_search = LocalDirectedGraphSearch(g, k_max)
    #g = fit(local_search, vars, mat)
    # Output to graph
    end
    p = plot(g, vars_names) # create TikZ plot with labels
    save(PDF(string(outfile[begin:end-4], ".pdf")), p) # save TikZ as PDF
    write_gph(g, Dict(enumerate(vars_names)), outfile)
end

function sub2ind(siz, x)
    # Identify which parental instantiation is relevant to a particular datapoint and variable
    k = vcat(1, cumprod(siz[1:end-1]))
    return dot(k, x .-1) + 1
    
end

# From Algorithm 4.1
function statistics(var, G, D)
    # statistics extracts counts from a discrete dataset D, assuming a Bayesian network with variables var and structure G.
    n = size(D, 1) # number of variables
    r = [var[i].r for i in 1:n]
    q = [isempty(inneighbors(G, i)) ? 1 : prod(r[j] for j in inneighbors(G, i)) for i in 1:n]
    M = [zeros(q[i], r[i]) for i in 1:n]
    for o in eachcol(D)
        for i in 1:n
            k = o[i]
            parents = inneighbors(G, i)
            j = 1
            if !isempty(parents)
                j = sub2ind(r[parents], o[parents])
            end
            M[i][j, k] += 1.0
        end
    end
    return M
end

function prior(vars, G)
    # Generate a prior a_{ijk} with all entries as 1
    n = length(vars)
    r = [vars[i].r for i in 1:n]
    q = [isempty(inneighbors(G, i)) ? 1 : prod(r[j] for j in inneighbors(G, i)) for i in 1:n]
    return [ones(q[i], r[i]) for i in 1:n]
end

function bayesian_score_component(M, a)
    p = sum(loggamma.(a + M))
    p -= sum(loggamma.(a))
    p += sum(loggamma.(sum(a,dims=2)))
    p -= sum(loggamma.(sum(a,dims=2) + sum(M,dims=2)))
    return p
end

function bayesian_score(vars, G, D) 
    n = length(vars)
    M = statistics(vars, G, D)
    a = prior(vars, G)
    return sum(bayesian_score_component(M[i], a[i]) for i in 1:n)
end 

function fit(method::K2Search, vars, D)
    G = SimpleDiGraph(length(vars))
    for (k, i) in enumerate(method.ordering[2:end])
        y = bayesian_score(vars, G, D)
        while true
            y_best, j_best = -Inf, 0
            for j in method.ordering[1:k]
                if !has_edge(G, j, i)
                    add_edge!(G, j, i)
                    y_p = bayesian_score(vars, G, D)
                    if y_p > y_best
                        y_best, j_best = y_p, j
                    end
                    rem_edge!(G, j, i)
                end
            end
            if y_best > y
                y = y_best
                add_edge!(G, j_best, i)
            else
                break
            end
        end
        println("Max_Score: ", y)
    end
    return G
end

function rand_graph_neighbor(G)
    n = nv(G)
    i = rand(1:n)
    j = mod1(i+rand(2:n)-1, n)
    G_p = copy(G)
    has_edge(G, i, j) ? rem_edge!(G_p, i, j) : add_edge!(G_p, i, j)
    return G_p
end

function fit(method::LocalDirectedGraphSearch, vars, D)
    G = method.G
    y = bayesian_score(vars, G, D)
    for k in 1:method.k_max
        G_p = rand_graph_neighbor(G)
        y_p = is_cyclic(G_p) ? -Inf : bayesian_score(vars, G_p, D)
        if y_p > y
            y, G = y_p, G_p
        end
    end
    println("Max_Score: ", y)
    return G
end

if length(ARGS) != 2
    error("usage: julia project1.jl <infile>.csv <outfile>.gph")
end

inputfilename = ARGS[1]
outputfilename = ARGS[2]

compute(inputfilename, outputfilename)
