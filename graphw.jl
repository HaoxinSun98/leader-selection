struct Graph_w
    n :: Int # |V|
    m :: Int # |E|
    u :: Array{Int, 1}
    v :: Array{Int, 1} # uv is an edge
    w :: Array{Array{Float64, 1}, 1} # weight of each edge
    nbr :: Array{Array{Int, 1}, 1}
end


function get_graph_w(ffname)
    n = 0
    Label = Dict{Int32, Int32}()
    Origin = Dict{Int32, Int32}()
    #E = Set{Tuple{Int32, Int32, Float32}}()

    getID(x :: Int) = haskey(Label, x) ? Label[x] : Label[x] = n += 1

    fname = string("data/",ffname)
    fin = open(fname, "r")


    str = readline(fin)
    str = split(str)
    #n   = parse(Int, str[1])
    m   = parse(Int, str[3])
    u = Int[]
    v = Int[]
    w = Float64[]

    tot = 0
    for i = 1 : m
        str = readline(fin)
        str = split(str)
        x   = parse(Int, str[1])
        y   = parse(Int, str[2])
        z   = parse(Float64, str[3])
        if x!=y
            u1 = getID(x)
            v1 = getID(y)
            Origin[u1] = x
            Origin[v1] = y
            push!(u, u1)
            push!(v, v1)
            push!(w,z)
            tot += 1
        end
    end
    nbr=[ [ ] for i in 1:n ]
    ww = [ [] for i in 1:n]
    for i=1:tot
        u1=u[i];
        v1=v[i];
        w1 = w[i];
        push!(nbr[u1],v1);
        push!(nbr[v1],u1);
        push!(ww[u1],w1);
        push!(ww[v1],w1)
    end

    close(fin)
    return Graph_w(n, tot, u, v,ww,nbr)
end


function lap_w(G :: Graph_w)
    F = zeros(G.n, G.n);
    for i = 1 : G.n
        for j = 1:length(G.nbr[i])
            F[i,G.nbr[i][j]] -= G.w[i][j];
            F[i,i] += G.w[i][j];
        end
    end
    return F
end
