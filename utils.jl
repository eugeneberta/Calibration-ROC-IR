using Plots, Printf

function crossentropy(ypred, ytrue)
    ϵ = eps(eltype(ypred))
    ce = -sum(ytrue .* log.(ypred .+ ϵ), dims=2)
    return sum(ce)/length(ce)
end

function regcrossentropy(ypred, ytrue, λ=0.05)
    ϵ = eps(eltype(ypred))
    ce = -sum(ytrue .* log.(ypred .+ ϵ) .+ λ*log.(ypred .+ ϵ), dims=2)
    return sum(ce)/length(ce)
end

function brierscore(ypred, ytrue)
    return sum((ypred - ytrue).^2)/size(ypred)[1]
end

function rocsurface(x, y, g)
    n, K = size(g)
    graph = zeros(n,K)
    for i in 1:n
        δ = x .- g[i,:]'
        ridxs = mapslices(argmax, δ, dims=2)[:]
        for k in 1:K
            rk = findall(ridxs .== k)
            graph[i,k] = sum(y[rk,k]) / sum(y[:,k])
        end
    end
    return unique(graph, dims=1)
end

function rocauc(ypred, ytrue)
    d = size(ypred)[2]
    if d==2
        g = vcat([[i 1000-i]; for i in -1:1001]...) ./ 1000
    elseif d==3
        g = vcat([[i j k]; for i in -51:151 for j in -51:151 for k in -51:151 if i+j+k==100]...) ./ 100
    elseif d==4
        g = vcat([[i j k l]; for i in -15:45 for j in -15:45 for k in -15:45 for l in -15:45 if i+j+k+l==30]...) ./ 30
    else
        println("ROC AUC NOT IMPLEMENTED FOR THIS DIMENSION")
    end
    r = rocsurface(ypred, ytrue, g)
    r = vcat(r, zeros(size(ypred)[2])')
    r = polyhedron(vrep(r), lib)
    removevredundancy!(r)
    return MCvolume(r)
end

function MCvolume(poly)
    dim = Polyhedra.fulldim(poly)
    if dim == 2
        nsamples=10000
    elseif dim == 3
        nsamples=20000
    else
        nsamples=30000
    end
    Random.seed!(42)
    grid = rand(Float64, (nsamples, dim))
    s = 0
    for i in 1:nsamples
        if in(grid[i,:], poly)
            s += 1
        end
    end
    return s/nsamples
end

# Functions for recursive tree splitting:
function monotone(x, y, g)
    N, K = size(g)
    for n in 1:N
        δ = x .- g[n,:]'
        rmax = maximum(δ, dims=2)[:]
        for i in 1:K
            ri = findall(δ[:,i] .== rmax)
            for j in 1:K
                rj = findall(δ[:,j] .== rmax)
                if i != j && length(ri) > 0 && length(rj) > 0
                    if minimum(y[ri,i]-y[ri,j]) < maximum(y[rj,i]-y[rj,j])
                        return false
                    end
                end
            end
        end
    end
    return true
end

function split(x, y, g, p)
    N, K = size(y)
    ysplit = zeros(N, K)
    ygsplit = zeros(size(g))

    δx = x .- p'
    rmax = maximum(δx, dims=2)[:]

    δg = g .- p'
    gmax = maximum(δg, dims=2)[:]

    xregions = []
    gregions = []
    for k in 1:K
        Rk = findall(δx[:,k] .== rmax)
        Gk = findall(δg[:,k] .== gmax)
        if length(Rk) > 0
            λ = 0.05
            mean = (sum(y[Rk,:], dims=1)/length(Rk) .+ λ) ./ (1+λ) # Laplace smoothing
            ysplit[Rk,:] .= mean
            ygsplit[Gk,:] .= mean
            push!(xregions, Rk)
            push!(gregions, Gk)
        end
    end
    return ysplit, ygsplit, xregions, gregions
end

function optimalsplit(x, y, g, yirp, ygirp, R, G, splitidxs)
    yirp_ = copy(yirp)
    ygirp_ = copy(ygirp)

    xregions, gregions = [], []
    splitidx = 0
    Msplit = 0
    # Find best split:
    for i in eachindex(G)
        # Test split:
        p = g[G[i],:]
        ysplit, _, splitregions, _ = split(x[R,:], y[R,:], g[G,:], p)

        if length(splitregions) >= 2
            # Evaluate split quality:
            yirp_[R,:] = ysplit
            M = sum(abs.(yirp_-yirp))

            # Evaluate monotony:
            push!(splitidxs, G[i])
            isotonic = monotone(x, yirp_, g[splitidxs,:])
            pop!(splitidxs)

            if M > Msplit && isotonic
                splitidx = G[i]
                Msplit = M
            end
        end
    end
    # Re-compute best split:
    if splitidx != 0
        p = g[splitidx,:]
        ysplit, ygsplit, xregions, gregions = split(x[R,:], y[R,:], g[G,:], p)
        yirp_[R,:] = ysplit
        ygirp_[G,:] = ygsplit
        xregions = [R[Rk] for Rk in xregions]
        gregions = [G[Gk] for Gk in gregions]
    end
    return yirp_, ygirp_, xregions, gregions, splitidx, Msplit
end

function MulticlassIRP(xcal, ycal, g, experiment=false, xtest=-1, ytest=-1)
    n, _ = size(ycal)
    m, _ = size(g)

    mean = sum(ycal, dims=1)/n
    ycalirp = repeat(mean, n)
    ygirp = repeat(mean, m)
    regions = [findall(ones(Bool, n))]
    grids = [findall(ones(Bool, m))]
    splitidxs = []

    if experiment
        ytestirp = repeat(mean, size(ytest)[1])

        listnregions = []
        listcalce = []
        listtestce = []
        nregions = length(regions)
        calce = regcrossentropy(ycalirp, ycal)
        testce = crossentropy(ytestirp, ytest)
        push!(listnregions, nregions)
        push!(listcalce, calce)
        push!(listtestce, testce)

        listcalAUC = []
        listtestAUC = []
        calAUC = rocauc(ycalirp, ycal)
        testAUC = rocauc(ytestirp, ytest)
        push!(listcalAUC, calAUC)
        push!(listtestAUC, testAUC)

        println("Num regions: $nregions, Cal CE: $(@sprintf("%.3f", calce)), Test CE: $(@sprintf("%.3f", testce))")
    end

    # Recursive splitting:
    while true
        bestidx = 0
        Msplit = 0
        xregions, gregions = [], []
        for i in eachindex(regions)
            _, _, _, _, splitidx, M = optimalsplit(
                xcal, ycal, g,
                ycalirp,
                ygirp,
                regions[i],
                grids[i],
                splitidxs
            )
            if splitidx != 0 && M > Msplit
                bestidx = i
                Msplit = M
            end
        end
        if bestidx != 0
            ycalirp, ygirp, xregions, gregions, splitidx, M = optimalsplit(
                xcal, ycal, g,
                ycalirp,
                ygirp,
                regions[bestidx],
                grids[bestidx],
                splitidxs
            )
            push!(splitidxs, splitidx)
            popat!(regions, bestidx)
            popat!(grids, bestidx)
            regions = vcat(regions, xregions)
            grids = vcat(grids, gregions)

            if experiment
                # Compute ytestirp:
                mask = vec(sum(ygirp, dims=2) .> 0.5)
                NNgrid = g[mask,:]
                ygirp_ = ygirp[mask,:]
                ytestirp = zeros(size(ytest))
                for i in 1:size(ytest)[1]
                    p = xtest[i,:]'
                    idx = argmin(vec(sum((NNgrid .- p).^2, dims=2)))
                    ytestirp[i,:] = ygirp_[idx,:]
                end

                nregions = length(regions)
                calce = regcrossentropy(ycalirp, ycal)
                testce = crossentropy(ytestirp, ytest)
                push!(listnregions, nregions)
                push!(listcalce, calce)
                push!(listtestce, testce)

                calAUC = rocauc(ycalirp, ycal)
                testAUC = rocauc(ytestirp, ytest)
                push!(listcalAUC, calAUC)
                push!(listtestAUC, testAUC)

                println("Num regions: $nregions, Cal CE: $(@sprintf("%.3f", calce)), Test CE: $(@sprintf("%.3f", testce))")
            end
        else
            break
        end
    end
    if experiment
        return ycalirp, ygirp, listnregions, listcalce, listtestce, listcalAUC, listtestAUC
    end
    return ycalirp, ygirp
end

function MulticlassBinning(xcal, ycal, g, depth=4, experiment=false, xtest=-1, ytest=-1)
    n, _ = size(ycal)
    m, _ = size(g)

    mean = sum(ycal, dims=1)/n
    ycalbin = repeat(mean, n)
    ygbin = repeat(mean, m)
    regions = [findall(ones(Bool, n))]
    grids = [findall(ones(Bool, m))]

    if experiment
        ytestbin = repeat(mean, size(ytest)[1])

        listnregions = []
        listcalce = []
        listtestce = []
        nregions = length(regions)
        calce = regcrossentropy(ycalbin, ycal)
        testce = crossentropy(ytestbin, ytest)
        push!(listnregions, nregions)
        push!(listcalce, calce)
        push!(listtestce, testce)

        listcalAUC = []
        listtestAUC = []
        calAUC = rocauc(ycalbin, ycal)
        testAUC = rocauc(ytestbin, ytest)
        push!(listcalAUC, calAUC)
        push!(listtestAUC, testAUC)

        println("Num regions: $nregions, Cal CE: $(@sprintf("%.3f", calce)), Test CE: $(@sprintf("%.3f", testce))")
    end

    nsplits = 1+sum([3^d for d in 1:(depth-1)])
    for _ in 1:nsplits
        r_ = popfirst!(regions)
        g_ = popfirst!(grids)

        p = vec(sum(g[g_,:], dims=1) ./ length(g_))
        y_, yg_, xregions_, gregions_ = split(xcal[r_,:], ycal[r_,:], g[g_,:], p)

        ycalbin[r_,:] = y_
        ygbin[g_,:] = yg_
        xregions = [r_[rk_] for rk_ in xregions_]
        gregions = [g_[gk_] for gk_ in gregions_]

        regions = vcat(regions, xregions)
        grids = vcat(grids, gregions)

        if experiment
            mask = vec(sum(ygbin, dims=2) .> 0.5)
            NNgrid = g[mask,:]
            ygbin_ = ygbin[mask,:]

            ytestbin = zeros(size(ytest))
            for i in 1:size(ytest)[1]
                p = xtest[i,:]'
                idx = argmin(vec(sum((NNgrid .- p).^2, dims=2)))
                ytestbin[i,:] = ygbin_[idx,:]
            end

            nregions = length(regions)
            calce = regcrossentropy(ycalbin, ycal)
            testce = crossentropy(ytestbin, ytest)
            push!(listnregions, nregions)
            push!(listcalce, calce)
            push!(listtestce, testce)

            calAUC = rocauc(ycalbin, ycal)
            testAUC = rocauc(ytestbin, ytest)
            push!(listcalAUC, calAUC)
            push!(listtestAUC, testAUC)

            println("Num regions: $nregions, Cal CE: $(@sprintf("%.3f", calce)), Test CE: $(@sprintf("%.3f", testce))")
        end
    end
    if experiment
        return ycalbin, ygbin, listnregions, listcalce, listtestce, listcalAUC, listtestAUC
    end
    return ycalbin, ygbin
end

# Functions to plot calibration maps:
function plotcal3D(x, y, title, markershape, markersize, markerstrokewidth)
    hover = string.(
        round.(x[:,1]*100)/100, ", ",
        round.(x[:,2]*100)/100, ", ",
        round.(x[:,3]*100)/100, " | ",
        round.(y[:,1]*100)/100, ", ",
        round.(y[:,2]*100)/100, ", ",
        round.(y[:,3]*100)/100
    )
    plot = Plots.plot(showaxis=false, grid=false, ticks=false, xlims=(-.1,1.1), ylims=(-.1,.√.75+.1), aspect_ratio=1)
    Plots.scatter!(
        x[:,1] + 0.5*x[:,2],
        √3*x[:,2]/2,
        mc=RGB.(y[:,1], y[:,2], y[:,3]),
        hover=hover,
        markershape=markershape,
        markersize=markersize,
        markerstrokewidth=markerstrokewidth,
        aspect_ratio=1
    )
    Plots.plot!([0,.5,1,0],[0,√.75,0,0], color=:black, markeralpha=0, label="", aspect_ratio=1)
    Plots.title!(title)
    return plot
end

function plotcal4D(x, y, title, markershape, markersize)
    plot = Plots.plot(showaxis=false, grid=false, ticks=false, xlims=(-.1,1.1), ylims=(-.1,.√.75+.1), aspect_ratio=1)
    Plots.scatter!(
        x[:,1] + 0.5*x[:,2] + 0.5*x[:,3],
        √3*x[:,2]/2 + 3*x[:,3]/8,
        √(5/8)*x[:,3],
        mc=RGB.(y[:,1], y[:,2], y[:,3]),
        markershape=markershape,
        markersize=markersize,
        markerstrokewidth=0.2,
        aspect_ratio=1
    )
    Plots.plot!([0, .5, 1, 0, .5, 1, .5, .5],[0, √3/2, 0, 0, 3/8, 0, √3/2, 3/8], [0, 0, 0, 0, √(5/8), 0, 0, √(5/8)], color=:black, markeralpha=0, label="", aspect_ratio=1)
    Plots.title!(title)
    return plot
end

function plotcalibrationmapchannels(x, y, markersize)
    hover = string.(round.(y[:,1]*100)/100)
    p1 = Plots.plot(showaxis=false, grid=false, ticks=false, xlims=(-.1,1.1), ylims=(-.1,.√.75+.1), aspect_ratio=1)
    Plots.scatter!(
        x[:,1] + 0.5*x[:,2],
        √.75*x[:,2],
        mc=RGB.(y[:,1]),
        markerstrokewidth=0.2,
        markershape = :hexagon,
        hover=hover,
        markersize=markersize,
        aspect_ratio=1
    )
    Plots.plot!([0,.5,1,0],[0,√.75,0,0], color=:black, markeralpha=0, label="", title="Channel: Red", aspect_ratio=1)

    hover = string.(round.(y[:,2]*100)/100)
    p2 = Plots.plot(showaxis=false, grid=false, ticks=false, xlims=(-.1,1.1), ylims=(-.1,.√.75+.1), aspect_ratio=1)
    Plots.scatter!(
        x[:,1] + 0.5*x[:,2],
        √.75*x[:,2],
        mc=RGB.(y[:,2]),
        markerstrokewidth=0.2,
        markershape = :hexagon,
        hover=hover,
        markersize=markersize,
        aspect_ratio=1
    )
    Plots.plot!([0,.5,1,0],[0,√.75,0,0], color=:black, markeralpha=0, label="", title="Channel: Green", aspect_ratio=1)

    hover = string.(round.(y[:,3]*100)/100)
    p3 = Plots.plot(showaxis=false, grid=false, ticks=false, xlims=(-.1,1.1), ylims=(-.1,.√.75+.1), aspect_ratio=1)
    Plots.scatter!(
        x[:,1] + 0.5*x[:,2],
        √.75*x[:,2],
        mc=RGB.(y[:,3]),
        markerstrokewidth=0.2,
        markershape = :hexagon,
        hover=hover,
        markersize=markersize,
        aspect_ratio=1
    )
    Plots.plot!([0,.5,1,0],[0,√.75,0,0], color=:black, markeralpha=0, label="", title="Channel: Blue", aspect_ratio=1)

    return p1, p2, p3
end
