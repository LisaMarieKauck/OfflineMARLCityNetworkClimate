# Utilities / config

const ACTIONS = [0,1,2,3]  # nothing, mitigation, adaptation, both
const K = length(ACTIONS)
#path = "test.csv"
#df_2 = load_data(path)

# Read dataset (change path as needed)
function load_data(path::Union{AbstractString, DataFrame})
    if path isa DataFrame
        df =  path
    else
        if !isfile(path)
            error("File $path not found. Please provide a valid CSV file path.")
        end
        df = CSV.read(path, DataFrame)
    end

    # Ensure types
    df.traj_id = string.(df.traj_id)
    df.t = Int.(df.t)
    df.action = Int.(df.action)
    df.reward = Float64.(df.reward)
    return df
end

# Extract state column names (assumes columns other than listed ones are state features)
function state_cols(df::DataFrame)
    reserved = Set(["traj_id","t","action","reward", "global_reward"])
    return [c for c in names(df) if !(String(c) in reserved)]
end

# Build per-row feature vector x(s,a) as concat of K blocks (state dims each)
# For action k, block k = s, other blocks = zeros
function build_feature_vector(s::Vector{Float64}, a::Int, K::Int)
    d = length(s)
    x = zeros(d*K)
    # locate action index (assume ACTIONS[i] == a)
    idx = findfirst(==(a), ACTIONS)
    if idx === nothing
        error("action $a not found in ACTIONS")
    end
    start = (idx-1)*d + 1
    x[start:start+d-1] .= s
    return x
end

# Build dataset by time-step organizations
"""
Return:
 - X[t] : matrix (n_t × (d*K)) of features at stage t
 - A[t] : vector actions (n_t)
 - R[t] : vector rewards observed at t
 - Snext[t] : matrix (n_t × d) of next-state features (for computing max_{a'} Q_{t+1}(s',a'))
 - traj_index[t] : vector of trajectory ids (for bootstrapping)
"""
function build_stage_matrices(df::DataFrame)
    scols = state_cols(df)
    d = length(scols)
    # discover T
    T = maximum(df.t)+1 - minimum(df.t)  
    # group by (traj_id, t) to ensure one row per step (if multiple, take first)
    g = groupby(df, [:traj_id, :t])
    df_unique = combine(g, first) 
    if nrow(df_unique) < nrow(df)
        @warn "Some (traj_id, t) pairs had multiple entries; taking first occurrence."
    end # safe simplification
    # index by t
    X = Vector{Matrix{Float64}}(undef, T)
    A = Vector{Vector{Int}}(undef, T)
    R = Vector{Vector{Float64}}(undef, T)
    Snext = Vector{Matrix{Float64}}(undef, T)
    traj_idx = Vector{Vector{String}}(undef, T)

    for t in minimum(df.t):maximum(df.t)
        rows = filter(row -> row.t == t, df_unique)
        n = nrow(rows)
        Xmat = zeros(n, d*K)
        Avec = zeros(Int, n)
        Rvec = zeros(Float64, n)
        Snext_mat = zeros(n, d)
        trajs = Vector{String}(undef, n)
        for (i,row) in enumerate(eachrow(rows))
            s = Float64.(collect(row[scols]))
            x = build_feature_vector(s, row.action, K)
            Xmat[i,:] = x
            Avec[i] = row.action
            Rvec[i] = row.reward
            trajs[i] = row.traj_id
            # next state: attempt to fetch state at t+1 for same traj
            next_row = findfirst(r -> (r.traj_id==row.traj_id && r.t==t+1), eachrow(df_unique))
            if next_row === nothing
                # either terminal next-state or missing — set next-state to zeros
                Snext_mat[i,:] .= 0.0
            else
                # next state fields
                r2 = df_unique[next_row, :]
                Snext_mat[i,:] = Float64.(collect(r2[scols]))
            end
        end
        index_to_write_to = t-minimum(df.t)+1
        X[index_to_write_to] = Xmat
        A[index_to_write_to] = Avec
        R[index_to_write_to] = Rvec
        Snext[index_to_write_to] = Snext_mat
        traj_idx[index_to_write_to] = trajs
        print(t)
    end
    return (X,A,R,Snext,traj_idx,scols)
end


# add mean-field features to dataframe

"""
Compute mean-field (action proportions) per time t and add as columns to df.
- df is expected to have columns :traj_id, :t, :action
- This produces columns mf_a_X where X is action value (0..K-1)
- Optionally exclude self when computing proportions (default: false).
"""
function add_mean_field_features!(df::DataFrame; exclude_self::Bool=false)
    # make sure df has unique rows per (traj_id, t)
    g = groupby(df, [:traj_id, :t])
    df_unique = combine(g, first)

    T = maximum(df_unique.t)
    # prepare storage for mf for each unique row
    mf_cols = ["mf_a_$(a)" for a in ACTIONS]
    for c in mf_cols
        if !(c in names(df))
            df[!, c] = zeros(Float64, nrow(df))
        end
    end

    # precompute action counts per t
    for t in 1:T
        rows_t = filter(r -> r.t == t, eachrow(df_unique))
        n_t = length(rows_t)
        # counts per action
        counts = zeros(Float64, K)
        for r in rows_t
            ai = findfirst(==(r.action), ACTIONS)
            counts[ai] += 1.0
        end
        props = counts ./ max(1.0, n_t)  # avoid div0
        # assign to original df rows with t
        # If exclude_self=true, we will correct per-row by subtracting self later
        for r in rows_t
            # map back to df indices (there might be duplicates but we added columns to df directly)
            # select all original rows with same traj_id and t
            sel = findall(row -> (row.traj_id==r.traj_id && row.t==t), eachrow(df))
            for idx in sel
                # default mean-field
                for (k,a) in enumerate(ACTIONS)
                    df[idx, Symbol(mf_cols[k])] = props[k]
                end
            end
        end
        # if exclude_self, correct each row by removing its own action contribution
        if exclude_self
            for r in rows_t
                sel = findall(row -> (row.traj_id==r.traj_id && row.t==t), eachrow(df))
                for idx in sel
                    ai = findfirst(==(df[idx,:action]), ACTIONS)
                    # subtract 1 from count then renormalize by (n_t-1)
                    if n_t > 1
                        for (k,a) in enumerate(ACTIONS)
                            corrected = (counts[k] - (k==ai ? 1.0 : 0.0)) / (n_t - 1)
                            df[idx, Symbol(mf_cols[k])] = corrected
                        end
                    else
                        # singleton trajectory at this t -> no other agents: set zeros
                        for (k,a) in enumerate(ACTIONS)
                            df[idx, Symbol(mf_cols[k])] = 0.0
                        end
                    end
                end
            end
        end
    end
    return df
end

# exclude_self=true optional

#################Common GLOBAL Goal

# compute global reward per (t): sum of reward across all agents at same t
function add_global_reward!(df::DataFrame; global_col::Symbol=:global_reward)
    if !(global_col in names(df))
        df[!, global_col] = zeros(Float64, nrow(df))
    end
    g = groupby(df, :t)
    for grp in g
        total = sum(grp.reward)
        # assign total to all rows in that group (same t)
        for row in eachrow(grp)
            #  matching indices in original df (traj_id,t)
            sel = findall(r -> (r.traj_id==row.traj_id && r.t==row.t), eachrow(df))
            for idx in sel
                df[idx, global_col] = total
            end
        end
    end
    return df
end

# build trajectories with globall rewward
function build_stage_matrices_global(df::DataFrame)
    scols = state_cols(df)
    d = length(scols)
    T = maximum(df.t)+1 - minimum(df.t)  
    # group by (traj_id, t) to ensure one row per step (if multiple, take first)
    g = groupby(df, [:traj_id, :t])
    df_unique = combine(g, first) 
    if nrow(df_unique) < nrow(df)
        @warn "Some (traj_id, t) pairs had multiple entries; taking first occurrence."
    end # safe simplification
    # index by t
    X = Vector{Matrix{Float64}}(undef, T)
    A = Vector{Vector{Int}}(undef, T)
    R = Vector{Vector{Float64}}(undef, T)
    Snext = Vector{Matrix{Float64}}(undef, T)
    traj_idx = Vector{Vector{String}}(undef, T)

    for t in minimum(df.t):maximum(df.t)
        rows = filter(row -> row.t == t, df_unique)
        n = nrow(rows)
        Xmat = zeros(n, d*K)
        Avec = zeros(Int, n)
        Rvec = zeros(Float64, n)
        Snext_mat = zeros(n, d)
        trajs = Vector{String}(undef, n)
        for (i,row) in enumerate(eachrow(rows))
            s = Float64.(collect(row[scols]))
            x = build_feature_vector(s, row.action, K)
            Xmat[i,:] = x
            Avec[i] = row.action
            Rvec[i] = row.global_reward 
            trajs[i] = row.traj_id
            # next state: attempt to fetch state at t+1 for same traj
            next_row = findfirst(r -> (r.traj_id==row.traj_id && r.t==t+1), eachrow(df_unique))
            if next_row === nothing
                # either terminal next-state or missing — set next-state to zeros
                Snext_mat[i,:] .= 0.0
            else
                # next state fields
                r2 = df_unique[next_row, :]
                Snext_mat[i,:] = Float64.(collect(r2[scols]))
            end
        end
        index_to_write_to = t-minimum(df.t)+1
        X[index_to_write_to] = Xmat
        A[index_to_write_to] = Avec
        R[index_to_write_to] = Rvec
        Snext[index_to_write_to] = Snext_mat
        traj_idx[index_to_write_to] = trajs
        print(t)
    end
    return (X,A,R,Snext,traj_idx,scols)
end


# Fitted Q-iteration (linear least squares)

"""
fitted_q_iteration(X,A,R,Snext; γ=1.0)
Inputs:
 - X[t] matrix n_t × (d*K)
 - A[t] actions vector length n_t
 - R[t] rewards vector length n_t
 - Snext[t] next states matrix n_t × d (raw state vectors; we'll create x(s',a') inside)
Returns:
 - betas: vector of beta_t (each is vector length d*K)
 - predict_Q(t, s, a) -> scalar Q
 - policy(s, t) -> argmax action
"""

function fitted_q_iteration(X,A,R,Snext; γ=1.0)
    T = length(X)
    dK = size(X[1],2)
    betas = Vector{Vector{Float64}}(undef, T)
    # last stage T
    X_T = X[T]
    y_T = R[T]
    # Solve least squares: beta = (X'X)^(-1) X'y, but use \ for numerical stability
    βT = X_T \ y_T
    betas[T] = βT
    # move backward
    for t in (T-1):-1:1
        # compute Q̃_t = R_t + γ * max_{a'} Q_{t+1}(s_{t+1}, a')
        n = size(X[t],1)
        d = div(dK, K)
        Qtilde = zeros(n)
        for i in 1:n
            s_next = Snext[t][i,:]
            # build x(s_next, a') for each a'
            maxq = -Inf
            for aprime in ACTIONS
                xnext = build_feature_vector(s_next, aprime, K)
                qval = dot(betas[t+1], xnext)
                maxq = max(maxq, qval)
            end
            Qtilde[i] = R[t][i] + γ * maxq
        end
        # regress Qtilde on X[t]
        βt = X[t] \ Qtilde
        betas[t] = βt
    end

    predict_Q = (t::Int, s::Vector{Float64}, a::Int) -> begin
        x = build_feature_vector(s,a,K)
        return dot(betas[t], x)
    end

    policy = (t::Int, s::Vector{Float64}) -> begin
        besta = ACTIONS[1]
        bestq = -Inf
        for a in ACTIONS
            q = predict_Q(t, s, a)
            if q > bestq
                bestq = q; besta = a
            end
        end
        return besta
    end

    return (betas, predict_Q, policy)
end

#fitted_q_iteration(X,A,R,Snext; γ=0.9)


# Bootstrap voting + CIs (approximate)

"""
bootstrap_fqi(df, B=200; rng=GLOBAL_RNG)
 - Resamples trajectories with replacement
 - Re-runs fitted_q_iteration on each bootstrap sample
Returns:
 - betas_boot[b][t] for each b and t
 - For a set of query states S_query (vector of state-vectors) and a chosen stage t0,
   compute:
    * p_win[action, s_idx] = proportion of bootstraps where action was best
    * Qvals_boot[b, a, s_idx] = distribution of Q(s,a) across bootstraps (for CIs)
"""
function bootstrap_fqi(df::DataFrame; B::Int=200, rng=Random.GLOBAL_RNG, γ=1.0)
    trajs = unique(df.traj_id)
    ntraj = length(trajs)
    betas_boot = Vector{Any}(undef, B)
    for b in 1:B
        println(b)
        sample_trajs = sample(trajs, Weights(fill(1.0,ntraj)), ntraj; replace=true)
        df_b = DataFrame()
        for (i,tj) in enumerate(sample_trajs)
            df_b = vcat(df_b, filter(row -> row.traj_id == tj, df))
        end
        X,A,R,Snext,_, = build_stage_matrices(df_b)[1:5]  # Actually build_stage_matrices returns 6 items, but no need for scols
        betas, predict_Q, policy = fitted_q_iteration(X,A,R,Snext; γ = γ)
        betas_boot[b] = betas
    end
    return betas_boot
end

# Compute p_win and bootstrap CIs for a chosen stage t and set of query states
function analyze_bootstrap_Q(betas_boot, S_query::Vector{Vector{Float64}}, t::Int; alpha=0.05)
    B = length(betas_boot)
    m = length(S_query)
    p_win = zeros(K, m)
    Qdist = Dict()  # map (a, s_idx) -> Vector of Q's across b
    for (si,s) in enumerate(S_query)
        for (ai,a) in enumerate(ACTIONS)
            arr = Float64[]
            for b in 1:B
                βt = betas_boot[b][t]  # vector
                x = build_feature_vector(s, a, K)
                push!(arr, dot(βt, x))
            end
            Qdist[(a, si)] = arr
            # p_win: proportion of bootstraps where action a is argmax across actions
        end
        for b in 1:B
            # evaluate all actions under bootstrap b
            qs = [dot(betas_boot[b][t], build_feature_vector(s,a,K)) for a in ACTIONS]
            arg = argmax(qs)
            p_win[arg, si] += 1
        end
    end
    p_win ./= B
    # compute percentile CIs
    CIs = Dict()
    for ((a,si), arr) in Qdist
        lo = quantile(arr, alpha/2)
        hi = quantile(arr, 1 - alpha/2)
        CIs[(a,si)] = (lo, hi)
    end
    return (p_win, CIs, Qdist)
end

if !("population" in scols)
    error("State variable 'population' not found in dataset!")
end

function population_grid(df)
# nun diskret!
    return sort(unique(df.population))
end


function make_population_s_sample(df, pop_grid, scols)
    med_state = [median(df[!,c]) for c in scols]
    pop_col = findfirst(==("population"), scols)
    s_sample = [begin
        s = copy(med_state)
        s[pop_col] = p
        s
    end for p in pop_grid]
    return s_sample
end
function population_grid(df; ngrid_small=40, ngrid_large=10)
    pop = sort(Float64.(df.population))

    small = filter(x -> x <= 50_000, pop)
    grid_small = quantile(small, range(0,1; length=ngrid_small))

    large = filter(x -> x > 50_000, pop)
    if !isempty(large)
        grid_large = range(minimum(large), maximum(large); length=ngrid_large)
    else
        grid_large = Float64[]
    end

    return vcat(grid_small, grid_large)
end

function make_population_s_sample(df, pop_grid, scols)
    med_state = [median(df[!,c]) for c in scols]
    pop_col = findfirst(==("population"), scols)
    s_sample = [begin
        s = copy(med_state)
        s[pop_col] = p
        s
    end for p in pop_grid]
    return s_sample
end



"""
Summarize bootstrap analysis results as a table.
Inputs:
  - p_win: K × m matrix (actions × states) of bootstrap win probabilities
  - CIs: Dict{(Int,Int), (Float64,Float64)} with confidence intervals
  - state_labels: optional names for query states
"""
function summarize_results(p_win, CIs; state_labels=nothing, backend=:text)
    rows = NamedTuple[]
    m = size(p_win, 2)
    for si in 1:m
        sname = isnothing(state_labels) ? "state $si" : state_labels[si]
        for (ai,a) in enumerate(ACTIONS)
            prob = round(p_win[ai, si], digits=2)
            ci = CIs[(a, si)]
            push!(rows, (State=sname, Action=a, p_win=prob,
                         CI_low=round(ci[1], digits=2), CI_high=round(ci[2], digits=2)))
        end
    end
    if backend == :latex
        pretty_table(rows, backend=Val(:latex))
    else
        pretty_table(rows)  # default text table
    end
end


"""
Bar plot of bootstrap win probabilities with CI whiskers.
Inputs:
  - p_win: K × m matrix (actions × states)
  - CIs: Dict of (a,si) => (lo,hi)
"""
function plot_action_preferences(p_win, CIs; state_labels=nothing)
    m = size(p_win, 2)
    for si in 1:m
        sname = isnothing(state_labels) ? "state $si" : state_labels[si]
        probs = [p_win[ai, si] for ai in 1:K]
        ci_lo = [CIs[(a,si)][1] for a in ACTIONS]
        ci_hi = [CIs[(a,si)][2] for a in ACTIONS]

        Plots.bar(
            string.(ACTIONS),
            probs,
            legend=false,
            ylabel="p_win",
            xlabel="Action",
            title="Bootstrap Action Preferences – $sname",
            ylim=(0,1)
        )
        # add CI whiskers as error bars (scaled to reward space)
        for (i,a) in enumerate(ACTIONS)
            plot!([i,i], [ci_lo[i], ci_hi[i]], color=:black, lw=2)
        end
        display(current())
    end
end

"""
Plot the L2 norm of β_t for each stage t, averaged over bootstraps.
"""
function plot_beta_stability(betas_boot)
    T = length(betas_boot[1])
    norms = zeros(length(betas_boot), T)
    for (b, betas) in enumerate(betas_boot)
        for t in 1:T
            norms[b,t] = norm(betas[t])
        end
    end
    mean_norms = mean(norms, dims=1)
    std_norms = std(norms, dims=1)

    stages = 1:T
    plot(
        stages,
        vec(mean_norms),
        ribbon=vec(std_norms),
        xlabel="Stage t",
        ylabel="‖β_t‖₂",
        title="Stability of β across bootstraps",
        lw=2,
        legend=false
    )
end

"""
Plot bootstrap Q-value distributions for a given state and stage.
"""
function plot_Q_distributions(Qdist, state_index::Int, stage::Int)
    plt = plot()  # empty figure
    for a in ACTIONS
        vals = Qdist[(a,state_index)]
        histogram!(
            plt,
            vals,
            alpha=0.5,
            bins=20,
            normalize=:pdf,
            label="Action $a"
        )
    end
    xlabel!(plt, "Q(s,a)")
    ylabel!(plt, "Density")
    title!(plt, "Bootstrap Q-value distribution – state $state_index, stage $stage")
    display(plt)
    return plt
end

#plot_Q_distributions(Qdist, 1, 2020)  # example state index and stage for now

"""
Compute Bellman errors for each stage and bootstrap sample.
Inputs:
  - betas_boot: vector of bootstrap β-sets
  - df: full dataset (DataFrame)
  - γ: discount factor
Returns:
  - errors[b,t] = mean absolute Bellman error in bootstrap b at stage t
"""
function compute_bellman_errors(betas_boot, df::DataFrame; γ=1.0)
    X,A,R,Snext,_, = build_stage_matrices(df)[1:5]
    T = length(X)
    B = length(betas_boot)
    errors = zeros(B,T)
    d = size(Snext[1],2)

    for b in 1:B
        betas = betas_boot[b]
        for t in 1:T
            n = size(X[t],1)
            errs = zeros(n)
            for i in 1:n
                s = Snext[t][i,:]
                r = R[t][i]
                a = A[t][i]
                # predicted Q_t(s,a)
                q_sa = dot(betas[t], X[t][i,:])
                # Bellman target
                if t < T
                    maxq = -Inf
                    for aprime in ACTIONS
                        xnext = build_feature_vector(s, aprime, K)
                        maxq = max(maxq, dot(betas[t+1], xnext))
                    end
                    target = r + γ*maxq
                else
                    target = r
                end
                errs[i] = target - q_sa
            end
            errors[b,t] = mean(abs.(errs))
        end
    end
    return errors
end

"""
Plot mean absolute Bellman error per stage with bootstrap variability.
"""
function plot_bellman_errors(errors)
    T = size(errors,2)
    mean_err = mean(errors, dims=1)
    std_err = std(errors, dims=1)
    stages = 1:T
    plot(
        stages,
        vec(mean_err),
        ribbon=vec(std_err),
        xlabel="Stage t",
        ylabel="Mean |Bellman error|",
        title="Bellman error across bootstraps",
        lw=2,
        legend=false
    )
end

errors = compute_bellman_errors(betas_boot, df; γ=0.6)
plot_bellman_errors(errors)



"""
Plot bootstrap voting probabilities over population with 95% CI ribbons.
Generate bootstrap voting probabilities as a function of population.
Inputs:
  - betas_boot: bootstrap β estimates
  - df: full dataset
  - t: stage index
  - ngrid: number of grid points across the population range
Returns:
  - pop_grid: vector of population values
  - p_win_grid: Dict(action => vector of p_win across grid)
"""
function population_policy_curve(betas_boot, df::DataFrame; t::Int=1, ngrid::Int=50)
    scols = state_cols(df)
    # Identify which column is population
    pop_col = findfirst(==("population"), scols)
    if pop_col === nothing
        error("No column named 'population' found in state variables.")
    end

    # Get ranges
    pop_vals = df[!,:population]
    pop_min, pop_max = minimum(pop_vals), maximum(pop_vals)
    pop_grid = range(pop_min, pop_max; length=ngrid)

    # Fix other state variables at their median
    med_state = [median(df[!,c]) for c in scols]

    # Storage: action × grid
    p_win_grid = Dict(a => zeros(ngrid) for a in ACTIONS)

    for (gi, p) in enumerate(pop_grid)
        # Construct query state vector
        s = copy(med_state)
        s[pop_col] = p
        # Evaluate bootstrap voting
        B = length(betas_boot)
        counts = zeros(Int, K)
        for b in 1:B
            qs = [dot(betas_boot[b][t], build_feature_vector(s,a,K)) for a in ACTIONS]
            best_a = ACTIONS[argmax(qs)]
            counts[findfirst(==(best_a), ACTIONS)] += 1
        end
        for (ai,a) in enumerate(ACTIONS)
            p_win_grid[a][gi] = counts[ai] / B
        end
    end
    return pop_grid, p_win_grid
end

function plot_population_policy_curve(pop_grid, p_win_grid; B::Int)
    plt = plot()
    for a in ACTIONS
        probs = p_win_grid[a]
        se = sqrt.(probs .* (1 .- probs) ./ B)
        lo = clamp.(probs .- 1.96 .* se, 0, 1)
        hi = clamp.(probs .+ 1.96 .* se, 0, 1)

        plot!(
            plt,
            pop_grid,
            probs,
            lw=2,
            label="Action $a",
        )
        plot!(
            plt,
            pop_grid,
            lo,
            ribbon=(hi .- lo),
            fillalpha=0.2,
            label="",
        )
    end
    xlabel!(plt, "Population")
    ylabel!(plt, "Bootstrap win probability")
    title!(plt, "Estimated Policy Stability across Population")
    display(plt)
    return plt
end

B_b = len
