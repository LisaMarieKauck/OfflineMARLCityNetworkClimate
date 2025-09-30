#Evaluiert Policy Performance über Trajektorien im Datensatz

function evaluate_policy_performance(df::DataFrame, predict_Q_func, policy_func, γ::Float64)
    trajs = unique(df.traj_id)
    scols = state_cols(df)
    returns = Float64[]
    
    for traj in trajs
        traj_data = filter(row -> row.traj_id == traj, df)
        sort!(traj_data, :t)
        
        total_return = 0.0
        discount = 1.0
        
        for row in eachrow(traj_data)
            s = Float64.(collect(row[scols]))
            t_idx = row.t - minimum(df.t) + 1  # Convert to 1-based index
            
            # Verwende gelernte Policy
            recommended_action = policy_func(t_idx, s)
            actual_reward = row.reward
            
            total_return += discount * actual_reward
            discount *= γ
        end
        push!(returns, total_return)
    end
    
    return returns
end


#Behavior Policy Performance (Baseline)

function evaluate_behavior_policy(df::DataFrame, γ::Float64)
    trajs = unique(df.traj_id)
    returns = Float64[]
    
    for traj in trajs
        traj_data = filter(row -> row.traj_id == traj, df)
        sort!(traj_data, :t)
        
        total_return = 0.0
        discount = 1.0
        
        for row in eachrow(traj_data)
            total_return += discount * row.reward
            discount *= γ
        end
        push!(returns, total_return)
    end
    
    return returns
end

# VALUE FUNCTION QUALITÄT



# Bellman Error für gelernte Q-Function

function compute_bellman_error(X, A, R, Snext, betas, γ::Float64)
    T = length(X)
    bellman_errors = Float64[]
    
    for t in 1:(T-1)  # Nicht für letzten Zeitschritt
        n = size(X[t], 1)
        d = div(size(X[t], 2), K)
        
        for i in 1:n
            # Aktueller Q-Wert
            s = zeros(d)  # Extract state from feature vector
            x_current = X[t][i, :]
            q_current = dot(betas[t], x_current)
            
            # Bellman Target: R + γ * max Q_{t+1}(s', a')
            s_next = Snext[t][i, :]
            max_q_next = -Inf
            for a_prime in ACTIONS
                x_next = build_feature_vector(s_next, a_prime, K)
                q_next = dot(betas[t+1], x_next)
                max_q_next = max(max_q_next, q_next)
            end
            bellman_target = R[t][i] + γ * max_q_next
            
            push!(bellman_errors, abs(q_current - bellman_target))
        end
    end
    
    return bellman_errors
end

# KONVERGENZ ANALYSE


"""
Analysiert Konvergenz während FQI Training
(Modifikation der ursprünglichen fitted_q_iteration Funktion nötig)
"""
function fitted_q_iteration_with_convergence(X, A, R, Snext; γ=1.0, max_iter=50, tol=1e-6)
    T = length(X)
    dK = size(X[1], 2)
    betas = Vector{Vector{Float64}}(undef, T)
    convergence_history = Vector{Vector{Float64}}()
    
    # Initialize randomly
    for t in 1:T
        betas[t] = randn(dK) * 0.1
    end
    
    for iter in 1:max_iter
        old_betas = deepcopy(betas)
        beta_changes = Float64[]
        
        # Backward iteration
        # Terminal stage
        X_T = X[T]
        y_T = R[T]
        betas[T] = X_T \ y_T
        
        # Previous stages
        for t in (T-1):-1:1
            n = size(X[t], 1)
            d = div(dK, K)
            Qtilde = zeros(n)
            
            for i in 1:n
                s_next = Snext[t][i, :]
                maxq = -Inf
                for aprime in ACTIONS
                    xnext = build_feature_vector(s_next, aprime, K)
                    qval = dot(betas[t+1], xnext)
                    maxq = max(maxq, qval)
                end
                Qtilde[i] = R[t][i] + γ * maxq
            end
            
            betas[t] = X[t] \ Qtilde
        end
        
        # Check convergence
        total_change = 0.0
        for t in 1:T
            change = norm(betas[t] - old_betas[t])
            total_change += change
        end
        
        push!(convergence_history, [total_change])
        
        if total_change < tol
            println("Converged after $iter iterations")
            break
        end
    end
    
    # Create prediction functions
    predict_Q = (t::Int, s::Vector{Float64}, a::Int) -> begin
        x = build_feature_vector(s, a, K)
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
    
    return (betas, predict_Q, policy, convergence_history)
end


# VISUALISIERUNGEN


"""
Erstellt Performance Comparison Plot für verschiedene Discount Factors
"""
function plot_performance_comparison(df, discount_factors, all_policies, all_predict_Qs)
    performance_data = DataFrame(
        gamma = Float64[],
        mean_return = Float64[],
        std_return = Float64[],
        median_return = Float64[],
        q25 = Float64[],
        q75 = Float64[]
    )
    
    # Behavior Policy Baseline
    behavior_returns = evaluate_behavior_policy(df, 0.8)  # Use γ=0.8 for baseline
    
    for (i, γ) in enumerate(discount_factors)
        returns = evaluate_policy_performance(df, all_predict_Qs[i], all_policies[i], γ)
        
        push!(performance_data, (
            γ,
            mean(returns),
            std(returns),
            median(returns),
            quantile(returns, 0.25),
            quantile(returns, 0.75)
        ))
    end
    
    # Plot erstellen
    p1 = plot(performance_data.gamma, performance_data.mean_return,
              yerr=performance_data.std_return,
              marker=:circle, linewidth=2, markersize=6,
              xlabel="Discount Factor γ", ylabel="Mean Return",
              title="Policy Performance vs Discount Factor",
              label="FQI Policy", legend=:topright)
              
    # Behavior Policy Baseline hinzufügen
    hline!([mean(behavior_returns)], linestyle=:dash, linewidth=2, 
           label="Behavior Policy Baseline", color=:red)
    
    # Confidence intervals
    p2 = plot(performance_data.gamma, performance_data.median_return,
              ribbon=(performance_data.median_return .- performance_data.q25,
                     performance_data.q75 .- performance_data.median_return),
              fillalpha=0.3, linewidth=2,
              xlabel="Discount Factor γ", ylabel="Return",
              title="Policy Performance Distribution",
              label="Median ± IQR")
    
    return plot(p1, p2, layout=(1,2), size=(800,400))
end


#Visualisiert Bellman Errors

function plot_bellman_errors(discount_factors, all_X, all_A, all_R, all_Snext, all_betas)
    bellman_data = DataFrame(
        gamma = Float64[],
        mean_error = Float64[],
        median_error = Float64[],
        q95_error = Float64[]
    )
    
    for (i, γ) in enumerate(discount_factors)
        errors = compute_bellman_error(all_X, all_A, all_R, all_Snext, all_betas[i], γ)
        
        push!(bellman_data, (
            γ,
            mean(errors),
            median(errors),
            quantile(errors, 0.95)
        ))
    end
    
    p = plot(bellman_data.gamma, bellman_data.mean_error,
             marker=:circle, linewidth=2, markersize=6,
             xlabel="Discount Factor γ", ylabel="Mean Bellman Error",
             title="Value Function Quality: Bellman Error",
             label="Mean Error", legend=:topright)
             
    plot!(bellman_data.gamma, bellman_data.median_error,
          marker=:square, linewidth=2, markersize=6,
          label="Median Error")
          
    plot!(bellman_data.gamma, bellman_data.q95_error,
          marker=:diamond, linewidth=2, markersize=6,
          label="95th Percentile", linestyle=:dash)
    
    return p
end



# Discount Factor Analyse für Fitted Q-Iteration

function compute_policy_entropy(df::DataFrame, policy_func, scols; n_samples=500)
    # Sample zufällige States aus dem Datensatz
    sampled_rows = sample(eachrow(df), n_samples, replace=true)
    action_counts = Dict(a => 0 for a in ACTIONS)
    
    for row in sampled_rows
        s = Float64.(collect(row[scols]))
        t_idx = row.t - minimum(df.t) + 1
        
        recommended_action = policy_func(t_idx, s)
        action_counts[recommended_action] += 1
    end
    
    # Shannon Entropy berechnen
    total = sum(values(action_counts))
    entropy = 0.0
    for count in values(action_counts)
        if count > 0
            p = count / total
            entropy -= p * log2(p)
        end
    end
    
    return entropy
end

"""
Sampelt Q-Values über verschiedene State-Action Paare
"""
function sample_q_values(df::DataFrame, predict_Q_func, scols; n_samples=200)
    sampled_rows = sample(eachrow(df), n_samples, replace=true)
    q_values = Float64[]
    
    for row in sampled_rows
        s = Float64.(collect(row[scols]))
        t_idx = row.t - minimum(df.t) + 1
        
        for a in ACTIONS
            q_val = predict_Q_func(t_idx, s, a)
            push!(q_values, q_val)
        end
    end
    
    return q_values
end


"""
Modifizierte FQI mit detailliertem Konvergenz-Tracking
"""
function fitted_q_iteration_convergence_tracking(X, A, R, Snext; γ=1.0, max_iter=100, tol=1e-8)
    T = length(X)
    dK = size(X[1], 2)
    
    # Initialize betas
    betas = Vector{Vector{Float64}}(undef, T)
    for t in 1:T
        betas[t] = zeros(dK)
    end
    
    convergence_metrics = Dict(
        :beta_changes => Float64[],
        :max_q_changes => Float64[],
        :policy_changes => Float64[],
        :bellman_residuals => Float64[]
    )
    
    # Für Policy Change Tracking
    old_policy_decisions = nothing
    
    for iter in 1:max_iter
        old_betas = deepcopy(betas)
        
        # Terminal stage
        X_T = X[T]
        y_T = R[T]
        betas[T] = X_T \ y_T
        
        # Backward iteration
        max_q_changes_iter = Float64[]
        
        for t in (T-1):-1:1
            n = size(X[t], 1)
            d = div(dK, K)
            Qtilde = zeros(n)
            old_q_values = Float64[]
            new_q_values = Float64[]
            
            for i in 1:n
                s_next = Snext[t][i, :]
                
                # Old Q-value
                x_current = X[t][i, :]
                old_q = dot(old_betas[t], x_current)
                push!(old_q_values, old_q)
                
                # Compute target
                maxq = -Inf
                for aprime in ACTIONS
                    xnext = build_feature_vector(s_next, aprime, K)
                    qval = dot(betas[t+1], xnext)
                    maxq = max(maxq, qval)
                end
                Qtilde[i] = R[t][i] + γ * maxq
            end
            
            betas[t] = X[t] \ Qtilde
            
            # New Q-values
            for i in 1:n
                x_current = X[t][i, :]
                new_q = dot(betas[t], x_current)
                push!(new_q_values, new_q)
            end
            
            # Track max Q-value change for this stage
            if !isempty(old_q_values) && !isempty(new_q_values)
                q_changes = abs.(new_q_values .- old_q_values)
                push!(max_q_changes_iter, maximum(q_changes))
            end
        end
        
        # Beta change
        total_beta_change = sum(norm(betas[t] - old_betas[t]) for t in 1:T)
        push!(convergence_metrics[:beta_changes], total_beta_change)
        
        # Max Q-value change
        avg_max_q_change = isempty(max_q_changes_iter) ? 0.0 : mean(max_q_changes_iter)
        push!(convergence_metrics[:max_q_changes], avg_max_q_change)
        
        # Policy change (simplified)
        current_policy_decisions = []
        if iter > 1
            # Sample some states to check policy consistency
            sample_states = [X[1][i, 1:div(dK, K)] for i in 1:min(50, size(X[1], 1))]  # Simplified state extraction
            for s in sample_states
                best_a = ACTIONS[1]
                best_q = -Inf
                for a in ACTIONS
                    x = build_feature_vector(s, a, K)
                    q = dot(betas[1], x)  # Use first stage for simplicity
                    if q > best_q
                        best_q = q
                        best_a = a
                    end
                end
                push!(current_policy_decisions, best_a)
            end
            
            if old_policy_decisions !== nothing
                policy_change_rate = mean(current_policy_decisions .!= old_policy_decisions)
                push!(convergence_metrics[:policy_changes], policy_change_rate)
            else
                push!(convergence_metrics[:policy_changes], 1.0)
            end
            old_policy_decisions = copy(current_policy_decisions)
        else
            push!(convergence_metrics[:policy_changes], 1.0)
        end
        
        # Bellman residual
        bellman_errors = compute_bellman_error(X, A, R, Snext, betas, γ)
        avg_bellman_residual = mean(bellman_errors)
        push!(convergence_metrics[:bellman_residuals], avg_bellman_residual)
        
        # Check convergence
        if total_beta_change < tol
            println("Converged after $iter iterations (γ=$γ)")
            break
        end
        
        if iter == max_iter
            println("Max iterations reached for γ=$γ")
        end
    end
    
    return betas, convergence_metrics
end


function create_discount_factor_plots(stability_results, γ_values)
    
    # 1. Performance vs γ mit Stabilität
    perf_data = DataFrame()
    for γ in γ_values
        returns_all_seeds = vcat(stability_results[γ][:returns]...)
        row = (gamma=γ, mean_return=mean(returns_all_seeds), 
               std_return=std(returns_all_seeds), 
               q25=quantile(returns_all_seeds, 0.25),
               q75=quantile(returns_all_seeds, 0.75))
        perf_data = vcat(perf_data, DataFrame([row]))
    end
    
    p1 = plot(perf_data.gamma, perf_data.mean_return,
              yerr=perf_data.std_return, marker=:circle, linewidth=3, markersize=8,
              xlabel="Discount Factor γ", ylabel="Mean Return",
              title="Performance vs Discount Factor\n(Multi-Seed Analysis)",
              label="Mean ± Std", legend=:topright, color=:blue)
    
    # 2. Stabilität Heatmap
    stability_matrix = zeros(length(γ_values), 4)  # 4 Metriken
    metric_names = ["Return Std", "Bellman Error Std", "Policy Entropy Std", "Q-Value Std"]
    
    for (i, γ) in enumerate(γ_values)
        # Return Variability
        returns_by_seed = [mean(ret) for ret in stability_results[γ][:returns]]
        stability_matrix[i, 1] = std(returns_by_seed)
        
        # Bellman Error Variability  
        bellman_by_seed = [mean(be) for be in stability_results[γ][:bellman_errors]]
        stability_matrix[i, 2] = std(bellman_by_seed)
        
        # Policy Entropy Variability
        entropy_by_seed = stability_results[γ][:policy_entropy]
        stability_matrix[i, 3] = std(entropy_by_seed)
        
        # Q-Value Variability
        qval_by_seed = [mean(qv) for qv in stability_results[γ][:q_values]]
        stability_matrix[i, 4] = std(qval_by_seed)
    end
    
    p2 = heatmap(metric_names, string.(γ_values), stability_matrix,
                 xlabel="Stability Metrics", ylabel="Discount Factor γ",
                 title="Multi-Seed Stability Analysis", 
                 color=:viridis, aspect_ratio=:auto)
    
    # 3. Policy Entropy vs γ
    entropy_data = DataFrame()
    for γ in γ_values
        entropies = stability_results[γ][:policy_entropy]
        row = (gamma=γ, mean_entropy=mean(entropies), std_entropy=std(entropies))
        entropy_data = vcat(entropy_data, DataFrame([row]))
    end
    
    p3 = plot(entropy_data.gamma, entropy_data.mean_entropy,
              yerr=entropy_data.std_entropy, marker=:square, linewidth=3, markersize=8,
              xlabel="Discount Factor γ", ylabel="Policy Entropy",
              title="Policy Exploration vs γ", label="Entropy ± Std",
              color=:green, legend=:topright)
    
    # Add theoretical maximum entropy line
    max_entropy = log2(length(ACTIONS))
    hline!([max_entropy], linestyle=:dash, linewidth=2, 
           label="Max Entropy", color=:red, alpha=0.7)
    
    return plot(p1, p2, p3, layout=(2,2), size=(900, 700))
end


#Konvergenz-Vergleich zwischen verschiedenen γ-Werten

function plot_convergence_comparison(df, γ_values; max_iter=50)
    convergence_data = Dict()
    
    # Für jeden γ-Wert Konvergenz tracken
    X, A, R, Snext, _, _ = build_stage_matrices(df)
    
    for γ in γ_values
        println("Tracking convergence for γ = $γ")
        _, conv_metrics = fitted_q_iteration_convergence_tracking(X, A, R, Snext; 
                                                                 γ=γ, max_iter=max_iter)
        convergence_data[γ] = conv_metrics
    end
    
    # Plot erstellen
    colors = [:blue, :red, :green, :orange, :purple]
    
    # Beta Changes
    p1 = plot(xlabel="Iteration", ylabel="Total β Change", 
              title="Parameter Convergence", yscale=:log10, legend=:topright)
    
    for (i, γ) in enumerate(γ_values)
        plot!(1:length(convergence_data[γ][:beta_changes]), 
              convergence_data[γ][:beta_changes],
              label="γ = $γ", linewidth=2, color=colors[i])
    end
    
    # Policy Changes
    p2 = plot(xlabel="Iteration", ylabel="Policy Change Rate",
              title="Policy Stability", legend=:topright)
    
    for (i, γ) in enumerate(γ_values)
        plot!(1:length(convergence_data[γ][:policy_changes]), 
              convergence_data[γ][:policy_changes],
              label="γ = $γ", linewidth=2, color=colors[i])
    end
    
    # Bellman Residuals
    p3 = plot(xlabel="Iteration", ylabel="Mean Bellman Error",
              title="Value Function Quality", yscale=:log10, legend=:topright)
    
    for (i, γ) in enumerate(γ_values)
        plot!(1:length(convergence_data[γ][:bellman_residuals]), 
              convergence_data[γ][:bellman_residuals],
              label="γ = $γ", linewidth=2, color=colors[i])
    end
    
    # Q-Value Changes
    p4 = plot(xlabel="Iteration", ylabel="Max Q-Value Change",
              title="Q-Function Stability", yscale=:log10, legend=:topright)
    
    for (i, γ) in enumerate(γ_values)
        plot!(1:length(convergence_data[γ][:max_q_changes]), 
              convergence_data[γ][:max_q_changes],
              label="γ = $γ", linewidth=2, color=colors[i])
    end
    
    return plot(p1, p2, p3, p4, layout=(2,2), size=(900, 700)), convergence_data
end


# Distributional Shift und Datensatz-Charakteristika Analyse für Offline RL


function analyze_dataset_characteristics(df::DataFrame)
    println("="^60)
    println("DATASET CHARACTERISTICS ANALYSIS")
    println("="^60)
    
    scols = state_cols(df)
    
    # Basic Statistics
    n_trajectories = length(unique(df.traj_id))
    n_timesteps = nrow(df)
    n_states = length(scols)
    trajectory_lengths = [nrow(group) for group in groupby(df, :traj_id)]
    
    println("Basic Dataset Statistics:")
    println("  Total trajectories: $n_trajectories")
    println("  Total timesteps: $n_timesteps")
    println("  State dimensions: $n_states")
    println("  Avg trajectory length: $(round(mean(trajectory_lengths), digits=2))")
    println("  Min/Max trajectory length: $(minimum(trajectory_lengths))/$(maximum(trajectory_lengths))")
    println("  State columns: $(scols)")
    println()
    
    # Action Distribution Analysis
    action_counts = countmap(df.action)
    action_probs = [action_counts[a] / n_timesteps for a in ACTIONS]
    
    println("Action Distribution (Behavior Policy):")
    for (i, a) in enumerate(ACTIONS)
        count = get(action_counts, a, 0)
        prob = count / n_timesteps
        println("  Action $a: $count times ($(round(prob*100, digits=1))%)")
    end
    println()
    
    # State Coverage Analysis
    state_data = Matrix(df[!, scols])
    
    # State space coverage metrics
    state_ranges = []
    state_stds = []
    for i in 1:n_states
        col_data = state_data[:, i]
        push!(state_ranges, maximum(col_data) - minimum(col_data))
        push!(state_stds, std(col_data))
        println("  $(scols[i]): range=$(round(state_ranges[i], digits=3)), std=$(round(state_stds[i], digits=3))")
    end
    println()
    
    # State-Action Coverage Matrix
    sa_coverage = zeros(length(ACTIONS), n_states)
    sa_counts = Dict()
    
    for row in eachrow(df)
        s = Float64.(collect(row[scols]))
        a = row.action
        key = (a, hash(s))  # Simple state discretization via hash
        sa_counts[key] = get(sa_counts, key, 0) + 1
    end
    
    unique_sa_pairs = length(sa_counts)
    avg_sa_visits = mean(values(sa_counts))
    
    println("State-Action Coverage:")
    println("  Unique (s,a) pairs: $unique_sa_pairs")
    println("  Average visits per (s,a): $(round(avg_sa_visits, digits=2))")
    println("  Coverage density: $(round(unique_sa_pairs / (n_timesteps * length(ACTIONS)), digits=4))")
    println()
    
    return Dict(
        :n_trajectories => n_trajectories,
        :n_timesteps => n_timesteps,
        :n_states => n_states,
        :trajectory_lengths => trajectory_lengths,
        :action_counts => action_counts,
        :action_probs => action_probs,
        :state_ranges => state_ranges,
        :state_stds => state_stds,
        :unique_sa_pairs => unique_sa_pairs,
        :sa_coverage_density => unique_sa_pairs / (n_timesteps * length(ACTIONS))
    )
end


function analyze_behavior_policy_quality(df::DataFrame)
    println("BEHAVIOR POLICY QUALITY ANALYSIS")
    println("="^40)
    
    # Return Distribution der Behavior Policy
    behavior_returns = evaluate_behavior_policy(df, 0.6)  # Use standard γ=0.6
    
    println("Behavior Policy Performance:")
    println("  Mean return: $(round(mean(behavior_returns), digits=3))")
    println("  Std return: $(round(std(behavior_returns), digits=3))")
    println("  Min/Max return: $(round(minimum(behavior_returns), digits=3))/$(round(maximum(behavior_returns), digits=3))")
    println("  25th/75th percentile: $(round(quantile(behavior_returns, 0.25), digits=3))/$(round(quantile(behavior_returns, 0.75), digits=3))")
    println()
    
    # Reward Distribution
    reward_stats = describe(df.reward)
    println("Reward Distribution:")
    println("  Mean: $(round(mean(df.reward), digits=3))")
    println("  Std: $(round(std(df.reward), digits=3))")
    println("  Min/Max: $(round(minimum(df.reward), digits=3))/$(round(maximum(df.reward), digits=3))")
    println("  Skewness: $(round(skewness(df.reward), digits=3))")
    println("  Kurtosis: $(round(kurtosis(df.reward), digits=3))")
    println()
    
    # Action Consistency Analysis (temporal correlations)
    action_transitions = Dict()
    prev_actions = Dict()
    
    for group in groupby(df, :traj_id)
        sorted_group = sort(group, :t)
        for i in 2:nrow(sorted_group)
            prev_a = sorted_group[i-1, :action]
            curr_a = sorted_group[i, :action]
            key = (prev_a, curr_a)
            action_transitions[key] = get(action_transitions, key, 0) + 1
        end
    end
    
    # Action persistence rate
    same_action_count = sum(get(action_transitions, (a,a), 0) for a in ACTIONS)
    total_transitions = sum(values(action_transitions))
    persistence_rate = same_action_count / total_transitions
    
    println("Temporal Action Patterns:")
    println("  Action persistence rate: $(round(persistence_rate*100, digits=1))%")
    println("  Total action transitions: $total_transitions")
    
    return Dict(
        :behavior_returns => behavior_returns,
        :reward_mean => mean(df.reward),
        :reward_std => std(df.reward),
        :persistence_rate => persistence_rate,
        :action_transitions => action_transitions
    )
end


function analyze_distributional_shift(df::DataFrame, all_policies::Vector, γ_values::Vector)
    println("DISTRIBUTIONAL SHIFT ANALYSIS")
    println("="^40)
    
    scols = state_cols(df)
    n_samples = min(1000, nrow(df))  # Sample für Effizienz
    sampled_indices = sample(1:nrow(df), n_samples, replace=false)
    
    shift_metrics = Dict()
    
    for (i, γ) in enumerate(γ_values)
        println("Analyzing distributional shift for γ = $γ")
        
        learned_policy = all_policies[i]
        behavior_actions = Int[]
        learned_actions = Int[]
        
        # Sample actions from both policies
        for idx in sampled_indices
            row = df[idx, :]
            s = Float64.(collect(row[scols]))
            t_idx = row.t - minimum(df.t) + 1
            
            behavior_action = row.action
            learned_action = learned_policy(t_idx, s)
            
            push!(behavior_actions, behavior_action)
            push!(learned_actions, learned_action)
        end
        
        # Compute distributional shift metrics
        
        # 1. Action Distribution Divergence (KL, JS, TV)
        behavior_dist = [count(==(a), behavior_actions) / length(behavior_actions) for a in ACTIONS]
        learned_dist = [count(==(a), learned_actions) / length(learned_actions) for a in ACTIONS]
        
        # Add small epsilon to avoid log(0)
        ε = 1e-8
        behavior_dist = behavior_dist .+ ε
        learned_dist = learned_dist .+ ε
        behavior_dist = behavior_dist ./ sum(behavior_dist)
        learned_dist = learned_dist ./ sum(learned_dist)
        
        # KL Divergence: D_KL(π_learned || π_behavior)
        kl_div = sum(learned_dist .* log.(learned_dist ./ behavior_dist))
        
        # JS Divergence (symmetric)
        m_dist = 0.5 * (behavior_dist + learned_dist)
        js_div = 0.5 * sum(behavior_dist .* log.(behavior_dist ./ m_dist)) + 
                 0.5 * sum(learned_dist .* log.(learned_dist ./ m_dist))
        
        # Total Variation Distance
        tv_dist = 0.5 * sum(abs.(learned_dist - behavior_dist))
        
        # 2. Policy Agreement Rate
        agreement_rate = mean(behavior_actions .== learned_actions)
        
        # 3. Action Distribution χ² test
        expected_counts = behavior_dist * length(learned_actions)
        observed_counts = [count(==(a), learned_actions) for a in ACTIONS]
        chi2_stat = sum((observed_counts - expected_counts).^2 ./ expected_counts)
        
        # 4. Per-Action Shift Analysis
        action_shifts = Dict()
        for a in ACTIONS
            behavior_freq = count(==(a), behavior_actions) / length(behavior_actions)
            learned_freq = count(==(a), learned_actions) / length(learned_actions)
            action_shifts[a] = learned_freq - behavior_freq
        end
        
        shift_metrics[γ] = Dict(
            :kl_divergence => kl_div,
            :js_divergence => js_div,
            :tv_distance => tv_dist,
            :agreement_rate => agreement_rate,
            :chi2_statistic => chi2_stat,
            :behavior_distribution => behavior_dist,
            :learned_distribution => learned_dist,
            :action_shifts => action_shifts
        )
        
        println("  KL Divergence: $(round(kl_div, digits=4))")
        println("  JS Divergence: $(round(js_div, digits=4))")
        println("  TV Distance: $(round(tv_dist, digits=4))")
        println("  Agreement Rate: $(round(agreement_rate*100, digits=1))%")
        println("  χ² Statistic: $(round(chi2_stat, digits=4))")
        println()
    end
    
    return shift_metrics
end

function document_hyperparameters(df::DataFrame, γ_values::Vector; 
                                 max_iterations::Int=50, 
                                 bootstrap_samples::Int=200,
                                 regularization::Float64=0.0)
    
    println("HYPERPARAMETER DOCUMENTATION")
    println("="^40)
    
    X, A, R, Snext, _, scols = build_stage_matrices(df)
    T = length(X)
    dK = size(X[1], 2)
    d = div(dK, K)
    
    # Algorithm Configuration
    println("Algorithm Configuration:")
    println("  Algorithm: Fitted Q-Iteration (FQI)")
    println("  Function Approximation: Linear (Least Squares)")
    println("  Feature Engineering: Concatenated state-action features")
    println("  Regularization: $regularization")
    println("  Max Iterations: $max_iterations")
    println()
    
    # Discount Factors
    println("Discount Factors:")
    for γ in γ_values
        println("  γ = $γ")
    end
    println()
    
    # Data Configuration
    println("Data Configuration:")
    println("  Training Samples: $(nrow(df))")
    println("  State Dimension: $d")
    println("  Action Space: $(ACTIONS)")
    println("  Feature Dimension: $dK ($(d) states × $(length(ACTIONS)) actions)")
    println("  Time Horizon: $T")
    println()
    
    # Feature Engineering Details
    println("Feature Engineering:")
    println("  Type: Block-structured state-action features")
    println("  Construction: x(s,a) = [0,...,0, s, 0,...,0] where s is placed in block corresponding to action a")
    println("  State Features: $(scols)")
    println("  Feature Normalization: None (raw features)")
    println()
    
    # Bootstrap Configuration
    println("Bootstrap Configuration:")
    println("  Bootstrap Samples: $bootstrap_samples")
    println("  Resampling Unit: Complete trajectories")
    println("  Confidence Level: 95% (α = 0.05)")
    println()
    
    # Optimization Details
    println("Optimization Details:")
    println("  Solver: Normal equations (X \\ y in Julia)")
    println("  Numerical Stability: Built-in pivoting in Julia's LAPACK")
    println("  Convergence Criterion: ||β_new - β_old|| < 1e-6")
    println("  Backward Iteration: Terminal stage → Stage 1")
    println()
    
    # Computational Environment
    println("Computational Environment:")
    println("  Language: Julia $(VERSION)")
    println("  Key Packages: LinearAlgebra, Statistics, Random")
    println("  Matrix Operations: BLAS/LAPACK optimized")
    println("  Random Seeds: [111, 222, 333, 444, 555] for reproducibility")
    println()
    
    return Dict(
        :algorithm => "Fitted Q-Iteration",
        :function_approximation => "Linear Least Squares",
        :gamma_values => γ_values,
        :max_iterations => max_iterations,
        :feature_dim => dK,
        :state_dim => d,
        :time_horizon => T,
        :bootstrap_samples => bootstrap_samples,
        :regularization => regularization
    )
end


function create_dataset_and_shift_plots(dataset_chars, df, behavior_quality, shift_metrics, γ_values)
    
    # 1. Datensatz Charakteristika
    
    # Action Distribution
    p1 = bar(ACTIONS, [dataset_chars[:action_counts][a] for a in ACTIONS],
             xlabel="Action", ylabel="Count", title="Behavior Policy Action Distribution",
             color=:lightblue, alpha=0.7, legend  = false)
    
    # Reward Distribution
    p3 = histogram(behavior_quality[:behavior_returns], bins=30,
                   xlabel="Return", ylabel="Frequency", 
                   title="Behavior Policy Return Distribution",
                   color=:orange, alpha=0.7, legend = false)

    # Durchschnittlicher Reward pro Zeit
    reward_time = combine(groupby(df, :t),
                        :reward => mean => :avg_reward,
                        :reward => (x -> count(>(0), x)) => :num_positive)

    p2 = plot(reward_time.t, reward_time.avg_reward,
            seriestype=:bar,
            xlabel="Time t", ylabel="Average Reward",
            title="Average Reward per Time Step",
            color=:purple, alpha=0.6, legend=false,
            xticks =([1, 2, 3, 4, 5], ["1990\n-2008", "2009\n-2012", "2013\n-2015", "2016\n-2019", "2020\n-2023"]))

    
    # 2. Distributional Shift Plots
    
    # Agreement Rate vs γ
    agreement_rates = [shift_metrics[γ][:agreement_rate] for γ in γ_values]
    p4 = plot(γ_values, agreement_rates, marker=:circle, linewidth=3, markersize=8,
              xlabel="Discount Factor γ", ylabel="Policy Agreement Rate",
              title="Learned vs Behavior Policy Agreement", color=:red,
              ylims=(0, 1), legend=false)
    
    # Divergence Metrics
    kl_divs = [shift_metrics[γ][:kl_divergence] for γ in γ_values]
    js_divs = [shift_metrics[γ][:js_divergence] for γ in γ_values]
    tv_dists = [shift_metrics[γ][:tv_distance] for γ in γ_values]
    
    p5 = plot(γ_values, kl_divs, marker=:circle, linewidth=2, label="KL Divergence",
              xlabel="Discount Factor γ", ylabel="Divergence", 
              title="Distributional Shift Metrics", legend=:topright)
    plot!(γ_values, js_divs, marker=:square, linewidth=2, label="JS Divergence")
    plot!(γ_values, tv_dists, marker=:diamond, linewidth=2, label="TV Distance")
    
    # Action Distribution Comparison (für einen γ-Wert)
    γ_example = γ_values[end-1]  # Use highest γ
    behavior_dist = shift_metrics[γ_example][:behavior_distribution]
    learned_dist = shift_metrics[γ_example][:learned_distribution]
    
    x_pos = 1:length(ACTIONS)
    p6 = groupedbar([behavior_dist learned_dist], 
                    bar_position=:dodge, bar_width=0.7,
                    xlabel="Action", ylabel="Probability",
                    title="Action Distributions (γ=$γ_example)",
                    label=["Behavior Policy" "Learned Policy"],
                    xticks=(x_pos, string.(ACTIONS)))
    
    return plot(p1, p3, p2, p4, p5, p6, layout=(2,3), size=(1200, 900))
end


function plot_state_action_coverage(df::DataFrame)
    scols = state_cols(df)
    
    # State-Action Heatmap (simplified for 2D visualization)
    if length(scols) >= 2
        # Use first two state dimensions
        s1_vals = df[!, scols[1]]
        s2_vals = df[!, scols[2]]
        actions = df.action
        
        # Create 2D histogram for each action
        plots_list = []
        for a in ACTIONS
            mask = actions .== a
            if sum(mask) > 0
                p = scatter(s1_vals[mask], s2_vals[mask], 
                           alpha=0.6, markersize=3,
                           xlabel=string(scols[1]), ylabel=string(scols[2]),
                           title="Action $a Coverage", legend=false)
                push!(plots_list, p)
            end
        end
        
        return plot(plots_list..., layout=(2,2), size=(800, 800))
    else
        # Fallback: 1D state distribution
        p = histogram(df[!, scols[1]], bins=50,
                     xlabel=string(scols[1]), ylabel="Frequency",
                     title="State Distribution", alpha=0.7)
        return p
    end
end

function analyze_trajectory_level_policies(df::DataFrame, all_policies::Vector, 
                                         γ_values::Vector; n_traj_sample::Int=50)
    
    println("TRAJECTORY-LEVEL POLICY ANALYSIS")
    println("="^50)
    
    scols = state_cols(df)
    all_trajectories = unique(df.traj_id)
    
    # Sample Trajektorien für detaillierte Analyse
    sample_trajectories = if length(all_trajectories) > n_traj_sample
        sample(all_trajectories, n_traj_sample, replace=false)
    else
        all_trajectories
    end
    
    trajectory_results = Dict()
    
    for traj_id in sample_trajectories
        traj_data = filter(row -> row.traj_id == traj_id, df)
        sort!(traj_data, :t)
        
        traj_results = Dict(
            :length => nrow(traj_data),
            :behavior_actions => traj_data.action,
            :behavior_rewards => traj_data.reward,
            :behavior_return => sum(traj_data.reward),  # Undiscounted für Vergleichbarkeit
            :states => [Float64.(collect(row[scols])) for row in eachrow(traj_data)],
            :learned_actions => Dict(),
            :learned_returns => Dict(),
            :action_differences => Dict(),
            :q_value_differences => Dict()
        )
        
        # Für jede gelernte Policy
        for (i, γ) in enumerate(γ_values)
            policy_func = all_policies[i]
            learned_actions = Int[]
            q_values_per_action = Dict(a => Float64[] for a in ACTIONS)
            
            for (step, row) in enumerate(eachrow(traj_data))
                s = Float64.(collect(row[scols]))
                t_idx = row.t - minimum(df.t) + 1
                
                # Gelernte Policy Action
                learned_action = policy_func(t_idx, s)
                push!(learned_actions, learned_action)
                
                # Q-Values für alle Aktionen in diesem State
                for a in ACTIONS
                    q_val = all_policies[i] isa Function ? 
                           compute_q_value_for_trajectory(s, a, t_idx, γ) : 0.0
                    push!(q_values_per_action[a], q_val)
                end
            end
            
            # Action Unterschiede zur Behavior Policy
            action_diffs = learned_actions .!= traj_results[:behavior_actions]
            diff_rate = mean(action_diffs)
            
            # Q-Value Spreads (als Proxy für Policy Confidence)
            q_spreads = Float64[]
            for step in 1:length(learned_actions)
                step_q_vals = [q_values_per_action[a][step] for a in ACTIONS]
                push!(q_spreads, maximum(step_q_vals) - minimum(step_q_vals))
            end
            
            traj_results[:learned_actions][γ] = learned_actions
            traj_results[:action_differences][γ] = diff_rate
            traj_results[:q_value_differences][γ] = mean(q_spreads)
        end
        
        trajectory_results[traj_id] = traj_results
    end
    
    return trajectory_results
end


function analyze_state_specific_qvalues(df::DataFrame, all_policies::Vector, 
                                       predict_Q_funcs::Vector, γ_values::Vector; 
                                       n_state_samples::Int=200)
    
    println("\nSTATE-SPECIFIC Q-VALUE ANALYSIS")
    println("="^40)
    
    scols = state_cols(df)
    state_samples = sample(eachrow(df), n_state_samples, replace=false)
    
    state_analysis = DataFrame(
        state_id = Int[],
        gamma = Float64[],
        behavior_action = Int[],
        learned_action = Int[],
        q_behavior = Float64[],
        q_learned = Float64[],
        q_max = Float64[],
        q_advantage = Float64[],
        action_changed = Bool[]
    )
    
    for (state_id, row) in enumerate(state_samples)
        s = Float64.(collect(row[scols]))
        t_idx = row.t - minimum(df.t) + 1
        behavior_action = row.action
        
        for (i, γ) in enumerate(γ_values)
            policy_func = all_policies[i]
            predict_Q = predict_Q_funcs[i]
            
            learned_action = policy_func(t_idx, s)
            
            # Q-Values berechnen
            q_behavior = predict_Q(t_idx, s, behavior_action)
            q_learned = predict_Q(t_idx, s, learned_action)
            
            # Alle Q-Values für diesen State
            all_q_vals = [predict_Q(t_idx, s, a) for a in ACTIONS]
            q_max = maximum(all_q_vals)
            q_advantage = q_learned - q_behavior
            
            action_changed = learned_action != behavior_action
            
            push!(state_analysis, (
                state_id, γ, behavior_action, learned_action,
                q_behavior, q_learned, q_max, q_advantage, action_changed
            ))
        end
    end
    
    return state_analysis
end


function compute_policy_agreement_matrix(df::DataFrame, all_policies::Vector, 
                                       γ_values::Vector; n_samples::Int=500)
    
    println("\nPOLICY AGREEMENT MATRIX")
    println("="^30)
    
    scols = state_cols(df)
    sampled_rows = sample(eachrow(df), n_samples, replace=false)
    
    # Policy Decisions sammeln
    policy_decisions = Dict()
    for (i, γ) in enumerate(γ_values)
        decisions = Int[]
        policy_func = all_policies[i]
        
        for row in sampled_rows
            s = Float64.(collect(row[scols]))
            t_idx = row.t - minimum(df.t) + 1
            action = policy_func(t_idx, s)
            push!(decisions, action)
        end
        policy_decisions[γ] = decisions
    end
    
    # Agreement Matrix berechnen
    n_γ = length(γ_values)
    agreement_matrix = zeros(n_γ, n_γ)
    
    for i in 1:n_γ
        for j in 1:n_γ
            if i == j
                agreement_matrix[i, j] = 1.0
            else
                agreement_rate = mean(policy_decisions[γ_values[i]] .== 
                                   policy_decisions[γ_values[j]])
                agreement_matrix[i, j] = agreement_rate
            end
        end
    end
    
    return agreement_matrix, policy_decisions
end


function analyze_temporal_policy_dynamics(df::DataFrame, all_policies::Vector, 
                                        γ_values::Vector)
    
    println("\nTEMPORAL POLICY DYNAMICS")
    println("="^30)
    
    scols = state_cols(df)
    time_points = sort(unique(df.t))
    
    temporal_analysis = Dict()
    
    for γ in γ_values
        policy_idx = findfirst(==(γ), γ_values)
        policy_func = all_policies[policy_idx]
        
        time_data = Dict(
            :action_distribution_over_time => Dict(),
            :policy_entropy_over_time => Float64[],
            :behavior_agreement_over_time => Float64[]
        )
        
        for t in time_points
            t_data = filter(row -> row.t == t, df)
            if nrow(t_data) == 0
                continue
            end
            
            t_idx = t - minimum(df.t) + 1
            
            # Sample aus diesem Zeitpunkt
            sample_size = min(100, nrow(t_data))
            sampled_rows = sample(eachrow(t_data), sample_size, replace=false)
            
            behavior_actions = [row.action for row in sampled_rows]
            learned_actions = Int[]
            
            for row in sampled_rows
                s = Float64.(collect(row[scols]))
                action = policy_func(t_idx, s)
                push!(learned_actions, action)
            end
            
            # Action Distribution
            action_counts = countmap(learned_actions)
            total = sum(values(action_counts))
            action_dist = Dict(a => get(action_counts, a, 0) / total for a in ACTIONS)
            time_data[:action_distribution_over_time][t] = action_dist
            
            # Policy Entropy
            probs = [action_dist[a] for a in ACTIONS]
            entropy = -sum(p > 0 ? p * log2(p) : 0.0 for p in probs)
            push!(time_data[:policy_entropy_over_time], entropy)
            
            # Behavior Agreement
            agreement = mean(learned_actions .== behavior_actions)
            push!(time_data[:behavior_agreement_over_time], agreement)
        end
        
        temporal_analysis[γ] = time_data
    end
    
    return temporal_analysis, time_points
end


function create_trajectory_level_plots(trajectory_results, state_analysis, 
                                     agreement_matrix, temporal_analysis, 
                                     γ_values, time_points)
    
    plots_list = []
    
    # 1. Action Change Rate per Trajectory Type
    traj_lengths = [traj[:length] for traj in values(trajectory_results)]
    length_categories = ["Short (≤$(quantile(traj_lengths, 0.33)))", 
                        "Medium", 
                        "Long (≥$(quantile(traj_lengths, 0.67)))"]
    
    change_rates_by_length = Dict()
    for γ in γ_values
        short_rates = Float64[]
        med_rates = Float64[]
        long_rates = Float64[]
        
        for traj in values(trajectory_results)
            rate = traj[:action_differences][γ]
            if traj[:length] <= quantile(traj_lengths, 0.33)
                push!(short_rates, rate)
            elseif traj[:length] >= quantile(traj_lengths, 0.67)
                push!(long_rates, rate)
            else
                push!(med_rates, rate)
            end
        end
        
        change_rates_by_length[γ] = [mean(short_rates), mean(med_rates), mean(long_rates)]
    end
    
    p1 = groupedbar([change_rates_by_length[γ] for γ in γ_values]',
                    bar_position=:dodge, xlabel="Trajectory Length Category",
                    ylabel="Mean Action Change Rate", 
                    title="Policy Deviation by Trajectory Length",
                    label=["γ=$(γ)" for γ in γ_values]',
                    xticks=(1:3, length_categories))
    push!(plots_list, p1)
    
    # 2. Q-Value Advantage Distribution
    p2 = violin(xlabel="Discount Factor γ", ylabel="Q-Value Advantage",
               title="Q-Value Advantage: Learned vs Behavior Action")
    for γ in γ_values
        γ_data = filter(row -> row.gamma == γ, state_analysis)
        violin!([string(γ)], γ_data.q_advantage, alpha=0.6, 
               label="γ=$γ", side=:right)
    end
    push!(plots_list, p2)
    
    # 3. Policy Agreement Heatmap
    p3 = heatmap(string.(γ_values), string.(γ_values), agreement_matrix,
                xlabel="Discount Factor γ", ylabel="Discount Factor γ",
                title="Policy Agreement Matrix", color=:viridis,
                aspect_ratio=:equal)
    
    # Annotate with values
    for i in 1:length(γ_values)
        for j in 1:length(γ_values)
            annotate!(j, i, text("$(round(agreement_matrix[i,j], digits=2))", 
                     8, :white, :center))
        end
    end
    push!(plots_list, p3)
    
    # 4. Temporal Policy Entropy
    p4 = plot(xlabel="Time", ylabel="Policy Entropy", 
             title="Policy Entropy Evolution Over Time", legend=:topright)
    
    colors = [:blue, :red, :green, :orange]
    for (i, γ) in enumerate(γ_values)
        if haskey(temporal_analysis, γ)
            entropies = temporal_analysis[γ][:policy_entropy_over_time]
            if length(entropies) == length(time_points)
                plot!(time_points, entropies, linewidth=2, marker=:circle,
                     label="γ=$γ", color=colors[i])
            end
        end
    end
    push!(plots_list, p4)
    
    # 5. Action Change vs Q-Advantage Scatter
    p5 = scatter(xlabel="Q-Value Advantage", ylabel="Action Changed",
                title="Action Changes vs Q-Value Advantage", alpha=0.6)
    
    for (i, γ) in enumerate(γ_values)
        γ_data = filter(row -> row.gamma == γ, state_analysis)
        y_jitter = γ_data.action_changed .+ randn(nrow(γ_data)) * 0.02
        scatter!(γ_data.q_advantage, y_jitter, label="γ=$γ", 
                alpha=0.6, markersize=3, color=colors[i])
    end
    push!(plots_list, p5)
    
    # 6. Temporal Behavior Agreement
    p6 = plot(xlabel="Time", ylabel="Behavior Agreement Rate",
             title="Agreement with Behavior Policy Over Time", legend=:topright)
    
    for (i, γ) in enumerate(γ_values)
        if haskey(temporal_analysis, γ)
            agreements = temporal_analysis[γ][:behavior_agreement_over_time]
            if length(agreements) == length(time_points)
                plot!(time_points, agreements, linewidth=2, marker=:square,
                     label="γ=$γ", color=colors[i])
            end
        end
    end
    hline!([1.0], linestyle=:dash, color=:black, alpha=0.5, label="Perfect Agreement")
    push!(plots_list, p6)
    
    return plot(plots_list..., layout=(3,2), size=(1200, 1000))
end


function trajectory_statistical_analysis(trajectory_results, state_analysis, γ_values)
    
    println("\n" * "="^60)
    println("TRAJECTORY-LEVEL STATISTICAL ANALYSIS")
    println("="^60)
    
    # 1. Action Change Rate Unterschiede
    println("\n1. Action Change Rate Analysis:")
    change_rates_by_gamma = Dict()
    for γ in γ_values
        rates = [traj[:action_differences][γ] for traj in values(trajectory_results)]
        change_rates_by_gamma[γ] = rates
        println("   γ=$γ: $(round(mean(rates)*100, digits=1))% ± $(round(std(rates)*100, digits=1))%")
    end
    
    # 2. Q-Value Advantage Analysis
    println("\n2. Q-Value Advantage Analysis:")
    for γ in γ_values
        γ_data = filter(row -> row.gamma == γ, state_analysis)
        positive_advantage = mean(γ_data.q_advantage .> 0)
        mean_advantage = mean(γ_data.q_advantage)
        
        println("   γ=$γ:")
        println("     Mean Q-advantage: $(round(mean_advantage, digits=4))")
        println("     States with positive advantage: $(round(positive_advantage*100, digits=1))%")
    end
    
    # 3. Policy Confidence (Q-spreads)
    println("\n3. Policy Confidence Analysis:")
    for γ in γ_values
        q_spreads = [mean(collect(values(traj[:q_value_differences]))) 
                    for traj in values(trajectory_results) 
                    if haskey(traj[:q_value_differences], γ)]
        
        if !isempty(q_spreads)
            println("   γ=$γ: Mean Q-spread: $(round(mean(q_spreads), digits=4))")
        end
    end
    
    return change_rates_by_gamma
end

function analyze_state_influences(betas_boot, df, scols; t::Int=1, ngrid::Int=5)

    results = Dict()

    # Fix all state variables on their median (numeric) or mode (categorical)
    fixed_state = Vector{Float64}(undef, length(scols))
    for (j,col) in enumerate(scols)
        x = df[!, col]
        if eltype(x) <: Number
            fixed_state[j] = median(skipmissing(x))
        else
            fixed_state[j] = mode(skipmissing(x))
        end
    end

    for (j,col) in enumerate(scols)
        println("Analyzing state variable: $col")

        # Define value grid
        x = df[!, col]
        values = nothing
        if eltype(x) <: Number
            if length(unique(x)) <= 10
                values = sort(unique(x))
            else
                values = collect(range(minimum(skipmissing(x)),
                                       maximum(skipmissing(x));
                                       length=ngrid))
            end
        else
            values = unique(x)
        end

        # Storage
        action_means = Dict(a => Float64[] for a in ACTIONS)  # Q-values
        policy_probs = Dict(a => Float64[] for a in ACTIONS)  # p_win

        for v in values
            s = copy(fixed_state)
            s[j] = v

            # Collect bootstrap Q-values
            Q_boot = [ [dot(b[t], build_feature_vector(s,a,length(ACTIONS))) for a in ACTIONS]
                       for b in betas_boot ]

            # Average Q-values per action
            for (ai,a) in enumerate(ACTIONS)
                push!(action_means[a], mean(q[ai] for q in Q_boot))
            end

            # p_win = how often each action is argmax
            wins = zeros(Int, length(ACTIONS))
            for q in Q_boot
                wins[argmax(q)] += 1
            end
            for (ai,a) in enumerate(ACTIONS)
                push!(policy_probs[a], wins[ai] / length(betas_boot))
            end
        end

        results[col] = (values, action_means, policy_probs)

        # --- Plot 1: Q-values
        plt1 = plot()
        for a in ACTIONS
            plot!(values, action_means[a], lw=2, marker=:circle, label="Action $a")
        end
        xlabel!(plt1, col)
        ylabel!(plt1, "Q(s,a)")
        title!(plt1, "Effect of $col on Q-values (stage $t)")
        display(plt1)

        # --- Plot 2: Policy choice probabilities (p_win)
        plt2 = plot()
        for a in ACTIONS
            plot!(values, policy_probs[a], lw=2, marker=:circle, label="Action $a")
        end
        xlabel!(plt2, col)
        ylabel!(plt2, "p_win (bootstrap share)")
        #ylim!(plt2, (0,1))
        title!(plt2, "Effect of $col on optimal policy (stage $t)")
        display(plt2)
    end

    return results
end


