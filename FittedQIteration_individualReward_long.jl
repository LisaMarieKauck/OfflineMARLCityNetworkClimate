function create_dataset_and_shift_plots_long(dataset_chars, df, behavior_quality, shift_metrics, γ_values)
    
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
            color=:purple, alpha=0.6, legend=false)

    
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

B=120
path = filled_df;  

Random.seed!(rng_seed);
df_ind_long = load_data(path); 
add_mean_field_features!(df_ind_long; exclude_self=false)
X_ind_long,A_ind_long,R_ind_long,Snext_ind_long, traj_idx_ind_long, scols_ind_long = build_stage_matrices(df_ind_long)
println("Trajectories build!")
betas_08_ind_long, predict_Q_08_ind_long, policy_08_ind_long = fitted_q_iteration(X_ind_long,A_ind_long,R_ind_long,Snext_ind_long; γ=0.8);
println("Learned betas for γ = 0.8,stages 1..$(length(betas_08_ind_long)). State cols: ", scols_ind_long)

betas_08_agg, predict_Q_08_agg, policy_08_agg = fitted_q_iteration(X_agg,A_agg,R_agg,Snext_agg; γ=0.8);
println("Learned betas for γ = 0.8,stages 1..$(length(betas_08_agg)). State cols: ", scols_agg)

betas_agg, predict_Q_agg, policy_agg = fitted_q_iteration(X_agg,A_agg,R_agg,Snext_agg; γ=0.6);
println("Learned betas for γ = 0.6,stages 1..$(length(betas_agg)). State cols: ", scols_agg)
betas_04_agg, predict_Q_04_agg, policy_04_agg = fitted_q_iteration(X_agg,A_agg,R_agg,Snext_agg; γ=0.4);
println("Learned betas for γ = 0.4,stages 1..$(length(betas_04_agg)). State cols: ", scols_agg)
betas_02_agg, predict_Q_02_agg, policy_02_agg = fitted_q_iteration(X_agg,A_agg,R_agg,Snext_agg; γ=0.2);
println("Learned betas for γ = 0.2,stages 1..$(length(betas_02_agg)). State cols: ", scols_agg)

######################## Ergebnisse visualisieren

discount_factors = [0.2, 0.4, 0.6, 0.8]
all_betas_ind_long = []
all_predict_Qs_ind_long = []
all_policies_ind_long = []

push!(all_betas_agg, betas_08_agg)
push!(all_betas_agg, betas_agg)
push!(all_betas_agg, betas_04_agg)
push!(all_betas_agg, betas_02_agg)
push!(all_predict_Qs_agg, predict_Q_08_agg)
push!(all_predict_Qs_agg, predict_Q_agg)
push!(all_predict_Qs_agg, predict_Q_04_agg)
push!(all_predict_Qs_agg, predict_Q_02_agg)
push!(all_policies_agg, policy_08_agg)
push!(all_policies_agg, policy_agg)
push!(all_policies_agg, policy_04_agg)
push!(all_policies_agg, policy_02_agg)


println("Starting Dataset and Distributional Shift Analysis...")

# 1. Datensatz Charakteristika
dataset_chars_ind_long = analyze_dataset_characteristics(df_ind_long)

# 2. Behavior Policy Qualität
behavior_quality_ind_long = analyze_behavior_policy_quality(df_ind_long)

# 3. Hyperparameter Dokumentation
hyperparams_ind_long = document_hyperparameters(df_ind_long, discount_factors)

# 4. Distributional Shift
shift_metrics_ind_long = analyze_distributional_shift(df_ind_long, all_policies_ind_long, discount_factors)

# 5. Visualisierungen
main_plots_ind_long = create_dataset_and_shift_plots(dataset_chars_ind_long, df_ind_long, behavior_quality_ind_long, 
                                            shift_metrics_ind_long, discount_factors)
coverage_plot_ind_long = plot_state_action_coverage(df_ind_long)

# 6. Summary für Paper
println("\n" * "="^60)
println("SUMMARY FOR PAPER REPORTING")
println("="^60)

println("Dataset Characteristics:")
println("- $(dataset_chars_ind_long[:n_trajectories]) trajectories, $(dataset_chars_ind_long[:n_timesteps]) total timesteps")
println("- $(dataset_chars_ind_long[:n_states]) state dimensions, $(length(ACTIONS)) actions") 
println("- State-action coverage: $(round(dataset_chars_ind_long[:sa_coverage_density]*100, digits=2))%")
println("- Action distribution: $(round.(dataset_chars_ind_long[:action_probs]*100, digits=1))%")
println()

println("Behavior Policy Quality:")
println("- Mean return: $(round(mean(behavior_quality_ind_long[:behavior_returns]), digits=3)) ± $(round(std(behavior_quality_ind_long[:behavior_returns]), digits=3))")
println("- Action persistence: $(round(behavior_quality_ind_long[:persistence_rate]*100, digits=1))%")
println()

println("Distributional Shift (Policy Agreement Rates):")
for γ in discount_factors
    agreement = shift_metrics_ind_long[γ][:agreement_rate]
    kl_div = shift_metrics_ind_long[γ][:kl_divergence]
    println("- γ=$γ: $(round(agreement*100, digits=1))% agreement, KL=$(round(kl_div, digits=4))")
end
    


# 2. Discount Factor 

# Ergebnisse zusammenfassen
println("\n" * "="^50)
println("CORE METRICS SUMMARY GLOBAL")
println("="^50)

for (i, γ) in enumerate(discount_factors)
    returns_ind_long = evaluate_policy_performance(df_ind_long, all_predict_Qs_ind_long[i], all_policies_ind_long[i], γ)
    bellman_errors_ind_long = compute_bellman_error(X_ind_long, A_ind_long, R_ind_long, Snext_ind_long, all_betas_ind_long[i], γ)

    println("γ = $γ:")
    println("  Mean Return: $(round(mean(returns_ind_long), digits=3)) ± $(round(std(returns_ind_long), digits=3))")
    println("  Mean Bellman Error: $(round(mean(bellman_errors_ind_long), digits=6))")
    println("  Median Bellman Error: $(round(median(bellman_errors_ind_long), digits=6))")
    println()
end

# Plots erstellen
println("Creating visualizations...")

p1 = plot_performance_comparison(df_ind_long, discount_factors, all_policies_ind_long, all_predict_Qs_ind_long)
p2 = plot_bellman_errors(discount_factors, X_ind_long, A_ind_long, R_ind_long, Snext_ind_long, all_betas_ind_long)

# Discount Factor Analyse für Fitted Q-Iteration



println("Starting Comprehensive Discount Factor Analysis...")
println("γ values: $discount_factors")

# 2. Konvergenz Analysis  
println("\n2. Running Convergence Analysis...")
convergence_plots_ind_long, convergence_data_ind_long = plot_convergence_comparison(df_ind_long, discount_factors)

    
println("Starting Comprehensive Trajectory-Level Analysis...")
println("γ values: $discount_factors")
println("Number of policies: $(length(all_policies_ind_long))")

# 1. Trajektorie-spezifische Analyse
println("\n1. Analyzing trajectory-specific policies...")
trajectory_results_ind_long = analyze_trajectory_level_policies(df_ind_long, all_policies_ind_long, discount_factors)

# 2. State-spezifische Q-Value Analyse
println("\n2. Analyzing state-specific Q-values...")
state_analysis_ind_long = analyze_state_specific_qvalues(df_ind_long, all_policies_ind_long, 
                                                all_predict_Qs_ind_long, discount_factors)

# 3. Policy Agreement Matrix
println("\n3. Computing policy agreement matrix...")
agreement_matrix_ind_long, policy_decisions_ind_long = compute_policy_agreement_matrix(df_ind_long, all_policies_ind_long, discount_factors)

# 4. Temporal Dynamics
println("\n4. Analyzing temporal policy dynamics...")
temporal_analysis_ind_long, time_points_ind_long = analyze_temporal_policy_dynamics(df_ind_long, all_policies_ind_long, discount_factors)

# 5. Visualisierungen -> debugging
println("\n5. Creating visualizations...")
#trajectory_plots_ind_long = create_trajectory_level_plots(trajectory_results_ind_long, state_analysis_ind_long,
                                                agreement_matrix_ind_long, temporal_analysis_ind_long,
                                                discount_factors, time_points_ind_long)

# 6. Statistische Analyse
println("\n6. Running statistical analysis...")
change_rates_ind_long = trajectory_statistical_analysis(trajectory_results_ind_long, state_analysis_ind_long , discount_factors)


pop_grid_ind_long = population_grid(df_ind_long; ngrid_small=20, ngrid_large=5)
s_sample_ind_long = make_population_s_sample(df_ind_long, pop_grid_ind_long, scols_ind_long)

println("We have a sample! Number of states: ", length(s_sample_ind_long))
# run bootstrap
B= 120;

println("Starting bootstrap FQI with B=$B samples and γ=0.6...")
betas_boot_ind_long = bootstrap_fqi(df_ind_long; B=B, γ=0.6)
println("Completed bootstrap FQI with $B samples and γ=0.6.")

p_win_ind_long, CIs_ind_long, Qdist_ind_long = analyze_bootstrap_Q(betas_boot_ind_long, s_sample_ind_long, 1)
println("p_win (actions × states): ", p_win_ind_long)
println("CIs example: ", CIs_ind_long)
#   return (betas, betas_boot, p_win, CIs, Qdist)

#betas, betas_boot, p_win, CIs, Qdist = main(filled_df; B=10, rng_seed=111)  # adjust B as needed (100 for quick test, 200+ for better accuracy)

println("GLOBAL AGGREGATED REWARD------------------------------------------")

summarize_results(p_win_ind_long, CIs_ind_long, backend=:latex)
summarize_results(p_win_ind_long, CIs_ind_long)



plot_action_preferences(p_win_ind_long, CIs_ind_long)
# Convergence Statistics


# L2 norm of β_t for each stage t, averaged over bootstraps.

plot_beta_stability(betas_boot_ind_long)


plot_Q_distributions(Qdist_ind_long, 1, 1)  # example state index and stage for now


bellman_errors_ind_long = compute_bellman_errors(betas_boot_ind_long, df_ind_long; γ=0.6)
plot_bellman_errors(bellman_errors_ind_long)


B_b_ind_long = length(betas_boot_ind_long)  # number of bootstrap samples
pop_grid_ind_long, p_win_grid_ind_long = population_policy_curve(betas_boot_ind_long, df_ind_long; t=1, ngrid=25)
plot_population_policy_curve(pop_grid_ind_long, p_win_grid_ind_long; B=B_b_ind_long)
