rng_seed = 111
#B=200
df_org_agg = copy(df_agg_discrete)
path = df_agg_discrete;  

Random.seed!(rng_seed);
df_agg = load_data(path);
add_mean_field_features!(df_agg; exclude_self=false); 
year = df_agg.t
select!(df_agg, Not(:t))  
df_agg = hcat(DataFrame(t=repeat(1:5, outer=length(unique(df_agg.traj_id)))), df_agg)
X_agg,A_agg,R_agg,Snext_agg, traj_idx_agg, scols_agg = build_stage_matrices(df_agg)
println("Trajectories build!")

betas_08_agg, predict_Q_08_agg, policy_08_agg = fitted_q_iteration(X_agg,A_agg,R_agg,Snext_agg; γ=0.8);
println("Learned betas for γ = 0.8,stages 1..$(length(betas_08_agg)). State cols: ", scols_agg)

betas_agg, predict_Q_agg, policy_agg = fitted_q_iteration(X_agg,A_agg,R_agg,Snext_agg; γ=0.6);
println("Learned betas for γ = 0.6,stages 1..$(length(betas_agg)). State cols: ", scols_agg)
betas_04_agg, predict_Q_04_agg, policy_04_agg = fitted_q_iteration(X_agg,A_agg,R_agg,Snext_agg; γ=0.4);
println("Learned betas for γ = 0.4,stages 1..$(length(betas_04_agg)). State cols: ", scols_agg)
betas_02_agg, predict_Q_02_agg, policy_02_agg = fitted_q_iteration(X_agg,A_agg,R_agg,Snext_agg; γ=0.2);
println("Learned betas for γ = 0.2,stages 1..$(length(betas_02_agg)). State cols: ", scols_agg)

######################## Ergebnisse analysieren und  visualisieren

discount_factors = [0.2, 0.4, 0.6, 0.8];
all_betas_agg = [];
all_predict_Qs_agg = [];
all_policies_agg = [];

push!(all_betas_agg, betas_08_agg);
push!(all_betas_agg, betas_agg);
push!(all_betas_agg, betas_04_agg);
push!(all_betas_agg, betas_02_agg);
push!(all_predict_Qs_agg, predict_Q_08_agg);
push!(all_predict_Qs_agg, predict_Q_agg);
push!(all_predict_Qs_agg, predict_Q_04_agg);
push!(all_predict_Qs_agg, predict_Q_02_agg);
push!(all_policies_agg, policy_08_agg);
push!(all_policies_agg, policy_agg);
push!(all_policies_agg, policy_04_agg);
push!(all_policies_agg, policy_02_agg);

  
println("Starting Dataset and Distributional Shift Analysis...")

# 1. Datensatz Charakteristika
dataset_chars_agg = analyze_dataset_characteristics(df_agg)

# 2. Behavior Policy Qualität
behavior_quality_agg = analyze_behavior_policy_quality(df_agg)

# 3. Hyperparameter Dokumentation
hyperparams_agg = document_hyperparameters(df_agg, discount_factors)

# 4. Distributional Shift
shift_metrics_agg = analyze_distributional_shift(df_agg, all_policies_agg, discount_factors)

# 5. Visualisierungen
main_plots_agg = create_dataset_and_shift_plots(dataset_chars_agg, df_agg, behavior_quality_agg, 
                                            shift_metrics_agg, discount_factors)
coverage_plot_agg = plot_state_action_coverage(df_agg)

# 6. Summary für Paper
println("\n" * "="^60)
println("SUMMARY FOR PAPER REPORTING")
println("="^60)

println("Dataset Characteristics:")
println("- $(dataset_chars_agg[:n_trajectories]) trajectories, $(dataset_chars_agg[:n_timesteps]) total timesteps")
println("- $(dataset_chars_agg[:n_states]) state dimensions, $(length(ACTIONS)) actions") 
println("- State-action coverage: $(round(dataset_chars_agg[:sa_coverage_density]*100, digits=2))%")
println("- Action distribution: $(round.(dataset_chars_agg[:action_probs]*100, digits=1))%")
println()

println("Behavior Policy Quality:")
println("- Mean return: $(round(mean(behavior_quality_agg[:behavior_returns]), digits=3)) ± $(round(std(behavior_quality_agg[:behavior_returns]), digits=3))")
println("- Action persistence: $(round(behavior_quality_agg[:persistence_rate]*100, digits=1))%")
println()

println("Distributional Shift (Policy Agreement Rates):")
for γ in discount_factors
    agreement = shift_metrics_agg[γ][:agreement_rate]
    kl_div = shift_metrics_agg[γ][:kl_divergence]
    println("- γ=$γ: $(round(agreement*100, digits=1))% agreement, KL=$(round(kl_div, digits=4))")
end
    

#  Discount Factor FUNKTION


# Ergebnisse zusammenfassen
println("\n" * "="^50)
println("CORE METRICS SUMMARY GLOBAL")
println("="^50)

for (i, γ) in enumerate(discount_factors)
    returns_agg = evaluate_policy_performance(df_agg, all_predict_Qs_agg[i], all_policies_agg[i], γ)
    bellman_errors_agg = compute_bellman_error(X_agg, A_agg, R_agg, Snext_agg, all_betas_agg[i], γ)

    println("γ = $γ:")
    println("  Mean Return: $(round(mean(returns_agg), digits=3)) ± $(round(std(returns_agg), digits=3))")
    println("  Mean Bellman Error: $(round(mean(bellman_errors_agg), digits=6))")
    println("  Median Bellman Error: $(round(median(bellman_errors_agg), digits=6))")
    println()
end

# Plots erstellen
println("Creating visualizations...")

p1 = plot_performance_comparison(df_agg, discount_factors, all_policies_agg, all_predict_Qs_agg)
p2 = plot_bellman_errors(discount_factors, X_agg, A_agg, R_agg, Snext_agg, all_betas_agg)

# Discount Factor Analyse für Fitted Q-Iteration



println("Starting Comprehensive Discount Factor Analysis...")
println("γ values: $discount_factors")

# 2. Konvergenz Analysis  
println("\n2. Running Convergence Analysis...")
convergence_plots_agg, convergence_data_agg = plot_convergence_comparison(df_agg, discount_factors)

    
println("Starting Comprehensive Trajectory-Level Analysis...")
println("γ values: $discount_factors")
println("Number of policies: $(length(all_policies_agg))")

# 1. Trajektorie-spezifische Analyse
println("\n1. Analyzing trajectory-specific policies...")
trajectory_results_agg = analyze_trajectory_level_policies(df_agg, all_policies_agg, discount_factors)

# 2. State-spezifische Q-Value Analyse
println("\n2. Analyzing state-specific Q-values...")
state_analysis_agg = analyze_state_specific_qvalues(df_agg, all_policies_agg, 
                                                all_predict_Qs_agg, discount_factors)

# 3. Policy Agreement Matrix
println("\n3. Computing policy agreement matrix...")
agreement_matrix_agg, policy_decisions_agg = compute_policy_agreement_matrix(df_agg, all_policies_agg, discount_factors)

# 4. Temporal Dynamics
println("\n4. Analyzing temporal policy dynamics...")
temporal_analysis_agg, time_points_agg = analyze_temporal_policy_dynamics(df_agg, all_policies_agg, discount_factors)

# 5. Visualisierungen -> debugging
println("\n5. Creating visualizations...")
#trajectory_plots_agg = create_trajectory_level_plots(trajectory_results_agg, state_analysis_agg,
                                          #      agreement_matrix_agg, temporal_analysis_agg,
                                           #     discount_factors, time_points_agg)

# 6. Statistische Analyse
println("\n6. Running statistical analysis...")
change_rates_agg = trajectory_statistical_analysis(trajectory_results_agg, state_analysis_agg , discount_factors)

pop_grid_agg = population_grid(df_agg)
s_sample_agg = make_population_s_sample(df_agg, pop_grid_agg, scols_agg)

println("We have a sample! Number of states: ", length(s_sample_agg))
# run bootstrap
B= 200;

println("Starting bootstrap FQI with B=$B samples and γ=0.6...")
betas_boot_agg = bootstrap_fqi(df_agg; B=B, γ=0.6)
println("Completed bootstrap FQI with $B samples and γ=0.6.")

p_win_agg, CIs_agg, Qdist_agg = analyze_bootstrap_Q(betas_boot_agg, s_sample_agg, 1)
println("p_win (actions × states): ", p_win_agg)
println("CIs example: ", CIs_agg)
 #   return (betas, betas_boot, p_win, CIs, Qdist)

#betas, betas_boot, p_win, CIs, Qdist = main(filled_df; B=10, rng_seed=111)  # adjust B as needed (100 for quick test, 200+ for better accuracy)

println("GLOBAL AGGREGATED REWARD------------------------------------------")

summarize_results(p_win_agg, CIs_agg, backend=:latex)
summarize_results(p_win_agg, CIs_agg)


plot_action_preferences(p_win_agg, CIs_agg)
# Convergence Statistics

# l-norm
plot_beta_stability(betas_boot_agg)


plot_Q_distributions(Qdist_agg, 1, 1)  # example state index and stage for now


bellman_errors_agg = compute_bellman_errors(betas_boot_agg, df_agg; γ=0.6)
plot_bellman_errors(bellman_errors_agg)



B_b_agg = length(betas_boot_agg)  # number of bootstrap samples
pop_grid_agg, p_win_grid_agg = population_policy_curve(betas_boot_agg, df_agg; t=1, ngrid=25)
plot_population_policy_curve(pop_grid_agg, p_win_grid_agg; B=B_b_agg)

