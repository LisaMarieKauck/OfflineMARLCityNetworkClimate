# Step 1: Signatories Data
## Read and prep data
function create_EU_data()

    wholedata = XLSX.readxlsx("Data/df1_Signatories_2024.xlsx")
    signatories = DataFrame(XLSX.readtable("Data/df1_Signatories_2024.xlsx", "1", infer_eltypes=true))

    replace!(signatories.organisation_latitude, "null" => missing)
    replace!(signatories.organisation_latitude, "-" => missing)

    replace!(signatories.organisation_longitude, "null" => missing)
    replace!(signatories.organisation_longitude, "-" => missing)

    signatories.organisation_latitude[isa.(signatories.organisation_latitude, String)] = parse.(Float64, replace.(signatories.organisation_latitude[isa.(signatories.organisation_latitude, String)],"," => "."))

    signatories.organisation_longitude[isa.(signatories.organisation_longitude, String)] = parse.(Float64, replace.(signatories.organisation_longitude[isa.(signatories.organisation_longitude, String)],"," => "."))

    replace!(signatories.population_adhesion, "-" => missing)

    ## filter EU and Europe signatories
    EU_country_codes = ["at", "be", "bg", "hr", "cy", "cz", "dk", "ee", "fi", "fr", "de", "gr", "hu", "ie", "it", "lv", "lt", "lu", "mt", "nl", "pl", "pt", "ro", "sk", "si", "es", "se"]
    Europe_country_codes = vcat(EU_country_codes, ["al", "ad", "am", "by", "ba", "fo", "ge", "gi", "is", "im", "xk", "li", "mk", "md", "mc", "me", "no", "ru", "sm", "rs", "ch", "tr", "ua", "gb", "va"])

    signatories_EU = filter(row -> lowercase(row.country_code) in EU_country_codes, signatories)
   # signatories_Europe = filter(row -> lowercase(row.country_code) in Europe_country_codes, signatories)

    return signatories_EU
end;

function create_monitoring_data()
    monitoring_id = DataFrame(XLSX.readtable("Data/df3_Monitoring.xlsx", "3", infer_eltypes=true))
    monitoring_id = select!(monitoring_id, :organisation_id, :action_plan_id, :monitoring_report_id, :monitoring_report_type, :monitoring_report_submission_status, :monitoring_report_submission_date)
    monitoring_id.monitoring_report_submission_date = [x == "-" ? missing : x for x in monitoring_id.monitoring_report_submission_date]

    monitoring_mitigation = DataFrame(XLSX.readtable("Data/df3_Monitoring.xlsx", "3A", infer_eltypes=true))
    monitoring_mitigation.mitigation_ongoing = [x == "-" ? missing : x for x in monitoring_mitigation.mitigation_ongoing]
    monitoring_mitigation.mitigation_completed = [x == "-" ? missing : x for x in monitoring_mitigation.mitigation_completed]
    monitoring_mitigation.mitigation_postponed = [x == "-" ? missing : x for x in monitoring_mitigation.mitigation_postponed]
    monitoring_mitigation.mitigation_notstarted = [x == "-" ? missing : x for x in monitoring_mitigation.mitigation_notstarted]

    monitoring_adaption = DataFrame(XLSX.readtable("Data/df3_Monitoring.xlsx", "3B_2", infer_eltypes=true))
    monitoring_adaption.adaptation_ongoing = [x == "-" ? missing : x for x in monitoring_adaption.adaptation_ongoing]
    monitoring_adaption.adaptation_completed = [x == "-" ? missing : x for x in monitoring_adaption.adaptation_completed]
    monitoring_adaption.adaptation_postponed = [x == "-" ? missing : x for x in monitoring_adaption.adaptation_postponed]
    monitoring_adaption.adaptation_notstarted = [x == "-" ? missing : x for x in monitoring_adaption.adaptation_notstarted]

    monitoring_energy_consumption = DataFrame(XLSX.readtable("Data/df3_Monitoring.xlsx", "3D_1", infer_eltypes=true))
    monitoring_energy_consumption = select!(monitoring_energy_consumption, :monitoring_report_id, :inventory_year, :population_in_the_inventory_year, :aggregated_carrier, :energy_measure, :emission_measure)

    monitoring_data = outerjoin(monitoring_id, monitoring_mitigation, monitoring_adaption, monitoring_energy_consumption, on = :monitoring_report_id, makeunique = true)
    return monitoring_data
end;

monitoring_data = create_monitoring_data();
signatories_EU = create_EU_data();

monitoring_data.adaptation_action_sector = [ismissing(x) ? "0" : x for x in monitoring_data.adaptation_action_sector]
hot_encode_action_sector = select(monitoring_data, [:adaptation_action_sector => ByRow(isequal(v))=> Symbol(v) for v in unique(monitoring_data.adaptation_action_sector)]).*1
hot_encode_action_sector = hot_encode_action_sector[: ,2:end]
for c in names(hot_encode_action_sector)
    hot_encode_action_sector[!, c] = coalesce.(hot_encode_action_sector[!, c], 0)
end
monitoring_data = hcat(monitoring_data, hot_encode_action_sector)



# Step x: Combine datasets for modeling
signatories_EU_description_data = select(signatories_EU, :organisation_id, :organisation_name, :country_name, :organisation_latitude, :organisation_longitude, :signatory_status)

model_data = select(signatories_EU, :organisation_id, :population_adhesion, :organisation_latitude, :organisation_longitude, :signatory_status, :adhesion_type, :date_of_adhesion, :group_profile)

model_data_2 = leftjoin(model_data, monitoring_data, on = :organisation_id)
#sort(unique(model_data_2.inventory_year))
for name in names(hot_encode_action_sector)
    print(", :", name)
end
model_data_subset = select(model_data_2, :organisation_id, :inventory_year, :signatory_status, :group_profile, :mitigation_action_number, :adaptation_action_number, :aggregated_carrier, :emission_measure, :population_in_the_inventory_year, :mitigation_completed, :mitigation_ongoing, :mitigation_postponed, :mitigation_notstarted, :adaptation_completed, :adaptation_ongoing, :adaptation_postponed, :adaptation_notstarted, :health, :environment, :land, :water, :buildings, :education, :civil, :agriculture, :transport, :energy, :waste, :tourism, :ict);

df = DataFrame();
df.traj_id = string.(model_data_subset.organisation_id);
df.t = model_data_subset.inventory_year;
model_data_subset.adaptation_action_number = coalesce.(model_data_subset.adaptation_action_number, 0);
model_data_subset.mitigation_action_number = coalesce.(model_data_subset.mitigation_action_number, 0);
df.action = ifelse.(model_data_subset.adaptation_action_number .> 0, 1, 0) .+ ifelse.(model_data_subset.mitigation_action_number .> 0, 2, 0);
#state variables
df.population = model_data_subset.population_in_the_inventory_year;
df.signatory_status = model_data_subset.signatory_status;
df.group_profile = model_data_subset.group_profile;
df.health = model_data_subset.health;
df.environment = model_data_subset.environment;
df.land = model_data_subset.land;
df.water = model_data_subset.water;
df.buildings = model_data_subset.buildings;
df.education = model_data_subset.education;
df.civil = model_data_subset.civil;
df.agriculture = model_data_subset.agriculture;
df.transport = model_data_subset.transport;
df.energy = model_data_subset.energy;
df.waste = model_data_subset.waste;
df.tourism = model_data_subset.tourism;
df.ict = model_data_subset.ict;
# reward but also state variable
df.reward = model_data_subset.emission_measure ./ model_data_subset.population_in_the_inventory_year;
df = dropmissing(df, :t);

#prüfen, ob es pro (traj_id, t) nur einen action und population wert gibt
gdf = groupby(df, [:traj_id, :t]);
summary_df = combine(gdf, 
    :action => (x -> length(unique(x))) => :n_unique_action,
    :population => (x -> length(unique(x))) => :n_unique_population
);
# Filtern, wo mehr als ein Wert für actionvorkommt
conflicts = summary_df[(summary_df.n_unique_action .> 1) .| (summary_df.n_unique_population .> 1), :]
df_subset = subset(df, :traj_id => (x -> x .∈ Ref(conflicts.traj_id)))

# lets group
df_grouped = combine(groupby(df, [:traj_id, :t]), 
    :reward => sum => :reward, 
    :action => maximum => :action,
    :population => mean => :population,
    :signatory_status => mode => :signatory_status,
    :group_profile => mode => :group_profile,
    :health => sum => :health,
    :environment => sum => :environment,
    :land => sum => :land,
    :water => sum => :water,
    :buildings => sum => :buildings,
    :education => sum => :education,
    :civil => sum => :civil,
    :agriculture => sum => :agriculture,
    :transport => sum => :transport,
    :energy => sum => :energy,
    :waste => sum => :waste,
    :tourism => sum => :tourism,
    :ict => sum => :ict)



function fill_missing_years(df)
    result = DataFrame(traj_id=eltype(df.traj_id)[], 
                       t=Int[],  
                       action=Int[], 
                       reward=Float64[],
                       population=Float64[],
                       emission_per_capita=Float64[],
                       signatory_status=String[],
                       group_profile=Bool[],
                       health=Int[],
                       environment=Int[],
                       land=Int[],
                       water=Int[],
                       buildings=Int[],
                       education=Int[],
                       civil=Int[],
                       agriculture=Int[],
                       transport=Int[],
                       energy=Int[],
                       waste=Int[],
                       tourism=Int[],
                       ict=Int[])

    for traj in unique(df.traj_id)
        subdf = df[df.traj_id .== traj, :]
        years = sort(subdf.t)
        rewards = subdf.reward[sortperm(subdf.t)]
        actions = subdf.action[sortperm(subdf.t)]
        #state variables
        populations = subdf.population[sortperm(subdf.t)]
        emission_per_capita = subdf.reward[sortperm(subdf.t)]
        signatory_status = subdf.signatory_status[sortperm(subdf.t)]
        group_profile = subdf.group_profile[sortperm(subdf.t)]
        health = subdf.health[sortperm(subdf.t)]
        environment = subdf.environment[sortperm(subdf.t)]
        land = subdf.land[sortperm(subdf.t)]
        water = subdf.water[sortperm(subdf.t)]
        buildings = subdf.buildings[sortperm(subdf.t)]
        education = subdf.education[sortperm(subdf.t)]
        civil = subdf.civil[sortperm(subdf.t)]
        agriculture = subdf.agriculture[sortperm(subdf.t)]
        transport = subdf.transport[sortperm(subdf.t)]
        energy = subdf.energy[sortperm(subdf.t)]
        waste = subdf.waste[sortperm(subdf.t)]
        tourism = subdf.tourism[sortperm(subdf.t)]
        ict = subdf.ict[sortperm(subdf.t)]


        first_year = minimum(years)
        last_year  = maximum(years)

        for year in 1990:2023
            if year in years
                # vorhandenes Jahr übernehmen
                idx = findfirst(==(year), years)
                push!(result, (traj, year, actions[idx], rewards[idx], populations[idx], rewards[idx], signatory_status[idx], group_profile[idx], health[idx], environment[idx], land[idx], water[idx], buildings[idx], education[idx], civil[idx], agriculture[idx], transport[idx], energy[idx], waste[idx], tourism[idx], ict[idx]))
            elseif year < first_year
                # Jahr vor erstem bekannten Jahr: ersten Wert verwenden
                push!(result, (traj, year, 0, 0, populations[1], rewards[1], "on_hold",  group_profile[1], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
            elseif year > last_year
                # Jahr nach letztem bekannten Jahr: letzten Wert verwenden
                push!(result, (traj, year, 0, 0, populations[end], rewards[end], "on_hold", group_profile[end], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
            else
                # Jahr zwischen bekannten Jahren: letzten bekannten Wert nehmen
                idx = findlast(<(year), years)
                push!(result, (traj, year, 0, 0, populations[idx], rewards[idx], "on_hold", group_profile[idx], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
            end
        end
    end
    
    return result
end;


filled_df = fill_missing_years(df_grouped)
filled_df.reward = zeros(Float64, nrow(filled_df))
filled_df = DataFrames.transform(groupby(filled_df, :traj_id), :emission_per_capita => (x -> -1 .* [0.0; diff(x)]) => :reward)
filled_df.signatory_status = ifelse.(filled_df.signatory_status .== "on_hold", 0, 1)
filled_df.group_profile = filled_df.group_profile .*1

df_discrete = DataFrame()
df_discrete.traj_id = filled_df.traj_id
df_discrete.t = filled_df.t
df_discrete.action = filled_df.action
df_discrete.reward = filled_df.reward
#Emission per capita: Quintile
df_discrete.emission_per_capita = Int64.(levelcode.(cut(filled_df.emission_per_capita, 5; labels=1:5)))
# countmap(df_discrete.emission_per_capita)
# Binäre bleibt binär
df_discrete.signatory_status = filled_df.signatory_status
df_discrete.group_profile = filled_df.group_profile
# population: same as reported in data section
function bin_population(pop)
    if pop < 10_000
        return 1
    elseif pop < 50_000
        return 2
    elseif pop < 100_000
        return 3
    else
        return 4
    end
end
df_discrete.population = bin_population.(filled_df.population)
counts = countmap(df_discrete.population)
vals = [counts[k] for k in sort(collect(keys(counts)))]
Plots.bar(vals ./ 34, legend=false, xlabel="Population Bin", ylabel="Count", title="Distribution of Population Bins", xticks=(1:4, ["<10k", "10k-50k", "50k-100k", ">100k"]), color=:lightblue, yformatter = :plain)
# action sectors
df_discrete.literacy = filled_df.education + filled_df.civil
df_discrete.infrastructure = filled_df.buildings + filled_df.transport + filled_df.energy + filled_df.health + filled_df.ict
df_discrete.environmental = filled_df.environment + filled_df.land + filled_df.water + filled_df.agriculture + filled_df.tourism + filled_df.waste
df_discrete.literacy = ifelse.(df_discrete.literacy .> 0, 1, 0)
df_discrete.infrastructure = ifelse.(df_discrete.infrastructure .> 0, 1, 0)
df_discrete.environmental = ifelse.(df_discrete.environmental .> 0, 1, 0)


#subset(df, :t => ByRow(ismissing))
#subset(subset(df, :reward => ByRow(ismissing)), :t => ByRow(!ismissing))
#subset(filled_df, :reward => ByRow(ismissing))

# aggregated reward over window W years
function aggregate_rewards(df::DataFrame, windows::Vector{Tuple{Int,Int}})
    # define aggregation specification
    agg_spec = [
        :reward => sum => :reward,
        :action => maximum => :action,
        :population => mean => :population,
        :emission_per_capita => mean => :emission_per_capita,
        :signatory_status => maximum => :signatory_status,
        :group_profile => maximum => :group_profile,
        :literacy => sum => :literacy,
        :infrastructure => sum => :infrastructure,
        :environmental => sum => :environmental
    ]

    df_agg = DataFrame()

 for gid in unique(df.traj_id)
        sub = filter(r -> r.traj_id == gid, df)
        for (start_year, end_year) in windows
            rows = filter(r -> (r.t >= start_year && r.t <= end_year), sub)
            if nrow(rows) == 0
                continue
            end
            grouped = combine(groupby(rows, [:traj_id]), agg_spec...)
            grouped.t .= start_year  # set aggregated time to window start
            append!(df_agg, grouped)
        end
    end

    # enforce column order
    desired_order = [
        :traj_id, :t, :action, :reward, :population, :emission_per_capita,
        :signatory_status, :group_profile, :literacy, :infrastructure, :environmental
    ]
    return df_agg[:, desired_order]
end

windows = [(1990,2008), (2009,2012), (2013,2015), (2016,2019), (2020,2023)]
df_agg_discrete = aggregate_rewards(df_discrete, windows)

#plot(filled_df.t, filled_df.reward)