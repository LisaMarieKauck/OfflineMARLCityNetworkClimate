#root = dirname(@__FILE__)
#joinpath(root, "Data", "databrief-signatories.csv")

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

end

## Create a scatter plot of the signatories in EU and Europe
scatter(
    signatories_EU.organisation_longitude, signatories_EU.organisation_latitude,
    proj = :mercator,
    xlims = (-10, 40), # Roughly covers Europe
    ylims = (35, 65),
    xlabel = "Longitude",
    ylabel = "Latitude",
    title = "Signatories in EU",
    legend = false,
    markersize = 4,
    markercolor = :blue,
    background_color = :lightgray
)

scatter(
    signatories_Europe.organisation_longitude, signatories_Europe.organisation_latitude,
    proj = :mercator,
    xlims = (-25, 50), # Roughly covers Europe
    ylims = (25, 70),
    xlabel = "Longitude",
    ylabel = "Latitude",
    title = "Signatories in Europe",
    legend = false,
    markersize = 4,
    markercolor = :blue,
    background_color = :lightgray
)

## Overview Population size in EU


bins = [0, 10000, 50000, 100000, 500000, 1000000, Inf]

# Labels for the bins
labels = [
    "0-10k",
    "10k-50k",
    "50k-100k",
    "100k-500k",
    "500k-1M",
    "> 1M"
]

pop_cut_vector = cut(signatories_EU.population_adhesion, bins, labels=labels)
prop_pop_vector = countmap(pop_cut_vector)
x = [prop_pop_vector[label] for label in labels] 
x = x./ sum(x)

# Plot pie chart
Plots.pie(labels, x, title="Population sizes of Signatories")
round.(x.*100, digits=2)

## country distribution
### number of cities per country
country_counts = countmap(lowercase.(signatories_EU.country_code))
Plots.bar(country_counts, title = "Cities per country")
findmax(country_counts)
sort(collect(country_counts), by = x -> x[2], rev = true)[1:10]
sort(collect(countmap(lowercase.(signatories_EU.country_name))), by = x -> x[2], rev = true)[1:10]
### population distribution per country
groupby(signatories_EU, "country_name")
sum_without_missing(col) = sum(skipmissing(col))
GCOM_pop_by_country_EU = combine(groupby(signatories_EU, "country_name"), :population_adhesion => sum_without_missing)

real_pop_by_country_EU = CSV.read("Data/EU_population_2024.csv", delim=',', DataFrame)
#### Mapping: Deutsche Namen → Englische Namen (EU-27)
de_to_en = Dict(
    "Belgien" => "Belgium",
    "Bulgarien" => "Bulgaria",
    "Dänemark" => "Denmark",
    "Deutschland" => "Germany",
    "Estland" => "Estonia",
    "Finnland" => "Finland",
    "Frankreich" => "France",
    "Griechenland" => "Greece",
    "Irland" => "Ireland",
    "Italien" => "Italy",
    "Kroatien" => "Croatia",
    "Lettland" => "Latvia",
    "Litauen" => "Lithuania",
    "Luxemburg" => "Luxembourg",
    "Malta" => "Malta",
    "Niederlande" => "Netherlands",
    "Österreich" => "Austria",
    "Polen" => "Poland",
    "Portugal" => "Portugal",
    "Rumänien" => "Romania",
    "Schweden" => "Sweden",
    "Slowakei" => "Slovakia",
    "Slowenien" => "Slovenia",
    "Spanien" => "Spain",
    "Tschechien" => "Czechia",
    "Ungarn" => "Hungary",
    "Zypern" => "Cyprus"
)
# Neue Spalte mit englischen Ländernamen
real_pop_by_country_EU.Land_en = [get(de_to_en, land, "UNKNOWN") for land in real_pop_by_country_EU.geo]

GCOM_pop_by_country_EU = leftjoin(GCOM_pop_by_country_EU, real_pop_by_country_EU[:, ["Land_en", "OBS_VALUE"]], on = "country_name" => "Land_en")
rename!(GCOM_pop_by_country_EU, "OBS_VALUE" => "real_population")
bar(GCOM_pop_by_country_EU.country_name, GCOM_pop_by_country_EU.real_population, label="Real Population", legend=:topright)
bar!(GCOM_pop_by_country_EU.country_name, GCOM_pop_by_country_EU.population_adhesion_sum_without_missing, label="Adhesion Sum")

bar(
    GCOM_pop_by_country_EU.country_name,
    [GCOM_pop_by_country_EU.population_adhesion_sum_without_missing GCOM_pop_by_country_EU.real_population],   
    bar_position = :stack,
    label = ["Adhesion Sum" "Real Population"],
    title = "Population Comparison per Country",
    xlabel = "Country",
    ylabel = "Population",
    bar_width = 0.7,
    legend = :topright,
    size = (1000, 600),
    rotation = 45,
    framestyle = :box
)

###signatory_status
signatory_status_counts = countmap(signatories_EU.signatory_status[ismissing.(signatories_EU.population_adhesion).==false])

#-------------------
# Step 2: Action Plans Data
## Read and prep data
wholedata2 = XLSX.readxlsx("Data/df2_Action_Plans.xlsx")
meta_action_plan = DataFrame(XLSX.readtable("Data/df2_Action_Plans.xlsx", "df2_METADATA", infer_eltypes=true))

# TODO Liste

function check_missings(data)
    for name in names(data)
        println(name, ": ", sum(data[!, name] .== "-"))
    end 
end

function create_model_data_action_plan()    
    action_plans_id = DataFrame(XLSX.readtable("Data/df2_Action_Plans.xlsx", "2", infer_eltypes=true))
    action_plans_id_subset = select(action_plans_id, :organisation_id, :action_plan_id, :number_of_group_members, :approval_date)

    action_mitigation = DataFrame(XLSX.readtable("Data/df2_Action_Plans.xlsx", "2A_2", infer_eltypes=true))
    action_mitigation = select!(action_mitigation, :action_plan_id, :co2_target, :action_sector, :mitigation_action_number, :estimates_co2_reduction, :estimates_energy_savings, :estimates_energy_production)
    action_mitigation.mitigation_action_number = [x == "-" ? missing : Int64(x) for x in action_mitigation.mitigation_action_number]
    action_mitigation.estimates_energy_savings = [x == "-" ? missing : Float64(x) for x in action_mitigation.estimates_energy_savings]
    action_mitigation.estimates_energy_production = [x == "-" ? missing : Float64(x) for x in action_mitigation.estimates_energy_production]

    action_adaption = DataFrame(XLSX.readtable("Data/df2_Action_Plans.xlsx", "2B_1", infer_eltypes=true))
    action_adaption = select!(action_adaption, :action_plan_id, :base_year, :target_year, :climate_hazard_addressed)
    action_adaption.climate_hazard_addressed = [x == "-" ? missing : x for x in action_adaption.climate_hazard_addressed]

    action_adaption_2 = DataFrame(XLSX.readtable("Data/df2_Action_Plans.xlsx", "2B_2", infer_eltypes=true))

    actions_details = DataFrame(XLSX.readtable("Data/df2_Action_Plans.xlsx", "2C", infer_eltypes=true))
    actions_details = select!(actions_details, :action_plan_id, :action_id, :mitigation_flag, :adaptation_flag, :energy_poverty_flag, :timeframe_start, :timeframe_end, :action_implementation_status, :action_stakeholders, :financing_sources, :mitigation_action_sector, :mitigation_action_area, :mitigation_action_instrument, :mitigation_estimated_impact_co2_reduction, :adaptation_hazards, :adaptation_sectors, :energy_poverty_details_macro_areas, :mitigation_details_vulnerability_group_targeter, :adaptation_details_vulnerability_group_targeter, :energy_poverty_details_vulnerability_group_targeter)
    actions_details.timeframe_start = [x == "-" ? missing : x for x in actions_details.timeframe_start]
    actions_details.timeframe_end = [x == "-" ? missing : x for x in actions_details.timeframe_end]
    actions_details.action_stakeholders = [x == "-" ? missing : x for x in actions_details.action_stakeholders]
    actions_details.financing_sources = [x == "-" ? missing : x for x in actions_details.financing_sources]
    actions_details.mitigation_action_sector = [x == "-" ? missing : x for x in actions_details.mitigation_action_sector]
    actions_details.mitigation_action_area = [x == "-" ? missing : x for x in actions_details.mitigation_action_area]
    actions_details.mitigation_action_instrument = [x == "-" ? missing : x for x in actions_details.mitigation_action_instrument]
    actions_details.mitigation_estimated_impact_co2_reduction = [x == "-" ? missing : Float64(x) for x in actions_details.mitigation_estimated_impact_co2_reduction]
    actions_details.adaptation_hazards = [x == "-" ? missing : x for x in actions_details.adaptation_hazards]
    actions_details.adaptation_sectors = [x == "-" ? missing : x for x in actions_details.adaptation_sectors]
    actions_details.energy_poverty_details_macro_areas = [x == "-" ? missing : x for x in actions_details.energy_poverty_details_macro_areas]
    actions_details.mitigation_details_vulnerability_group_targeter = [x == "-" ? missing : x for x in actions_details.mitigation_details_vulnerability_group_targeter]
    actions_details.adaptation_details_vulnerability_group_targeter = [x == "-" ? missing : x for x in actions_details.adaptation_details_vulnerability_group_targeter]
    actions_details.energy_poverty_details_vulnerability_group_targeter = [x == "-" ? missing : x for x in actions_details.energy_poverty_details_vulnerability_group_targeter]

    inventory_energy_consumption = DataFrame(XLSX.readtable("Data/df2_Action_Plans.xlsx", "2D_1", infer_eltypes=true))
    inventory_energy_consumption = select!(inventory_energy_consumption, :action_plan_id, :inventory_year, :population_in_the_inventory_year, :aggregated_sector, :aggregated_carrier, :energy_measure, :emission_measure)

    inventory_energy_supply = DataFrame(XLSX.readtable("Data/df2_Action_Plans.xlsx", "2D_2", infer_eltypes=true))
    inventory_energy_supply = select!(inventory_energy_supply, :action_plan_id, :inventory_year, :energy_supply_type, :energy_carrier, :energy_output, :emission_measure)

    inventory_adaption_1 = DataFrame(XLSX.readtable("Data/df2_Action_Plans.xlsx", "2E_1", infer_eltypes=true))
    inventory_adaption_2 = DataFrame(XLSX.readtable("Data/df2_Action_Plans.xlsx", "2E_2", infer_eltypes=true))
    inventory_adaption_2.step2_rva_vulnerable_sector_level = [x == "-" ? missing : x for x in inventory_adaption_2.step2_rva_vulnerable_sector_level]
    inventory_adaption_3 = DataFrame(XLSX.readtable("Data/df2_Action_Plans.xlsx", "2E_3", infer_eltypes=true))
    inventory_adaption_3.step3_rva_adaptive_capacity_level = [x == "-" ? missing : x for x in inventory_adaption_3.step3_rva_adaptive_capacity_level]
    inventory_adaption_4 = DataFrame(XLSX.readtable("Data/df2_Action_Plans.xlsx", "2E_4", infer_eltypes=true))

    inventory_adaption_stepboard = DataFrame(XLSX.readtable("Data/df2_Action_Plans.xlsx", "2E_6", infer_eltypes=true))
    inventory_adaption_stepboard = select!(inventory_adaption_stepboard, :action_plan_id, :adaptation_scoreboard_actions, :adaptation_scoreboard_self_check_of_the_status)

    energy_poverty = DataFrame(XLSX.readtable("Data/df2_Action_Plans.xlsx", "2F_2", infer_eltypes=true))
    energy_poverty = select!(energy_poverty, :action_plan_id, :energy_poverty_macro_area, :energy_poverty_action_number)

    inventory_energy_poverty = DataFrame(XLSX.readtable("Data/df2_Action_Plans.xlsx", "2G", infer_eltypes=true))
    inventory_energy_poverty = select!(inventory_energy_poverty, :action_plan_id, :macro_area, :current_level, :target_level)
    inventory_energy_poverty = inventory_energy_poverty[inventory_energy_poverty.current_level .!= "Yes", :]
    inventory_energy_poverty = inventory_energy_poverty[inventory_energy_poverty.target_level .!= "Yes", :]
    inventory_energy_poverty.current_level = [x == "-" ? missing : x for x in inventory_energy_poverty.current_level]
    inventory_energy_poverty.target_level = [x == "-" ? missing : x for x in inventory_energy_poverty.target_level]


    model_data_action_plan = outerjoin(action_plans_id_subset, 
        action_mitigation, 
        action_adaption, 
        on = :action_plan_id)

    model_data_action_plan = outerjoin(action_plans_id_subset, 
        action_adaption_2, 
        actions_details,
        on = :action_plan_id)
        
    model_data_action_plan = outerjoin(model_data_action_plan,
        inventory_energy_consumption,
        inventory_energy_supply,
        inventory_adaption_1,
        inventory_adaption_2,
        inventory_adaption_3,
        inventory_adaption_4,
        on = :action_plan_id,
        makeunique = true)

    model_data_action_plan = outerjoin(model_data_action_plan,
        inventory_adaption_stepboard,
        energy_poverty,
        inventory_energy_poverty,
        on = :action_plan_id)

    return model_data_action_plan

end

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
end

model_data_action_plan = create_model_data_action_plan() 
monitoring_data = create_monitoring_data()
signatories_EU = create_EU_data()


