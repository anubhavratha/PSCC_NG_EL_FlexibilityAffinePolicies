# Running out-of-sample simulations to determine MRR for the system

using JuMP, Distributions, LinearAlgebra, DataFrames, Mosek, MosekTools

using CSV, DataFrames

#Import test data DataFrame
N_tr = 1000     #TrainingDataSamples
N_tot = 10000   #TotalNumberDataSamples
wind_realizations = CSV.read("CS1_24bus/oos_data/OOS_WindRealizations_TestData.csv", header=false)
# Transform wind_realizations dataframe to be used well
wind_realizations[!,:WFNum] = repeat([1], (N_tot-N_tr)*2)
wind_realizations[(N_tot-N_tr)+1:(N_tot-N_tr)*2,:WFNum] = repeat([2],(N_tot-N_tr))
wind_realizations[!, :ScenNum] = 1:(N_tot-N_tr)*2
wind_realizations[(N_tot-N_tr)+1:(N_tot-N_tr)*2, :ScenNum] = 1:(N_tot-N_tr)

# Case Study 1: 24el + 12ng Bus System : Prepare and load data
include("CS1_24bus/CS1_data_load_script_PSCC.jl")
(elBus_data, gen_data, elLine_data ,B , f̅, ν, π, refbus, ng_prods_data, ngBus_data, ngLine_data, wind_data) = load_data()
hourly_demand = CSV.read("CS1_24bus/data/24el_12ng/hourlyDemand.csv")


Np = length(gen_data)           #number of generators
Ng = length(ng_prods_data)      #number of gas producers
Nw = length(wind_data)          #number of wind power producers
Nel_line = length(elLine_data)  #number of power lines (l)
Nel_bus = length(elBus_data)    #number of power buses (elnode)
Nng_line = length(ngLine_data)  #number of gas pipelines (pl)
Nng_bus = length(ngBus_data)    #number of gas buses (gnode)

Nt = 24                       #Time periods

C_shed = 2000
C_spill = 0

include("M1_Deterministic_EL_NG_Coordination.jl")

function determine_MRR_Deterministic(m1_elprod, m1_elflows, m1_windgen, m1_ng_lmp, Scenario)
    m_rt = Model(with_optimizer(Mosek.Optimizer, MSK_IPAR_LOG=1, MSK_IPAR_INTPNT_SOLVE_FORM=MSK_SOLVE_PRIMAL, MSK_DPAR_INTPNT_CO_TOL_REL_GAP=1.0e-10, MSK_IPAR_PRESOLVE_ELIMINATOR_MAX_NUM_TRIES = 0))
    #EL Variables
    @variable(m_rt, r_act[1:Np, 1:Nt])
    @variable(m_rt, l_shed[1:Nel_bus, 1:Nt])
    @variable(m_rt, w_spill[1:Nw, 1:Nt])
    @variable(m_rt, f_adj[1:Nel_line, 1:Nt])
    @variable(m_rt, θ_adj[1:Nel_bus, 1:Nt])

    @objective(m_rt, Min, sum(sum(r_act[i,t]*gen_data[i].c for i=1:Np if gen_data[i].ngBusNum==0) + sum(r_act[i,t]*gen_data[i].ngConvEff*m1_ng_lmp[gen_data[i].ngBusNum,t] for i=1:Np if gen_data[i].ngBusNum>0) + sum(C_shed*l_shed[elnode,t] for elnode = 1: Nel_bus) + sum(C_spill*w_spill[i,t] for i = 1:Nw) for t=1:Nt))

    @constraint(m_rt, ϕ_r_act[i=1:Np, t=1:Nt], -(m1_elprod[(m1_elprod.gen .== i) .& (m1_elprod.hour .== t), :r_dn][1]) <=  r_act[i,t] <= m1_elprod[(m1_elprod.gen .== i) .& (m1_elprod.hour .== t), :r_up][1])
    @constraint(m_rt, ϕ_p_lims[i=1:Np, t=1:Nt], gen_data[i].p̲ <= m1_elprod[(m1_elprod.gen .== i) .& (m1_elprod.hour .== t), :p][1] + r_act[i,t] <= gen_data[i].p̅)
    @constraint(m_rt, ϕ_l_shed[elnode=1:Nel_bus, t=1:Nt], 0 <= l_shed[elnode,t] <= elBus_data[elnode].elLoadShare*hourly_demand[t,2])
    @constraint(m_rt, ϕ_w_spill[i=1:Nw, t=1:Nt], 0 <= w_spill[i,t] <= wind_realizations[(wind_realizations.WFNum.==i) .& (wind_realizations.ScenNum .== Scenario), t][1])

    @constraint(m_rt, el_f_lim_rt[l=1:Nel_line,t=1:Nt], -f̅[elLine_data[l].b_f, elLine_data[l].b_t] <= f_adj[l,t] <= f̅[elLine_data[l].b_f, elLine_data[l].b_t])

    @constraint(m_rt, λ_el_rt[elnode=1:Nel_bus, t=1:Nt], sum(r_act[i,t] for i in 1:Np if gen_data[i].elBusNum==elnode)
                                                    + sum((wind_realizations[(wind_realizations.WFNum.==i) .& (wind_realizations.ScenNum .== Scenario), t][1] - m1_windgen[i,t] - w_spill[i,t]) for i in 1:Nw if wind_data[i].elBusNum==elnode)
                                                   + l_shed[elnode,t]  - sum(B[elnode,r]*θ_adj[r,t] for r=1:Nel_bus) == 0)

    @constraint(m_rt, el_f_def[l=1:Nel_line, t=1:Nt], f_adj[l,t] == ν[elLine_data[l].b_f,elLine_data[l].b_t]*(θ_adj[elLine_data[l].b_f,t] - θ_adj[elLine_data[l].b_t,t]))
    @constraint(m_rt, el_f_lim[l=1:Nel_line,t=1:Nt], -f̅[elLine_data[l].b_f, elLine_data[l].b_t] <= (m1_elflows[l,t] + f_adj[l,t]) <= f̅[elLine_data[l].b_f, elLine_data[l].b_t])

    #5. El Reference bus
    @constraint(m_rt, ref_el_rt[t=1:Nt], θ_adj[refbus, t] == 0)

    @time optimize!(m_rt)
    status = termination_status(m_rt)
    println(status)
    println(raw_status(m_rt))

    @info("Deterministic RT Redispatch Model status ---> $(status)")

    return status, JuMP.objective_value(m_rt), round.(JuMP.value.(r_act), digits=2), JuMP.value.(l_shed), JuMP.value.(w_spill), JuMP.value.(f_adj), JuMP.dual.(λ_el_rt)
end

#(status, redispatch_cost, res_act, lshed, wspill, flow_adj, rt_shadow) = determine_MRR_Deterministic(m1_elprod, m1_elflows, m1_windgen, 2)


### Running Deterministic Model for DayAhead Optimization: Returns setpoints for power and natural gas systems ###

MRRvalue = [150,200,250,300,350,400,450,500]
#=
scen_res=DataFrame(MRRVal=Int[], ScenNum=Int[], CostRedispatch=Float64[], WindSpilled=Float64[], LoadShed=Float64[])
for i in MRRvalue
    (m1_status, m1_cost, m1_el_lmp, m1_elflows, m1_elprod, m1_windgen, m1_vangs, m1_ng_inflows, m1_ng_outflows, m1_ng_flows, m1_linepack_amount, m1_ng_pre, m1_ng_prod, m1_ng_lmp) = unidir_deterministic_SOCP_EL_NG(i)
    if m1_status != MOI.OPTIMAL
        println("Not DA feasible for MRR = ",i)
        snapshot=[i, 0, 0, 0, 0]
    elseif m1_status == MOI.OPTIMAL
        for Scenario = 1:500
            (status, redispatch_cost, res_act, lshed, wspill, flow_adj) = determine_MRR_Deterministic(m1_elprod, m1_elflows, m1_windgen, m1_ng_lmp, Scenario)
            snapshot=[i, Scenario, redispatch_cost, sum(wspill), sum(lshed)]
            push!(scen_res,snapshot)
        end
    end
end
@show scen_res
CSV.write("MRR_Results.csv", scen_res)
=#

### ========= PART 2: Process the output CSV file ====== #

SimResults_with_MRR = CSV.read("MRR_Results.csv")
NumScenarios = 500
MRR_results_summary = DataFrame(MRRValue=Int[], ExpRedispatchCost = Float64[], TotalDA_ExpRedispatchCost=Float64[], ExpWindSpillage=Float64[], ExpLoadShed=Float64[])

for i in MRRvalue
    (m1_status, m1_cost, m1_el_lmp, m1_elflows, m1_elprod, m1_windgen, m1_vangs, m1_ng_inflows, m1_ng_outflows, m1_ng_flows, m1_linepack_amount, m1_ng_pre, m1_ng_prod, m1_ng_lmp) = unidir_deterministic_SOCP_EL_NG(i)
    if m1_status != MOI.OPTIMAL
        m1_cost = 0
    end
    snapshot = [i, sum(SimResults_with_MRR[(SimResults_with_MRR.MRRVal.==i),:CostRedispatch])/NumScenarios, m1_cost + sum(SimResults_with_MRR[(SimResults_with_MRR.MRRVal.==i),:CostRedispatch])/NumScenarios, sum(SimResults_with_MRR[(SimResults_with_MRR.MRRVal.==i),:WindSpilled])/NumScenarios, sum(SimResults_with_MRR[(SimResults_with_MRR.MRRVal.==i),:LoadShed])/NumScenarios]
    push!(MRR_results_summary, snapshot)
end
@show MRR_results_summary
