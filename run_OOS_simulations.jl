# Running out of Sample Simulations and gathering results

using JuMP, Distributions, LinearAlgebra, DataFrames, Mosek, MosekTools

using CSV, DataFrames

#Import test data DataFrame
N_tr = 1000     #TrainingDataSamples
N_tot = 10000   #TotalNumberDataSamples
wind_realizations = CSV.read("CS1_24bus/oos_data/OOS_WindRealizations_TestData.csv", header=false)
# Transform wind_realizations
wind_realizations[!,:WFNum] = repeat([1], (N_tot-N_tr)*2)
wind_realizations[(N_tot-N_tr)+1:(N_tot-N_tr)*2,:WFNum] = repeat([2],(N_tot-N_tr))
wind_realizations[!, :ScenNum] = 1:(N_tot-N_tr)*2
wind_realizations[(N_tot-N_tr)+1:(N_tot-N_tr)*2, :ScenNum] = 1:(N_tot-N_tr)

#Prepare In-Sample Data
wind_traindata = CSV.read("CS1_24bus/oos_data/InSample_WindRealizations_TrainData.csv", header=false)
#transform the dataset
wind_traindata[!,:WFNum] = repeat([1], (N_tr)*2)
wind_traindata[N_tr+1:N_tr*2,:WFNum] = repeat([2],N_tr)
wind_traindata[!, :ScenNum] = 1:N_tr*2
wind_traindata[N_tr+1:N_tr*2, :ScenNum] = 1:N_tr

# Case Study 1: 24el + 12ng Bus System : Prepare and load data
include("CS1_24bus/CS1_data_load_script_PSCC.jl")
(elBus_data, gen_data, elLine_data ,B , f̅, ν, π, refbus, ng_prods_data, ngBus_data, ngLine_data, wind_data, Bflow, PTDF, PTDF_gens, PTDF_wind, PTDF_load) = load_data()
w_hat = CSV.read("CS1_24bus/data/UncertaintyMoments/PSCC_Point_Forecast_Values.csv", header=false)        #Point forecast

Np = length(gen_data)           #number of generators
Ng = length(ng_prods_data)      #number of gas producers
Nw = length(wind_data)          #number of wind power producers
Nel_line = length(elLine_data)  #number of power lines (l)
Nel_bus = length(elBus_data)    #number of power buses (elnode)
Nng_line = length(ngLine_data)  #number of gas pipelines (pl)
Nng_bus = length(ngBus_data)    #number of gas buses (gnode)

Nt = 24                     #Time periods

C_shed_el = 5000
C_shed_ng = 5000
C_spill = 0

##==== Network Data Pre-processing for PTDF Formulation =====##
Cgens = zeros(Nel_bus,Np)       #Generator matrix
for i=1:Np
    ThisGenBusNum = gen_data[i].elBusNum
    Cgens[ThisGenBusNum,i] = 1
end
Cwind = zeros(Nel_bus,Nw)       #Wind PP Matrix
for i=1:Nw
    ThisWindPPBusNum = wind_data[i].elBusNum
    Cwind[ThisWindPPBusNum,i] = 1
end
Cload = zeros(Nel_bus, size(PTDF_load,2))   #load matrix
for d in 1:size(PTDF_load,2)
    ThisLoadBusNum = [elBus_data[elnode].ind for elnode=1:Nel_bus if elBus_data[elnode].elLoadNum == d][1]
    Cload[ThisLoadBusNum, d] = 1
end
PL = zeros(Nel_line,1)      #Vector of Line Flow Limits
for l in 1:Nel_line
    PL[l,1] = elLine_data[l].f̅
end
LoadShare = zeros(size(PTDF_load,2), 1)  #Vector of LoadShares of each load
for d in 1:size(PTDF_load,2)
    LoadShare[d,1] = [elBus_data[elnode].elLoadShare for elnode=1:Nel_bus if elBus_data[elnode].elLoadNum == d][1]
end


#=
## ========================== REDISPATCH FOR DETERMINISTIC : M1 ================================= ##
### Running Deterministic Model for DayAhead Optimization: Returns setpoints for power and natural gas systems ###
include("M1_Deterministic_EL_NG_Coordination.jl")
(m1_status, m1_cost, m1_el_lmp, m1_elflows, m1_elprod, m1_windgen, m1_vangs, m1_ng_inflows, m1_ng_outflows, m1_ng_flows, m1_linepack_amount, m1_ng_pre, m1_ng_prod, m1_ng_lmp) = unidir_deterministic_SOCP_EL_NG(300)

function unidir_deterministic_redispatch(m1_elprod, m1_elflows, m1_windgen, m1_ng_lmp, m1_ng_pre, m1_ng_flows, m1_ng_prod, Scenario)
    m1_rt = Model(with_optimizer(Mosek.Optimizer, MSK_IPAR_LOG=1, MSK_IPAR_INTPNT_SOLVE_FORM=MSK_SOLVE_PRIMAL, MSK_DPAR_INTPNT_CO_TOL_REL_GAP=1.0e-10, MSK_IPAR_PRESOLVE_ELIMINATOR_MAX_NUM_TRIES = 0))

    #EL adjustment variables
    @variable(m1_rt, r_act[1:Np, 1:Nt])
    @variable(m1_rt, l_shed[1:Nel_bus, 1:Nt])
    @variable(m1_rt, w_spill[1:Nw, 1:Nt])
    @variable(m1_rt, f_adj[1:Nel_line, 1:Nt])
    @variable(m1_rt, θ_adj[1:Nel_bus, 1:Nt])

    #NG adjustment variables
    @variable(m1_rt, pr_adj[1:Nng_bus, t=1:Nt])
    @variable(m1_rt, q_adj[1:Nng_line, t=1:Nt])
    @variable(m1_rt, q_in_adj[1:Nng_line, t=1:Nt])
    @variable(m1_rt, q_out_adj[1:Nng_line, t=1:Nt])
    @variable(m1_rt, h_rt[1:Nng_line, t=1:Nt])
    @variable(m1_rt, g_adj[1:Ng, t=1:Nt])


    @objective(m1_rt, Min, sum(sum(r_act[i,t]*gen_data[i].c for i=1:Np if gen_data[i].ngBusNum==0) + sum(r_act[i,t]*gen_data[i].ngConvEff*m1_ng_lmp[gen_data[i].ngBusNum,t] for i=1:Np if gen_data[i].ngBusNum>0) + sum(C_shed*l_shed[elnode,t] for elnode = 1: Nel_bus) + sum(C_spill*w_spill[i,t] for i = 1:Nw) for t=1:Nt))
    #@objective(m_rt, Min, sum(sum(r_act[i,t]*gen_data[i].c for i=1:Np if gen_data[i].ngBusNum==0) + sum(g_adj[k,t]*ng_prods_data[k].C_gas for k= 1:Ng) + sum(C_shed*l_shed[elnode,t] for elnode = 1: Nel_bus) + sum(C_spill*w_spill[i,t] for i = 1:Nw) for t=1:Nt))

    ###-----EL Constraints----###
    @constraint(m1_rt, ϕ_r_act[i=1:Np, t=1:Nt], -(m1_elprod[(m1_elprod.gen .== i) .& (m1_elprod.hour .== t), :r_dn][1]) <=  r_act[i,t] <= m1_elprod[(m1_elprod.gen .== i) .& (m1_elprod.hour .== t), :r_up][1])
    @constraint(m1_rt, ϕ_p_lims[i=1:Np, t=1:Nt], gen_data[i].p̲ <= m1_elprod[(m1_elprod.gen .== i) .& (m1_elprod.hour .== t), :p][1] + r_act[i,t] <= gen_data[i].p̅)
    @constraint(m1_rt, ϕ_l_shed[elnode=1:Nel_bus, t=1:Nt], 0 <= l_shed[elnode,t] <= elBus_data[elnode].elLoadShare*hourly_demand[t,2])
    @constraint(m1_rt, ϕ_w_spill[i=1:Nw, t=1:Nt], 0 <= w_spill[i,t] <= wind_realizations[(wind_realizations.WFNum.==i) .& (wind_realizations.ScenNum .== Scenario), t][1])

    @constraint(m1_rt, λ_el_rt[elnode=1:Nel_bus, t=1:Nt], sum(r_act[i,t] for i in 1:Np if gen_data[i].elBusNum==elnode)
                                                    + sum((wind_realizations[(wind_realizations.WFNum.==i) .& (wind_realizations.ScenNum .== Scenario), t][1] - m1_windgen[i,t] - w_spill[i,t]) for i in 1:Nw if wind_data[i].elBusNum==elnode)
                                                   + l_shed[elnode,t]  - sum(B[elnode,r]*θ_adj[r,t] for r=1:Nel_bus) == 0)

    @constraint(m1_rt, el_f_def[l=1:Nel_line, t=1:Nt], f_adj[l,t] == ν[elLine_data[l].b_f,elLine_data[l].b_t]*(θ_adj[elLine_data[l].b_f,t] - θ_adj[elLine_data[l].b_t,t]))
    @constraint(m1_rt, el_f_lim[l=1:Nel_line,t=1:Nt], -f̅[elLine_data[l].b_f, elLine_data[l].b_t] <= (m1_elflows[l,t] + f_adj[l,t]) <= f̅[elLine_data[l].b_f, elLine_data[l].b_t])

    #5. El Reference bus
    @constraint(m1_rt, ref_el_rt[t=1:Nt], θ_adj[refbus, t] == 0)

    ###-----NG Constraints----###
    @constraint(m1_rt, ϕ_ng_rt[k=1:Ng, t=1:Nt], ng_prods_data[k].G̲ <= (m1_ng_prod[k,t] + g_adj[k,t]) <= ng_prods_data[k].G̅)

    #2. Nodal Pressure Constraints
    @constraint(m1_rt, ϕ_pr_rt[gnode=1:Nng_bus, t=1:Nt], ngBus_data[gnode].ngPreMin <= (m1_ng_pre[gnode,t] + pr_adj[gnode,t]) <= ngBus_data[gnode].ngPreMax)

    #3. Pipelines with Compressors
    for pl=1:Nng_line
        if ngLine_data[pl].Γ_mu != 1
            @constraint(m1_rt,[t=1:Nt], (m1_ng_pre[ngLine_data[pl].ng_t,t] + pr_adj[ngLine_data[pl].ng_t,t]) <= ngLine_data[pl].Γ_mu*(m1_ng_pre[ngLine_data[pl].ng_f,t] + pr_adj[ngLine_data[pl].ng_f,t]))
        end
    end

    #5. Definition of average flow in a pipeline
    @constraint(m1_rt, q_value_rt[pl=1:Nng_line, t=1:Nt], q_adj[pl,t] == 0.5*(q_in_adj[pl,t] + q_out_adj[pl,t]))

    #6a. Weymouth equation - convex relaxation of equality into a SOC, ignoring the concave part of the cone
    #uncomment if using MOSEK - SecondOrderCone special formulation
    @constraint(m1_rt, wm_soc_rt[pl=1:Nng_line, t=1:Nt], [ngLine_data[pl].K_mu*(m1_ng_pre[ngLine_data[pl].ng_f,t] + pr_adj[ngLine_data[pl].ng_f,t]), (m1_ng_flows[pl,t] + q_adj[pl,t]), ngLine_data[pl].K_mu*(m1_ng_pre[ngLine_data[pl].ng_t,t] + pr_adj[ngLine_data[pl].ng_t,t])] in SecondOrderCone())

    #7. Linepack Definition
    @constraint(m1_rt, lp_def_rt[pl=1:Nng_line,t=1:Nt], h_rt[pl,t] == ngLine_data[pl].K_h*0.5*((m1_ng_pre[ngLine_data[pl].ng_f,t] + pr_adj[ngLine_data[pl].ng_f,t]) + (m1_ng_pre[ngLine_data[pl].ng_t,t] + pr_adj[ngLine_data[pl].ng_t,t])))

    #8. Linepack Operation Dynamics Constraints: for t=1, for t>1 and for t=T
    for pl=1:Nng_line
        for t̂=1:Nt
            if t̂ == 1      #First Hour
                @constraint(m1_rt, h_rt[pl,t̂] == ngLine_data[pl].H_ini + (m1_ng_inflows[pl,t̂] + q_in_adj[pl,t̂]) - (m1_ng_outflows[pl,t̂] + q_out_adj[pl,t̂]))
            end
            if t̂ != 1 #All hours other than first
                @constraint(m1_rt, h_rt[pl,t̂] == h_rt[pl,t̂-1] + (m1_ng_inflows[pl,t̂] + q_in_adj[pl,t̂]) - (m1_ng_outflows[pl,t̂] + q_out_adj[pl,t̂]))
            end
            if t̂ == Nt #Final Hour
                @constraint(m1_rt, h_rt[pl,t̂] >= ngLine_data[pl].H_ini)
            end
        end
    end

    #9. Nodal NG balance
    @constraint(m1_rt, λ_ng_rt[gnode=1:Nng_bus, t=1:Nt], sum(g_adj[k,t] for k in 1:Ng if ng_prods_data[k].ngProdBusNum==gnode) - sum(gen_data[i].ngConvEff*r_act[i,t] for i in 1:Np if gen_data[i].ngBusNum==gnode)
                                                - (sum(q_in_adj[pl,t] for pl in 1:Nng_line if ngLine_data[pl].ng_f==gnode) - sum(q_out_adj[pl,t] for pl in 1:Nng_line if ngLine_data[pl].ng_t==gnode))
                                                == 0)

    #println(m)
    @time optimize!(m1_rt)
    status = termination_status(m1_rt)
    println(status)
    println(raw_status(m1_rt))

    @info("Deterministic RT EL Redispatch Model status ---> $(status)")
    return status,JuMP.objective_value(m1_rt), round.(JuMP.value.(r_act), digits=2), round.(JuMP.value.(l_shed), digits=2),  round.(JuMP.value.(w_spill), digits=2), JuMP.value.(q_in_adj), JuMP.value.(q_out_adj), JuMP.value.(q_adj), JuMP.value.(h_rt), JuMP.value.(pr_adj), JuMP.value.(g_adj)
end
#Testing the function, single run only
(m1_rd_status, m1_rd_cost, m1_rd_p_adj, m1_lshed, m1_wspill, m1_rd_qin_adj, m1_rd_qout_adj, m1_rd_q_adj, m1_rd_h_adj, m1_rd_pr_adj, m1_g_adj) = unidir_deterministic_redispatch(m1_elprod, m1_elflows, m1_windgen, m1_ng_lmp, m1_ng_pre, m1_ng_flows, m1_ng_prod, 5)
=#
#Redispatch in each of the scenarios and pressure adjustments
#=
#running for multiple scenarios
m1_scen_res=DataFrame(ScenNum=Int[], CostRedispatch=Float64[], WindSpilled=Float64[], LoadShed=Float64[])
for Scenario = 1:50
    (m1_rd_status, m1_rd_cost, m1_rd_p_adj, m1_lshed, m1_wspill, m1_rd_qin_adj, m1_rd_qout_adj, m1_rd_q_adj, m1_rd_h_adj, m1_rd_pr_adj, m1_g_adj) = unidir_deterministic_redispatch(m1_elprod, m1_elflows, m1_windgen, m1_ng_lmp, m1_ng_pre, m1_ng_flows, m1_ng_prod, Scenario)
    if m1_rd_status != MOI.OPTIMAL
        println("Not RT feasible for Scenario =", Scenario)
        snapshot=[Scenario, Inf, 0, 0]
        push!(m1_scen_res,snapshot)
    elseif m1_rd_status == MOI.OPTIMAL
        snapshot=[Scenario, m1_rd_cost, sum(m1_wspill), sum(m1_lshed)]
        push!(m1_scen_res,snapshot)
    end
end

@show m1_scen_res
=#

## ========================== REDISPATCH FOR DRCC: M2 ================================= ##

include("M2a_Final_DRCC_McCormick.jl")
(status, cost, m2_el_prod, m2_el_alpha, m2_el_lmp_da, m2_el_lmp_rt, m2_ng_prod, m2_ng_beta, m2_ng_pre, m2_ng_rho, m2_ng_flows, m2_ng_gamma, m2_ng_inflows, m2_ng_gamma_in, m2_ng_outflows, m2_ng_gamma_out, ng_lmp_da, ng_lmp_rt, linepack) = unidir_DRCC_McCormick_SOCP_EL_NG(0)

function undir_DRCC_rt_operation(InSample, w_hat, m2_el_prod, m2_el_alpha, m2_ng_prod, m2_ng_beta, m2_ng_pre, m2_ng_rho, m2_ng_flows, m2_ng_gamma, m2_ng_inflows, m2_ng_gamma_in, m2_ng_outflows, m2_ng_gamma_out, Scenario)
    if InSample == 1
        wind_simdata = wind_traindata
    elseif InSample == 0
        wind_simdata = wind_realizations
    end
    Δ = []     #Realized net system deviation
    for t=1:Nt
        push!(Δ, (sum(w_hat[i,t] for i in 1:Nw) - (sum(wind_simdata[(wind_simdata.WFNum.==i) .& (wind_simdata.ScenNum .== Scenario), t][1] for i = 1:Nw))))
    end

    #Check Feasibility of line-flows
    #m2_rt = Model(with_optimizer(Ipopt.Optimizer))
    m2_rt = Model(with_optimizer(Mosek.Optimizer, MSK_IPAR_LOG=1, MSK_IPAR_INTPNT_SOLVE_FORM=MSK_SOLVE_PRIMAL, MSK_DPAR_INTPNT_CO_TOL_REL_GAP=1.0e-10, MSK_IPAR_PRESOLVE_ELIMINATOR_MAX_NUM_TRIES = 0,  MSK_IPAR_INFEAS_REPORT_AUTO=MSK_ON))


    #EL adjustment variables
    @variable(m2_rt, p_rt[1:Np, 1:Nt])
    @variable(m2_rt, l_shed[1:Nel_bus, 1:Nt])
    @variable(m2_rt, w_spill[1:Nw, 1:Nt])
    @variable(m2_rt, f_rt[1:Nel_line, 1:Nt])
    @variable(m2_rt, θ_rt[1:Nel_bus, 1:Nt])

    #NG adjustment variables
    @variable(m2_rt, pr_rt[1:Nng_bus, t=1:Nt])
    @variable(m2_rt, q_rt[1:Nng_line, t=1:Nt] >=0)
    @variable(m2_rt, q_in_rt[1:Nng_line, t=1:Nt] >=0)
    @variable(m2_rt, q_out_rt[1:Nng_line, t=1:Nt] >=0)
    @variable(m2_rt, h_rt[1:Nng_line, t=1:Nt])
    @variable(m2_rt, g_rt[1:Ng, t=1:Nt])
    @variable(m2_rt, g_shed[1:Nng_bus, t=1:Nt])

    @objective(m2_rt, Min, sum(sum(C_shed_el*l_shed[elnode,t] for elnode = 1: Nel_bus)
                                    + sum(C_spill*w_spill[i,t] for i = 1:Nw)
                                    + sum(C_shed_ng*g_shed[gnode,t] for gnode = 1:Nng_bus)
                                    for t=1:Nt))

    ###-----EL Constraints----###
    @constraint(m2_rt, ϕ_r_act[i=1:Np, t=1:Nt], p_rt[i,t] == m2_el_prod[i,t] + Δ[t]*m2_el_alpha[i,t])
    @constraint(m2_rt, ϕ_ract_bds[i=1:Np, t=1:Nt], gen_data[i].p̲ <= p_rt[i,t] <= gen_data[i].p̅)
    @constraint(m2_rt, ϕ_l_shed[elnode=1:Nel_bus, t=1:Nt], 0 <= l_shed[elnode,t] <= elBus_data[elnode].elLoadShare*hourly_demand[t,2])
    @constraint(m2_rt, ϕ_w_spill[i=1:Nw, t=1:Nt], 0 <= w_spill[i,t] <= wind_simdata[(wind_simdata.WFNum.==i) .& (wind_simdata.ScenNum .== Scenario), t][1])


    @constraint(m2_rt, el_f_def[l=1:Nel_line, t=1:Nt], f_rt[l,t] == ν[elLine_data[l].b_f,elLine_data[l].b_t]*(θ_rt[elLine_data[l].b_f,t] - θ_rt[elLine_data[l].b_t,t]))
    @constraint(m2_rt, el_f_lim_rt[l=1:Nel_line,t=1:Nt], -f̅[elLine_data[l].b_f, elLine_data[l].b_t] <= f_rt[l,t] <= f̅[elLine_data[l].b_f, elLine_data[l].b_t])



    @constraint(m2_rt, λ_el_rt[elnode=1:Nel_bus, t=1:Nt], sum(p_rt[i,t] for i in 1:Np if gen_data[i].elBusNum==elnode)
                                                        + sum(wind_simdata[(wind_simdata.WFNum.==i) .& (wind_simdata.ScenNum .== Scenario), t][1] for i in 1:Nw if wind_data[i].elBusNum==elnode)
                                                        - sum(w_spill[i,t] for i=1:Nw if wind_data[i].elBusNum==elnode)
                                                        + l_shed[elnode,t]
                                                        - elBus_data[elnode].elLoadShare*hourly_demand[t,2]
                                                        == sum(B[elnode,r]*θ_rt[r,t] for r=1:Nel_bus))

    #5. El Reference bus
    @constraint(m2_rt, ref_el_rt[t=1:Nt], θ_rt[refbus, t] == 0)

    ###-----NG Constraints----###
    @constraint(m2_rt, ϕ_ng_rt[k=1:Ng, t=1:Nt], g_rt[k,t] == m2_ng_prod[k,t] + Δ[t]*m2_ng_beta[k,t])
    @constraint(m2_rt, ϕ_ng_lshed[gnode=1:Nng_bus, t=1:Nt], 0 <= g_shed[gnode,t] <= ngBus_data[gnode].ngLoadShare*hourly_demand[t,3])

    #2. Nodal Pressure Constraints
    #@constraint(m2_rt, ϕ_pr_rt[gnode=1:Nng_bus, t=1:Nt], pr_rt[gnode,t] == m2_ng_pre[gnode,t] + Δ[t]*m2_ng_rho[gnode,t])

    #3. Pipelines with Compressors
    for pl=1:Nng_line
        if ngLine_data[pl].Γ_mu != 1
            @constraint(m2_rt,[t=1:Nt], pr_rt[ngLine_data[pl].ng_t,t] <= ngLine_data[pl].Γ_mu*pr_rt[ngLine_data[pl].ng_f,t])
        end
    end

    #5. Definition of average flow in a pipeline
    @constraint(m2_rt, q_value_rt[pl=1:Nng_line, t=1:Nt], q_rt[pl,t] == m2_ng_flows[pl,t] + Δ[t]*m2_ng_gamma[pl,t])
    @constraint(m2_rt, q_value_rt_try[pl=1:Nng_line, t=1:Nt], q_rt[pl,t] == 0.5*(q_in_rt[pl,t] + q_out_rt[pl,t]))

    #@constraint(m2_rt, q_in_value_rt[pl=1:Nng_line, t=1:Nt], q_in_rt[pl,t] == m2_ng_inflows[pl,t] + Δ[t]*m2_ng_gamma_in[pl,t])
    #@constraint(m2_rt, q_out_value_rt[pl=1:Nng_line, t=1:Nt], q_out_rt[pl,t] == m2_ng_outflows[pl,t] + Δ[t]*m2_ng_gamma_out[pl,t])

    #6a. Weymouth equation - convex relaxation of equality into a SOC, ignoring the concave part of the cone
    #uncomment if using MOSEK - SecondOrderCone special formulation
    #@constraint(m2_rt, wm_rt[pl=1:Nng_line, t=1:Nt], q_rt[pl,t]^2 ==  (ngLine_data[pl].K_mu)^2*(pr_rt[ngLine_data[pl].ng_f,t]^2  - pr_rt[ngLine_data[pl].ng_t,t]^2))
    #@constraint(m2_rt, wm_rt_eq[pl=1:Nng_line, t=1:Nt], [ngLine_data[pl].K_mu*pr_rt[ngLine_data[pl].ng_f,t], q_rt[pl,t], ngLine_data[pl].K_mu*pr_rt[ngLine_data[pl].ng_t,t]] in SecondOrderCone())

    #7. Linepack Definition
    @constraint(m2_rt, lp_def_rt[pl=1:Nng_line,t=1:Nt], h_rt[pl,t] == ngLine_data[pl].K_h*0.5*(pr_rt[ngLine_data[pl].ng_f,t] + pr_rt[ngLine_data[pl].ng_t,t]))

    #8. Linepack Operation Dynamics Constraints: for t=1, for t>1 and for t=T
    for pl=1:Nng_line
        for t̂=1:Nt
            if t̂ == 1      #First Hour
                @constraint(m2_rt, h_rt[pl,t̂] == ngLine_data[pl].H_ini + q_in_rt[pl,t̂] - q_out_rt[pl,t̂])
            end
            if t̂ != 1 #All hours other than first
                @constraint(m2_rt, h_rt[pl,t̂] == h_rt[pl,t̂-1] + q_in_rt[pl,t̂] - q_out_rt[pl,t̂])
            end
            if t̂ == Nt #Final Hour
                @constraint(m2_rt, h_rt[pl,t̂] >= ngLine_data[pl].H_ini)
            end
        end
    end


    #9. Nodal NG balance
    @constraint(m2_rt, λ_ng_rt[gnode=1:Nng_bus, t=1:Nt], sum(g_rt[k,t] for k in 1:Ng if ng_prods_data[k].ngProdBusNum==gnode)
                                                - sum(gen_data[i].ngConvEff*p_rt[i,t] for i in 1:Np if gen_data[i].ngBusNum==gnode)
                                                - (sum(q_in_rt[pl,t] for pl in 1:Nng_line if ngLine_data[pl].ng_f==gnode) - sum(q_out_rt[pl,t] for pl in 1:Nng_line if ngLine_data[pl].ng_t==gnode))
                                                + g_shed[gnode,t]
                                                == ngBus_data[gnode].ngLoadShare*hourly_demand[t,3])


    @time optimize!(m2_rt)
    status = termination_status(m2_rt)
    println(status)
    println(raw_status(m2_rt))

    @info("DRCC RT EL Redispatch Model status ---> $(status)")
    #println("Inflows RT:", minimum(JuMP.value.(pr_rt)))
    #return status
    return m2_rt, Δ, status, JuMP.objective_value(m2_rt), JuMP.value.(p_rt), JuMP.value.(l_shed), JuMP.value.(w_spill), JuMP.value.(θ_rt), JuMP.value.(g_shed), JuMP.value.(pr_rt), JuMP.value.(g_rt), JuMP.value.(q_in_rt), JuMP.value.(q_out_rt), JuMP.value.(q_rt)
end

#Testing the function - single run only
#(status) = undir_DRCC_rt_operation(1, w_hat,  m2_el_prod, m2_el_alpha, m2_ng_prod, m2_ng_beta, m2_ng_pre, m2_ng_rho, m2_ng_flows, m2_ng_gamma, m2_ng_inflows, m2_ng_gamma_in, m2_ng_outflows, m2_ng_gamma_out, 2)
(model, Δ, m2_rd_status, m2_rd_cost, m2_rd_p_adj, m2_lshed, m2_wspill, m2_vangs, m2_gshed, m2_ng_pre_rt, m2_ng_prod_rt, m2_ng_q_in_rt, m2_ng_q_out_rt, m2_ng_q_rt) = undir_DRCC_rt_operation(0, w_hat,  m2_el_prod, m2_el_alpha, m2_ng_prod, m2_ng_beta, m2_ng_pre, m2_ng_rho, m2_ng_flows, m2_ng_gamma, m2_ng_inflows, m2_ng_gamma_in, m2_ng_outflows, m2_ng_gamma_out, 20)

#calculate quality of exactness of approximation :
wm_exact_rt=DataFrame(t=Any[],pl=Any[], LHS=Any[], RHS=Any[], diff=[], diffPer=Any[])
for hour = 1:Nt
    for pl = 1:Nng_line
        lhs_val= round(m2_ng_q_rt[pl,hour]^2,digits=2)
        rhs_val = round(ngLine_data[pl].K_mu^2*(m2_ng_pre_rt[ngLine_data[pl].ng_f,hour]^2 - m2_ng_pre_rt[ngLine_data[pl].ng_t,hour]^2), digits=2)
        push!(wm_exact_rt, [hour, pl, lhs_val, rhs_val, abs(lhs_val - rhs_val), 100*abs(lhs_val - rhs_val)/(lhs_val)])
    end
end
#@show wm_exact
println("Total Absolute Error RT Flows:", sum(wm_exact_rt[:,5]))
println("RMS Error RT Flows:", sqrt(sum(wm_exact_rt[:,5])/(Nt+Nng_line)))
println("NRMS Error RT Flows:", sqrt(sum(wm_exact_rt[:,5])/(Nt+Nng_line))/mean(sqrt.(abs.(wm_exact_rt[:,4]))))


#Checking the value of Linepack
h_rt_val = zeros(1:Nng_line,1:Nt)
for hour=1:Nt
    for pl=1:Nng_line
        if hour ==1
            h_rt_val[pl,hour] = ngLine_data[pl].H_ini + m2_ng_q_in_rt[pl,hour] - m2_ng_q_out_rt[pl,hour]
        else
            h_rt_val[pl,hour] =  h_rt_val[pl,hour-1] + m2_ng_q_in_rt[pl,hour] - m2_ng_q_out_rt[pl,hour]
        end
    end
end
#=
#running for multiple scenarios
m2_scen_res=DataFrame(ScenNum=Int[], RedispatchCost=Float64[], WindSpilled=Float64[], ELLoadShed=Float64[], NGLoadShed=Float64[])
InSample = 1
for Scenario = 1:100
    (Δ, m2_rd_status, m2_rd_cost, m2_rd_p_adj, m2_lshed, m2_wspill, m2_vangs, m2_gshed, m2_ng_pre_rt, m2_ng_prod_rt, m2_ng_linepack_rt) = undir_DRCC_rt_operation(InSample, w_hat,  m2_el_prod, m2_el_alpha, m2_ng_prod, m2_ng_beta, m2_ng_pre, m2_ng_rho, m2_ng_flows, m2_ng_gamma, m2_ng_inflows, m2_ng_gamma_in, m2_ng_outflows, m2_ng_gamma_out, Scenario)
    println("Actual System Deviation is: ", Δ)
    if m2_rd_status != MOI.OPTIMAL
        println("Not RT feasible for Scenario =", Scenario)
        snapshot=[Scenario, Inf, 0, 0, 0]
        push!(m2_scen_res,snapshot)
    elseif m2_rd_status == MOI.OPTIMAL
        print(m2_lshed)
        snapshot=[Scenario, m2_rd_cost, sum(m2_wspill), sum(m2_lshed), sum(m2_gshed)]
        push!(m2_scen_res,snapshot)
    end
end
@show m2_scen_res
=#



#=
##======== OOS Simulation for No Natural Gas Case, M0 ===========##
#include("M0_PowerGens_Affine_Policies_DRCC.jl")
include("M0_PTDF_Policies_PowerSystems_PTDF_Formulation.jl")
(m0_status, m0_cost, m0_pvals, m0_alphavals, m0_el_lmp_da, m0_el_lmp_rt) = DRCC_PTDF_EL_PolicyReserves()
#uncomment for M0 copper plate model
#(m0_status, m0_cost, m0_pvals, m0_alphavals, m0_el_lmp_da, m0_el_lmp_rt) = DRCC_EL_PolicyReserves_CopperPlate()
#uncomment for M0 networked power system model
#(m0_status, m0_cost, m0_pvals, m0_alphavals, m0_el_lmp_da, m0_el_lmp_rt) = DRCC_EL_PolicyReserves()

function DRCC_EL_Reserves_OOS(InSample, w_hat, m0_el_prod, m0_el_alpha, Scenario)
    if InSample == 1
        wind_simdata = wind_traindata
    elseif InSample == 0
        wind_simdata = wind_realizations
    end
    print(wind_simdata[(wind_simdata.ScenNum .== Scenario), :])

    Δ = []     #Realized net system deviation
    for t=1:Nt
        push!(Δ, (sum(w_hat[i,t] for i in 1:Nw) - (sum(wind_simdata[(wind_simdata.WFNum.==i) .& (wind_simdata.ScenNum .== Scenario), t][1] for i = 1:Nw))))
        #push!(Δ, 0)
    end

    #Check Feasibility of line-flows
    m0_rt = Model(with_optimizer(Mosek.Optimizer, MSK_IPAR_LOG=1, MSK_IPAR_INTPNT_SOLVE_FORM=MSK_SOLVE_PRIMAL, MSK_DPAR_INTPNT_CO_TOL_REL_GAP=1.0e-10, MSK_IPAR_PRESOLVE_ELIMINATOR_MAX_NUM_TRIES = 0,  MSK_IPAR_INFEAS_REPORT_AUTO=MSK_ON))

    #EL adjustment variables
    @variable(m0_rt, p_rt[1:Np, 1:Nt])
    @variable(m0_rt, l_shed[1:Nel_bus, 1:Nt])
    @variable(m0_rt, w_spill[1:Nw, 1:Nt])
    @variable(m0_rt, f_rt[1:Nel_line, 1:Nt])
    @variable(m0_rt, θ_rt[1:Nel_bus, 1:Nt])

    @objective(m0_rt, Min, sum(sum(C_shed*l_shed[elnode,t] for elnode = 1: Nel_bus) + sum(C_spill*w_spill[i,t] for i = 1:Nw) for t=1:Nt))

    @constraint(m0_rt, ϕ_r_act[i=1:Np, t=1:Nt], p_rt[i,t] == m0_el_prod[i,t] + Δ[t]*m0_el_alpha[i,t])
    @constraint(m0_rt, ϕ̅_p[i=1:Np, t=1:Nt], gen_data[i].p̲ <= p_rt[i,t] <= gen_data[i].p̅)        #Deterministic

    @constraint(m0_rt, ϕ_l_shed[elnode=1:Nel_bus, t=1:Nt], 0 <= l_shed[elnode,t] <= elBus_data[elnode].elLoadShare*hourly_demand[t,2])
    #@constraint(m0_rt, ϕ_w_spill[i=1:Nw, t=1:Nt], 0 <= w_spill[i,t] <= w_hat[i,t])
    @constraint(m0_rt, ϕ_w_spill[i=1:Nw, t=1:Nt], 0 <= w_spill[i,t] <= wind_simdata[(wind_simdata.WFNum.==i) .& (wind_simdata.ScenNum .== Scenario), t][1])

   # Uncomment the constraints below for networked system
    @constraint(m0_rt, el_f_def[l=1:Nel_line, t=1:Nt], f_rt[l,t] == ν[elLine_data[l].b_f,elLine_data[l].b_t]*(θ_rt[elLine_data[l].b_f,t] - θ_rt[elLine_data[l].b_t,t]))
    @constraint(m0_rt, el_f_lim_rt[l=1:Nel_line,t=1:Nt], -f̅[elLine_data[l].b_f, elLine_data[l].b_t] <= f_rt[l,t] <= f̅[elLine_data[l].b_f, elLine_data[l].b_t])

    @constraint(m0_rt, λ_el_rt[elnode=1:Nel_bus, t=1:Nt], sum(p_rt[i,t] for i in 1:Np if gen_data[i].elBusNum==elnode)
                                                        #+ sum(w_hat[i,t] for i in 1:Nw if wind_data[i].elBusNum==elnode)
                                                        + sum(wind_simdata[(wind_simdata.WFNum.==i) .& (wind_simdata.ScenNum .== Scenario), t][1] for i in 1:Nw if wind_data[i].elBusNum==elnode)
                                                        - sum(w_spill[i,t] for i=1:Nw if wind_data[i].elBusNum==elnode)
                                                        + l_shed[elnode,t]
                                                        - elBus_data[elnode].elLoadShare*hourly_demand[t,2]
                                                        == sum(B[elnode,r]*θ_rt[r,t] for r=1:Nel_bus))


    @constraint(m0_rt, ref_el_rt[t=1:Nt], θ_rt[refbus, t] == 0)

    #=
    for t=1:Nt
        for l in 1:Nel_line
            @constraint(m0_rt, -PL[l] .<= PTDF*(Cgens*p_rt[:,t] + Cwind*w_hat[:,t] - Cload*(LoadShare*hourly_demand[t,2])) .<= PL[l])
        end
    end
    =#

    #=
    #Uncomment for M0 copper plate system
    @constraint(m0_rt, λ_el_rt_nonw[t=1:Nt], sum(p_rt[i,t] for i in 1:Np)
                                                + sum(w_hat[i,t] for i in 1:Nw)
                                                #+ sum(wind_simdata[(wind_simdata.WFNum.==i) .& (wind_simdata.ScenNum .== Scenario), t][1] for i in 1:Nw)
                                                - sum(w_spill[i,t] for i=1:Nw)
                                                + sum(l_shed[elnode,t] for elnode=1:Nel_bus)
                                                ==sum(elBus_data[elnode].elLoadShare*hourly_demand[t,2] for elnode=1:Nel_bus))
    =#

    @time optimize!(m0_rt)
    status = termination_status(m0_rt)
    println(status)
    println(raw_status(m0_rt))

    @info("DRCC RT EL Only Redispatch ---> $(status)")

    return Δ, status, JuMP.objective_value(m0_rt), round.(JuMP.value.(p_rt), digits=2), round.(JuMP.value.(l_shed), digits=2),  round.(JuMP.value.(w_spill), digits=2)
end

#Testing the function - single run only
(Δ, m0_rt_status, m0_rt_cost, m0_rt_p_adj, m0_lshed, m0_wspill) = DRCC_EL_Reserves_OOS(1, w_hat, m0_pvals, m0_alphavals, 1)

#=
#running for multiple scenarios
m0_scen_res=DataFrame(ScenNum=Int[], RedispatchCost=Float64[], WindSpilled=Float64[], LoadShed=Float64[])
InSample = 0
for Scenario = 1:50
    (Δ, m0_rt_status, m0_rt_cost, m0_rt_p_adj, m0_rt_elflows, m0_lshed, m0_wspill) = DRCC_EL_Reserves_OOS(InSample, w_hat,  m0_pvals, m0_alphavals, Scenario)
    println("Actual System Deviation is: ", Δ)
    if m0_rt_status != MOI.OPTIMAL
        println("Not RT feasible for Scenario =", Scenario)
        snapshot=[Scenario, Inf, 0, 0]
        push!(m0_scen_res,snapshot)
    elseif m0_rt_status == MOI.OPTIMAL
        snapshot=[Scenario, m0_rt_cost, sum(m0_wspill), sum(m0_lshed)]
        push!(m0_scen_res,snapshot)
    end
end
@show m0_scen_res
=#
=#
