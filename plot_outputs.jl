#Getting results for plotting
using JuMP, Distributions, LinearAlgebra, DataFrames, Mosek, MosekTools, Ipopt

using CSV, DataFrames

# Case Study 1: 24el + 12ng Bus System : Prepare and load data
include("CS1_24bus/CS1_data_load_script_PSCC.jl")
(elBus_data, gen_data, elLine_data ,B , f̅, ν, π, refbus, ng_prods_data, ngBus_data, ngLine_data, wind_data) = load_data()

include("main.jl")


#Import test data DataFrame (Out-of-Sample)
N_tr = 1000     #TrainingDataSamples
N_tot = 10000   #TotalNumberDataSamples
wind_realizations = CSV.read("CS1_24bus/oos_data/OOS_WindRealizations_TestData.csv", header=false)
# Transform wind_realizations
wind_realizations[!,:WFNum] = repeat([1], (N_tot-N_tr)*2)
wind_realizations[(N_tot-N_tr)+1:(N_tot-N_tr)*2,:WFNum] = repeat([2],(N_tot-N_tr))
wind_realizations[!, :ScenNum] = 1:(N_tot-N_tr)*2
wind_realizations[(N_tot-N_tr)+1:(N_tot-N_tr)*2, :ScenNum] = 1:(N_tot-N_tr)

#Prepare In-Sample Data (In-Sample)
wind_traindata = CSV.read("CS1_24bus/oos_data/InSample_WindRealizations_TrainData.csv", header=false)
#transform the dataset
wind_traindata[!,:WFNum] = repeat([1], (N_tr)*2)
wind_traindata[N_tr+1:N_tr*2,:WFNum] = repeat([2],N_tr)
wind_traindata[!, :ScenNum] = 1:N_tr*2
wind_traindata[N_tr+1:N_tr*2, :ScenNum] = 1:N_tr

#Point Forecast Data
w_hat = CSV.read("CS1_24bus/data/UncertaintyMoments/PSCC_Point_Forecast_Values.csv", header=false)        #Point forecast


Np = length(gen_data)           #number of generators
Ng = length(ng_prods_data)      #number of gas producers
Nw = length(wind_data)          #number of wind power producers
Nel_line = length(elLine_data)  #number of power lines (l)
Nel_bus = length(elBus_data)    #number of power buses (elnode)
Nng_line = length(ngLine_data)  #number of gas pipelines (pl)
Nng_bus = length(ngBus_data)    #number of gas buses (gnode)

Nt = 24    #Time periods for Simulation Horizon

#=
## DRCC Model Figure 2: Risk factor vs. DA Cost
m2_riskfactor = DataFrame(Epsilon=Float64[], Confidence=Float64[], Feasibility=Int[], DAExpCost=Float64[])
for RiskFactor in range(0.04, 0.25, step=0.01)
    (m2_da_status, m2_da_cost, m2_el_prod, m2_el_alpha, m2_el_lmp_da, m2_el_lmp_rt, m2_ng_prod, m2_ng_beta, m2_ng_pre, m2_ng_rho, m2_ng_flows, m2_ng_gamma, m2_ng_inflows, m2_ng_gamma_in, m2_ng_outflows, m2_ng_gamma_out, ng_lmp_da, ng_lmp_rt, linepack) = unidir_DRCC_McCormick_SOCP_EL_NG(RiskFactor)
    if m2_da_status != MOI.OPTIMAL
        println("Not DA feasible for RiskFactor =", RiskFactor)
        push!(m2_riskfactor,[RiskFactor, (1-RiskFactor), 0, Inf])
    elseif m2_da_status == MOI.OPTIMAL
        push!(m2_riskfactor,[RiskFactor, (1-RiskFactor), 1, m2_da_cost])
    end
end
@show m2_riskfactor
=#

# Fig.3 : Gather results for the optimal policies allocations plot
RiskFactor = 0.05
(m2_da_status, m2_da_cost, m2_el_prod, m2_el_alpha, m2_el_lmp_da, m2_el_lmp_rt, m2_ng_prod, m2_ng_beta, m2_ng_pre, m2_ng_rho, m2_ng_flows, m2_ng_gamma, m2_ng_inflows, m2_ng_gamma_in, m2_ng_outflows, m2_ng_gamma_out, ng_lmp_da, ng_lmp_rt, linepack) = unidir_DRCC_McCormick_SOCP_EL_NG(RiskFactor)

#Gather indices of generators which are NGFPPs and which are non-NGFPPs
p_nonGF = []
α_nonGF = []
p_GF = []
α_GF = []
for t=1:Nt
    push!(α_nonGF, sum(m2_el_alpha[i,t] for i=1:Np if gen_data[i].ngBusNum==0))
    push!(p_nonGF, sum(m2_el_prod[i,t] for i=1:Np if gen_data[i].ngBusNum==0))
    push!(α_GF, sum(m2_el_alpha[i,t] for i=1:Np if gen_data[i].ngBusNum!=0))
    push!(p_GF, sum(m2_el_prod[i,t] for i=1:Np if gen_data[i].ngBusNum!=0))
end

total_linepack = []
for t=1:Nt
    push!(total_linepack, round(sum(linepack[:,t]), digits=2))
end
