#Getting results for plotting
using JuMP, Distributions, LinearAlgebra, DataFrames, Mosek, MosekTools, Ipopt

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


## DRCC Model
include("M2a_Final_DRCC_McCormick.jl")
m2_riskfactor = DataFrame(RiskFactor=Float64[], Feasible=Int[], DACost=Float64[])
for RiskFactor in range(0.025, 0.25, step=0.005)
    (m2_da_status, m2_da_cost, m2_el_prod, m2_el_alpha, m2_el_lmp_da, m2_el_lmp_rt, m2_ng_prod, m2_ng_beta, m2_ng_pre, m2_ng_rho, m2_ng_flows, m2_ng_gamma, m2_ng_inflows, m2_ng_gamma_in, m2_ng_outflows, m2_ng_gamma_out, ng_lmp_da, ng_lmp_rt, linepack) = unidir_DRCC_McCormick_SOCP_EL_NG(RiskFactor)
    if m2_da_status != MOI.OPTIMAL
        println("Not DA feasible for RiskFactor =", RiskFactor)
        push!(m2_riskfactor,[RiskFactor, 0, Inf])
    elseif m2_da_status == MOI.OPTIMAL
        push!(m2_riskfactor,[RiskFactor, 1, m2_da_cost])
    end
end
@show m2_riskfactor
