# Running out of Sample Simulations and gathering results

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


include("M2a_DRCC_McCormick_Chebyshev.jl")

RiskFactor = 0.05
(status, cost, m2_el_prod, m2_el_alpha, m2_el_lmp_da, m2_el_lmp_rt, m2_ng_prod, m2_ng_beta, m2_ng_pre, m2_ng_rho, m2_ng_flows, m2_ng_gamma, m2_ng_inflows, m2_ng_gamma_in, m2_ng_outflows, m2_ng_gamma_out, ng_lmp_da, ng_lmp_rt, linepack) = unidir_DRCC_McCormick_SOCP_EL_NG(RiskFactor)

function undir_exAnte_CC_ViolationCheck(InSample, w_hat, m2_el_prod, m2_el_alpha, m2_ng_prod, m2_ng_beta, m2_ng_pre, m2_ng_rho, m2_ng_flows, m2_ng_gamma, m2_ng_inflows, m2_ng_gamma_in, m2_ng_outflows, m2_ng_gamma_out, Scenario)
    if InSample == 1
        wind_simdata = wind_traindata
    elseif InSample == 0
        wind_simdata = wind_realizations
    end
    Δ = []     #Realized net system deviation
    for t=1:Nt
        push!(Δ, (sum(w_hat[i,t] for i in 1:Nw) - (sum(wind_simdata[(wind_simdata.WFNum.==i) .& (wind_simdata.ScenNum .== Scenario), t][1] for i = 1:Nw))))
    end


    violation=0
    #EL Generators Limit Check
    for t=1:Nt, i=1:Np
        if(m2_el_prod[i,t] + Δ[t]*m2_el_alpha[i,t] < gen_data[i].p̲)
            push!(GenLimsCC, [Scenario, 1, 1, m2_el_prod[i,t] + Δ[t]*m2_el_alpha[i,t] - gen_data[i].p̲, 0, 0])
            violation = 1
        elseif(m2_el_prod[i,t] + Δ[t]*m2_el_alpha[i,t] > gen_data[i].p̅)
            push!(GenLimsCC, [Scenario, 1, 0, 0, 1, m2_el_prod[i,t] + Δ[t]*m2_el_alpha[i,t] - gen_data[i].p̅])
            violation = 1
        else
            push!(GenLimsCC, [Scenario, 0, 0, 0, 0, 0])
        end
    end

    #LineFlow Limits Check



    push!(CCViolations, [Scenario, violation])
    return Δ, GenLimsCC, CCViolations
end
#Testing the function - single run only
#(Δ, GenLimsCC) = undir_exAnte_CC_ViolationCheck(1, w_hat,  m2_el_prod, m2_el_alpha, m2_ng_prod, m2_ng_beta, m2_ng_pre, m2_ng_rho, m2_ng_flows, m2_ng_gamma, m2_ng_inflows, m2_ng_gamma_in, m2_ng_outflows, m2_ng_gamma_out, 2)


#running for multiple scenarios
CCViolations = DataFrame(ScenNum=[], AnyViolation=Int[])
GenLimsCC = DataFrame(ScenNum=Int[], AnyViol = Int[], LBViol = Int[], LBViolAmount=Float64[], UBViol = Int[], UBViolAmount=Float64[])
InSample = 1
for Scenario = 1:200
    (Δ, GenLimsCC) = undir_exAnte_CC_ViolationCheck(InSample, w_hat,  m2_el_prod, m2_el_alpha, m2_ng_prod, m2_ng_beta, m2_ng_pre, m2_ng_rho, m2_ng_flows, m2_ng_gamma, m2_ng_inflows, m2_ng_gamma_in, m2_ng_outflows, m2_ng_gamma_out, Scenario)
end
@show CCViolations
