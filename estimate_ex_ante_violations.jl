# Model M2a: DRCC with Chebyshev Inequality => Ex-Ante Violation Probability Calculations, Figure 2b and Figure 4 (PSCC paper)
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


include("main.jl")


function undir_exAnte_CC_ViolationCheck(InSample, w_hat, m2_el_prod, m2_el_alpha, m2_ng_prod, m2_ng_beta, m2_ng_pre, m2_ng_rho, m2_ng_flows, m2_ng_gamma, m2_ng_inflows, m2_ng_gamma_in, m2_ng_outflows, m2_ng_gamma_out, m2_linepack, Scenario)
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
            push!(GenLimsCC, [Scenario, i, t, 1, 1, m2_el_prod[i,t] + Δ[t]*m2_el_alpha[i,t] - gen_data[i].p̲, 0, 0])
            violation = 1
        elseif(m2_el_prod[i,t] + Δ[t]*m2_el_alpha[i,t] > gen_data[i].p̅)
            push!(GenLimsCC, [Scenario, i, t, 1, 0, 0, 1, m2_el_prod[i,t] + Δ[t]*m2_el_alpha[i,t] - gen_data[i].p̅])
            violation = 1
        else
            push!(GenLimsCC, [Scenario, i, t, 0, 0, 0, 0, 0])
        end
    end

    #LineFlow Limits Check
    for t=1:Nt
        P_inj = Cgens*(m2_el_prod[:,t] + Δ[t]*m2_el_alpha[:,t]) + Cwind*(wind_simdata[(wind_simdata.ScenNum .== Scenario),t]) - Cload*(LoadShare*hourly_demand[t,2])
        for l = 1:Nel_line
            if((PTDF*P_inj)[l] < -PL[l])
                violation = 1
                push!(LineLimsCC, [Scenario, l, -1, 1])
            elseif((PTDF*P_inj)[l] > PL[l])
                violation = 1
                push!(LineLimsCC, [Scenario, l, 1, 1])
            else
                push!(LineLimsCC, [Scenario, l, 0, 0])
            end
        end
    end

    #GasProducers
    for t=1:Nt, k=1:Ng
        if(m2_ng_prod[k,t] + Δ[t]*m2_ng_beta[k,t] < ng_prods_data[k].G̲)
            push!(GasProdLimsCC, [Scenario, 1, 1, m2_ng_prod[k,t] + Δ[t]*m2_ng_beta[k,t] - ng_prods_data[k].G̲, 0, 0])
            violation = 1
        elseif(m2_ng_prod[k,t] + Δ[t]*m2_ng_beta[k,t] > ng_prods_data[k].G̅)
            push!(GasProdLimsCC, [Scenario, 1, 0, 0, 1, m2_ng_prod[k,t] + Δ[t]*m2_ng_beta[k,t] - ng_prods_data[k].G̅])
            violation = 1
        else
            push!(GasProdLimsCC, [Scenario, 0, 0, 0, 0, 0])
        end
    end

    #Gas Nodes Pressure
    for t=1:Nt, gnode=1:Nng_bus
        if(m2_ng_pre[gnode,t] + Δ[t]*m2_ng_rho[gnode,t] < ngBus_data[gnode].ngPreMin)
            push!(GasNodePreLimsCC, [Scenario, gnode, -1, 1])
            violation = 1
        elseif(m2_ng_pre[gnode,t] + Δ[t]*m2_ng_rho[gnode,t] > ngBus_data[gnode].ngPreMax)
            push!(GasNodePreLimsCC, [Scenario, gnode, 1, 1])
            violation = 1
        else
            push!(GasNodePreLimsCC, [Scenario, gnode, 0, 0])
        end
    end

    #Gas Flow Directions
    for t=1:Nt, pl=1:Nng_line
        if m2_ng_flows[pl,t] + Δ[t]*m2_ng_gamma[pl,t] < 0 || m2_ng_inflows[pl,t] + Δ[t]*m2_ng_gamma_in[pl,t] < 0 || m2_ng_outflows[pl,t] + Δ[t]*m2_ng_gamma_out[pl,t] < 0
            push!(GasLineFlowsCC, [Scenario, pl, t, 1])
            violation = 1
        else
            push!(GasLineFlowsCC, [Scenario, pl, t, 0])
        end
    end
    #@constraint(m,[t=1:Nt], pr[ngLine_data[pl].ng_t,t] <= ngLine_data[pl].Γ_mu*pr[ngLine_data[pl].ng_f,t])

    #Compressors
    for t=1:Nt, pl=1:Nng_line
        if ngLine_data[pl].Γ_mu != 1
            if((m2_ng_pre[ngLine_data[pl].ng_t,t] + Δ[t]*m2_ng_rho[ngLine_data[pl].ng_t,t]) > ngLine_data[pl].Γ_mu*(m2_ng_pre[ngLine_data[pl].ng_t,t] + Δ[t]*m2_ng_rho[ngLine_data[pl].ng_t,t]))
                push!(ComprViolCC, [Scenario, pl, t, 1])
                violation = 1
            else
                push!(ComprViolCC, [Scenario, pl, t, 0])
            end
        end
    end


    #Final Linepack
    for pl=1:Nng_line
        if(m2_linepack[pl,24] < ngLine_data[pl].H_ini)
            violation =1
            push!(LPFinalCC,[Scenario, pl, 1])
        else
            push!(LPFinalCC,[Scenario, pl, 0])
        end
    end


    push!(CCViolations, [Scenario, violation])
    return Δ, CCViolations, GenLimsCC, LineLimsCC, GasProdLimsCC, GasNodePreLimsCC, GasLineFlowsCC, LPFinalCC
end
#Testing the function - single run only
#(Δ, GenLimsCC) = undir_exAnte_CC_ViolationCheck(1, w_hat,  m2_el_prod, m2_el_alpha, m2_ng_prod, m2_ng_beta, m2_ng_pre, m2_ng_rho, m2_ng_flows, m2_ng_gamma, m2_ng_inflows, m2_ng_gamma_in, m2_ng_outflows, m2_ng_gamma_out, 2)

CCViolations = DataFrame(ScenNum=Int[], AnyViolation=Int[])
GenLimsCC = DataFrame(ScenNum=Int[], GenNum=Int[], Hour=[], AnyViol = Int[], LBViol = Int[], LBViolAmount=Float64[], UBViol = Int[], UBViolAmount=Float64[])
LineLimsCC = DataFrame(ScenNum=Int[], LineNum=Int[], ViolSense = Int[], AnyViol = Int[])
GasProdLimsCC = DataFrame(ScenNum=Int[], AnyViol = Int[], LBViol = Int[], LBViolAmount=Float64[], UBViol = Int[], UBViolAmount=Float64[])
GasNodePreLimsCC = DataFrame(ScenNum=Int[], GasNode=Int[], ViolLense=Int[], AnyViol=Int[])
GasLineFlowsCC = DataFrame(ScenNum=Int[], Pipeline=Int[], Hour=Int[], AnyViol=Int[])
LPFinalCC = DataFrame(ScenNum=Int[], PipeLine=Int[], AnyViol=Int[])
ComprViolCC = DataFrame(ScenNum=Int[], PipeLine=Int[], Hour =Int[], AnyViol = Int[])

InSample = 0
ExAnteNumScenarios = 1000

# Single Run for checking program correctness

RiskFactor = 0.15
(status, cost, m2_el_prod, m2_el_alpha, m2_el_lmp_da, m2_el_lmp_rt, m2_ng_prod, m2_ng_beta, m2_ng_pre, m2_ng_rho, m2_ng_flows, m2_ng_gamma, m2_ng_inflows, m2_ng_gamma_in, m2_ng_outflows, m2_ng_gamma_out, ng_lmp_da, ng_lmp_rt, m2_linepack) = unidir_DRCC_McCormick_SOCP_EL_NG(RiskFactor)
for Scenario = 1:ExAnteNumScenarios
    (Δ, CCViolations, GenLimsCC, LineLimsCC, GasProdLimsCC, GasNodePreLimsCC, GasLineFlowsCC, LPFinalCC) = undir_exAnte_CC_ViolationCheck(InSample, w_hat,  m2_el_prod, m2_el_alpha, m2_ng_prod, m2_ng_beta, m2_ng_pre, m2_ng_rho, m2_ng_flows, m2_ng_gamma, m2_ng_inflows, m2_ng_gamma_in, m2_ng_outflows, m2_ng_gamma_out, m2_linepack, Scenario)
end
@show CCViolations
#@show LineLimsCC


#=
## Subroutine to estimate overall violation probability with different values of epsilon (Figure 2b)
#Note: Remember to clear Workspace before running this snippet!!
OverallViolationProb = DataFrame(Epsilon=Float64[], Confidence = Float64[], ViolProb=Float64[])
for RiskFactor in [0.05, 0.1, 0.15, 0.2, 0.25]
    #running for multiple scenarios
    (status, cost, m2_el_prod, m2_el_alpha, m2_el_lmp_da, m2_el_lmp_rt, m2_ng_prod, m2_ng_beta, m2_ng_pre, m2_ng_rho, m2_ng_flows, m2_ng_gamma, m2_ng_inflows, m2_ng_gamma_in, m2_ng_outflows, m2_ng_gamma_out, ng_lmp_da, ng_lmp_rt, m2_linepack) = unidir_DRCC_McCormick_SOCP_EL_NG(RiskFactor)
    if status != MOI.OPTIMAL
        println("Not feasible for Epsilon =", RiskFactor)
        push!(OverallViolationProb,[RiskFactor, (1-RiskFactor), Inf])
    elseif status == MOI.OPTIMAL
        println("Now running for Epsilon=", RiskFactor)
        for Scenario = 1:ExAnteNumScenarios
            (Δ, CCViolations, GenLimsCC, LineLimsCC, GasProdLimsCC, GasNodePreLimsCC, GasLineFlowsCC, LPFinalCC) = undir_exAnte_CC_ViolationCheck(InSample, w_hat,  m2_el_prod, m2_el_alpha, m2_ng_prod, m2_ng_beta, m2_ng_pre, m2_ng_rho, m2_ng_flows, m2_ng_gamma, m2_ng_inflows, m2_ng_gamma_in, m2_ng_outflows, m2_ng_gamma_out, m2_linepack, Scenario)
        end
        push!(OverallViolationProb, [RiskFactor, (1-RiskFactor), sum(CCViolations[:,2])/ExAnteNumScenarios])
        deleterows!(CCViolations,1:ExAnteNumScenarios)
        deleterows!(GenLimsCC,1:ExAnteNumScenarios*Np*Nt)
        deleterows!(LineLimsCC,1:ExAnteNumScenarios*Nel_line*Nt)
        deleterows!(GasProdLimsCC,1:ExAnteNumScenarios*Ng*Nt)
        deleterows!(GasNodePreLimsCC,1:ExAnteNumScenarios*Nng_bus*Nt)
        deleterows!(GasLineFlowsCC,1:ExAnteNumScenarios*Nng_line*Nt)
        deleterows!(LPFinalCC,1:ExAnteNumScenarios)
    end
end
@show OverallViolationProb
=#




## Figure 4: Processing the DataFrame to calculate violation probabilities for each set of Chance Constraints
#Must be run separately for each epsilon value
OverallViolationProbability = sum(CCViolations[:,2]/ExAnteNumScenarios)

CountGenViol = []
for s = 1:ExAnteNumScenarios
    ThisScenData = GenLimsCC[(GenLimsCC.ScenNum.==s), 4]
    if(any(x->x>0, ThisScenData))
        push!(CountGenViol, 1)
    end
end
GenViolProb = sum(CountGenViol)/ExAnteNumScenarios


CountLineFlowViol = []
for s=1:ExAnteNumScenarios
    ThisScenData = LineLimsCC[(LineLimsCC.ScenNum).==s, 4]
    if(any(x->x>0, ThisScenData))
        push!(CountLineFlowViol, 1)
    end
end
if(!isempty(CountLineFlowViol))
    LineFlowViolProb = sum(CountLineFlowViol)/ExAnteNumScenarios
end


CountGasProdViol = []
for s=1:ExAnteNumScenarios
    ThisScenData = GasProdLimsCC[(GasProdLimsCC.ScenNum.==s),2]
    if(any(x->x>0, ThisScenData))
        push!(CountGasProdViol, 1)
    end
end
if(!isempty(CountGasProdViol))
    GasProdViolProb = sum(CountGasProdViol)/ExAnteNumScenarios
end

CountGasPreNodeViol = []
for s=1:ExAnteNumScenarios
    ThisScenData = GasNodePreLimsCC[(GasNodePreLimsCC.ScenNum.==s),4]
    if(any(x->x>0, ThisScenData))
        push!(CountGasPreNodeViol, 1)
    end
end
if(!isempty(CountGasPreNodeViol))
    GasNodePreViolProb = sum(CountGasPreNodeViol)/ExAnteNumScenarios
end

CountGasFlowViol = []
for s=1:ExAnteNumScenarios
    ThisScenData = GasLineFlowsCC[(GasLineFlowsCC.ScenNum.==s),4]
    if(any(x->x>0, ThisScenData))
        push!(CountGasFlowViol, 1)
    end
end
if(!isempty(CountGasFlowViol))
    GasLineFlowViolProb = sum(CountGasFlowViol)/ExAnteNumScenarios
end

CountLPFinalViol =[]
for s=1:ExAnteNumScenarios
    ThisScenData = LPFinalCC[(LPFinalCC.ScenNum.==s),3]
    if(any(x->x>0, ThisScenData))
        push!(CountLPFinalViol,1)
    end
end
if(!isempty(CountLPFinalViol))
    LPFinalViolProb = sum(CountLPFinalViol)/ExAnteNumScenarios
end


CountComprViol = []
for s=1:ExAnteNumScenarios
    ThisScenData = ComprViolCC[(ComprViolCC.ScenNum.==s), 4]
    if(any(x->x>0, ThisScenData))
        push!(CountComprViol,1)
    end
end
if(!isempty(CountComprViol))
    CompViolProb = sum(CountComprViol)/ExAnteNumScenarios
end
