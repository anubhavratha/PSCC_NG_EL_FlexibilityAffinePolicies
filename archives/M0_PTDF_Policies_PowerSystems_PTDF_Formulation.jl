#Model M0: DRCC Affine Policies with Chebyshev Inequality - Considering Only Power Systems Flexibility. Cost data for NGFPPs are modified inline.
#PTDF Formulation is used for power flows modeling

using JuMP, Distributions, LinearAlgebra, DataFrames, Mosek, MosekTools
using DataFrames, Statistics

# Case Study 1: 24el + 12ng Bus System : Prepare and load data
include("CS1_24bus/CS1_data_load_script_PSCC.jl")
(elBus_data, gen_data, elLine_data ,B , f̅, ν, π, refbus, ng_prods_data, ngBus_data, ngLine_data, wind_data, Bflow, PTDF, PTDF_gens, PTDF_wind, PTDF_load) = load_data()
#--Reading Uncertainty Data - Wind Farm Forecast Errors
w_hat = CSV.read("CS1_24bus/data/UncertaintyMoments/PSCC_Point_Forecast_Values.csv", header=false)        #Point forecast
Σ = CSV.read("CS1_24bus/data/UncertaintyMoments/PSCC_Covariance_Matrix_Data.csv", header=false)                                  #Large Spatial Temporal Covariance Matrix
Σ_t = CSV.read("CS1_24bus/data/UncertaintyMoments/PSCC_TemporallyFolded_Covariance_Matrix_Data.csv", header=false)
hourly_demand = CSV.read("CS1_24bus/data/24el_12ng/hourlyDemand.csv")

Np = length(gen_data)           #number of generators
Nw = length(wind_data)          #number of wind power producers
Nel_line = length(elLine_data)  #number of power lines (l)
Nel_bus = length(elBus_data)    #number of power buses (elnode)

Nt =24  #Time periods for Simulation Horizon

ϵ_i = 0.05              #risk tolerance factor for generators for CC violation
z_i = ((1-ϵ_i)/ϵ_i)     #Fixed multiplier for risk tolerance

z_nr=z_i

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
    PL[l,1] = elLine_data[l].f̅ + 50
end
LoadShare = zeros(size(PTDF_load,2), 1)  #Vector of LoadShares of each load
for d in 1:size(PTDF_load,2)
    LoadShare[d,1] = [elBus_data[elnode].elLoadShare for elnode=1:Nel_bus if elBus_data[elnode].elLoadNum == d][1]
end

function DRCC_PTDF_EL_PolicyReserves()
    m = Model(with_optimizer(Mosek.Optimizer, MSK_IPAR_LOG=1, MSK_IPAR_INTPNT_SOLVE_FORM=MSK_SOLVE_PRIMAL, MSK_DPAR_INTPNT_CO_TOL_REL_GAP=1.0e-10, MSK_IPAR_PRESOLVE_ELIMINATOR_MAX_NUM_TRIES = 0))

    gen_data[1].c = 40
    gen_data[2].c = 35
    gen_data[5].c = 22
    gen_data[6].c = 23
    gen_data[7].c = 44
    gen_data[10].c = 24
    gen_data[11].c = 21

    #EL Variables
    @variable(m, p[1:Np, 1:Nt])        #Production from power generators
    @variable(m, f[1:Nel_line, 1:Nt])  #Flow in power lines
    @variable(m, θ[1:Nel_bus, 1:Nt])   #Bus angles

    #Affine Response Variables
    @variable(m, 1 >= α[1:Np, 1:Nt] >= 0)         #Affine response from power Generators

    #OBJECTIVE function
    @objective(m, Min, sum(sum(p[i,t]*gen_data[i].c for i=1:Np) for t=1:Nt))

    ###-----EL Constraints----###
    #1. All generator Power Limits
    @constraint(m, ϕ̅_p[i=1:Np, t=1:Nt], gen_data[i].p̲ <= p[i,t] <= gen_data[i].p̅)        #Deterministic
    for t=1:Nt
        CovarMat = convert(Matrix, Σ[t*Nw-1:t*Nw,t*Nw-1:t*Nw])
        agen1 = -α[:,t]*ones(1,Nw)
        agen2 = α[:,t]*ones(1,Nw)
        bgen = -p[:,t]
        for i=1:Np
            @constraint(m, [sqrt(1/z_i)*( bgen[i] + gen_data[i].p̅); sqrt(CovarMat)*agen1[i,:]] in SecondOrderCone())
            @constraint(m, [sqrt(1/z_i)*(-bgen[i] - gen_data[i].p̲); sqrt(CovarMat)*agen2[i,:]] in SecondOrderCone())
        end
    end

    #4. Power flow in the lines - PTDF formulation
    for t=1:Nt
        CovarMat = convert(Matrix, Σ[t*Nw-1:t*Nw,t*Nw-1:t*Nw])
        awinline = [PTDF*(-Cgens*α[:,t]*ones(1,Nw) + Cwind); -PTDF*(-Cgens*α[:,t]*ones(1,Nw) + Cwind)]
        bwinline = [PL + PTDF*(Cload*(LoadShare*hourly_demand[t,2]) - Cgens*p[:,t] - Cwind*w_hat[:,t]); PL - PTDF*(Cload*(LoadShare*hourly_demand[t,2]) - Cgens*p[:,t] - Cwind*w_hat[:,t])]
        for l = 1:Nel_line*2
            @constraint(m, [sqrt(1/z_nr)*bwinline[l] ; sqrt(CovarMat)*(awinline[l,:])] in SecondOrderCone())
        end
    end

    #EL_Power Balance
    @constraint(m, λ_el_da[t=1:Nt], sum(p[i,t] for i = 1:Np) + sum(w_hat[j,t] for j=1:Nw) == sum(elBus_data[elnode].elLoadShare*hourly_demand[t,2] for elnode=1:Nel_bus))
    @constraint(m, λ_el_sys[t=1:Nt], sum(α[i,t] for i=1:Np) == 1)

    @time optimize!(m)
    status = termination_status(m)
    println(status)
    println(raw_status(m))

    @info("DRCC Model M0 PTDF status ---> $(status)")

    return status, JuMP.objective_value(m), round.(JuMP.value.(p), digits=2), round.(JuMP.value.(α), digits=2), round.(JuMP.dual.(λ_el_da), digits=2), round.(JuMP.dual.(λ_el_sys), digits=2)
end

(status, cost, pvals, alphavals, el_lmp_da, el_lmp_rt) = DRCC_PTDF_EL_PolicyReserves()

Pflows = zeros(Nel_line, Nt)
for t=1:Nt
    Pflows[:,t] = PTDF*(Cgens*pvals[:,t] + Cwind*w_hat[:,t] - Cload*(LoadShare*hourly_demand[t,2]))
end
