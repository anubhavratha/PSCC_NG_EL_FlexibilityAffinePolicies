#M2a: WORK-IN_PROGRESS (30-09-2019): Enhancement of McCormick Relaxation using the method proposed by Pang, Mitchel et al.
#Helper function to determine the non-trivially calculated bounds for McCormick Envelope Tightening
using JuMP, Distributions, LinearAlgebra, DataFrames, Mosek, MosekTools
using DataFrames, Statistics

# Case Study 1: 24el + 12ng Bus System : Prepare and load data
include("CS1_24bus/CS1_data_load_script_PSCC.jl")
(elBus_data, gen_data, elLine_data ,B , f̅, ν, π, refbus, ng_prods_data, ngBus_data, ngLine_data, wind_data) = load_data()
#--Reading Uncertainty Data - Wind Farm Forecast Errors
w_hat = CSV.read("CS1_24bus/data/UncertaintyMoments/PSCC_Point_Forecast_Values.csv", header=false)        #Point forecast
Σ = CSV.read("CS1_24bus/data/UncertaintyMoments/PSCC_Covariance_Matrix_Data.csv", header=false)                                  #Large Spatial Temporal Covariance Matrix
Σ_t = CSV.read("CS1_24bus/data/UncertaintyMoments/PSCC_TemporallyFolded_Covariance_Matrix_Data.csv", header=false)
hourly_demand = CSV.read("CS1_24bus/data/24el_12ng/hourlyDemand.csv")

Np = length(gen_data)           #number of generators
Ng = length(ng_prods_data)      #number of gas producers
Nw = length(wind_data)          #number of wind power producers
Nel_line = length(elLine_data)  #number of power lines (l)
Nel_bus = length(elBus_data)    #number of power buses (elnode)
Nng_line = length(ngLine_data)  #number of gas pipelines (pl)
Nng_bus = length(ngBus_data)    #number of gas buses (gnode)

Nt = 24    #Time periods for Simulation Horizon



#McCormick Bounds
qMin = 0
qMax = 8000
wMax = 1000


#Parameters for jong-shi-pang-et-al: Enhancement of McCormick Relaxation
τ = 1
η = 1
allow_Trivial_McCormickTightening = 1


function determine_upper_bounds()

    m_ub = Model(with_optimizer(Mosek.Optimizer, MSK_IPAR_LOG=1, MSK_IPAR_INTPNT_SOLVE_FORM=MSK_SOLVE_PRIMAL, MSK_DPAR_INTPNT_CO_TOL_REL_GAP=1.0e-10, MSK_IPAR_PRESOLVE_ELIMINATOR_MAX_NUM_TRIES = 0))

    #EL Variables
    @variable(m_ub, p[1:Np, 1:Nt])        #Production from power generators

    #NG Variables
    @variable(m_ub, g[1:Ng, 1:Nt] >= 0)             #Gas production from gas producers
    @variable(m_ub, q_in[1:Nng_line, 1:Nt] >= 0)    #Gas inflow in the PipeLine
    @variable(m_ub, q_out[1:Nng_line, 1:Nt] >= 0)   #Gas outflow in the PipeLine
    @variable(m_ub, q[1:Nng_line, 1:Nt] >=0)       #Avg. Gas inflow in the PipeLine
    @variable(m_ub, h[1:Nng_line, 1:Nt] >=0)       #Line pack amount
    @variable(m_ub, pr[1:Nng_bus, 1:Nt] >=0)       #Pressure in gas bus nodes

    @variable(m_ub, γ[1:Nng_line, 1:Nt] >= 0)            #Affine change in average gas flow in pipeline
    @variable(m_ub, γ_in[1:Nng_line, 1:Nt] >= 0)         #Affine change in gas inflow in the pipleine
    @variable(m_ub, γ_out[1:Nng_line, 1:Nt] >= 0)        #Affine change in pipeline inflow pressure
    @variable(m_ub, ρ[1:Nng_bus, 1:Nt])             #Affine change in nodal gas pressure

    #Auxiliary Variables I: For NG Weymouth Equation - McCormick bounds
    @variable(m_ub, σ[1:Nng_bus, 1:Nt])    #Bilinear term of product of pr and ρ for each gas node
    @variable(m_ub, λ[1:Nng_line, 1:Nt])   #Bilinear term of product of flow q and γ for each gas pipeline


    @objective(m_ub, Min, pr[12,12] - 50*ρ[12,12])

    @constraint(m_ub, sum(sum(p[i,t]*gen_data[i].c for i=1:Np if gen_data[i].ngBusNum==0) + sum(g[k,t]*ng_prods_data[k].C_gas for k= 1:Ng) for t=1:Nt) <= 1.6e6)
    @constraint(m_ub, ϕ_pr[gnode=1:Nng_bus, t=1:Nt], ngBus_data[gnode].ngPreMin <= pr[gnode,t] <= ngBus_data[gnode].ngPreMax)
    @constraint(m_ub, ϕ_ρ[gnode=1:Nng_bus, t=1:Nt], 0 <= ρ[gnode, t] <= (ngBus_data[gnode].ngPreMax - ngBus_data[gnode].ngPreMin)/wMax)

    #Rectangular bounds on σ[from_ngbus,t] = pr[from_ngbus,t]* ρ[from_ngbus,t] for each pipeline pl
    @constraint(m_ub, wp_mcm_bounds_f_LL[pl=1:Nng_line, t=1:Nt], 0*pr[ngLine_data[pl].ng_f,t] + ngBus_data[ngLine_data[pl].ng_f].ngPreMin*ρ[ngLine_data[pl].ng_f,t] <= σ[ngLine_data[pl].ng_f,t] + 0*ngBus_data[ngLine_data[pl].ng_f].ngPreMin)
    @constraint(m_ub, wp_mcm_bounds_f_UU[pl=1:Nng_line, t=1:Nt], ((ngBus_data[ngLine_data[pl].ng_f].ngPreMax-ngBus_data[ngLine_data[pl].ng_f].ngPreMin)/wMax)*pr[ngLine_data[pl].ng_f,t] + ngBus_data[ngLine_data[pl].ng_f].ngPreMax*ρ[ngLine_data[pl].ng_f,t]  <= σ[ngLine_data[pl].ng_f,t] + ngBus_data[ngLine_data[pl].ng_f].ngPreMax*(ngBus_data[ngLine_data[pl].ng_f].ngPreMax-ngBus_data[ngLine_data[pl].ng_f].ngPreMin)/wMax)
    @constraint(m_ub, wp_mcm_bounds_f_UL[pl=1:Nng_line, t=1:Nt], 0*pr[ngLine_data[pl].ng_f,t] + ngBus_data[ngLine_data[pl].ng_f].ngPreMax*ρ[ngLine_data[pl].ng_f,t] <= σ[ngLine_data[pl].ng_f,t] + ngBus_data[ngLine_data[pl].ng_f].ngPreMax*0)
    @constraint(m_ub, wp_mcm_bounds_f_LU[pl=1:Nng_line, t=1:Nt], ((ngBus_data[ngLine_data[pl].ng_f].ngPreMax-ngBus_data[ngLine_data[pl].ng_f].ngPreMin)/wMax)*pr[ngLine_data[pl].ng_f,t] + ngBus_data[ngLine_data[pl].ng_f].ngPreMin*ρ[ngLine_data[pl].ng_f,t] <= σ[ngLine_data[pl].ng_f,t] +  ngBus_data[ngLine_data[pl].ng_f].ngPreMin * (ngBus_data[ngLine_data[pl].ng_f].ngPreMax-ngBus_data[ngLine_data[pl].ng_f].ngPreMin)/wMax)

    #6b: Tightening of McCormick bounds for the convex direction
    @constraint(m_ub, wp_mcm_tight_pr_from[pl=1:Nng_line, t=1:Nt],
                        [4*σ[ngLine_data[pl].ng_f,t]*τ + ((ngBus_data[ngLine_data[pl].ng_f].ngPreMin - τ*((ngBus_data[ngLine_data[pl].ng_f].ngPreMax - ngBus_data[ngLine_data[pl].ng_f].ngPreMin)/wMax)) + (ngBus_data[ngLine_data[pl].ng_f].ngPreMax - τ*0))*(pr[ngLine_data[pl].ng_f,t] - τ*ρ[ngLine_data[pl].ng_f,t]) - (ngBus_data[ngLine_data[pl].ng_f].ngPreMin - τ*((ngBus_data[ngLine_data[pl].ng_f].ngPreMax - ngBus_data[ngLine_data[pl].ng_f].ngPreMin)/wMax))*(ngBus_data[ngLine_data[pl].ng_f].ngPreMax - τ*0), (pr[ngLine_data[pl].ng_f,t] + τ*ρ[ngLine_data[pl].ng_f,t])] in SecondOrderCone())

    #println(λ_el_rt[1])
    @time optimize!(m_ub)
    status = termination_status(m_ub)
    println(status)
    println(raw_status(m_ub))

    @info("Upper Bound Problem ---> $(status)")

    return status, JuMP.objective_value(m_ub)

end

(status, cost) = determine_upper_bounds()
