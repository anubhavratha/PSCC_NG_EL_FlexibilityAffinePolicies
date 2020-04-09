#Model M2a: DRCC Co-optimization with Chebyshev Inequality : (Ignoring Concave part of the Weymouth equations)
#Implemented McCormick Tightening using Trivial bounds (explore results further ) - 30-09-2019
using JuMP, Distributions, LinearAlgebra, DataFrames, Mosek, MosekTools
using DataFrames, Statistics
using Plots, StatsPlots, LaTeXStrings         #Plotting Engines
#using PGFPlots

#Change folder to current
cd("/Users/anubhav/Dropbox/2_PhD_Projects_Repo/PSCC_Gas_EL_UncertaintyAware_CC/")


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


#McCormick Bounds
qMin = 0
wMax = 1000

# Pipeline Specific Maximum for qMax obtained from Deterministic Problem
#=
qMax= [
5876.782479825875,
5625.674018111602,
7696.0074465190455,
2537.235251014365,
5363.920227305744,
5166.947505748216,
1671.1226037380395,
2139.1392467122887,
2217.5034716931327,
2231.861160239421,
1753.5660660696262,
2386.23698663083]
=#

qMax = 7696*ones(12,1)

#Parameters for jong-shi-pang-et-al: Enhancement of McCormick Relaxation
τ = 1
η = 1
allow_Trivial_McCormickTightening = 0


function unidir_DRCC_McCormick_SOCP_EL_NG(rf)

    ## DRCC Risk Factor for CC Violation, #Minimum value ~ 0.025, Maximum value = 1
    if(rf == 0)
        ϵ_i = 0.05  #risk tolerance factor for generators for CC violation
        ϵ_g = 0.05
    else
        ϵ_i = rf
        ϵ_g = rf
    end
    z_i = ((1-ϵ_i)/ϵ_i)     #Fixed multiplier for risk tolerance for power gens
    z_g = ((1-ϵ_g)/ϵ_g)
    z_mu = z_nr = z_g

    m = Model(with_optimizer(Mosek.Optimizer, MSK_IPAR_LOG=1, MSK_IPAR_INTPNT_SOLVE_FORM=MSK_SOLVE_PRIMAL, MSK_DPAR_INTPNT_CO_TOL_REL_GAP=1.0e-10, MSK_IPAR_PRESOLVE_ELIMINATOR_MAX_NUM_TRIES = 0))

    #EL Variables
    @variable(m, p[1:Np, 1:Nt])        #Production from power generators

    #NG Variables
    @variable(m, g[1:Ng, 1:Nt] >= 0)             #Gas production from gas producers
    @variable(m, q_in[1:Nng_line, 1:Nt] >= 0)    #Gas inflow in the PipeLine
    @variable(m, q_out[1:Nng_line, 1:Nt] >= 0)   #Gas outflow in the PipeLine
    @variable(m, q[1:Nng_line, 1:Nt] >=0)       #Avg. Gas inflow in the PipeLine
    @variable(m, h[1:Nng_line, 1:Nt] >=0)       #Line pack amount
    @variable(m, pr[1:Nng_bus, 1:Nt] >=0)       #Pressure in gas bus nodes

    #Affine Response Variables
    @variable(m, -1 <= α[1:Np, 1:Nt] <= 1)                  #Affine response from power Generators
    @variable(m, β[1:Ng, 1:Nt])                  #Affine response from gas producers
    @variable(m, γ[1:Nng_line, 1:Nt])            #Affine change in average gas flow in pipeline
    @variable(m, γ_in[1:Nng_line, 1:Nt])         #Affine change in gas inflow in the pipleine
    @variable(m, γ_out[1:Nng_line, 1:Nt])        #Affine change in pipeline inflow pressure
    @variable(m, ρ[1:Nng_bus, 1:Nt])             #Affine change in nodal gas pressure

    #Auxiliary Variables I: For NG Weymouth Equation - McCormick bounds
    @variable(m, σ[1:Nng_bus, 1:Nt])    #Bilinear term of product of pr and ρ for each gas node
    @variable(m, λ[1:Nng_line, 1:Nt])   #Bilinear term of product of flow q and γ for each gas pipeline

    #OBJECTIVE function
    #Uncomment for co-optimization of el and ng
    @objective(m, Min, sum(sum(p[i,t]*gen_data[i].c for i=1:Np if gen_data[i].ngBusNum==0) + sum(g[k,t]*ng_prods_data[k].C_gas for k= 1:Ng) for t=1:Nt))
    #uncomment for just EL
    #@objective(m, Min, sum(sum(p[i,t]*gen_data[i].c for i=1:Np if gen_data[i].ngBusNum==0) for t=1:Nt))

    ###-----EL Constraints----###
    #1. All generator Power Limits
    @constraint(m, ϕ̅_p[i=1:Np, t=1:Nt], gen_data[i].p̲ <= p[i,t] <= gen_data[i].p̅)        #Deterministic
    for t=1:Nt
        CovarMat = convert(Matrix, Σ[t*Nw-1:t*Nw,t*Nw-1:t*Nw])
        agen1 = α[:,t]*ones(1,Nw)
        agen2 = -α[:,t]*ones(1,Nw)
        bgen = p[:,t]
        for i=1:Np
            @constraint(m, [sqrt(1/z_i)*(-bgen[i] + gen_data[i].p̅); sqrt(CovarMat)*agen1[i,:]] in SecondOrderCone())
            @constraint(m, [sqrt(1/z_i)*(bgen[i] - gen_data[i].p̲); sqrt(CovarMat)*agen2[i,:]] in SecondOrderCone())
        end
    end

    #4. Power flow in the lines - PTDF formulation
    for t=1:Nt
        CovarMat = convert(Matrix, Σ[t*Nw-1:t*Nw,t*Nw-1:t*Nw])
        awinline = [PTDF*(Cgens*α[:,t]*ones(1,Nw) - Cwind); -PTDF*(Cgens*α[:,t]*ones(1,Nw) - Cwind)]
        bwinline = [PL + PTDF*(Cload*(LoadShare*hourly_demand[t,2]) - Cgens*p[:,t] - Cwind*w_hat[:,t]); PL - PTDF*(Cload*(LoadShare*hourly_demand[t,2]) - Cgens*p[:,t] - Cwind*w_hat[:,t])]
        for l = 1:Nel_line*2
            @constraint(m, [sqrt(1/z_nr)*bwinline[l] ; sqrt(CovarMat)*(awinline[l,:])] in SecondOrderCone())
        end
    end

    #EL_Power Balance
    @constraint(m, λ_el_da[t=1:Nt], sum(p[i,t] for i = 1:Np) + sum(w_hat[j,t] for j=1:Nw) == sum(elBus_data[elnode].elLoadShare*hourly_demand[t,2] for elnode=1:Nel_bus))
    @constraint(m, λ_el_sys[t=1:Nt], sum(α[i,t] for i=1:Np) == 1)

    ###-----NG Constraints----###
    #1. Gas Producers Constraints
    #@constraint(m, ϕ_ng[k=1:Ng, t=1:Nt], ng_prods_data[k].G̲ <= g[k,t] <= ng_prods_data[k].G̅)     #original
    for t=1:Nt
        CovarMat = convert(Matrix, Σ[t*Nw-1:t*Nw,t*Nw-1:t*Nw])
        AngPro1 = β[:,t]*ones(1,Nw)
        AngPro2 = - β[:,t]*ones(1,Nw)
        BngPro = g[:,t]
        for k=1:Ng
            @constraint(m, [sqrt(1/z_g)*(-BngPro[k] + ng_prods_data[k].G̅); sqrt(CovarMat)*AngPro1[k,:]] in SecondOrderCone())
            @constraint(m, [sqrt(1/z_g)*(BngPro[k] - ng_prods_data[k].G̲); sqrt(CovarMat)*AngPro2[k,:]] in SecondOrderCone())
        end
    end

    #2. Nodal Pressure Constraints
    #@constraint(m, ϕ_pr[gnode=1:Nng_bus, t=1:Nt], ngBus_data[gnode].ngPreMin <= pr[gnode,t] <= ngBus_data[gnode].ngPreMax) #original
    for t=1:Nt
        CovarMat = convert(Matrix, Σ[t*Nw-1:t*Nw,t*Nw-1:t*Nw])
        Apre1 = ρ[:,t]*ones(1,Nw)
        Apre2 = -ρ[:,t]*ones(1,Nw)
        Bpre = pr[:,t]
        for gnode=1:Nng_bus
            @constraint(m, [sqrt(1/z_g)*(-Bpre[gnode] + ngBus_data[gnode].ngPreMax); sqrt(CovarMat)*Apre1[gnode,:]] in SecondOrderCone())
            @constraint(m, [sqrt(1/z_g)*( Bpre[gnode] - ngBus_data[gnode].ngPreMin); sqrt(CovarMat)*Apre2[gnode,:]] in SecondOrderCone())
        end
    end

    #3. Pipelines with Compressors
    for pl=1:Nng_line
        if ngLine_data[pl].Γ_mu != 1
            #@constraint(m,[t=1:Nt], pr[ngLine_data[pl].ng_t,t] <= ngLine_data[pl].Γ_mu*pr[ngLine_data[pl].ng_f,t])
            #@constraint(m, [t=1:Nt], [-(pr[ngLine_data[pl].ng_t,t] - ngLine_data[pl].Γ_mu*pr[ngLine_data[pl].ng_f,t]), sqrt(z_mu*Σ_t[t,t])*(ρ[ngLine_data[pl].ng_t,t]), sqrt(z_mu*Σ_t[t,t]*ngLine_data[pl].Γ_mu)*ρ[ngLine_data[pl].ng_f,t]] in SecondOrderCone())
            for t=1:Nt
                CovarMat = convert(Matrix, Σ[t*Nw-1:t*Nw,t*Nw-1:t*Nw])
                f_node = ngLine_data[pl].ng_f
                t_node = ngLine_data[pl].ng_t
                Acomp = (ρ[t_node,t] - (ngLine_data[pl].Γ_mu)*ρ[f_node,t])
                Bcomp = ngLine_data[pl].Γ_mu*pr[f_node,t] - pr[t_node,t]
                @constraint(m, [sqrt(1/z_mu)*Bcomp ; sqrt(CovarMat)*[Acomp;Acomp]] in SecondOrderCone())
            end
        end
    end

    #4. Uni-directionality of flows in all pipelines
    for t=1:Nt
        CovarMat = convert(Matrix, Σ[t*Nw-1:t*Nw,t*Nw-1:t*Nw])
        a_flow_avg= -γ[:,t]*ones(1,Nw)
        a_flow_in = -γ_in[:,t]*ones(1,Nw)
        a_flow_out = -γ_out[:,t]*ones(1,Nw)
        b_flow_avg = q[:,t]
        b_flow_in = q_in[:,t]
        b_flow_out = q_out[:,t]
        for pl=1:Nng_line
            @constraint(m, [sqrt(1/z_g)*b_flow_avg[pl]; sqrt(CovarMat)*a_flow_avg[pl,:]] in SecondOrderCone())
            @constraint(m, [sqrt(1/z_g)*b_flow_in[pl]; sqrt(CovarMat)*a_flow_in[pl,:]] in SecondOrderCone())
            @constraint(m, [sqrt(1/z_g)*b_flow_out[pl]; sqrt(CovarMat)*a_flow_out[pl,:]] in SecondOrderCone())
        end
    end


    #5. Definition of average flow in a pipeline : for DA and real-time
    @constraint(m, q_value[pl=1:Nng_line, t=1:Nt], q[pl,t] == 0.5*(q_in[pl,t] + q_out[pl,t]))
    @constraint(m, γ_values[pl=1:Nng_line, t=1:Nt], γ[pl,t] == 0.5*(γ_in[pl,t] + γ_out[pl,t]))

    #6a. Weymouth equations - convex relaxation of equality into a SOC
    #SOC constraint for the terms in LHS and RHS that are independent of uncertainty
    @constraint(m, wm_soc_DA[pl=1:Nng_line, t=1:Nt], [ngLine_data[pl].K_mu*pr[ngLine_data[pl].ng_f,t], q[pl,t], ngLine_data[pl].K_mu*pr[ngLine_data[pl].ng_t,t]] in SecondOrderCone())
    #SOC constraint for the terms in LHS and RHS that are quadratic in uncertainty
    @constraint(m, wm_soc_RT[pl=1:Nng_line, t=1:Nt], [ngLine_data[pl].K_mu*ρ[ngLine_data[pl].ng_f,t], γ[pl,t], ngLine_data[pl].K_mu*ρ[ngLine_data[pl].ng_t,t]] in SecondOrderCone())


    #McCormick Relaxations for the terms in LHS and RHS that are linear in uncertainty
    @constraint(m, wp_mcm_1[pl=1:Nng_line,t=1:Nt], λ[pl,t] - ngLine_data[pl].K_mu*ngLine_data[pl].K_mu*σ[ngLine_data[pl].ng_f,t] + ngLine_data[pl].K_mu*ngLine_data[pl].K_mu*σ[ngLine_data[pl].ng_t,t] == 0)

    #Rectangular bounds on σ[from_ngbus,t] = pr[from_ngbus,t]* ρ[from_ngbus,t] for each pipeline pl
    @constraint(m, wp_mcm_bounds_f_LL[pl=1:Nng_line,t=1:Nt], -((ngBus_data[ngLine_data[pl].ng_f].ngPreMax-ngBus_data[ngLine_data[pl].ng_f].ngPreMin)/wMax)*pr[ngLine_data[pl].ng_f,t] + ngBus_data[ngLine_data[pl].ng_f].ngPreMin*ρ[ngLine_data[pl].ng_f,t] <= σ[ngLine_data[pl].ng_f,t] + 0*ngBus_data[ngLine_data[pl].ng_f].ngPreMin)
    @constraint(m, wp_mcm_bounds_f_UU[pl=1:Nng_line,t=1:Nt], ((ngBus_data[ngLine_data[pl].ng_f].ngPreMax-ngBus_data[ngLine_data[pl].ng_f].ngPreMin)/wMax)*pr[ngLine_data[pl].ng_f,t] + ngBus_data[ngLine_data[pl].ng_f].ngPreMax*ρ[ngLine_data[pl].ng_f,t]  <= σ[ngLine_data[pl].ng_f,t] + ngBus_data[ngLine_data[pl].ng_f].ngPreMax*(ngBus_data[ngLine_data[pl].ng_f].ngPreMax-ngBus_data[ngLine_data[pl].ng_f].ngPreMin)/wMax)
    @constraint(m, wp_mcm_bounds_f_UL[pl=1:Nng_line,t=1:Nt], -((ngBus_data[ngLine_data[pl].ng_f].ngPreMax-ngBus_data[ngLine_data[pl].ng_f].ngPreMin)/wMax)*pr[ngLine_data[pl].ng_f,t] + ngBus_data[ngLine_data[pl].ng_f].ngPreMax*ρ[ngLine_data[pl].ng_f,t] >= σ[ngLine_data[pl].ng_f,t] + ngBus_data[ngLine_data[pl].ng_f].ngPreMax*-((ngBus_data[ngLine_data[pl].ng_f].ngPreMax-ngBus_data[ngLine_data[pl].ng_f].ngPreMin)/wMax))
    @constraint(m, wp_mcm_bounds_f_LU[pl=1:Nng_line,t=1:Nt], ((ngBus_data[ngLine_data[pl].ng_f].ngPreMax-ngBus_data[ngLine_data[pl].ng_f].ngPreMin)/wMax)*pr[ngLine_data[pl].ng_f,t] + ngBus_data[ngLine_data[pl].ng_f].ngPreMin*ρ[ngLine_data[pl].ng_f,t] >= σ[ngLine_data[pl].ng_f,t] +  ngBus_data[ngLine_data[pl].ng_f].ngPreMin * (ngBus_data[ngLine_data[pl].ng_f].ngPreMax-ngBus_data[ngLine_data[pl].ng_f].ngPreMin)/wMax)

    #Rectangular bounds on σ[to_ngbus,t] = pr[to_ngbus,t]* ρ[to_ngbus,t] for each pipeline pl
    @constraint(m, wp_mcm_bounds_t_LL[pl=1:Nng_line,t=1:Nt], -((ngBus_data[ngLine_data[pl].ng_t].ngPreMax-ngBus_data[ngLine_data[pl].ng_t].ngPreMin)/wMax)*pr[ngLine_data[pl].ng_t,t] + ngBus_data[ngLine_data[pl].ng_t].ngPreMin*ρ[ngLine_data[pl].ng_t,t] <= σ[ngLine_data[pl].ng_t,t] + -((ngBus_data[ngLine_data[pl].ng_t].ngPreMax-ngBus_data[ngLine_data[pl].ng_t].ngPreMin)/wMax)*ngBus_data[ngLine_data[pl].ng_t].ngPreMin)
    @constraint(m, wp_mcm_bounds_t_UU[pl=1:Nng_line, t=1:Nt], ((ngBus_data[ngLine_data[pl].ng_t].ngPreMax-ngBus_data[ngLine_data[pl].ng_t].ngPreMin)/wMax)*pr[ngLine_data[pl].ng_t,t] + ngBus_data[ngLine_data[pl].ng_t].ngPreMax*ρ[ngLine_data[pl].ng_t,t]  <= σ[ngLine_data[pl].ng_t,t] + ngBus_data[ngLine_data[pl].ng_t].ngPreMax*(ngBus_data[ngLine_data[pl].ng_t].ngPreMax-ngBus_data[ngLine_data[pl].ng_t].ngPreMin)/wMax)
    @constraint(m, wp_mcm_bounds_t_UL[pl=1:Nng_line, t=1:Nt], -((ngBus_data[ngLine_data[pl].ng_t].ngPreMax-ngBus_data[ngLine_data[pl].ng_t].ngPreMin)/wMax)*pr[ngLine_data[pl].ng_t,t] + ngBus_data[ngLine_data[pl].ng_t].ngPreMax*ρ[ngLine_data[pl].ng_t,t] >= σ[ngLine_data[pl].ng_t,t] + ngBus_data[ngLine_data[pl].ng_t].ngPreMax*-((ngBus_data[ngLine_data[pl].ng_t].ngPreMax-ngBus_data[ngLine_data[pl].ng_t].ngPreMin)/wMax))
    @constraint(m, wp_mcm_bounds_t_LU[pl=1:Nng_line, t=1:Nt], ((ngBus_data[ngLine_data[pl].ng_t].ngPreMax-ngBus_data[ngLine_data[pl].ng_t].ngPreMin)/wMax)*pr[ngLine_data[pl].ng_t,t] + ngBus_data[ngLine_data[pl].ng_t].ngPreMin*ρ[ngLine_data[pl].ng_t,t] >= σ[ngLine_data[pl].ng_t,t] +  ngBus_data[ngLine_data[pl].ng_t].ngPreMin * (ngBus_data[ngLine_data[pl].ng_t].ngPreMax-ngBus_data[ngLine_data[pl].ng_t].ngPreMin)/wMax)

    #Rectangular bounds on λ[pl, t] = q[pl,t]*γ[pl,t]
    @constraint(m, wp_mcm_bounds_pl_LL[pl=1:Nng_line,t=1:Nt], -((qMax[pl] - qMin)/wMax)*q[pl,t] + qMin*γ[pl,t] <= λ[pl,t] + qMin*-((qMax[pl] - qMin)/wMax))
    @constraint(m, wp_mcm_bounds_pl_UU[pl=1:Nng_line,t=1:Nt], ((qMax[pl] - qMin)/wMax)*q[pl,t] + qMax[pl]*γ[pl,t] <= λ[pl,t] + qMax[pl]*((qMax[pl] - qMin)/wMax))
    @constraint(m, wp_mcm_bounds_pl_UL[pl=1:Nng_line,t=1:Nt], -((qMax[pl] - qMin)/wMax)*q[pl,t] + ((qMax[pl] - qMin)/wMax)*γ[pl,t] >=  λ[pl,t] + qMax[pl]*-((qMax[pl] - qMin)/wMax))
    @constraint(m, wp_mcm_bounds_pl_LU[pl=1:Nng_line,t=1:Nt], ((qMax[pl] - qMin)/wMax)*q[pl,t] +  qMin*γ[pl,t] >= λ[pl,t] + qMin*((qMax[pl] - qMin)/wMax))


    #6b: Tightening of McCormick bounds for the convex direction
    if allow_Trivial_McCormickTightening == 1
        @constraint(m, wp_mcm_tight_pr_from[pl=1:Nng_line, t=1:Nt],
                        [4*σ[ngLine_data[pl].ng_f,t]*τ + ((ngBus_data[ngLine_data[pl].ng_f].ngPreMin - τ*((ngBus_data[ngLine_data[pl].ng_f].ngPreMax - ngBus_data[ngLine_data[pl].ng_f].ngPreMin)/wMax)) + (ngBus_data[ngLine_data[pl].ng_f].ngPreMax - τ*0))*(pr[ngLine_data[pl].ng_f,t] - τ*ρ[ngLine_data[pl].ng_f,t]) - (ngBus_data[ngLine_data[pl].ng_f].ngPreMin - τ*((ngBus_data[ngLine_data[pl].ng_f].ngPreMax - ngBus_data[ngLine_data[pl].ng_f].ngPreMin)/wMax))*(ngBus_data[ngLine_data[pl].ng_f].ngPreMax - τ*0), (pr[ngLine_data[pl].ng_f,t] + τ*ρ[ngLine_data[pl].ng_f,t])] in SecondOrderCone())

        @constraint(m, wp_mcm_tight_pr_to[pl=1:Nng_line, t=1:Nt],
                        [4*σ[ngLine_data[pl].ng_t,t]*τ + ((ngBus_data[ngLine_data[pl].ng_t].ngPreMin - τ*((ngBus_data[ngLine_data[pl].ng_t].ngPreMax - ngBus_data[ngLine_data[pl].ng_t].ngPreMin)/wMax)) + (ngBus_data[ngLine_data[pl].ng_t].ngPreMax - τ*0))*(pr[ngLine_data[pl].ng_t,t] - τ*ρ[ngLine_data[pl].ng_t,t]) - (ngBus_data[ngLine_data[pl].ng_t].ngPreMin - τ*((ngBus_data[ngLine_data[pl].ng_t].ngPreMax - ngBus_data[ngLine_data[pl].ng_t].ngPreMin)/wMax))*(ngBus_data[ngLine_data[pl].ng_t].ngPreMax - τ*0), (pr[ngLine_data[pl].ng_t,t] + τ*ρ[ngLine_data[pl].ng_t,t])] in SecondOrderCone())

        @constraint(m, wp_mcm_tight_flow_pl[pl=1:Nng_line, t=1:Nt],
                        [4*λ[pl,t]*η + (qMin - η*(ngLine_data[pl].K_mu*(ngBus_data[ngLine_data[pl].ng_f].ngPreMax - ngBus_data[ngLine_data[pl].ng_t].ngPreMin)/wMax) + (qMax -800) - η*0)*(q[pl,t] - η*γ[pl,t]) - (qMin - η*(ngLine_data[pl].K_mu*(ngBus_data[ngLine_data[pl].ng_f].ngPreMax - ngBus_data[ngLine_data[pl].ng_t].ngPreMin)/wMax))*((qMax - 800) - η*0), (q[pl,t] + η*γ[pl,t])] in SecondOrderCone())
    end

    #6c. Concave Sense of Weymouth Equations - Yet to Come

    #7. Linepack Definition
    @constraint(m, lp_def[pl=1:Nng_line,t=1:Nt], h[pl,t] == ngLine_data[pl].K_h*0.5*(pr[ngLine_data[pl].ng_f,t] + pr[ngLine_data[pl].ng_t,t]))

    #8. Linepack Operation Dynamics Constraints: for t=1, for t>1 and for t=T
    for pl=1:Nng_line
        for t̂=1:Nt
            if t̂ == 1      #First Hour
                @constraint(m, h[pl,t̂] == ngLine_data[pl].H_ini + q_in[pl,t̂] - q_out[pl,t̂])      #linepack in DA
            end
            if t̂ != 1 #All hours other than first
                @constraint(m, h[pl,t̂] == h[pl,t̂-1] + q_in[pl,t̂] - q_out[pl,t̂])
                @constraint(m, ngLine_data[pl].K_h*0.5*(ρ[ngLine_data[pl].ng_f,t̂] + ρ[ngLine_data[pl].ng_t,t̂] - ρ[ngLine_data[pl].ng_f,t̂-1] - ρ[ngLine_data[pl].ng_t,t̂-1]) == γ_in[pl,t̂] - γ_out[pl,t̂])
            end
            if t̂ == Nt #Final Hour
                #@constraint(m, h[pl,t̂] >= ngLine_data[pl].H_ini)       #original
                #@constraint(m, [sqrt(1/z_mu)*(h[pl,t̂] - ngLine_data[pl].H_ini), -ngLine_data[pl].K_h*0.5*sqrt(Σ_t[t̂,t̂])*ρ[ngLine_data[pl].ng_f,t̂], -ngLine_data[pl].K_h*0.5*sqrt(Σ_t[t̂,t̂])*ρ[ngLine_data[pl].ng_t,t̂]] in SecondOrderCone())
                CovarMat = convert(Matrix, Σ[t̂*Nw-1:t̂*Nw, t̂*Nw-1:t̂*Nw])
                alp = -sqrt(0.5*ngLine_data[pl].K_h)*(ρ[ngLine_data[pl].ng_f,t̂] + ρ[ngLine_data[pl].ng_t,t̂])
                blp = (h[pl,t̂] - ngLine_data[pl].H_ini)
                @constraint(m, [sqrt(1/z_mu)*blp; sqrt(CovarMat)*[alp; alp]] in SecondOrderCone())
            end
        end
    end


    #9. Nodal NG balance
    @constraint(m, λ_ng_da[gnode=1:Nng_bus, t=1:Nt], sum(g[k,t] for k in 1:Ng if ng_prods_data[k].ngProdBusNum==gnode)
                                                - sum(gen_data[i].ngConvEff*p[i,t] for i in 1:Np if gen_data[i].ngBusNum==gnode)
                                                - (sum(q_in[pl,t] for pl in 1:Nng_line if ngLine_data[pl].ng_f==gnode) - sum(q_out[pl,t] for pl in 1:Nng_line if ngLine_data[pl].ng_t==gnode))
                                                == ngBus_data[gnode].ngLoadShare*hourly_demand[t,3])



    @constraint(m, λ_ng_rt[gnode=1:Nng_bus, t=1:Nt], sum(β[k,t] for k in 1:Ng if ng_prods_data[k].ngProdBusNum==gnode)
                                                - sum(gen_data[i].ngConvEff*α[i,t] for i in 1:Np if gen_data[i].ngBusNum==gnode)
                                                - (sum(γ_in[pl,t] for pl in 1:Nng_line if ngLine_data[pl].ng_f==gnode) - sum(γ_out[pl,t] for pl in 1:Nng_line if ngLine_data[pl].ng_t==gnode))
                                                == 0)
    @time optimize!(m)
    status = termination_status(m)
    println(status)
    println(raw_status(m))


    #println(wm_soc[:,1])
    @info("DRCC Model M2 status ---> $(status)")

    return status, JuMP.objective_value(m), JuMP.value.(p), JuMP.value.(α), round.(JuMP.dual.(λ_el_da), digits=2), round.(JuMP.dual.(λ_el_sys), digits=2), JuMP.value.(g), JuMP.value.(β), JuMP.value.(pr),
    JuMP.value.(ρ), JuMP.value.(q), JuMP.value.(γ), JuMP.value.(q_in), JuMP.value.(γ_in), JuMP.value.(q_out), JuMP.value.(γ_out), round.(JuMP.dual.(λ_ng_da),digits=2), round.(JuMP.dual.(λ_ng_rt),digits=2), JuMP.value.(h)

end



##Single Run of the function for testing and to evaluate Exactness of the SOC Relaxation
(status, cost, el_prod, el_alpha, el_lmp_da, el_lmp_rt, ng_prod, ng_beta, ng_pre, ng_rho, ng_flows, ng_gamma, ng_inflows, ng_gamma_in, ng_outflows, ng_gamma_out, ng_lmp_da, ng_lmp_rt, linepack) = unidir_DRCC_McCormick_SOCP_EL_NG(0)
#(status, cost, el_prod, el_alpha, el_lmp, vangs) = unidir_DRCC_McCormick_SOCP_EL_NG()
println("EL + NG System Cost:", cost)
println("Gas Response:", any(x->abs(x)>0.01, ng_beta))

function calculate_exactness(ng_pre,ng_flows,ng_rho,ng_gamma)
    NRMS_relaxation_gap = []
    #calculate quality of exactness of approximation : Uncertainty Independent (nominal)
    wm_exact_nom=DataFrame(t=Int[],pl=Int[], LHS=Float64[], RHS=Float64[], AbsoluteError=Float64[], NormalizedError=Float64[])
    wm_exact_nom_export = DataFrame(t=Int[], pl=Int[], NormalizedError = Float64[])
    for hour = 1:Nt
        for pl = 1:Nng_line
            lhs_val= round(ng_flows[pl,hour]^2, digits=4)
            rhs_val = round(ngLine_data[pl].K_mu^2*(ng_pre[ngLine_data[pl].ng_f,hour]^2 - ng_pre[ngLine_data[pl].ng_t,hour]^2), digits=4)
            push!(wm_exact_nom, [hour, pl, lhs_val, rhs_val, (rhs_val - lhs_val), round((rhs_val - lhs_val)/(rhs_val), digits=4)])
            push!(wm_exact_nom_export, [hour, pl, round((rhs_val - lhs_val)/(rhs_val), digits=4)])
        end
    end
    #@show wm_exact
    #Replace any NaNs due to normalization
    replace_nan(v) = map(x -> isnan(x) ? zero(x) : x, v)
    wm_exact_exact_nom = DataFrame(map(replace_nan, eachrow(wm_exact_nom)))
    wm_exact_nom_export = DataFrame(map(replace_nan, eachrow(wm_exact_nom_export)))

    println("Mean Absolute Error Nominal Flows:", sum(wm_exact_nom[:,5])/(Nt+Nng_line))
    println("NRMS Error Nominal Flows:", sqrt(sum((wm_exact_nom[:,5]/mean(wm_exact_nom[:,3])).^2)/(Nt+Nng_line)))
    push!(NRMS_relaxation_gap, sqrt(sum((wm_exact_nom[:,5]/mean(wm_exact_nom[:,3])).^2)/(Nt+Nng_line)))

    p1=plot(reshape(round.(wm_exact_nom[:,6],digits=2),(12,24)), st=:heatmap, color=cgrad([:white, :yellow, :orange, :red]), colorbar_title="Normalized Error - Nom")

    println("======AFFINE RESPONSES : Quadratic Term=======")

    #calculate quality of exactness of approximation : Uncertainty Dependent - quadratic (response)
    wm_exact_resp_quad=DataFrame(t=Int[],pl=Int[], LHS=Float64[], RHS=Float64[], AbsoluteError=Float64[], NormalizedError=Float64[])
    wm_exact_resp_quad_export = DataFrame(t=Int[], pl=Int[], NormalizedError = Float64[])
    for hour = 1:Nt
        for pl = 1:Nng_line
            lhs_val= round(ng_gamma[pl,hour]^2, digits=4)
            rhs_val = round(ngLine_data[pl].K_mu^2*(ng_rho[ngLine_data[pl].ng_f,hour]^2 - ng_rho[ngLine_data[pl].ng_t,hour]^2), digits=4)
            push!(wm_exact_resp_quad, [hour, pl, lhs_val, rhs_val, rhs_val - lhs_val, round((rhs_val - lhs_val)/rhs_val, digits=4)])
            push!(wm_exact_resp_quad_export, [hour, pl, round((rhs_val - lhs_val)/rhs_val, digits=4)])
        end
    end
    #Replace any NaNs due to normalization
    replace_nan(v) = map(x -> isnan(x) ? zero(x) : x, v)
    wm_exact_resp_quad = DataFrame(map(replace_nan, eachrow(wm_exact_resp_quad)))
    wm_exact_resp_quad_export = DataFrame(map(replace_nan, eachrow(wm_exact_resp_quad_export)))
    #@show wm_exact
    println("Mean Absolute Error Response (Quad) Flows:", sum(wm_exact_resp_quad[:,5])/(Nt+Nng_line))
    println("NRMS Error Response (Quad) Flows:", sqrt(sum(wm_exact_resp_quad[:,6].^2)/(Nt+Nng_line)))
    push!(NRMS_relaxation_gap,  sqrt(sum(wm_exact_resp_quad[:,6].^2)/(Nt+Nng_line)))
    p2=plot(reshape(round.(wm_exact_resp_quad[:,6],digits=2),(12,24)), st=:heatmap, color=:heat, colorbar_title="Normalized Error - Quad")


    println("======AFFINE RESPONSES : :Linear Term=======")

    #calculate quality of exactness of approximation : Uncertainty Dependent - linear (response)
    wm_exact_resp_lin=DataFrame(t=Int[],pl=Int[], LHS=Float64[], RHS=Float64[], AbsoluteError=Float64[], NormalizedError=Float64[])
    wm_exact_resp_lin_export = DataFrame(t=Int[], pl=Int[], NormalizedError = Float64[])
    for hour = 1:Nt
        for pl = 1:Nng_line
            lhs_val= round(ng_gamma[pl,hour]*ng_flows[pl,hour], digits=4)
            rhs_val = round(ngLine_data[pl].K_mu^2*(ng_rho[ngLine_data[pl].ng_f,hour]*ng_pre[ngLine_data[pl].ng_f,hour] - ng_rho[ngLine_data[pl].ng_t,hour]*ng_pre[ngLine_data[pl].ng_t,hour]), digits=4)
            push!(wm_exact_resp_lin, [hour, pl, lhs_val, rhs_val, rhs_val - lhs_val, round((rhs_val - lhs_val)/rhs_val, digits=4)])
            push!(wm_exact_resp_lin_export, [hour, pl, round((rhs_val - lhs_val)/rhs_val, digits=4)])
        end
    end
    #Replace any NaNs due to normalization
    replace_nan(v) = map(x -> isnan(x) ? zero(x) : x, v)
    wm_exact_resp_lin = DataFrame(map(replace_nan, eachrow(wm_exact_resp_lin)))
    wm_exact_resp_lin_export = DataFrame(map(replace_nan, eachrow(wm_exact_resp_lin_export)))
    #@show wm_exact
    println("Mean Absolute Error Response (Lin) Flows:", sum(wm_exact_resp_lin[:,5])/(Nt+Nng_line))
    println("NRMS Error Response (Lin) Flows:", sqrt(sum(wm_exact_resp_lin[:,6].^2)/(Nt+Nng_line)))
    push!(NRMS_relaxation_gap, sqrt(sum(wm_exact_resp_lin[:,6].^2)/(Nt+Nng_line)))
    p3=plot(reshape(round.(wm_exact_resp_lin[:,6],digits=2),(12,24)), st=:heatmap, color=cgrad([:white, :yellow, :orange, :red]), colorbar_title="Normalized Error - Lin")

    return NRMS_relaxation_gap
end
#=
#% To view Plot in Julia
lay = @layout [a ; b ; c]
plot(p1,p2,p3, layout=lay)
# To export data to csv files
CSV.write("Nomval_WM.csv", wm_exact_nom_export)
CSV.write("Quadval_WM.csv", wm_exact_resp_quad_export)
CSV.write("Linval_WM.csv", wm_exact_resp_lin_export)
=#





#=
#Table and Plots to Analyze Results
#groupedbar(el_alpha', bar_position = :stack, bar_width=0.7)

#Fig 1: Wind Forecast Values
groupedbar(convert(Matrix,w_hat)', bar_position = :stack, bar_width = 0.7)
totalWindForecast = sum(convert(Matrix,w_hat),dims=1)

#Fig 2:Power Dispatch and Alpha
p_nonGF = []
α_nonGF = []
p_GF = []
α_GF = []
for t=1:Nt
    push!(α_nonGF, sum(el_alpha[i,t] for i=1:Np if gen_data[i].ngBusNum==0))
    push!(p_nonGF, sum(el_prod[i,t] for i=1:Np if gen_data[i].ngBusNum==0))
    push!(α_GF, sum(el_alpha[i,t] for i=1:Np if gen_data[i].ngBusNum!=0))
    push!(p_GF, sum(el_prod[i,t] for i=1:Np if gen_data[i].ngBusNum!=0))
end
#areaplot(p_GF + p_nonGF + totalWindForecast', color =:gray)
#areaplot!(p_GF + p_nonGF, color =:blue)
#areaplot!(p_GF, color =:red)
=#
