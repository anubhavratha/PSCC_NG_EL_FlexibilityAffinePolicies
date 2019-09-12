#Model M2a: [Incomplete] DRCC Co-optimization with Chebyshev Inequality : Ignoring Concave part of the Weymouth equations
# Yet to come: McCormick Relaxation for the linear terms of uncertainty
using JuMP, Distributions, LinearAlgebra, DataFrames, Mosek, MosekTools


# Case Study 1: 24el + 12ng Bus System : Prepare and load data
include("CS1_24bus/CS1_data_load_script_PSCC.jl")
(elBus_data, gen_data, elLine_data ,B , f̅, ν, π, refbus, ng_prods_data, ngBus_data, ngLine_data, wind_data) = load_data()
#--Reading Uncertainty Data - Wind Farm Forecast Errors
w_hat = CSV.read("CS1_24bus/data/UncertaintyMoments/PSCC_Point_Forecast_Values.csv", header=false)        #Point forecast
#Σ = CSV.read("data/Covariance_Matrix_Data.csv", header=false)                                  #Large Spatial Temporal Covariance Matrix
Σ_t = CSV.read("CS1_24bus/data/UncertaintyMoments/PSCC_TemporallyFolded_Covariance_Matrix_Data.csv", header=false)
hourly_demand = CSV.read("CS1_24bus/data/24el_12ng/hourlyDemand.csv")


#=
# Case Study 2: 188el + 48ng Bus System : Prepare and load data
include("CS2_118bus/CS2_data_load_script_PSCC.jl")
(elBus_data, gen_data, elLine_data ,B , f̅, ν, π, refbus, ng_prods_data, ngBus_data, ngLine_data, wind_data) = load_data()
#--Reading Uncertainty Data - Wind Farm Forecast Errors
w_hat = CSV.read("CS2_118bus/data/UncertaintyMoments/PSCC_Point_Forecast_Values.csv", header=false)        #Point forecast
#Σ = CSV.read("data/Covariance_Matrix_Data.csv", header=false)                                  #Large Spatial Temporal Covariance Matrix
Σ_t = CSV.read("CS2_118bus/data/UncertaintyMoments/PSCC_TemporallyFolded_Covariance_Matrix_Data.csv", header=false)
hourly_demand = CSV.read("CS2_118bus/data/118el_48ng/hourlyDemand.csv")
=#



Np = length(gen_data)           #number of generators
Ng = length(ng_prods_data)      #number of gas producers
Nw = length(wind_data)          #number of wind power producers
Nel_line = length(elLine_data)  #number of power lines (l)
Nel_bus = length(elBus_data)    #number of power buses (elnode)
Nng_line = length(ngLine_data)  #number of gas pipelines (pl)
Nng_bus = length(ngBus_data)    #number of gas buses (gnode)

Nt = 24       #Time periods for Simulation Horizon
wind_buses = [5,7]

#Minimum value ~ 0.025
ϵ_i = 0.05  #risk tolerance factor for generators for CC violation
z_i = ((1-ϵ_i)/ϵ_i)     #Fixed multiplier for risk tolerance

ϵ_g = 0.05
z_g = ((1-ϵ_g)/ϵ_g)
z_mu = z_nr = z_g

#McCormick Bounds
qMin = 0
qMax = 2000
wMax = 1000

function unidir_DRCC_McCormick_SOCP_EL_NG()
    m = Model(with_optimizer(Mosek.Optimizer, MSK_IPAR_LOG=1, MSK_IPAR_INTPNT_SOLVE_FORM=MSK_SOLVE_PRIMAL, MSK_DPAR_INTPNT_CO_TOL_REL_GAP=1.0e-10, MSK_IPAR_PRESOLVE_ELIMINATOR_MAX_NUM_TRIES = 0))

    #EL Variables
    @variable(m, p[1:Np, 1:Nt])        #Production from power generators
    @variable(m, f[1:Nel_line, 1:Nt])  #Flow in power lines
    @variable(m, θ[1:Nel_bus, 1:Nt])   #Bus angles

    #NG Variables
    @variable(m, g[1:Ng, 1:Nt] >= 0)             #Gas production from gas producers
    @variable(m, q_in[1:Nng_line, 1:Nt] >= 0)    #Gas inflow in the PipeLine
    @variable(m, q_out[1:Nng_line, 1:Nt] >= 0)   #Gas outflow in the PipeLine
    @variable(m, q[1:Nng_line, 1:Nt] >=0)       #Avg. Gas inflow in the PipeLine
    @variable(m, h[1:Nng_line, 1:Nt] >=0)       #Line pack amount
    @variable(m, pr[1:Nng_bus, 1:Nt] >=0)       #Pressure in gas bus nodes

    #Affine Response Variables
    @variable(m, α[1:Np, 1:Nt] >= 0)         #Affine response from power Generators
    @variable(m, β[1:Ng, 1:Nt] >= 0)         #Affine response from gas producers
    @variable(m, γ[1:Nng_line, 1:Nt] >= 0)            #Affine change in average gas flow in pipeline
    @variable(m, γ_in[1:Nng_line, 1:Nt] >= 0)         #Affine change in gas inflow in the pipleine
    @variable(m, γ_out[1:Nng_line, 1:Nt] >= 0)        #Affine change in pipeline inflow pressure
    @variable(m, ρ[1:Nng_bus, 1:Nt] >= 0)             #Affine change in nodal gas pressure

    #Auxiliary Variables I: For EL Flows in real-time
    @variable(m, κ[1:Nel_bus, 1:Nt])        #Variable for computational ease of real-time flow

    #Auxiliary Variables II: For NG Weymouth Equation - McCormick bounds
    @variable(m, σ[1:Nng_bus, 1:Nt])    #Bilinear term of product of pr and ρ for each gas node
    @variable(m, λ[1:Nng_line, 1:Nt])   #Bilinear term of product of flow q and γ for each gas pipeline


    #OBJECTIVE function
    #Uncomment for co-optimization of el and ng
    @objective(m, Min, sum(sum(p[i,t]*gen_data[i].c for i=1:Np if gen_data[i].ngBusNum==0) + sum(g[k,t]*ng_prods_data[k].C_gas for k= 1:Ng) for t=1:Nt))
    #uncomment for just EL
    #@objective(m, Min, sum(sum(p[i,t]*gen_data[i].c for i=1:Np if gen_data[i].ngBusNum==0) for t=1:Nt))

    ###-----EL Constraints----###
    #1. All generator Power Limits
    #@constraint(m, ϕ_p[i=1:Np, t=1:Nt], gen_data[i].p̲ <= p[i,t] <= gen_data[i].p̅)        %Original deterministic
    @constraint(m, ϕ̅_p[i=1:Np, t=1:Nt], [gen_data[i].p̅ - p[i,t], sqrt(z_i*Σ_t[t,t])*α[i,t]] in SecondOrderCone())
    @constraint(m, ϕ̲_p[i=1:Np, t=1:Nt], [p[i,t] - gen_data[i].p̲, -sqrt(z_i*Σ_t[t,t])*α[i,t]] in SecondOrderCone())


    #3. Power flow along lines
    for t=1:Nt
        for l=1:Nel_line
            if ν[elLine_data[l].b_f,elLine_data[l].b_t] != 0
                norm_args = []
                for b = wind_buses
                    norm_args=push!(norm_args,@expression(m, sqrt((1/z_nr)*Σ_t[t,t])*ν[elLine_data[l].b_f,elLine_data[l].b_t]*(π[elLine_data[l].b_f,b]-π[elLine_data[l].b_t,b]-κ[elLine_data[l].b_f,t]+κ[elLine_data[l].b_t,t])))
                end
                network_SOCP=vcat(f̅[elLine_data[l].b_f, elLine_data[l].b_t] - ν[elLine_data[l].b_f,elLine_data[l].b_t]*(θ[elLine_data[l].b_f,t] - θ[elLine_data[l].b_t,t]),norm_args)
                @constraint(m, network_SOCP in SecondOrderCone())
            end
        end
    end

    #4. El nodal power balance
    @constraint(m, λ_el_da[elnode=1:Nel_bus, t=1:Nt], sum(p[i,t] for i in 1:Np if gen_data[i].elBusNum==elnode)
                                                    + sum(w_hat[i,t] for i in 1:Nw if wind_data[i].elBusNum==elnode)
                                                    - elBus_data[elnode].elLoadShare*hourly_demand[t,2]
                                                    == sum(B[elnode,r]*θ[r,t] for r=1:Nel_bus))

    @constraint(m, λ_el_sys[t=1:Nt], sum(α[i,t] for i=1:Np) == 1)
    @constraint(m, λ_el_rt[elnode=setdiff(1:Nel_bus,refbus), t=1:Nt], sum(α[i,t] for i=1:Np if gen_data[i].elBusNum==elnode) - sum(B[elnode,j]*κ[j,t] for j=setdiff(1:Nel_bus,refbus)) == 0)

    #5. El Reference bus
    @constraint(m,ref_el_θ[t=1:Nt], θ[refbus,t] == 0)
    @constraint(m, ref_el_κ[t=1:Nt], κ[refbus,t] == 0)


    ###-----NG Constraints----###
    #1. Gas Producers Constraints
    #@constraint(m, ϕ_ng[k=1:Ng, t=1:Nt], ng_prods_data[k].G̲ <= g[k,t] <= ng_prods_data[k].G̅)     #original
    @constraint(m, ϕ̅_ng[k=1:Ng, t=1:Nt], [ng_prods_data[k].G̅ - g[k,t], sqrt(z_g*Σ_t[t,t])*β[k,t]] in SecondOrderCone())
    @constraint(m, ϕ̲_ng[k=1:Ng, t=1:Nt], [g[k,t] - ng_prods_data[k].G̲, -sqrt(z_g*Σ_t[t,t])*β[k,t]] in SecondOrderCone())


    #2. Nodal Pressure Constraints
    #@constraint(m, ϕ_pr[gnode=1:Nng_bus, t=1:Nt], ngBus_data[gnode].ngPreMin <= pr[gnode,t] <= ngBus_data[gnode].ngPreMax) #original
    @constraint(m, ϕ̅_pr[gnode=1:Nng_bus,t=1:Nt], [ngBus_data[gnode].ngPreMax - pr[gnode,t], sqrt(z_g*Σ_t[t,t])*ρ[gnode,t]] in SecondOrderCone())
    @constraint(m, ϕ̲_pr[gnode=1:Nng_bus,t=1:Nt], [pr[gnode,t] - ngBus_data[gnode].ngPreMin, -sqrt(z_g*Σ_t[t,t])*ρ[gnode,t]] in SecondOrderCone())

    #3. Pipelines with Compressors
    for pl=1:Nng_line
        if ngLine_data[pl].Γ_mu != 1
            #@constraint(m,[t=1:Nt], pr[ngLine_data[pl].ng_t,t] <= ngLine_data[pl].Γ_mu*pr[ngLine_data[pl].ng_f,t])
            #@constraint(m, [t=1:Nt], [-(pr[ngLine_data[pl].ng_t,t] - ngLine_data[pl].Γ_mu*pr[ngLine_data[pl].ng_f,t]), sqrt(z_mu*Σ_t[t,t])*(ρ[ngLine_data[pl].ng_t,t] - ngLine_data[pl].Γ_mu*ρ[ngLine_data[pl].ng_f,t])] in SecondOrderCone())
            #Alternate Correct Conic Reformulation
            @constraint(m, [t=1:Nt], [-(pr[ngLine_data[pl].ng_t,t] - ngLine_data[pl].Γ_mu*pr[ngLine_data[pl].ng_f,t]), sqrt(z_mu*Σ_t[t,t])*(ρ[ngLine_data[pl].ng_t,t]), sqrt(z_mu*Σ_t[t,t]*ngLine_data[pl].Γ_mu)*ρ[ngLine_data[pl].ng_f,t]] in SecondOrderCone())
        end
    end

    #4. Uni-directionality of flows in all pipelines
    @constraint(m, q_nonneg[pl=1:Nng_line, t=1:Nt], [q[pl,t], -(sqrt(z_g*Σ_t[t,t]))*γ[pl,t]] in SecondOrderCone())
    @constraint(m, qin_nonneg[pl=1:Nng_line, t=1:Nt], [q_in[pl,t], -(sqrt(z_g*Σ_t[t,t]))*γ_in[pl,t]] in SecondOrderCone())
    @constraint(m, qout_nonneg[pl=1:Nng_line, t=1:Nt], [q_out[pl,t], -(sqrt(z_g*Σ_t[t,t]))*γ_out[pl,t]] in SecondOrderCone())


    #5. Definition of average flow in a pipeline : for DA and real-time
    @constraint(m, q_value[pl=1:Nng_line, t=1:Nt], q[pl,t] == 0.5*(q_in[pl,t] + q_out[pl,t]))
    @constraint(m, γ_values[pl=1:Nng_line, t=1:Nt], γ[pl,t] == 0.5*(γ_in[pl,t] + γ_out[pl,t]))

    #6a. Weymouth equations - convex relaxation of equality into a SOC
    #SOC constraint for the terms in LHS and RHS that are independent of uncertainty
    @constraint(m, wm_soc_DA[pl=1:Nng_line, t=1:Nt], [ngLine_data[pl].K_mu*pr[ngLine_data[pl].ng_f,t], q[pl,t], ngLine_data[pl].K_mu*pr[ngLine_data[pl].ng_t,t]] in SecondOrderCone())
    #SOC constraint for the terms in LHS and RHS that are quadratic in uncertainty
    @constraint(m, wm_soc_RT[pl=1:Nng_line, t=1:Nt], [ngLine_data[pl].K_mu*ρ[ngLine_data[pl].ng_f,t], γ[pl,t], ngLine_data[pl].K_mu*ρ[ngLine_data[pl].ng_t,t]] in SecondOrderCone())
    #=
    #McCormick Relaxations for the terms in LHS and RHS that are linear in uncertainty
    @constraint(m, wp_mcm_1[pl=1:Nng_line, t=1:Nt], λ[pl,t] - ngLine_data[pl].K_mu*(σ[ngLine_data[pl].ng_f,t] - σ[ngLine_data[pl].ng_t,t]) <= 0)
    #Rectangular bounds on σ[from_ngbus,t] = pr[from_ngbus,t]* ρ[from_ngbus,t] for each pipeline pl
    @constraint(m, wp_mcm_bounds_f_LL[pl=1:Nng_line, t=1:Nt], 0*pr[ngLine_data[pl].ng_f,t] + ngBus_data[ngLine_data[pl].ng_f].ngPreMin*ρ[ngLine_data[pl].ng_f,t] <= σ[ngLine_data[pl].ng_f,t] + 0*ngBus_data[ngLine_data[pl].ng_f].ngPreMin)
    @constraint(m, wp_mcm_bounds_f_UU[pl=1:Nng_line, t=1:Nt], ((ngBus_data[ngLine_data[pl].ng_f].ngPreMax-ngBus_data[ngLine_data[pl].ng_f].ngPreMin)/wMax)*pr[ngLine_data[pl].ng_f,t] + ngBus_data[ngLine_data[pl].ng_f].ngPreMax*ρ[ngLine_data[pl].ng_f,t]  <= σ[ngLine_data[pl].ng_f,t] + ngBus_data[ngLine_data[pl].ng_f].ngPreMax*(ngBus_data[ngLine_data[pl].ng_f].ngPreMax-ngBus_data[ngLine_data[pl].ng_f].ngPreMin)/wMax)
    @constraint(m, wp_mcm_bounds_f_UL[pl=1:Nng_line, t=1:Nt], 0*pr[ngLine_data[pl].ng_f,t] + ngBus_data[ngLine_data[pl].ng_f].ngPreMax*ρ[ngLine_data[pl].ng_f,t] <= σ[ngLine_data[pl].ng_f,t] + ngBus_data[ngLine_data[pl].ng_f].ngPreMax*0)
    @constraint(m, wp_mcm_bounds_f_LU[pl=1:Nng_line, t=1:Nt], ((ngBus_data[ngLine_data[pl].ng_f].ngPreMax-ngBus_data[ngLine_data[pl].ng_f].ngPreMin)/wMax)*pr[ngLine_data[pl].ng_f,t] + ngBus_data[ngLine_data[pl].ng_f].ngPreMin*ρ[ngLine_data[pl].ng_f,t] <= σ[ngLine_data[pl].ng_f,t] +  ngBus_data[ngLine_data[pl].ng_f].ngPreMin * (ngBus_data[ngLine_data[pl].ng_f].ngPreMax-ngBus_data[ngLine_data[pl].ng_f].ngPreMin)/wMax)
    #Rectangular bounds on σ[to_ngbus,t] = pr[to_ngbus,t]* ρ[to_ngbus,t] for each pipeline pl
    @constraint(m, wp_mcm_bounds_t_LL[pl=1:Nng_line, t=1:Nt], 0*pr[ngLine_data[pl].ng_t,t] + ngBus_data[ngLine_data[pl].ng_t].ngPreMin*ρ[ngLine_data[pl].ng_t,t] <= σ[ngLine_data[pl].ng_t,t] + 0*ngBus_data[ngLine_data[pl].ng_t].ngPreMin)
    @constraint(m, wp_mcm_bounds_t_UU[pl=1:Nng_line, t=1:Nt], ((ngBus_data[ngLine_data[pl].ng_t].ngPreMax-ngBus_data[ngLine_data[pl].ng_t].ngPreMin)/wMax)*pr[ngLine_data[pl].ng_t,t] + ngBus_data[ngLine_data[pl].ng_t].ngPreMax*ρ[ngLine_data[pl].ng_t,t]  <= σ[ngLine_data[pl].ng_t,t] + ngBus_data[ngLine_data[pl].ng_t].ngPreMax*(ngBus_data[ngLine_data[pl].ng_t].ngPreMax-ngBus_data[ngLine_data[pl].ng_t].ngPreMin)/wMax)
    @constraint(m, wp_mcm_bounds_t_UL[pl=1:Nng_line, t=1:Nt], 0*pr[ngLine_data[pl].ng_t,t] + ngBus_data[ngLine_data[pl].ng_t].ngPreMax*ρ[ngLine_data[pl].ng_t,t] <= σ[ngLine_data[pl].ng_t,t] + ngBus_data[ngLine_data[pl].ng_t].ngPreMax*0)
    @constraint(m, wp_mcm_bounds_t_LU[pl=1:Nng_line, t=1:Nt], ((ngBus_data[ngLine_data[pl].ng_t].ngPreMax-ngBus_data[ngLine_data[pl].ng_t].ngPreMin)/wMax)*pr[ngLine_data[pl].ng_t,t] + ngBus_data[ngLine_data[pl].ng_t].ngPreMin*ρ[ngLine_data[pl].ng_t,t] <= σ[ngLine_data[pl].ng_t,t] +  ngBus_data[ngLine_data[pl].ng_t].ngPreMin * (ngBus_data[ngLine_data[pl].ng_t].ngPreMax-ngBus_data[ngLine_data[pl].ng_t].ngPreMin)/wMax)
    #Rectangular bounds on λ[pl, t] = q[pl,t]*γ[pl,t]
    @constraint(m, wp_mcm_bounds_pl_LL[pl=1:Nng_line, t=1:Nt], 0*q[pl,t] + qMin*γ[pl,t] <= λ[pl,t] + qMin*0)
    @constraint(m, wp_mcm_bounds_pl_UU[pl=1:Nng_line, t=1:Nt], ((ngBus_data[ngLine_data[pl].ng_f].ngPreMax - ngBus_data[ngLine_data[pl].ng_t].ngPreMin)/wMax)*q[pl,t] + qMax*γ[pl,t] <= λ[pl,t] + qMax*((ngBus_data[ngLine_data[pl].ng_f].ngPreMax - ngBus_data[ngLine_data[pl].ng_t].ngPreMin)/wMax))
    @constraint(m, wp_mcm_bounds_pl_UL[pl=1:Nng_line, t=1:Nt], 0*q[pl,t] + wMax*γ[pl,t] <=  λ[pl,t] + qMax*0)
    @constraint(m, wp_mcm_bounds_pl_LU[pl=1:Nng_line, t=1:Nt], ((ngBus_data[ngLine_data[pl].ng_f].ngPreMax - ngBus_data[ngLine_data[pl].ng_t].ngPreMin)/wMax)*q[pl,t] +  qMin*γ[pl,t] <= λ[pl,t] + qMin*((ngBus_data[ngLine_data[pl].ng_f].ngPreMax - ngBus_data[ngLine_data[pl].ng_t].ngPreMin)/wMax))
    =#
    #6b. Concave Part of Weymouth Equations - Yet to Come

    #7. Linepack Definition
    @constraint(m, lp_def[pl=1:Nng_line,t=1:Nt], h[pl,t] == ngLine_data[pl].K_h*0.5*(pr[ngLine_data[pl].ng_f,t] + pr[ngLine_data[pl].ng_t,t]))

    #8. Linepack Operation Dynamics Constraints: for t=1, for t>1 and for t=T
    for pl=1:Nng_line
        for t̂=1:Nt
            if t̂ == 1      #First Hour
                @constraint(m, h[pl,t̂] == ngLine_data[pl].H_ini + q_in[pl,t̂] - q_out[pl,t̂])      #linepack in DA
                @constraint(m, ngLine_data[pl].K_h*0.5*(ρ[ngLine_data[pl].ng_f,t̂] + ρ[ngLine_data[pl].ng_t,t̂]) == γ_in[pl,t̂] - γ_out[pl,t̂])  #Linepack in RT
            end
            if t̂ != 1 #All hours other than first
                @constraint(m, h[pl,t̂] == h[pl,t̂-1] + q_in[pl,t̂] - q_out[pl,t̂])
                @constraint(m, ngLine_data[pl].K_h*0.5*(ρ[ngLine_data[pl].ng_f,t̂] + ρ[ngLine_data[pl].ng_t,t̂] - ρ[ngLine_data[pl].ng_f,t̂-1] - ρ[ngLine_data[pl].ng_t,t̂-1]) == γ_in[pl,t̂] - γ_out[pl,t̂])
            end
            if t̂ == Nt #Final Hour
                #@constraint(m, h[pl,t̂] >= ngLine_data[pl].H_ini)       #original
                @constraint(m, [h[pl,t̂] - ngLine_data[pl].H_ini, -(sqrt(z_mu*Σ_t[t̂,t̂])*ngLine_data[pl].K_h*0.5)*(ρ[ngLine_data[pl].ng_f,t̂] + ρ[ngLine_data[pl].ng_t,t̂])] in SecondOrderCone())
                #@constraint(m, [h[pl,t̂] - ngLine_data[pl].H_ini, -sqrt(z_mu*Σ_t[t̂,t̂]*ngLine_data[pl].K_h*0.5)*ρ[ngLine_data[pl].ng_f,t̂], -sqrt(z_mu*Σ_t[t̂,t̂]*ngLine_data[pl].K_h*0.5)*ρ[ngLine_data[pl].ng_f,t̂]] in SecondOrderCone())
            end
        end
    end


    #9. Nodal NG balance
    @constraint(m, λ_ng_da[gnode=1:Nng_bus, t=1:Nt], sum(g[k,t] for k in 1:Ng if ng_prods_data[k].ngProdBusNum==gnode)
                                                - sum(gen_data[i].ngConvEff*p[i,t] for i in 1:Np if gen_data[i].ngBusNum==gnode)
                                                - (sum(q_in[pl,t] for pl in 1:Nng_line if ngLine_data[pl].ng_f==gnode) - sum(q_out[pl,t] for pl in 1:Nng_line if ngLine_data[pl].ng_t==gnode))
                                                == ngBus_data[gnode].ngLoadShare*hourly_demand[t,3])



    @constraint(m, λ_ng_rt[gnode=1:Nng_bus, t=1:Nt], -sum(β[k,t] for k in 1:Ng if ng_prods_data[k].ngProdBusNum==gnode)
                                                + sum(gen_data[i].ngConvEff*α[i,t] for i in 1:Np if gen_data[i].ngBusNum==gnode)
                                                + (sum(γ_in[pl,t] for pl in 1:Nng_line if ngLine_data[pl].ng_f==gnode) - sum(γ_out[pl,t] for pl in 1:Nng_line if ngLine_data[pl].ng_t==gnode))
                                                == 0)

    #System level response balance - alternative
    #=
    @constraint(m, λ_ng_rt_sys[gnode=1:Nng_bus, t=1:Nt], sum(β[k,t] for k in 1:Ng)
                                                - sum(gen_data[i].ngConvEff*α[i,t] for i in 1:Np)
                                                == 0) =#

    #println(λ_el_rt[1])
    @time optimize!(m)
    status = termination_status(m)
    println(status)
    println(raw_status(m))

    #println(wm_soc[:,1])
    @info("DRCC Model M2 status ---> $(status)")

    return status, JuMP.objective_value(m), round.(JuMP.value.(p), digits=2), round.(JuMP.value.(α), digits=2), round.(JuMP.dual.(λ_el_da), digits=2), round.(JuMP.value.(g),digits=2), round.(JuMP.value.(β),digits=2), round.(JuMP.value.(pr),digits=2), round.(JuMP.value.(ρ),digits=2), round.(JuMP.value.(q),digits=2), round.(JuMP.value.(γ),digits=2), round.(JuMP.dual.(λ_ng_rt),digits=2), round.(JuMP.value.(h), digits=2)
    #return status, JuMP.objective_value(m), round.(JuMP.value.(p), digits=2), round.(JuMP.value.(α), digits=2), round.(JuMP.dual.(λ_el_da), digits=2), JuMP.value.(θ)

end

(status, cost, el_prod, el_alpha, el_lmp, ng_prod, ng_beta, ng_pre, ng_rho, ng_flows, ng_gamma, ng_lmp, linepack) = unidir_DRCC_McCormick_SOCP_EL_NG()
#(status, cost, el_prod, el_alpha, el_lmp, vangs) = unidir_DRCC_McCormick_SOCP_EL_NG()
println("EL + NG System Cost:", cost)

#calculate quality of exactness of approximation
wm_exact=DataFrame(t=Any[],pl=Any[], LHS=Any[], RHS=Any[], diff=[], diffPer=Any[])
for hour = 1:Nt
    for pl = 1:Nng_line
        lhs_val= round(ng_flows[pl,hour]^2,digits=2)
        rhs_val = round(ngLine_data[pl].K_mu^2*(ng_pre[ngLine_data[pl].ng_f,hour]^2 - ng_pre[ngLine_data[pl].ng_t,hour]^2), digits=2)
        push!(wm_exact, [hour, pl, lhs_val, rhs_val, abs(lhs_val - rhs_val), 100*abs(lhs_val - rhs_val)/(lhs_val)])
    end
end
#@show wm_exact
println("Total Absolute Error:", sum(wm_exact[:,5]))
println("RMS Error:", sqrt(sum(wm_exact[:,5])/(Nt+Nng_line)))
println("NRMS Error:", sqrt(sum(wm_exact[:,5])/(Nt+Nng_line))/mean(sqrt.(abs.(wm_exact[:,4]))))

println("Gas Response:", any(x->x > 0, ng_beta))
