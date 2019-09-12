#Model M2b: DRCC Co-optimization with Exact SOC Reformulation (Xie and Ahmed) : Ignoring Concave part
using JuMP, Distributions, LinearAlgebra, DataFrames, Mosek, MosekTools

#Prepare and load data
include("CS1_24bus/CS1_data_load_script_PSCC.jl")
(elBus_data, gen_data, elLine_data ,B , f̅, ν, π, refbus, ng_prods_data, ngBus_data, ngLine_data, wind_data) = load_data()

hourly_demand = CSV.read("CS1_24bus/data/24el_12ng/hourlyDemand.csv")
wind_factors = CSV.read("CS1_24bus/data/24el_12ng/wind_factors.csv")

Np = length(gen_data)           #number of generators
Ng = length(ng_prods_data)      #number of gas producers
Nw = length(wind_data)          #number of wind power producers
Nel_line = length(elLine_data)  #number of power lines (l)
Nel_bus = length(elBus_data)    #number of power buses (elnode)
Nng_line = length(ngLine_data)  #number of gas pipelines (pl)
Nng_bus = length(ngBus_data)    #number of gas buses (gnode)

Nt = 24       #Time periods for Simulation Horizon
wind_buses = [5,7]

#--Reading Uncertainty Data - Wind Farm Forecast Errors
w_hat = CSV.read("CS1_24bus/data/UncertaintyMoments/PSCC_Point_Forecast_Values.csv", header=false)        #Point forecast
#Σ = CSV.read("data/Covariance_Matrix_Data.csv", header=false)                                  #Large Spatial Temporal Covariance Matrix
Σ_t = CSV.read("CS1_24bus/data/UncertaintyMoments/PSCC_TemporallyFolded_Covariance_Matrix_Data.csv", header=false)

ϵ_i = 0.05  #risk tolerance factor for generators for CC violation

ϵ_ng = ϵ_pr = ϵ_nr = ϵ_i

z_i = ((1-ϵ_i)/ϵ_i)     #Fixed multiplier for risk tolerance
z_g = z_mu = z_nr = z_i

#McCormick Bounds
qMin = 0
qMax = 10000
wMax = 1000

function unidir_DRCC_McCormick_SOCP_EL_NG()
    m = Model(with_optimizer(Mosek.Optimizer, MSK_IPAR_LOG=1, MSK_IPAR_INTPNT_SOLVE_FORM=MSK_SOLVE_PRIMAL, MSK_DPAR_INTPNT_CO_TOL_REL_GAP=1.0e-10, MSK_IPAR_PRESOLVE_ELIMINATOR_MAX_NUM_TRIES = 0))

    #EL Variables
    @variable(m, p[1:Np, 1:Nt])        #Production from power generators
    @variable(m, f[1:Nel_line, 1:Nt])  #Flow in power lines
    @variable(m, θ[1:Nel_bus, 1:Nt])   #Bus angles

    #NG Variables
    @variable(m, g[1:Ng, 1:Nt] >=0)             #Gas production from gas producers
    @variable(m, q_in[1:Nng_line, 1:Nt] >=0)    #Gas inflow in the PipeLine
    @variable(m, q_out[1:Nng_line, 1:Nt] >=0)   #Gas outflow in the PipeLine
    @variable(m, q[1:Nng_line, 1:Nt] >=0)       #Avg. Gas inflow in the PipeLine
    @variable(m, h[1:Nng_line, 1:Nt])       #Line pack amount
    @variable(m, pr[1:Nng_bus, 1:Nt])       #Pressure in gas bus nodes

    #Affine Response Variables
    @variable(m, α[1:Np, 1:Nt] >=0)         #Affine response from power Generators
    @variable(m, β[1:Ng, 1:Nt] >=0)         #Affine response from gas producers
    @variable(m, γ[1:Nng_line, 1:Nt] >= 0)            #Affine change in average gas flow in pipeline
    @variable(m, γ_in[1:Nng_line, 1:Nt] >= 0)         #Affine change in gas inflow in the pipleine
    @variable(m, γ_out[1:Nng_line, 1:Nt] >= 0)        #Affine change in pipeline inflow pressure
    @variable(m, ρ[1:Nng_bus, 1:Nt] >= 0)             #Affine change in nodal gas pressure

    #Auxiliary Variables I: For EL Flows in real-time
    @variable(m, κ[1:Nel_bus, 1:Nt])        #Variable for computational ease of real-time flow

    #Auxiliary Variables II: For Exact SOC formulation of DRCC
    @variable(m, p_y[1:Np, 1:Nt] >=0)
    @variable(m, p_π[1:Np, 1:Nt] >=0)
    @variable(m, f_y[1:Nel_line, 1:Nt] >= 0)
    @variable(m, f_π[1:Nel_line, 1:Nt]>= 0)
    @variable(m, g_y[1:Ng, 1:Nt] >= 0)
    @variable(m, g_π[1:Ng, 1:Nt] >= 0)
    @variable(m, pr_y[1:Nng_bus, 1:Nt] >= 0)
    @variable(m, pr_π[1:Nng_bus, 1:Nt] >= 0)

    #Auxiliary Variables III: For NG Weymouth Equation - McCormick bounds
    @variable(m, σ[1:Nng_bus, 1:Nt])    #Bilinear term of product of pr and ρ for each gas node
    @variable(m, λ[1:Nng_line, 1:Nt])   #Bilinear term of product of flow q and γ for each gas pipeline

    #OBJECTIVE function
    #Uncomment for co-optimization of el and ng
    @objective(m, Min, sum(sum(p[i,t]*gen_data[i].c for i=1:Np if gen_data[i].ngBusNum==0) + sum(g[k,t]*ng_prods_data[k].C_gas for k= 1:Ng) for t=1:Nt))
    #uncomment for just EL
    #@objective(m, Min, sum(sum(p[i,t]*gen_data[i].c for i=1:Np if gen_data[i].ngBusNum==0) for t=1:Nt))

    ###-----EL Constraints----###
    #1. All generator Power Limits
    @constraint(m, ϕ_p̅[i=1:Np, t=1:Nt], p[i,t] - p_y[i,t] - p_π[i,t] <= 0.5*(gen_data[i].p̅ + gen_data[i].p̲))
    @constraint(m, ϕ_p̲[i=1:Np, t=1:Nt], - p[i,t] - p_y[i,t] - p_π[i,t] <= -0.5*(gen_data[i].p̅ + gen_data[i].p̲))
    @constraint(m, ϕ_p_soc[i=1:Np, t=1:Nt], [sqrt(ϵ_i)*(0.5*(gen_data[i].p̅ - gen_data[i].p̲) - p_π[i,t]), p_y[i,t], α[i,t]*sqrt(Σ_t[t,t])] in SecondOrderCone())
    @constraint(m, ϕ_p_pi[i=1:Np, t=1:Nt], p_π[i,t] <= 0.5*(gen_data[i].p̅ - gen_data[i].p̲))


    #2. Constraints on alpha: deviation balance, refbus, bounds on alpha
    @constraint(m, α_bounds[i=1:Np, t=1:Nt], 0 <= α[i,t] <= 1)

    #3. Power flow along lines

    for t=1:Nt
        for l=1:Nel_line
            if ν[elLine_data[l].b_f,elLine_data[l].b_t] != 0
                norm_args = []
                for b = wind_buses
                    norm_args=push!(norm_args,@expression(m, (sqrt((1/z_nr)*Σ_t[t,t]))*ν[elLine_data[l].b_f,elLine_data[l].b_t]*(π[elLine_data[l].b_f,b]-π[elLine_data[l].b_t,b]-κ[elLine_data[l].b_f,t]+κ[elLine_data[l].b_t,t])))
                end
                network_SOCP=vcat(f̅[elLine_data[l].b_f, elLine_data[l].b_t] - ν[elLine_data[l].b_f,elLine_data[l].b_t]*(θ[elLine_data[l].b_f,t] - θ[elLine_data[l].b_t,t]),norm_args)
                @constraint(m, network_SOCP in SecondOrderCone())
            end
        end
    end


    #Exact reform for line flows - trial
    #=
    for t=1:Nt
        for l=1:Nel_line
            if ν[elLine_data[l].b_f,elLine_data[l].b_t] != 0
                @constraint(m, [sqrt(ϵ_nr)*(f̅[elLine_data[l].b_f, elLine_data[l].b_t] - f_π[l,t]), f_y[l,t], ν[elLine_data[l].b_f,elLine_data[l].b_t]*sqrt(Σ_t[t,t])*(π[elLine_data[l].b_f,b]-π[elLine_data[l].b_t,b]-κ[elLine_data[l].b_f,t]+κ[elLine_data[l].b_t,t])] in SecondOrderCone())
                @constraint(m, ν[elLine_data[l].b_f,elLine_data[l].b_t]*(θ[elLine_data[l].b_f,t] - θ[elLine_data[l].b_t,t]) <= f_y[l,t] + f_π[l,t])
                @constraint(m, ν[elLine_data[l].b_f,elLine_data[l].b_t]*(θ[elLine_data[l].b_t,t] - θ[elLine_data[l].b_f,t]) <= f_y[l,t] + f_π[l,t])
                @constraint(m, f_π[l,t] <= f̅[elLine_data[l].b_f, elLine_data[l].b_t])
            end
        end
    end
    =#

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
    @constraint(m, ϕ̅_ng[k=1:Ng, t=1:Nt], g[k,t] - g_y[k,t] - g_π[k,t] <= 0.5*(ng_prods_data[k].G̅ + ng_prods_data[k].G̲))
    @constraint(m, ϕ̲_ng[k=1:Ng, t=1:Nt], - g[k,t] - g_y[k,t] - g_π[k,t] <= -0.5*(ng_prods_data[k].G̅ + ng_prods_data[k].G̲))
    @constraint(m, ϕ_ng_soc[k=1:Ng, t=1:Nt], [sqrt(ϵ_ng)*(0.5*(ng_prods_data[k].G̅ - ng_prods_data[k].G̲) - g_π[k,t]), g_y[k,t], β[k,t]*sqrt(Σ_t[t,t])] in SecondOrderCone())
    @constraint(m, ϕ_ng_π[k=1:Ng, t=1:Nt], g_π[k,t] <= 0.5*(ng_prods_data[k].G̅ - ng_prods_data[k].G̲))


    #2. Nodal Pressure Constraints
    #@constraint(m, ϕ_pr[gnode=1:Nng_bus, t=1:Nt], ngBus_data[gnode].ngPreMin <= pr[gnode,t] <= ngBus_data[gnode].ngPreMax) #original
    @constraint(m, ϕ̅_pr[gnode=1:Nng_bus, t=1:Nt], pr[gnode,t] - pr_y[gnode,t] - pr_π[gnode,t] <= 0.5*(ngBus_data[gnode].ngPreMax + ngBus_data[gnode].ngPreMin))
    @constraint(m, ϕ̲_pr[gnode=1:Nng_bus, t=1:Nt], - pr[gnode,t] - pr_y[gnode,t] - pr_π[gnode,t] <= -0.5*(ngBus_data[gnode].ngPreMax + ngBus_data[gnode].ngPreMin))
    @constraint(m, ϕ_pr_soc[gnode=1:Nng_bus, t=1:Nt], [sqrt(ϵ_pr)*(0.5*(ngBus_data[gnode].ngPreMax - ngBus_data[gnode].ngPreMin) - pr_π[gnode,t]), pr_y[gnode,t], ρ[gnode,t]*sqrt(Σ_t[t,t])] in SecondOrderCone())
    @constraint(m,ϕ_pr_π[gnode=1:Nng_bus, t=1:Nt], pr_π[gnode,t] <= 0.5*(ngBus_data[gnode].ngPreMax - ngBus_data[gnode].ngPreMin))

    #3. Pipelines with Compressors
    for pl=1:Nng_line
        if ngLine_data[pl].Γ_mu != 1
            #@constraint(m,[t=1:Nt], pr[ngLine_data[pl].ng_t,t] <= ngLine_data[pl].Γ_mu*pr[ngLine_data[pl].ng_f,t])
            @constraint(m, [t=1:Nt], [-(pr[ngLine_data[pl].ng_t,t] - ngLine_data[pl].Γ_mu*pr[ngLine_data[pl].ng_f,t]), (sqrt(z_mu*Σ_t[t,t]))*(ρ[ngLine_data[pl].ng_t,t]) - ngLine_data[pl].Γ_mu*ρ[ngLine_data[pl].ng_f,t]] in SecondOrderCone())
        end
    end

    #4. Uni-directionality of flows in all pipelines
    @constraint(m, q_nonneg[pl=1:Nng_line, t=1:Nt], [q[pl,t], -(sqrt(z_g*Σ_t[t,t]))*γ[pl,t]] in SecondOrderCone())
    @constraint(m, qin_nonneg[pl=1:Nng_line, t=1:Nt], [q_in[pl,t], -(sqrt(z_g*Σ_t[t,t]))*γ_in[pl,t]] in SecondOrderCone())
    @constraint(m, qout_nonneg[pl=1:Nng_line, t=1:Nt], [q_out[pl,t], -(sqrt(z_g*Σ_t[t,t]))*γ_out[pl,t]] in SecondOrderCone())


    #5. Definition of average flow in a pipeline : for DA and real-time
    @constraint(m, q_value[pl=1:Nng_line, t=1:Nt], q[pl,t] == 0.5*(q_in[pl,t] + q_out[pl,t]))
    @constraint(m, γ_values[pl=1:Nng_line, t=1:Nt], γ[pl,t] == 0.5*(γ_in[pl,t] + γ_out[pl,t]))

    #6a. Weymouth equations - convex relaxation of equality into a SOC, ignoring the concave part of the cone
    #SOC constraint for the terms in LHS and RHS that are independent of uncertainty
    @constraint(m, wm_soc_DA[pl=1:Nng_line, t=1:Nt], [ngLine_data[pl].K_mu*pr[ngLine_data[pl].ng_f,t], q[pl,t], ngLine_data[pl].K_mu*pr[ngLine_data[pl].ng_t,t]] in SecondOrderCone())
    #SOC constraint for the terms in LHS and RHS that are quadratic in uncertainty
    @constraint(m, wm_soc_RT[pl=1:Nng_line, t=1:Nt], [ngLine_data[pl].K_mu*ρ[ngLine_data[pl].ng_f,t], γ[pl,t], ngLine_data[pl].K_mu*ρ[ngLine_data[pl].ng_t,t]] in SecondOrderCone())
    #=
    #Non-convex constraint formed by terms in LHS and RHS that are linear in uncertainty
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


    #println(λ_el_rt[1])
    @time optimize!(m)
    status = termination_status(m)
    println(status)
    println(raw_status(m))
    println(λ_ng_rt[:,1])

    @info("DRCC Model M2b status ---> $(status)")

    return status, JuMP.objective_value(m), round.(JuMP.value.(p), digits=2), round.(JuMP.value.(α), digits=2), round.(JuMP.dual.(λ_el_da), digits=2), round.(JuMP.value.(g),digits=2), round.(JuMP.value.(β),digits=2), round.(JuMP.value.(pr),digits=2), round.(JuMP.value.(ρ),digits=2), round.(JuMP.value.(q),digits=2), round.(JuMP.value.(γ),digits=2), round.(JuMP.dual.(λ_ng_da),digits=2), round.(JuMP.value.(h), digits=2)

end

(status, cost, el_prod, el_alpha, el_lmp, ng_prod, ng_beta, ng_pre, ng_rho, ng_flows, ng_gamma, ng_lmp, linepack) = unidir_DRCC_McCormick_SOCP_EL_NG()
println(cost)
