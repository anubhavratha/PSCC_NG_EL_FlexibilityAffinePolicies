## Test problem no-network case - OOS analysis for DRCC problems - power only##

#Model M0: DRCC Affine Policies with Chebyshev Inequality
using JuMP, Distributions, LinearAlgebra, DataFrames, Mosek, MosekTools
using DataFrames, Statistics


# Case Study 1: 24el + 12ng Bus System : Prepare and load data
include("CS1_24bus/CS1_data_load_script_PSCC.jl")
(elBus_data, gen_data, elLine_data ,B , f̅, ν, π, refbus, ng_prods_data, ngBus_data, ngLine_data, wind_data) = load_data()
#--Reading Uncertainty Data - Wind Farm Forecast Errors
w_hat = CSV.read("CS1_24bus/data/UncertaintyMoments/PSCC_Point_Forecast_Values.csv", header=false)        #Point forecast
#Σ = CSV.read("data/Covariance_Matrix_Data.csv", header=false)                                  #Large Spatial Temporal Covariance Matrix
Σ_t = CSV.read("CS1_24bus/data/UncertaintyMoments/PSCC_TemporallyFolded_Covariance_Matrix_Data.csv", header=false)
hourly_demand = CSV.read("CS1_24bus/data/24el_12ng/hourlyDemand.csv")

Np = length(gen_data)           #number of generators
Nw = length(wind_data)          #number of wind power producers
Nel_line = length(elLine_data)  #number of power lines (l)
Nel_bus = length(elBus_data)    #number of power buses (elnode)

Nt = 24    #Time periods for Simulation Horizon
wind_buses = [5,7]

#Minimum value ~ 0.025
ϵ_i = 0.05  #risk tolerance factor for generators for CC violation
z_i = ((1-ϵ_i)/ϵ_i)     #Fixed multiplier for risk tolerance

z_nr=z_i


function DRCC_EL_PolicyReserves()
    m = Model(with_optimizer(Mosek.Optimizer, MSK_IPAR_LOG=1, MSK_IPAR_INTPNT_SOLVE_FORM=MSK_SOLVE_PRIMAL, MSK_DPAR_INTPNT_CO_TOL_REL_GAP=1.0e-10, MSK_IPAR_PRESOLVE_ELIMINATOR_MAX_NUM_TRIES = 0))

    gen_data[1].c = 40
    gen_data[2].c = 85
    gen_data[5].c = 22
    gen_data[6].c = 23
    gen_data[7].c = 43
    gen_data[10].c = 82
    gen_data[11].c = 21


    #EL Variables
    @variable(m, p[1:Np, 1:Nt])        #Production from power generators
    @variable(m, f[1:Nel_line, 1:Nt])  #Flow in power lines
    @variable(m, θ[1:Nel_bus, 1:Nt])   #Bus angles

    #Affine Response Variables
    @variable(m, α[1:Np, 1:Nt] >= 0)         #Affine response from power Generators

    #Auxiliary Variables I: For EL Flows in real-time
    @variable(m, κ[1:Nel_bus, 1:Nt])        #Variable for computational ease of real-time flow


    #OBJECTIVE function
    @objective(m, Min, sum(sum(p[i,t]*gen_data[i].c for i=1:Np) for t=1:Nt))

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

    @time optimize!(m)
    status = termination_status(m)
    println(status)
    println(raw_status(m))

    @info("DRCC Model M0 status ---> $(status)")

    return status, JuMP.objective_value(m), round.(JuMP.value.(p), digits=2), round.(JuMP.value.(α), digits=2), round.(JuMP.dual.(λ_el_da), digits=2), round.(JuMP.dual.(λ_el_sys), digits=2)
end


function DRCC_EL_PolicyReserves_CopperPlate()
    m_cp = Model(with_optimizer(Mosek.Optimizer, MSK_IPAR_LOG=1, MSK_IPAR_INTPNT_SOLVE_FORM=MSK_SOLVE_PRIMAL, MSK_DPAR_INTPNT_CO_TOL_REL_GAP=1.0e-10, MSK_IPAR_PRESOLVE_ELIMINATOR_MAX_NUM_TRIES = 0))

    gen_data[1].c = 40
    gen_data[2].c = 85
    gen_data[5].c = 22
    gen_data[6].c = 23
    gen_data[7].c = 44
    gen_data[10].c = 82
    gen_data[11].c = 21


    #EL Variables
    @variable(m_cp, p[1:Np, 1:Nt])        #Production from power generators
    #Affine Response Variables
    @variable(m_cp, α[1:Np, 1:Nt] >= 0)         #Affine response from power Generators


    #OBJECTIVE function
    @objective(m_cp, Min, sum(sum(p[i,t]*gen_data[i].c for i=1:Np) for t=1:Nt))

    ###-----EL Constraints----###
    #1. All generator Power Limits
    @constraint(m_cp, ϕ̅_p[i=1:Np, t=1:Nt], [gen_data[i].p̅ - p[i,t], sqrt(z_i*Σ_t[t,t])*α[i,t]] in SecondOrderCone())
    @constraint(m_cp, ϕ̲_p[i=1:Np, t=1:Nt], [p[i,t] - gen_data[i].p̲, -sqrt(z_i*Σ_t[t,t])*α[i,t]] in SecondOrderCone())


    #4. El nodal power balance
    @constraint(m_cp, λ_el_da[t=1:Nt], sum(p[i,t] for i in 1:Np)
                                                    + sum(w_hat[i,t] for i in 1:Nw)
                                                    == sum(elBus_data[elnode].elLoadShare*hourly_demand[t,2] for elnode=1:Nel_bus))

    @constraint(m_cp, λ_el_sys[t=1:Nt], sum(α[i,t] for i=1:Np) == 1)

    @time optimize!(m_cp)
    status = termination_status(m_cp)
    println(status)
    println(raw_status(m_cp))

    @info("DRCC Model M0 Copper Plate status ---> $(status)")

    return status, JuMP.objective_value(m_cp), round.(JuMP.value.(p), digits=2), round.(JuMP.value.(α), digits=2), round.(JuMP.dual.(λ_el_da), digits=2), round.(JuMP.dual.(λ_el_sys), digits=2)
end


#(status, cost, pvals, alphavals, el_lmp_da, el_lmp_rt) = DRCC_EL_PolicyReserves()
