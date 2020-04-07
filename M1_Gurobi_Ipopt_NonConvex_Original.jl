#Model M1: Deterministic Co-optimization (Anna PowerTech Paper)
using JuMP, Distributions, LinearAlgebra, DataFrames, Ipopt

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


function unidir_deterministic_SOCP_EL_NG()
    #Uncomment for Gurobi - SOC Relaxation
    #m = Model(with_optimizer(Gurobi.Optimizer, Presolve=0, OutputFlag=1, DualReductions=0))
    #Uncomment for Ipopt - Nonconvex Solver
    m = Model(with_optimizer(Ipopt.Optimizer, max_cpu_time=600.0))


    #EL Variables
    @variable(m, p[1:Np, 1:Nt])        #Production from power generators
    @variable(m, w[1:Nw, 1:Nt])        #Production from wind farms
    @variable(m, f[1:Nel_line, 1:Nt])  #Flow in power lines
    @variable(m, θ[1:Nel_bus, 1:Nt])   #Bus angles

    #NG Variables
    @variable(m, g[1:Ng, 1:Nt] >=0)             #Gas production from gas producers
    @variable(m, q_in[1:Nng_line, 1:Nt] >=0)    #Gas inflow in the PipeLine
    @variable(m, q_out[1:Nng_line, 1:Nt] >=0)   #Gas outflow in the PipeLine
    @variable(m, q[1:Nng_line, 1:Nt] >= 0)       #Avg. Gas inflow in the PipeLine
    @variable(m, h[1:Nng_line, 1:Nt] >= 0 )       #Line pack amount
    @variable(m, pr[1:Nng_bus, 1:Nt] >= 0)       #Pressure in gas bus nodes

    #OBJECTIVE function
    @objective(m, Min, sum(sum(p[i,t]*gen_data[i].c for i=1:Np if gen_data[i].ngBusNum==0) + sum(g[k,t]*ng_prods_data[k].C_gas for k= 1:Ng) for t=1:Nt))

    #Uncomment if testing only for el system
    #@objective(m, Min, sum(sum(p[i,t]*gen_data[i].c for i=1:Np if gen_data[i].ngBusNum==0) for t=1:Nt))

    ###-----EL Constraints----###
    #1. All generator Power Limits
    @constraint(m, ϕ_p[i=1:Np, t=1:Nt], gen_data[i].p̲ <= p[i,t] <= gen_data[i].p̅)
    #2. Wind Production bound by Capacity and Wind Factors
    @constraint(m, ϕ_w[j=1:Nw, t=1:Nt], 0 <= w[j,t] <= wind_data[j].Ŵ*wind_factors[t,j+1])
    #3. Definition of power flows and flow limits in a line
    @constraint(m, el_f_def[l=1:Nel_line, t=1:Nt], f[l,t] == ν[elLine_data[l].b_f,elLine_data[l].b_t]*(θ[elLine_data[l].b_f,t] - θ[elLine_data[l].b_t,t]))
    @constraint(m, el_f_lim[l=1:Nel_line,t=1:Nt], -f̅[elLine_data[l].b_f, elLine_data[l].b_t] <= f[l,t] <= f̅[elLine_data[l].b_f, elLine_data[l].b_t])
    #4. El nodal power balance
    @constraint(m, λ_el[elnode=1:Nel_bus, t=1:Nt], sum(p[i,t] for i in 1:Np if gen_data[i].elBusNum==elnode) +
                                                    sum(w[i,t] for i in 1:Nw if wind_data[i].elBusNum==elnode)
                                                    - elBus_data[elnode].elLoadShare*hourly_demand[t,2]
                                                    == sum(B[elnode,r]*θ[r,t] for r=1:Nel_bus))
    #5. El Reference bus
    @constraint(m,ref_el[t=1:Nt], θ[refbus,t] == 0)
    @constraint(m, ang_lims[elnode=1:Nel_bus, t=1:Nt], -Base.MathConstants.pi <= θ[elnode, t] <= Base.MathConstants.pi)

    ###-----NG Constraints----###
    #1. Gas Producers Constraints
    @constraint(m, ϕ_ng[k=1:Ng, t=1:Nt], ng_prods_data[k].G̲ <= g[k,t] <= ng_prods_data[k].G̅)

    #2. Nodal Pressure Constraints
    @constraint(m, ϕ_pr[gnode=1:Nng_bus, t=1:Nt], ngBus_data[gnode].ngPreMin <= pr[gnode,t] <= ngBus_data[gnode].ngPreMax)

    #3. Pipelines with Compressors
    for pl=1:Nng_line
        if ngLine_data[pl].Γ_mu != 1
            @constraint(m,[t=1:Nt], pr[ngLine_data[pl].ng_t,t] <= ngLine_data[pl].Γ_mu*pr[ngLine_data[pl].ng_f,t])
        end
    end

    #5. Definition of average flow in a pipeline
    @constraint(m, q_value[pl=1:Nng_line, t=1:Nt], q[pl,t] == 0.5*(q_in[pl,t] + q_out[pl,t]))

    #6a. Weymouth equation - convex relaxation of equality into a SOC, ignoring the concave part of the cone
    #uncomment if using Gurobi
    #@constraint(m, wm_soc_gurobi[pl=1:Nng_line, t=1:Nt], q[pl,t]*q[pl,t] == ngLine_data[pl].K_mu*ngLine_data[pl].K_mu*(pr[ngLine_data[pl].ng_f,t]*pr[ngLine_data[pl].ng_f,t] - pr[ngLine_data[pl].ng_t,t]*pr[ngLine_data[pl].ng_t,t]))
    #uncomment if using Ipopt
    @constraint(m, wm_nonconvex_ipopt[pl=1:Nng_line, t=1:Nt], q[pl,t]*q[pl,t] == ngLine_data[pl].K_mu*ngLine_data[pl].K_mu*(pr[ngLine_data[pl].ng_f,t]*pr[ngLine_data[pl].ng_f,t] - pr[ngLine_data[pl].ng_t,t]*pr[ngLine_data[pl].ng_t,t]))

    #7. Linepack Definition
    @constraint(m, lp_def[pl=1:Nng_line,t=1:Nt], h[pl,t] == ngLine_data[pl].K_h*0.5*(pr[ngLine_data[pl].ng_f,t] + pr[ngLine_data[pl].ng_t,t]))

    #8. Linepack Operation Dynamics Constraints: for t=1, for t>1 and for t=T
    for pl=1:Nng_line
        for t̂=1:Nt
            if t̂ == 1      #First Hour
                @constraint(m, h[pl,t̂] == ngLine_data[pl].H_ini + q_in[pl,t̂] - q_out[pl,t̂])
            end
            if t̂ != 1 #All hours other than first
                @constraint(m, h[pl,t̂] == h[pl,t̂-1] + q_in[pl,t̂] - q_out[pl,t̂])
            end
            if t̂ == Nt #Final Hour
                @constraint(m, h[pl,t̂] >= ngLine_data[pl].H_ini)
            end
        end
    end

    #9. Nodal NG balance
    @constraint(m, λ_ng[gnode=1:Nng_bus, t=1:Nt], sum(g[k,t] for k in 1:Ng if ng_prods_data[k].ngProdBusNum==gnode)
                                                - sum(gen_data[i].ngConvEff*p[i,t] for i in 1:Np if gen_data[i].ngBusNum==gnode)
                                                - (sum(q_in[pl,t] for pl in 1:Nng_line if ngLine_data[pl].ng_f==gnode) - sum(q_out[pl,t] for pl in 1:Nng_line if ngLine_data[pl].ng_t==gnode))
                                                == ngBus_data[gnode].ngLoadShare*hourly_demand[t,3])

    @time optimize!(m)
    status = termination_status(m)
    println(status)
    println(raw_status(m))


    #println(wm_soc[:,1])
    @info("Deterministic Model status ---> $(status)")
    #uncomment if solving ng+el system
    return status,JuMP.objective_value(m), JuMP.value.(f), round.(JuMP.value.(p),digits=2),JuMP.value.(w), JuMP.value.(θ), JuMP.value.(q_in), JuMP.value.(q_out), JuMP.value.(q), JuMP.value.(h), JuMP.value.(pr), JuMP.value.(g)
    #Uncomment if solving only el system
    #return status,JuMP.objective_value(m),JuMP.dual.(λ_el), JuMP.value.(f), round.(JuMP.value.(p),digits=2),JuMP.value.(w), JuMP.value.(θ)
end

#uncomment if solving ng+el system
(status,cost, elflows, elprod, windgen, vangs, ng_q_inflows, ng_q_outflows, ng_flows, linepack_amount, ng_pre, ng_prod) = unidir_deterministic_SOCP_EL_NG()
#Uncomment if solving only el system
#(status,cost,el_lmp, elflows, elprod, windgen, vangs) = unidir_deterministic_SOCP_EL_NG()
println(cost)


#calculate quality of exactness of approximation
wm_exact=DataFrame(t=Any[],pl=Any[], LHS=Any[], RHS=Any[], diff=[], diffPer=Any[])
for hour = 1:Nt
    for pl = 1:Nng_line
        lhs_val= round(ng_flows[pl,hour]^2,digits=2)
        rhs_val = round(ngLine_data[pl].K_mu^2*(ng_pre[ngLine_data[pl].ng_f,hour]^2 - ng_pre[ngLine_data[pl].ng_t,hour]^2), digits=2)
        push!(wm_exact, [hour, pl, lhs_val, rhs_val, abs(lhs_val - rhs_val), 100*abs(lhs_val - rhs_val)/(lhs_val)])
    end
end
@show wm_exact
println("Total Absolute Error:" , sum(wm_exact[:,5]))
println("RMS Error:", sqrt(sum(wm_exact[:,5])/(Nt+Nng_line)))
println("NRMS Error:", sqrt(sum(wm_exact[:,5])/(Nt+Nng_line))/mean(sqrt.(abs.(wm_exact[:,4]))))
