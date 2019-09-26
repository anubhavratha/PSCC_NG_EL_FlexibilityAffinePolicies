function undir_DRCC_rt_operation(w_hat, m2_el_prod, m2_el_alpha, m2_ng_prod, m2_ng_beta, m2_ng_pre, m2_ng_rho, m2_ng_flows, m2_ng_gamma, m2_ng_inflows, m2_ng_gamma_in, m2_ng_outflows, m2_ng_gamma_out, Scenario)

    Δ = []
    #Realized net system deviation
    for t=1:Nt
        push!(Δ, sum(w_hat[i,t] for i in 1:Nw) - (sum(wind_realizations[(wind_realizations.WFNum.==i) .& (wind_realizations.ScenNum .== Scenario), t][1] for i = 1:Nw)))
    end

    #Check Feasibility of line-flows
    m2_rt = Model(with_optimizer(Mosek.Optimizer, MSK_IPAR_LOG=1, MSK_IPAR_INTPNT_SOLVE_FORM=MSK_SOLVE_PRIMAL, MSK_DPAR_INTPNT_CO_TOL_REL_GAP=1.0e-10, MSK_IPAR_PRESOLVE_ELIMINATOR_MAX_NUM_TRIES = 0))

    #EL adjustment variables
    @variable(m2_rt, r_act[1:Np, 1:Nt])
    @variable(m2_rt, l_shed[1:Nel_bus, 1:Nt])
    @variable(m2_rt, w_spill[1:Nw, 1:Nt])
    @variable(m2_rt, f_adj[1:Nel_line, 1:Nt])
    @variable(m2_rt, θ_adj[1:Nel_bus, 1:Nt])

    #NG adjustment variables
    @variable(m2_rt, pr_adj[1:Nng_bus, t=1:Nt])
    @variable(m2_rt, q_adj[1:Nng_line, t=1:Nt])
    @variable(m2_rt, q_in_adj[1:Nng_line, t=1:Nt])
    @variable(m2_rt, q_out_adj[1:Nng_line, t=1:Nt])
    @variable(m2_rt, h_rt[1:Nng_line, t=1:Nt])
    @variable(m2_rt, g_adj[1:Ng, t=1:Nt])


    @objective(m2_rt, Min, 0)
    #@objective(m2_rt, Min, sum(sum(C_shed*l_shed[elnode,t] for elnode = 1: Nel_bus) + sum(C_spill*w_spill[i,t] for i = 1:Nw) for t=1:Nt))


    ###-----EL Constraints----###
    @constraint(m2_rt, ϕ_r_act[i=1:Np, t=1:Nt], r_act[i,t] == Δ[t]*m2_el_alpha[i,t])
    @constraint(m2_rt, ϕ_l_shed[elnode=1:Nel_bus, t=1:Nt], 0 <= l_shed[elnode,t] <= elBus_data[elnode].elLoadShare*hourly_demand[t,2])
    @constraint(m2_rt, ϕ_w_spill[i=1:Nw, t=1:Nt], 0 <= w_spill[i,t] <= wind_realizations[(wind_realizations.WFNum.==i) .& (wind_realizations.ScenNum .== Scenario), t][1])

    @constraint(m2_rt, el_f_lim_rt[l=1:Nel_line,t=1:Nt], -f̅[elLine_data[l].b_f, elLine_data[l].b_t] <= f_adj[l,t] <= f̅[elLine_data[l].b_f, elLine_data[l].b_t])


    @constraint(m2_rt, λ_el_rt[elnode=1:Nel_bus, t=1:Nt], sum(r_act[i,t] for i in 1:Np if gen_data[i].elBusNum==elnode)
                                                    + sum((wind_realizations[(wind_realizations.WFNum.==i) .& (wind_realizations.ScenNum .== Scenario), t][1] - w_hat[i,t] - w_spill[i,t]) for i in 1:Nw if wind_data[i].elBusNum==elnode)
                                                   + l_shed[elnode,t]  - sum(B[elnode,r]*θ_adj[r,t] for r=1:Nel_bus) == 0)

    #@constraint(m2_rt, el_f_def[l=1:Nel_line, t=1:Nt], f_adj[l,t] == ν[elLine_data[l].b_f,elLine_data[l].b_t]*(θ_adj[elLine_data[l].b_f,t] - θ_adj[elLine_data[l].b_t,t]))
    #@constraint(m2_rt, el_f_lim[l=1:Nel_line,t=1:Nt], -f̅[elLine_data[l].b_f, elLine_data[l].b_t] <= (m2_elflows[l,t] + f_adj[l,t]) <= f̅[elLine_data[l].b_f, elLine_data[l].b_t])

    #5. El Reference bus
    @constraint(m2_rt, ref_el_rt[t=1:Nt], θ_adj[refbus, t] == 0)

    ###-----NG Constraints----###
    @constraint(m2_rt, ϕ_ng_rt[k=1:Ng, t=1:Nt], g_adj[k,t] == Δ[t]*m2_ng_beta[k,t])

    #2. Nodal Pressure Constraints
    @constraint(m2_rt, ϕ_pr_rt[gnode=1:Nng_bus, t=1:Nt], pr_adj[gnode,t] == Δ[t]*m2_ng_rho[gnode,t])

    #3. Pipelines with Compressors
    for pl=1:Nng_line
        if ngLine_data[pl].Γ_mu != 1
            @constraint(m2_rt,[t=1:Nt], (m2_ng_pre[ngLine_data[pl].ng_t,t] + pr_adj[ngLine_data[pl].ng_t,t]) <= ngLine_data[pl].Γ_mu*(m2_ng_pre[ngLine_data[pl].ng_f,t] + pr_adj[ngLine_data[pl].ng_f,t]))
        end
    end

    #5. Definition of average flow in a pipeline
    @constraint(m2_rt, q_value_rt[pl=1:Nng_line, t=1:Nt], q_adj[pl,t] == Δ[t]*m2_ng_gamma[pl,t])
    @constraint(m2_rt, q_in_value_rt[pl=1:Nng_line, t=1:Nt], q_in_adj[pl,t] == Δ[t]*m2_ng_gamma_in[pl,t])
    @constraint(m2_rt, q_out_value_rt[pl=1:Nng_line, t=1:Nt], q_out_adj[pl,t] == Δ[t]*m2_ng_gamma_out[pl,t])


    #6a. Weymouth equation - convex relaxation of equality into a SOC, ignoring the concave part of the cone
    #uncomment if using MOSEK - SecondOrderCone special formulation
    @constraint(m2_rt, wm_soc_rt[pl=1:Nng_line, t=1:Nt], [ngLine_data[pl].K_mu*(m2_ng_pre[ngLine_data[pl].ng_f,t] + pr_adj[ngLine_data[pl].ng_f,t]), (m2_ng_flows[pl,t] + q_adj[pl,t]), ngLine_data[pl].K_mu*(m2_ng_pre[ngLine_data[pl].ng_t,t] + pr_adj[ngLine_data[pl].ng_t,t])] in SecondOrderCone())

    #7. Linepack Definition
    @constraint(m2_rt, lp_def_rt[pl=1:Nng_line,t=1:Nt], h_rt[pl,t] == ngLine_data[pl].K_h*0.5*((m2_ng_pre[ngLine_data[pl].ng_f,t] + pr_adj[ngLine_data[pl].ng_f,t]) + (m2_ng_pre[ngLine_data[pl].ng_t,t] + pr_adj[ngLine_data[pl].ng_t,t])))

    #8. Linepack Operation Dynamics Constraints: for t=1, for t>1 and for t=T
    for pl=1:Nng_line
        for t̂=1:Nt
            if t̂ == 1      #First Hour
                @constraint(m2_rt, h_rt[pl,t̂] == ngLine_data[pl].H_ini + (m2_ng_inflows[pl,t̂] + q_in_adj[pl,t̂]) - (m2_ng_outflows[pl,t̂] + q_out_adj[pl,t̂]))
            end
            if t̂ != 1 #All hours other than first
                @constraint(m2_rt, h_rt[pl,t̂] == h_rt[pl,t̂-1] + (m2_ng_inflows[pl,t̂] + q_in_adj[pl,t̂]) - (m2_ng_outflows[pl,t̂] + q_out_adj[pl,t̂]))
            end
            if t̂ == Nt #Final Hour
                @constraint(m2_rt, h_rt[pl,t̂] >= ngLine_data[pl].H_ini)
            end
        end
    end

    #9. Nodal NG balance
    @constraint(m2_rt, λ_ng_rt[gnode=1:Nng_bus, t=1:Nt], sum(g_adj[k,t] for k in 1:Ng if ng_prods_data[k].ngProdBusNum==gnode) - sum(gen_data[i].ngConvEff*r_act[i,t] for i in 1:Np if gen_data[i].ngBusNum==gnode)
                                                - (sum(q_in_adj[pl,t] for pl in 1:Nng_line if ngLine_data[pl].ng_f==gnode) - sum(q_out_adj[pl,t] for pl in 1:Nng_line if ngLine_data[pl].ng_t==gnode))
                                                == 0)

    @time optimize!(m2_rt)
    status = termination_status(m2_rt)
    println(status)
    println(raw_status(m2_rt))

    @info("DRCC RT EL Redispatch Model status ---> $(status)")


    return Δ, status,JuMP.objective_value(m2_rt), round.(JuMP.value.(r_act), digits=2), round.(JuMP.value.(l_shed), digits=2),  round.(JuMP.value.(w_spill), digits=2), JuMP.value.(q_in_adj), JuMP.value.(q_out_adj), JuMP.value.(q_adj), JuMP.value.(h_rt), JuMP.value.(pr_adj), JuMP.value.(g_adj)
end


(Δ, m2_rd_status, m2_rd_cost, m2_rd_p_adj, m2_lshed, m2_wspill, m2_rd_qin_adj, m2_rd_qout_adj, m2_rd_q_adj, m2_rd_h_adj, m2_rd_pr_adj, m2_g_adj) = undir_DRCC_rt_operation(w_hat,  m2_el_prod, m2_el_alpha, m2_ng_prod, m2_ng_beta, m2_ng_pre, m2_ng_rho, m2_ng_flows, m2_ng_gamma, m2_ng_inflows, m2_ng_gamma_in, m2_ng_outflows, m2_ng_gamma_out, 1)
