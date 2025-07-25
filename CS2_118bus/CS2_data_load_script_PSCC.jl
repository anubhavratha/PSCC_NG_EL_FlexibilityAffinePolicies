using CSV, DataFrames
using DataStructures: SortedDict

mutable struct PowerGenerators
   ind::Int
   c::Any
   p̅::Any
   p̲::Any
   α̅::Any
   α̲::Any
   elBusNum::Int
   ngBusNum::Int
   ngConvEff::Any
   function PowerGenerators(ind,c,p̅,p̲,α̅,α̲,elBusNum,ngBusNum,ngConvEff)
      i = new()
      i.ind  = ind
      i.c = c
      i.p̅ = p̅
      i.p̲ = p̲
      i.α̅ = α̅
      i.α̲ = α̲
      i.elBusNum = elBusNum
      i.ngBusNum = ngBusNum
      i.ngConvEff = ngConvEff
      return i
   end
end

mutable struct elBus
   ind::Int
   elLoadNum::Int
   elLoad::Float64
   function elBus(ind,elLoadNum,elLoad)
      i = new()
      i.ind  = ind
      i.elLoadNum = elLoadNum
      i.elLoad = elLoad
      return i
   end
end

mutable struct elLine
   ind::Int
   b_f::Int
   b_t::Int
   ν::Float64
   f̅::Float64
   function elLine(ind,b_f,b_t,ν,f̅)
      i = new()
      i.ind  = ind
      i.b_f = b_f
      i.b_t = b_t
      i.ν = ν
      i.f̅ = f̅
      return i
   end
end

mutable struct ngBus
   ind::Int
   ngLoad::Float64
   ngPreMin::Float64
   ngPreMax::Float64
   ngPreInit::Float64
   function ngBus(ind,ngLoad,ngPreMin,ngPreMax,ngPreInit)
      i = new()
      i.ind  = ind
      i.ngLoad = ngLoad
      i.ngPreMin = ngPreMin
      i.ngPreMax = ngPreMax
      i.ngPreInit = ngPreInit
      return i
   end
end

mutable struct ngLine
   ind::Int
   ng_f::Int
   ng_t::Int
   K_mu::Any
   K_h::Any
   function ngLine(ind,ng_f,ng_t,K_mu,K_h)
      i = new()
      i.ind  = ind
      i.ng_f = ng_f
      i.ng_t = ng_t
      i.K_mu = K_mu
      i.K_h = K_h
      return i
   end
end

mutable struct GasProducers
    ind::Int
    ngProdBusNum::Int
    G̅::Int
    G̲::Int
    C_gas::Float64
    function GasProducers(ind,ngProdBusNum,G̅,G̲,C_gas)
        i = new()
        i.ind = ind
        i.G̅ = G̅
        i.G̲ = G̲
        i.C_gas = C_gas
        i.ngProdBusNum = ngProdBusNum
        return i
    end
end

mutable struct ngCompressors
    ind::Int
    ngCompInNode:: Int
    ngCompOutNode:: Int
    ngComp_R̲:: Float64
    ngComp_R̅:: Float64
    ngComp_Eff:: Float64
    ngComp_Fmax:: Float64
    function ngCompressors(ind, ngCompInNode, ngCompOutNode, ngComp_R̲, ngComp_R̅, ngComp_Eff,ngComp_Fmax)
        i = new()
        i.ind = ind
        i.ngCompInNode = ngCompInNode
        i.ngCompOutNode = ngCompOutNode
        i.ngComp_R̲ = ngComp_R̲
        i.ngComp_R̅ = ngComp_R̅
        i.ngComp_Eff = ngComp_Eff
        i.ngComp_Fmax = ngComp_Fmax
        return i
    end
end



mutable struct WindGenerators
    ind::Int
    Ŵ::Int
    elBusNum::Int
    function WindGenerators(ind,elBusNum,Ŵ)
        i = new()
        i.ind = ind
        i.Ŵ = Ŵ
        i.elBusNum = elBusNum
        return i
    end
end

function load_data()
    all_gens_data = CSV.read("data/118el_48ng/all_gens.csv")
    el_bus_data = CSV.read("data/118el_48ng/el_bus_data.csv")
    el_line_data = CSV.read("data/118el_48ng/el_line_data.csv")

    #Dict for generators data extracted from CSV file
    gen_data = Dict()
    for i in 1:nrow(all_gens_data)
        ind = all_gens_data[i, :UnitNum]
        c = all_gens_data[i, :C_1]
        p̅ = all_gens_data[i, :PG_max]
        p̲ = all_gens_data[i, :PG_min]
        α̅ = 1
        α̲ = 0
        elBusNum = all_gens_data[i, :elBusNum]
        ngBusNum = all_gens_data[i, :ngBusNum]
        ngConvEff = all_gens_data[i, :ng_ConvEff]
        add_generator = PowerGenerators(ind,c,p̅,p̲,α̅,α̲,elBusNum,ngBusNum,ngConvEff)
        gen_data[add_generator.ind] = add_generator
    end

    #Dicts for electrical bus and line data extracted from CSV file
    elBus_data = Dict()
    for i in 1:nrow(el_bus_data)
        ind = el_bus_data[i, :elBusNum]
        elLoadNum = el_bus_data[i, :loadNum]
        elLoad = el_bus_data[i, :P_dem]
        add_bus = elBus(ind,elLoadNum,elLoad)
        elBus_data[add_bus.ind] = add_bus
    end
    elLine_data = Dict()
    for i in 1:nrow(el_line_data)
        ind = el_line_data[i, :LineNum]
        b_f = el_line_data[i, :From]
        b_t = el_line_data[i, :To]
        ν = 1/el_line_data[i, :adm]
        f̅ = el_line_data[i, :f_max]
        add_line = elLine(ind,b_f,b_t,ν,f̅)
        elLine_data[add_line.ind] = add_line
    end


    # sort dictionaries
    gen_data=SortedDict(gen_data)
    elBus_data=SortedDict(elBus_data)
    elLine_data=SortedDict(elLine_data)

    Nb = length(elBus_data)
    line_set=collect(keys(elLine_data))

    B = zeros(Nb,Nb)        #Bus Susceptance Matrix
    #Filling of diagonal entries of the BSM
    for i in 1:Nb
        for l in line_set
            if elLine_data[l].b_f == i || elLine_data[l].b_t  == i
                B[i,i] += elLine_data[l].ν
            end
        end
    end
    #Filling of off-diagonal elements
    for l in line_set
        B[elLine_data[l].b_f,elLine_data[l].b_t] = -elLine_data[l].ν
        B[elLine_data[l].b_t,elLine_data[l].b_f] = -elLine_data[l].ν
    end

    #Max flow limits - if a line exists between two buses, it takes the value of the flow limit, else 0
    f̅ = zeros(Nb,Nb)
    for l in line_set
        f̅[elLine_data[l].b_f,elLine_data[l].b_t] = elLine_data[l].f̅
        f̅[elLine_data[l].b_t,elLine_data[l].b_f] = elLine_data[l].f̅
    end

    #Susceptance of line
    ν = zeros(Nb,Nb)
    for l in line_set
        ν[elLine_data[l].b_f,elLine_data[l].b_t] = elLine_data[l].ν
        ν[elLine_data[l].b_t,elLine_data[l].b_f] = elLine_data[l].ν
    end

    function remove_col_and_row(B,refbus)
        @assert size(B,1) == size(B,2)
        n = size(B,1)
        return B[1:n .!= refbus, 1:n .!= refbus]
    end

    function build_B̆(B̂inv,refbus)
        Nb = size(B̂inv,1)+1
        B̆ = zeros(Nb,Nb)
        for i in 1:Nb, j in 1:Nb
            if i < refbus && j < refbus
                B̆[i,j] = B̂inv[i,j]
            end
            if i > refbus && j > refbus
                B̆[i,j] = B̂inv[i-1,j-1]
            end
            if i > refbus && j < refbus
                B̆[i,j] = B̂inv[i-1,j]
            end
            if i < refbus && j > refbus
                B̆[i,j] = B̂inv[i,j-1]
            end
        end
        return B̆
    end

    refbus = 24
    B̂=remove_col_and_row(B,refbus)
    B̂inv = inv(B̂)
    π=build_B̆(B̂inv,refbus)

    ng_bus_data = CSV.read("data/118el_48ng/ng_bus_data.csv")
    ng_line_data = CSV.read("data/118el_48ng/ng_line_data.csv")
    ng_producers_data = CSV.read("data/118el_48ng/ng_producers.csv")
    ng_compressors_data = CSV.read("data/118el_48ng/ng_compressors.csv")

    #Dict for gas producers data extracted from CSV file
    ng_prods_data = Dict()
    for i in 1:nrow(ng_producers_data)
        ind = ng_producers_data[i, :ngProdNum﻿]
        C_gas = ng_producers_data[i, :C_prod]
        G̅ = ng_producers_data[i, :Prod_max]
        G̲ = ng_producers_data[i, :Prod_min]
        ngProdBusNum = ng_producers_data[i, :Gnode]
        add_gasproducer = GasProducers(ind, ngProdBusNum, G̅, G̲, C_gas)
        ng_prods_data[add_gasproducer.ind] = add_gasproducer
    end

    ngBus_data = Dict()
    for i in 1:nrow(ng_bus_data)
        ind = ng_bus_data[i, :ngBusNum﻿]
        ngLoad = ng_bus_data[i, :G_dem]
        ngPreMax = ng_bus_data[i, :Pre_max]
        ngPreMin = ng_bus_data[i, :Pre_min]
        ngPreInit = ng_bus_data[i, :Pre_ini]
        add_ngbus = ngBus(ind,ngLoad,ngPreMin,ngPreMax,ngPreInit)
        ngBus_data[add_ngbus.ind] = add_ngbus
    end

    ngLine_data = Dict()
    for i in 1:nrow(ng_line_data)
        ind = ng_line_data[i, :PipeLineNum]
        ng_f = ng_line_data[i, :From]
        ng_t = ng_line_data[i, :To]
        K_mu = ng_line_data[i, :Kmu]
        K_h = ng_line_data[i, :K_h]
        add_ngline = ngLine(ind, ng_f, ng_t, K_mu, K_h)
        ngLine_data[add_ngline.ind] = add_ngline
    end

    #Dict for ng compressors
    ngComp_data = Dict()
    for i in 1:nrow(ng_compressors_data)
        ind = ng_compressors_data[i, :CompNum]
        in_node = ng_compressors_data[i, :In]
        out_node = ng_compressors_data[i, :Out]
        comp_ratio_min = ng_compressors_data[i, :R_min]
        comp_ratio_max = ng_compressors_data[i, :R_max]
        comp_eff = ng_compressors_data[i, :Eff]
        comp_flow_max = ng_compressors_data[i, :Fcom_max]
        add_ngComp = ngCompressors(ind, in_node, out_node, comp_ratio_min, comp_ratio_max, comp_eff, comp_flow_max)
        ngComp_data[add_ngComp.ind] = add_ngComp
    end

    #=
    #Wind Generators and Multiplier parameters
    wind_gens_data = CSV.read("data/24el_12ng/wind_gens.csv")
    wind_data = Dict()
    for i in 1:nrow(wind_gens_data)
        ind = wind_gens_data[i, :WindNum]
        Ŵ = wind_gens_data[i, :W_instcap]
        elBusNum = wind_gens_data[i, :elBusNum]
        add_windgen = WindGenerators(ind,elBusNum,Ŵ)
        wind_data[add_windgen.ind] = add_windgen
    end
    =#

    # sort dictionaries
    #ng_prods_data=SortedDict(ng_prods_data)
    ngBus_data=SortedDict(ngBus_data)
    ngLine_data=SortedDict(ngLine_data)
    ngComp_data = SortedDict(ngComp_data)
#   wind_gens_data=SortedDict(wind_gens_data)


    return elBus_data, gen_data, elLine_data, B, f̅, ν, π, refbus, ng_prods_data, ngBus_data, ngLine_data, ngComp_data
end

#Uncomment to test the data generation file
(elBus_data,gen_data,elLine_data,B,f̅,ν,π,refbus,ng_prods_data,ngBus_data,ngLine_data,ngComp_data) = load_data()
