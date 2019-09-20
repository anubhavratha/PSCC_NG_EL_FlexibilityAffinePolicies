##Coordination of Power and Natural Gas Systems: Convexification Approaches of Linepack Modeling
##Copyright (C) 2018 Anna Schwele
##
##This program is free software: you can redistribute it and/or modify
##t under the terms of the GNU General Public License as published by
##the Free Software Foundation, either version 3 of the License, or
##(at your option) any later version.
##
##This program is distributed in the hope that it will be useful,
##but WITHOUT ANY WARRANTY; without even the implied warranty of
##MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##GNU General Public License for more details.

import numpy as np
import pandas as pd
import gurobipy as gb
import pprint
import csv

def input():

	solution = {

	'PP':{
	'i1':{'Pmax':152,'ELnode':'n1', 'phi':12.65, 'NGnode':'m12','type':'GFPP'},
	'i2':{'Pmax':152,'ELnode':'n2', 'phi':20.25, 'NGnode':'m12','type':'GFPP'},
	'i3':{'Pmax':300,'ELnode':'n7', 'C_E':65.61,'type':'TPP'},
	'i4':{'Pmax':591,'ELnode':'n13', 'C_E':30.82,'type':'TPP'},
	'i5':{'Pmax':60,'ELnode':'n15', 'phi':11.12, 'NGnode':'m10','type':'GFPP'},
	'i6':{'Pmax':155,'ELnode':'n15', 'phi':15.1, 'NGnode':'m10','type':'GFPP'},
	'i7':{'Pmax':155,'ELnode':'n16', 'phi':14.88, 'NGnode':'m7','type':'GFPP'},
	'i8':{'Pmax':400,'ELnode':'n18', 'C_E':20.84,'type':'TPP'},
	'i9':{'Pmax':400,'ELnode':'n21', 'C_E':26.9,'type':'TPP'},
	'i10':{'Pmax':300,'ELnode':'n22', 'phi':13.3, 'NGnode':'m6','type':'GFPP'},
	'i11':{'Pmax':60,'ELnode':'n23', 'phi':16.8, 'NGnode':'m6','type':'GFPP'},
	'i12':{'Pmax':350,'ELnode':'n23', 'C_E':32.22,'type':'TPP'}},

	'Windfarm':{
	'j1':{'Wmax':500, 'ELnode': 'n5'},
	'j2':{'Wmax':1000, 'ELnode': 'n7'}},

	'WindFactor':{
	't1':{'j1':0.8,'j2':0.9},
	't2':{'j1':0.9,'j2':0.85},
	't3': {'j1':0.85,'j2':0.75},
	't4': {'j1':0.6,'j2':0.7},
	't5': {'j1':0.7,'j2':0.8},
	't6':{'j1':0.68,'j2':0.58},
	't7': {'j1':0.5,'j2':0.56},
	't8': {'j1':0.3,'j2':0.4},
	't9': {'j1':0.45,'j2':0.2},
	't10':{'j1':0.85,'j2':0.75},
	't11': {'j1':0.7,'j2':0.75},
	't12':{'j1':0.15,'j2':0.35},
	't13': {'j1':0.2,'j2':0.25},
	't14':{'j1':0.4,'j2':0.3},
	't15': {'j1':0.35,'j2':0.25},
	't16':{'j1':0.65,'j2':0.7},
	't17': {'j1':0.5,'j2':0.2},
	't18':{'j1':0.25,'j2':0.1},
	't19': {'j1':0.1,'j2':0.15},
	't20': {'j1':0.2,'j2':0.15},
	't21': {'j1':0.15,'j2':0.1},
	't22': {'j1':0.15,'j2':0.1},
	't23': {'j1':0.5,'j2':0.7},
	't24': {'j1':0.65,'j2':0.77}},

	'GasSupply':{
	'k1':{'Gmax':6000, 'C_G':2, 'NGnode':'m1'},
	'k2':{'Gmax':8000, 'C_G':2.4, 'NGnode':'m3'},
	'k3':{'Gmax':15000, 'C_G':3.2, 'NGnode':'m11'}},

	'ELDemand':{
	'd1' : {'share' : 0.038,'node': 'n1'},
	'd2' : {'share' : 0.034,'node': 'n2'},
	'd3' : {'share' : 0.063,'node': 'n3'},
	'd4' : {'share' : 0.026,'node': 'n4'},
	'd5' : {'share' : 0.025,'node': 'n5'},
	'd6' : {'share' : 0.048,'node': 'n6'},
	'd7' : {'share' : 0.044,'node': 'n7'},
	'd8' : {'share' : 0.060,'node': 'n8'},
	'd9' : {'share' : 0.061,'node': 'n9'},
	'd10' : {'share' : 0.068,'node': 'n10'},
	'd11' : {'share' : 0.093,'node': 'n13'},
	'd12' : {'share' : 0.068,'node': 'n14'},
	'd13' : {'share' : 0.111,'node': 'n15'},
	'd14' : {'share' : 0.035,'node': 'n16'},
	'd15' : {'share' : 0.117,'node': 'n18'},
	'd16' : {'share' : 0.064,'node': 'n19'},
	'd17' : {'share' : 0.045,'node': 'n20'}},

	'NGDemand':{
	'd1' : {'share' : 0.25,'node': 'm5'},
	'd2' : {'share' : 0.25,'node': 'm7'},
	'd3' : {'share' : 0.35,'node': 'm6'},
	'd4' : {'share' : 0.15,'node': 'm12'}},

	'Demand':{
	't1':{'EL':2108.73, 'NG':7000},
	't2':{'EL':2002.63, 'NG':6700,'prev':'t1'},
	't3':{'EL':1573.98, 'NG':6400,'prev':'t2'},
	't4':{'EL':1230.90, 'NG':6500,'prev':'t3'},
	't5':{'EL':1251.00, 'NG':6600,'prev':'t4'},
	't6':{'EL':1282.04, 'NG':6700,'prev':'t5'},
	't7':{'EL':1778.60, 'NG':7750,'prev':'t6'},
	't8':{'EL':2344.03, 'NG':7800,'prev':'t7'},
	't9':{'EL':2461.77, 'NG':8550,'prev':'t8'},
	't10':{'EL':2702.58, 'NG':8700,'prev':'t9'},
	't11':{'EL':2741.14, 'NG':8700,'prev':'t10'},
	't12':{'EL':2535.38, 'NG':8550,'prev':'t11'},
	't13':{'EL':2482.55, 'NG':8550,'prev':'t12'},
	't14':{'EL':2367.35, 'NG':8550,'prev':'t13'},
	't15':{'EL':2741.30, 'NG':8400,'prev':'t14'},
	't16':{'EL':2875.31, 'NG':8550,'prev':'t15'},
	't17':{'EL':2597.63, 'NG':9000,'prev':'t16'},
	't18':{'EL':2715.35, 'NG':9000,'prev':'t17'},
	't19':{'EL':2879.81, 'NG':9000,'prev':'t18'},
	't20':{'EL':2989.47, 'NG':8700,'prev':'t19'},
	't21':{'EL':3000.00, 'NG':8250,'prev':'t20'},
	't22':{'EL':2713.75, 'NG':7500,'prev':'t21'},
	't23':{'EL':2610.42, 'NG':6600,'prev':'t22'},
	't24':{'EL':2437.59, 'NG':6700,'prev':'t23'}},

	'Lines': {
	'l1' : {'From' : 'n1', 'To' : 'n2','B' : 0.0146, 'capacity' : 175},
	'l2' : {'From' : 'n1', 'To' : 'n3','B' : 0.2253, 'capacity' : 175},
	'l3' : {'From' : 'n1', 'To' : 'n5','B' : 0.0907, 'capacity' : 500},
	'l4' : {'From' : 'n2', 'To' : 'n4','B' : 0.1356, 'capacity' : 175},
	'l5' : {'From' : 'n2', 'To' : 'n6','B' : 0.205, 'capacity' : 175},
	'l6' : {'From' : 'n3', 'To' : 'n9','B' : 0.1271, 'capacity' : 175},
	'l7' : {'From' : 'n3', 'To' : 'n24','B' : 0.084, 'capacity' : 400},
	'l8' : {'From' : 'n4', 'To' : 'n9','B' : 0.111, 'capacity' : 175},
	'l9' : {'From' : 'n5', 'To' : 'n10','B' : 0.094, 'capacity' : 500},
	'l10' : {'From' : 'n6', 'To' : 'n10','B' : 0.0642, 'capacity' : 175},
	'l11' : {'From' : 'n7', 'To' : 'n8','B' : 0.0652, 'capacity' : 1000},
	'l12' : {'From' : 'n8', 'To' : 'n9','B' : 0.1762, 'capacity' : 175},
	'l13' : {'From' : 'n8', 'To' : 'n10','B' : 0.1762, 'capacity' : 175},
	'l14' : {'From' : 'n9', 'To' : 'n11','B' : 0.084, 'capacity' : 400},
	'l15' : {'From' : 'n9', 'To' : 'n12','B' : 0.084, 'capacity' : 400},
	'l16' : {'From' : 'n10', 'To' : 'n11','B' : 0.084, 'capacity' : 400},
	'l17' : {'From' : 'n10', 'To' : 'n12','B' : 0.084, 'capacity' : 400},
	'l18' : {'From' : 'n11', 'To' : 'n13','B' : 0.0488, 'capacity' : 500},
	'l19' : {'From' : 'n11', 'To' : 'n14','B' : 0.0426, 'capacity' : 500},
	'l20' : {'From' : 'n12', 'To' : 'n13','B' : 0.0488, 'capacity' : 500},
	'l21' : {'From' : 'n12', 'To' : 'n23','B' : 0.0985, 'capacity' : 500},
	'l22' : {'From' : 'n13', 'To' : 'n23','B' : 0.0884, 'capacity' : 500},
	'l23' : {'From' : 'n14', 'To' : 'n16','B' : 0.0594, 'capacity' : 500},
	'l24' : {'From' : 'n15', 'To' : 'n16','B' : 0.0172, 'capacity' : 500},
	'l25' : {'From' : 'n15', 'To' : 'n21','B' : 0.0249, 'capacity' : 500},
	'l26' : {'From' : 'n15', 'To' : 'n24','B' : 0.0529, 'capacity' : 500},
	'l27' : {'From' : 'n16', 'To' : 'n17','B' : 0.0263, 'capacity' : 500},
	'l28' : {'From' : 'n16', 'To' : 'n19','B' : 0.0234, 'capacity' : 500},
	'l29' : {'From' : 'n17', 'To' : 'n18','B' : 0.0143, 'capacity' : 500},
	'l30' : {'From' : 'n17', 'To' : 'n22','B' : 0.1069, 'capacity' : 500},
	'l31' : {'From' : 'n18', 'To' : 'n21','B' : 0.0132, 'capacity' : 1000},
	'l32' : {'From' : 'n19', 'To' : 'n20','B' : 0.0203, 'capacity' : 1000},
	'l33' : {'From' : 'n20', 'To' : 'n23','B' : 0.0112, 'capacity' : 1000},
	'l34' : {'From' : 'n21', 'To' : 'n22','B' : 0.0692, 'capacity' : 500}},

	'Pipelines': {
	'z1':{'From':'m1','To':'m2','K_f':28,'type':'passive','Gamma':1,'K_h':121,'H_ini':39300},
	'z2':{'From':'m2','To':'m4','K_f':28,'type':'passive','Gamma':1.2,'K_h':121,'H_ini':39300},
	'z3':{'From':'m3','To':'m5','K_f':28,'type':'passive','Gamma':1,'K_h':150,'H_ini':49300},
	'z4':{'From':'m4','To':'m5','K_f':21,'type':'passive','Gamma':1,'K_h':186,'H_ini':59300},
	'z5':{'From':'m5','To':'m6','K_f':21,'type':'passive','Gamma':1,'K_h':189,'H_ini':54300},
	'z6':{'From':'m4','To':'m7','K_f':21,'type':'passive','Gamma':1,'K_h':184,'H_ini':54300},
	'z7':{'From':'m6','To':'m8','K_f':28,'type':'passive','Gamma':1,'K_h':150,'H_ini':39300},
	'z8':{'From':'m7','To':'m8','K_f':21,'type':'passive','Gamma':1,'K_h':179,'H_ini':44300},
	'z9':{'From':'m8','To':'m9','K_f':28,'type':'passive','Gamma':1.3,'K_h':149,'H_ini':39300},
	'z10':{'From':'m9','To':'m10','K_f':28,'type':'passive','Gamma':1,'K_h':148,'H_ini':39300},
	'z11':{'From':'m10','To':'m11','K_f':28,'type':'passive','Gamma':1,'K_h':150,'H_ini':39300},
	'z12':{'From':'m11','To':'m12','K_f':28,'type':'passive','Gamma':1,'K_h':130,'H_ini':29300}},

	'ELNodes' : {'n1','n2','n3','n4','n5','n6','n7','n8','n9','n10','n11','n12','n13','n14','n15','n16','n17','n18','n19','n20','n21','n22','n23','n24'},

	'NGNodes':{'m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12'},

#	'NGPr':{
#	'm1':{'minPr': 350, 'maxPr': 470},
#	'm2':{'minPr': 310, 'maxPr': 430},
#	'm3':{'minPr': 350, 'maxPr': 470},
#	'm4':{'minPr': 330, 'maxPr': 460},
#	'm5':{'minPr': 290, 'maxPr': 430},
#	'm6':{'minPr': 250, 'maxPr': 390},
#	'm7':{'minPr': 240, 'maxPr': 360},
#	'm8':{'minPr': 230, 'maxPr': 350},
#	'm9':{'minPr': 260, 'maxPr': 400},
#	'm10':{'minPr': 240, 'maxPr': 360},
#	'm11':{'minPr': 250, 'maxPr': 380},
#	'm12':{'minPr': 100, 'maxPr': 350}},

	'NGPr':{
	'm1':{'minPr': 100, 'maxPr': 500},
	'm2':{'minPr': 100, 'maxPr': 500},
	'm3':{'minPr': 100, 'maxPr': 500},
	'm4':{'minPr': 100, 'maxPr': 500},
	'm5':{'minPr': 100, 'maxPr': 500},
	'm6':{'minPr': 100, 'maxPr': 500},
	'm7':{'minPr': 100, 'maxPr': 500},
	'm8':{'minPr': 100, 'maxPr': 500},
	'm9':{'minPr': 100, 'maxPr': 500},
	'm10':{'minPr': 100, 'maxPr': 500},
	'm11':{'minPr': 100, 'maxPr': 500},
	'm12':{'minPr': 100, 'maxPr': 500}}
	}

	return solution

def del_keys(data, key):
	#  Delete column from dictionary:
	return {a:{c:d for c, d in b.items() if c != key} for a, b in data.items()}

def units_in_node(data,key):
	# Mapping:
	temp_dict= {}
	for k, v in data.items():
		   for k2, v2 in v.items():
				      temp_dict[(k, k2)] = v2

	length = sum(value == key for value in temp_dict.values())
	solution=[]

	for g in range(length):
			solution.append([k for k, v in temp_dict.items() if v == key][g][0])

	return solution

# Model
def SOCP(data):

	# Create new model
	m = gb.Model()

	# Create indices
	I = [i for i, i_info in data['PP'].items()]
	C = [i for i, i_info in data['PP'].items() if i_info['type']=='TPP']
	G = [i for i, i_info in data['PP'].items() if i_info['type']=='GFPP']
	J = [j for j, j_info in data['Windfarm'].items()]
	L = [l for l, l_info in data['Lines'].items()]
	N = list(data['ELNodes'])
	R_E = [d for d, d_info in data['ELDemand'].items()]

	l_r = del_keys(data['Lines'],'From')                 #lines ending at nodes
	l_n = del_keys(data['Lines'],'To')                   #lines starting from nodes

	T = [t for t, t_info in data['Demand'].items()]
	K = [k for k, k_info in data['GasSupply'].items()]
	Z = [z for z, z_info in data['Pipelines'].items()]
	MZ = [mz for mz, mz_info in data['NGPr'].items()]
	R_G = [d for d, d_info in data['NGDemand'].items()]

	z_m = del_keys(data['Pipelines'],'To')                   #pipelines starting from nodes
	z_u = del_keys(data['Pipelines'],'From')                 #pipelines ending at nodes

	previous = del_keys(data['Demand'],'prev')                 #previous time period

	# Create variables
	p = {}
	for i in I:
		for t in T:
			p[(i,t)] = m.addVar(lb=0.0, ub=data['PP'][i]['Pmax'], name='Generation of power plant; {}; {};'.format(i,t))
	p

	w = {}
	for j in J:
		for t in T:
			w[(j,t)] = m.addVar(lb=0.0, ub=data['Windfarm'][j]['Wmax']*data['WindFactor'][t][j], name='Generation of wind farm; {}; {};'.format(j,t))
	w

	f = {}
	for l in L:
		for t in T:
			f[(l,t)] =  m.addVar(lb=-data['Lines'][l]['capacity'], ub=data['Lines'][l]['capacity'], name='Flow in line; {}; {};'.format(l,t))
	f

	theta= {}
	for n in N:
		for t in T:
			theta[(n,t)] = m.addVar(lb=-np.pi, ub=np.pi, name='Voltage angle at node; {}; {};'.format(n,t))
	theta

	g= {}
	for k in K:
		for t in T:
			g[(k,t)] = m.addVar(lb=0.0, ub=data['GasSupply'][k]['Gmax'], name='Production of gas supply; {}; {};'.format(k,t))
	g

	q_in_plus= {}
	for z in Z:
		for t in T:
			q_in_plus[(z,t)] = m.addVar(lb=0, ub=gb.GRB.INFINITY, name='Inflow to pipeline; {}; {};'.format(z,t))
	q_in_plus

	q_out_plus= {}
	for z in Z:
		for t in T:
			q_out_plus[(z,t)] = m.addVar(lb=0, ub=gb.GRB.INFINITY, name='Outflow from pipeline; {}; {};'.format(z,t))
	q_out_plus


	h= {}
	for z in Z:
		for t in T:
			h[(z,t)] = m.addVar(lb=0, ub=gb.GRB.INFINITY, name='Linepack in pipeline; {}; {};'.format(z,t))
	h

	pr= {}
	for mz in MZ:
		for t in T:
			pr[(mz,t)] = m.addVar(lb=data['NGPr'][mz]['minPr'], ub=data['NGPr'][mz]['maxPr'], name='pressure {};'.format(mz,t))
	pr

	q= {}
	for z in Z:
		for t in T:
			q[(z,t)] = m.addVar(lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY, name='Flow over pipeline from m to u; {}; {};'.format(z,t))
	q
	
#	slack= {}
#	for z in Z:
#		for t in T:
#			slack[(z,t)] = m.addVar(lb=0, ub=gb.GRB.INFINITY, name='slack; {}; {};'.format(z,t))
#	slack
	


	m.update()

	#Objective function
	m.setObjective(gb.quicksum(data['PP'][i]['C_E']*p[(i,t)] for i in C for t in T)  # Cost of running conventionals
		+ gb.quicksum(data['GasSupply'][k]['C_G'] * g[(k,t)] for k in K for t in T)
		#+ gb.quicksum(1000* slack[(z,t)] for z in Z for t in T)
		, gb.GRB.MINIMIZE)  # Cost of gas supply
		
	m.update()

	#Constraints:
	# Reference node constraint
	for t in T:
		ref_n = 'n1'
		m.addConstr(theta[(ref_n,t)],gb.GRB.EQUAL,0 ,name="reference node{}".format(t))

	# Transmission capacity
	for l in L:
		n_node = data['Lines'][l]['From']
		r_node = data['Lines'][l]['To']

		for t in T:
			m.addConstr(f[(l,t)],gb.GRB.EQUAL, (1/data['Lines'][l]['B'])*250*(theta[(n_node,t)]-theta[(r_node,t)]) ,name="power flow over line{}{}".format(l,t))

	# Nodal power balance
	for n in N:
		A_I = units_in_node(data['PP'],n)
		A_J = units_in_node(data['Windfarm'],n)
		A_DE = units_in_node(data['ELDemand'],n)
		FromLines = units_in_node(l_n,n)        #Lines that start from node n
		ToLines = units_in_node(l_r,n)          #Lines that end in node n
		for t in T:
			m.addConstr(gb.quicksum(p[(i,t)] for i in A_I) + gb.quicksum(w[(j,t)] for j in A_J) - gb.quicksum(f[(l,t)] for l in FromLines) 
			+ gb.quicksum(f[(l,t)] for l in ToLines) ,gb.GRB.EQUAL, gb.quicksum(data['ELDemand'][d]['share']*data['Demand'][t]['EL'] for d in A_DE) ,name="power balance node{}{}".format(n,t))
	
	
	# Nodal gas balance
	for mz in MZ:
		A_K = units_in_node(data['GasSupply'],mz)
		A_DG = units_in_node(data['NGDemand'],mz)
		A_G = units_in_node(data['PP'],mz)
		FromPipelines = units_in_node(z_m,mz)        #Lines that start from node n
		ToPipelines = units_in_node(z_u,mz)          #Lines that end in node n
		for t in T:
			m.addConstr(gb.quicksum(g[(k,t)] for k in A_K)
			 - gb.quicksum(q_in_plus[(z,t)] for z in FromPipelines) 
			 + gb.quicksum(q_out_plus[(z,t)] for z in ToPipelines)
			 - gb.quicksum(data['PP'][i]['phi']*p[(i,t)] for i in A_G)
			 ,gb.GRB.EQUAL, gb.quicksum(data['NGDemand'][d]['share']*data['Demand'][t]['NG'] for d in A_DG) ,name="gasbalance node{}{}".format(m,t))

	for z in Z:
		m_node = data['Pipelines'][z]['From']
		u_node = data['Pipelines'][z]['To']
		print(u_node)
		for t in T:
			#Compressor
			if data['Pipelines'][z]['Gamma']!=1:
				m.addConstr((pr[(u_node,t)]), gb.GRB.LESS_EQUAL, data['Pipelines'][z]['Gamma'] *(pr[(m_node,t)]) ,name="compressor{}{}".format(z,t))
			#Bounds on auxiliary variables Phi
			#m.addConstr((data['NGPr'][m_node]['minPr']+data['NGPr'][u_node]['minPr']), gb.GRB.LESS_EQUAL, phi_plus[(z,t)] ,name="Phi plus lower bound{}{}".format(z,t))
			#m.addConstr(phi_plus[(z,t)], gb.GRB.LESS_EQUAL,(data['NGPr'][m_node]['maxPr']+data['NGPr'][u_node]['maxPr']) ,name="Phi plus upper bound{}{}".format(z,t))
			#m.addConstr((data['NGPr'][m_node]['minPr']-data['NGPr'][u_node]['maxPr']), gb.GRB.LESS_EQUAL, phi_minus[(z,t)] ,name="Phi minus lower bound{}{}".format(z,t))
			#m.addConstr(phi_minus[(z,t)], gb.GRB.LESS_EQUAL,(data['NGPr'][m_node]['maxPr']-data['NGPr'][u_node]['minPr']) ,name="Phi minus upper bound{}{}".format(z,t))
			#McCormick envelopes
			#m.addConstr(psi[(z,t)], gb.GRB.GREATER_EQUAL, (data['NGPr'][m_node]['minPr']+data['NGPr'][u_node]['minPr'])*phi_minus[(z,t)] + phi_plus[(z,t)]*(data['NGPr'][m_node]['minPr']-data['NGPr'][u_node]['maxPr']) -(data['NGPr'][m_node]['minPr']+data['NGPr'][u_node]['minPr'])*(data['NGPr'][m_node]['minPr']-data['NGPr'][u_node]['maxPr']) ,name="McC1{}{}".format(z,t))
			#m.addConstr(psi[(z,t)], gb.GRB.GREATER_EQUAL, (data['NGPr'][m_node]['maxPr']+data['NGPr'][u_node]['maxPr'])*phi_minus[(z,t)] + phi_plus[(z,t)]*(data['NGPr'][m_node]['maxPr']-data['NGPr'][u_node]['minPr']) -(data['NGPr'][m_node]['maxPr']+data['NGPr'][u_node]['maxPr'])*(data['NGPr'][m_node]['maxPr']-data['NGPr'][u_node]['minPr']) ,name="McC2{}{}".format(z,t))
			#m.addConstr(psi[(z,t)], gb.GRB.LESS_EQUAL, (data['NGPr'][m_node]['maxPr']+data['NGPr'][u_node]['maxPr'])*phi_minus[(z,t)] + phi_plus[(z,t)]*(data['NGPr'][m_node]['minPr']-data['NGPr'][u_node]['maxPr']) -(data['NGPr'][m_node]['maxPr']+data['NGPr'][u_node]['maxPr'])*(data['NGPr'][m_node]['minPr']-data['NGPr'][u_node]['maxPr']) ,name="McC3{}{}".format(z,t))
			#m.addConstr(psi[(z,t)], gb.GRB.LESS_EQUAL, (data['NGPr'][m_node]['minPr']+data['NGPr'][u_node]['minPr'])*phi_minus[(z,t)] + phi_plus[(z,t)]*(data['NGPr'][m_node]['maxPr']-data['NGPr'][u_node]['minPr']) -(data['NGPr'][m_node]['minPr']+data['NGPr'][u_node]['minPr'])*(data['NGPr'][m_node]['maxPr']-data['NGPr'][u_node]['minPr']) ,name="McC4{}{}".format(z,t))
			#MISOC constraints
			m.addConstr(q[(z,t)]*q[(z,t)],gb.GRB.LESS_EQUAL,data['Pipelines'][z]['K_f']*data['Pipelines'][z]['K_f'] *(pr[(m_node,t)]*pr[(m_node,t)]-pr[(u_node,t)]*pr[(u_node,t)]) ,name="wey1 {}{}".format(z,t))
			#m.addConstr(q[(z,t)]*q[(z,t)],gb.GRB.LESS_EQUAL,data['Pipelines'][z]['K_f']*data['Pipelines'][z]['K_f'] *(-psi[(z,t)]) + data['BigM']*data['BigM']*(y[(z,t)]),name="wey2 {}{}".format(z,t))
			#Bidirectional flow
			#m.addConstr(q[(z,t)],gb.GRB.LESS_EQUAL,data['BigM']*y[(z,t)],name="bin1 {}{}".format(z,t))
			#m.addConstr(q[(z,t)],gb.GRB.GREATER_EQUAL,-data['BigM']*(1-y[(z,t)]),name="bin2 {}{}".format(z,t))
			#m.addConstr(q[z,t],gb.GRB.EQUAL,q_plus[(z,t)]-q_minus[(z,t)],name="flow in pipeline {}{}".format(z,t))
			#m.addConstr(q_plus[z,t],gb.GRB.LESS_EQUAL,data['BigM']*y[z,t],name="binary1 {}{}".format(z,t))
			#m.addConstr(q_minus[z,t],gb.GRB.LESS_EQUAL,data['BigM']*(1-y[z,t]),name="binary2 {}{}".format(z,t))
			#Average flow
			m.addConstr(q[(z,t)],gb.GRB.EQUAL,0.5*(q_in_plus[(z,t)]+q_out_plus[(z,t)]),name="average flow from m to u {}{}".format(z,t))
			#m.addConstr(q_minus[(z,t)],gb.GRB.EQUAL,0.5*(q_in_minus[(z,t)]+q_out_minus[(z,t)]),name="average flow from u to m {}{}".format(z,t))
			#Linepack 
			m.addConstr(h[(z,t)],gb.GRB.EQUAL,data['Pipelines'][z]['K_h']*0.5*(pr[(m_node,t)]+pr[(u_node,t)] ),name="lineflow from m to u {}{}".format(z,t))
			#m.addConstr(q_in_plus[(z,t)],gb.GRB.EQUAL,q_out_plus[(z,t)],name="in and out flow from m to u {}{}".format(z,t))
			#m.addConstr(q_in_minus[(z,t)],gb.GRB.EQUAL,q_out_minus[(z,t)],name="in and out flow from u to m {}{}".format(z,t))
		#Linepack mass conservation
		for t in T:
			#Initial linepack
			if t=='t1':
				m.addConstr(h[(z,t)],gb.GRB.EQUAL, data['Pipelines'][z]['H_ini']+ q_in_plus[z,t] - q_out_plus[z,t],name="initial lineflow from m to u {}".format(z))
			#Linepack mass
			if t!='t1':
				previous = data['Demand'][t]['prev']
				m.addConstr(h[(z,t)],gb.GRB.EQUAL,h[(z,previous)] + q_in_plus[z,t] - q_out_plus[z,t] ,name="lineflow mass conservation from m to u {}{}".format(z,t))
			#Final linepack
			if t=='t24':
				m.addConstr(h[(z,t)],gb.GRB.GREATER_EQUAL, data['Pipelines'][z]['H_ini'],name="final lineflow from m to u {}".format(z))
	
	#m.Params.FeasibilityTol=0.01
	#m.Params.OptimalityTol=0.01
	#m.Params.Gap= 0.5
	m.optimize()
	#m.printAttr('X')

	return m.status, ([np.round(v.x,2) for v in m.getVars() if "Flow in line" in v.varName ]), ([np.round(v.x,2) for v in m.getVars() if "pressure" in v.varName ])
data=input()
status,elflows,pressures =SOCP(data)
print(status)
print("====")
print(elflows)
print("====")


#Calculate quality of exactness of approximation
wm_exact = pd.DataFrame(columns = ["t", "pl", "LHS", "RHS", "diff", "diffPer"])
T = [t for t, t_info in data['Demand'].items()]
Z = [z for z, z_info in data['Pipelines'].items()]
MZ = [mz for mz, mz_info in data['NGPr'].items()]
flows_changed = np.reshape(flows, (len(Z),len(T)))
pressure_changed = np.reshape(pressures,(len(MZ),len(T)))
print(pressure_changed)
iter=0
for t in T:
    for pl in Z:
        lhs_val = flows_changed[int(pl.replace('z',''))-1,int(t.replace('t',''))-1]*flows_changed[int(pl.replace('z',''))-1,int(t.replace('t',''))-1]
        m_node = data['Pipelines'][pl]['From']
        u_node = data['Pipelines'][pl]['To']
        print(m_node)
        print(u_node)
        rhs_val = data['Pipelines'][pl]['K_f']*data['Pipelines'][pl]['K_f']*(pressure_changed[int(m_node.replace('m',''))-1,int(t.replace('t',''))-1]*pressure_changed[int(m_node.replace('m',''))-1,int(t.replace('t',''))-1] - pressure_changed[int(u_node.replace('m',''))-1,int(t.replace('t',''))-1]*pressure_changed[int(u_node.replace('m',''))-1,int(t.replace('t',''))-1])
        wm_exact.loc[iter] = [t, pl, lhs_val, rhs_val, abs(lhs_val - rhs_val),100*abs(lhs_val - rhs_val)/(lhs_val)]
        iter=iter+1
print(wm_exact)
print("Total Absolute Error:{}".format(sum(wm_exact['diff'])/1e6))
print("RMS Error: {}".format(np.sqrt(sum(wm_exact['diff'])/(len(T)+len(Z)))))
print("NRMS Error: {}".format(np.sqrt(sum(wm_exact['diff'])/(len(T)+len(Z)))/np.mean(np.sqrt(abs(wm_exact['LHS'])))))
