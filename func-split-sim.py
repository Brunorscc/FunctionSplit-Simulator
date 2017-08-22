import simpy
import random
import functools
import sys
import argparse
import logging
import pandas as pd
from collections import defaultdict
import os 
import math
from itertools import izip
from operator import itemgetter
dir_path = os.path.dirname(os.path.realpath(__file__))

""" For cloudified C-RAN architecture see 'Radio Base Stations in the Cloud' paper """

# _ means 'per'
# DOWNLINK is NOT simulated yet
#### PHY layer parameters assumptions
CPRI_data=8.0
CPRI_total=10.0
CPRI_coding=CPRI_total/CPRI_data

n_RB= 100.0
n_SUB_SYM= 14.0
n_RB_SC= 12.0
n_Data_Sym=12.0

nAnt=2.0 		# number of antennas (2x2MIMO=4 antenna, 2 TX(DL) and 2 RX(UL))
n_Sector= 4		# 2x2MIMO = 4 setor
A = nAnt * n_Sector

QmPUSCH=4.0 	# 16QAM
LayersUL=1.0	# Could also be 2 (2x2 mimo). But, BW calc of splits 4 and 5 would return the same bandwidth..
RefSym_RES=6.0	# Number of REs for reference signals per RB per sub-frame (for 1 or 2 antennas) 
nIQ=32.0 		#16I + 16Q bits
PUCCH_RBs=4.0 #RBs allocated for PUCCH
nLLR=8.0
nSIC=1.0		# No SIC

#### L2/L3 layers parameters assumptions
HdrPDCP=2.0
HdrRLC=5.0
HdrMAC=2.0
IPpkt=1500.0
nTBS_UL_TTI= 2.0	# 1 Layer
Sched=0.5 		# Scheduler overhead per UE
FAPI_UL=1.0		# Uplink FAPI overhead per UE in Mbps
MCS_UL=28		# Dynamic tho. Attribute of each 'packet generated'

# Tables
mcs_2_tbs=[0,1,2,3,4,5,6,7,8,9,9,10,11,12,13,14,15,15,16,17,18,19,20,21,22,23,24,25,26]
# TBS_TABLE USAGE: tbs_table[resourceblocks][TBS_Index] -> Where TBS_Index == mcs_2_tbs[MCS]
tbs_table= pd.read_excel(dir_path + "/tabelas/TBS-table.xlsx")


# -->> FOR DOWNLINK ONLY <<--
# #### PHY layer parameters assumptions
# LayersDL=2
# CFI=1.0		# CFI symbols
# QmPDSCH=6.0 	# 64QAM
# QmPCFICH=2.0 	# QPSK
# QmPDCCH=2.0  	# QPSK
# PDSCH_REs= n_RB *(n_RB_SC *(n_SUB_SYM-CFI)-(RefSym_RES*nAnt)) # =150
# PCFICH_REs=16.0 # Regardless of System Bandwidth, PCFICH is always carried by 4 REGs (16 REs) at the first symbol of each subframe
# PHICH_REs=12.0 	#one PHICH group
# PDCCH_REs=144.0 	#aggregation lvl 4
# nUE=1 	# number of users per TTI
# MCS_DL=28
#
# #### L2/L3 layers parameters assumptions
# TBS_DL=75376 	# Transport block size (in bits) 
# IP_TTI_DL= (TBS_DL)/((IPpkt + HdrPDCP + HdrRLC + HdrMAC) *8) 
# nTBS_DL_TTI= 2 	# 2 Layers
# FAPI_DL=1.5
# ---------------------------

## PACKET INCOMING
# For each CPRI type we have different set of variables for calcs.
# CPRI options 4 and 6 do not fit 2x2MIMO LTE traffic perfectly, its necessary\
# to set custom Fs (Mhz) for each antenna in order achieve a better LTE<->CPRI fit 
used_CPRIs=[1,2,3,5,7]
# Table below used for calcs
CPRI={\
1:{'Mhz':1.25,'Fs':1.92,'PRB':6},\
2:{'Mhz':2.5,'Fs':3.84,'PRB':12},\
3:{'Mhz':5.0,'Fs':7.68,'PRB':25},\
4:5,\
5:{'Mhz':10.0,'Fs':15.36,'PRB':50},\
6:10,\
7:{'Mhz':5,'Fs':30.72,'PRB':100}}

splits_info = defaultdict( lambda: defaultdict( lambda: defaultdict( lambda: defaultdict(float))))

# Multilevel nested dict \/
# nested_dict = lambda: defaultdict(nested_dict)
# nest = nested_dict()

for coding in range(0,29):
	for cpri_option in used_CPRIs:

		#----------- Changing CPRI option, we change:
		# 	PRB 
		#	sampling frequency MHz
		#	TBS
		# 	IP pkt TTI

		#print "Coding: %d 	CPRI OPTION: %d" % (coding,cpri_option)
		TBS_UL = tbs_table[CPRI[cpri_option]['PRB'] - PUCCH_RBs][mcs_2_tbs[MCS_UL]]
		#print "TBS: %.3f" % TBS_UL
		
		IP_TTI_UL= (TBS_UL)/((IPpkt + HdrPDCP + HdrRLC + HdrMAC) *8)
		#print "IP_TTI_UL: %.3f" % IP_TTI_UL
		#--------------------------------------------


		#--------------BW & GOPS CALCS---------------
		# SPLIT 1 -> PHY SPLIT IV - SCF
		a1_UL = nIQ * CPRI_coding
		r1_UL= CPRI[cpri_option]['Fs'] * nAnt * n_Sector * a1_UL
		# nIQ= (2*(15+1))= 32 -> facilita aproximacao nas contas
		split1=1
		splits_info[coding][cpri_option][1]['bw'] = r1_UL
		
		gops_1 = int((cpri_option*nAnt*n_Sector*a1_UL)/10)
		splits_info[coding][cpri_option][1]['gops'] = gops_1
		#print "Split1 : %.3f Mbps	GOPS:%d |" % (r1_UL, gops_1),

		# SPLIT 2 -> PHY SPLIT IIIb - SCF
		a2_UL= nIQ
		r2_UL= CPRI[cpri_option]['Fs'] * nAnt * n_Sector * a2_UL
		splits_info[coding][cpri_option][2]['bw'] = r2_UL
		
		gops_2 = int((cpri_option*nAnt*n_Sector*a2_UL*nIQ)/100)
		splits_info[coding][cpri_option][2]['gops'] = gops_2
		#print "Split2 : %.3f Mbps  GOPS:%d |" % (r2_UL,gops_2)
				
		# SPLIT 3 -> PHY SPLIT III - SCF
		a3_UL = n_RB_SC * n_Data_Sym * nIQ # <- *1000 / 1000000
		r3_UL = (a3_UL * nAnt * n_Sector * CPRI[cpri_option]['PRB'])/1000
		gops_3 = int(r3_UL/10)
		splits_info[coding][cpri_option][3]['bw'] = r3_UL
		splits_info[coding][cpri_option][3]['gops'] = gops_3
		#print "Split3 : %.3f Mbps	GOPS:%d |" % (r3_UL,gops_3),

		#SPLIT 4 -> PHY SPLIT II - SCF
		r4_UL = (n_Data_Sym * n_RB_SC * (CPRI[cpri_option]['PRB'] - PUCCH_RBs) * nAnt * nIQ)/1000
		splits_info[coding][cpri_option][4]['bw'] = r4_UL

		# Can be better represented.. -> insert every variable of r4_UL in the calculation (insert PUCCH_RBs)
		gops_4= int(2*gops_2) 
		
		splits_info[coding][cpri_option][4]['gops'] = gops_4
		#print "Split4 : %.3f Mbps   GOPS:%d |" % (r4_UL, gops_4)

		# SPLIT 5 -> PHY SPLIT I - SCF
		r5_UL = (n_Data_Sym * n_RB_SC * (CPRI[cpri_option]['PRB'] - PUCCH_RBs) * QmPUSCH * LayersUL * nSIC * nLLR) / 1000
		if coding > 3: 
			gops_5= int((gops_4) * (coding**2/(1+coding+(coding*math.sqrt(coding)))))
		else: #  In coding <= 3, split 5 has lower gops than split 4 
			gops_5= int(gops_4) #lower limit is equal to split 4
		splits_info[coding][cpri_option][5]['gops'] = gops_5
		splits_info[coding][cpri_option][5]['bw'] = r5_UL
		#print "Split5 : %.3f Mbps 	GOPS:%d | " % (r5_UL, gops_5),

		# SPLIT 6 -> SPLIT MAC-PHY - SCF
		a6_UP = (IP_TTI_UL* (IPpkt + HdrPDCP + HdrRLC + HdrMAC) * nTBS_UL_TTI)/ 125
		r6_UL = a6_UP * LayersUL + FAPI_UL
		gops_6 = int(a6_UP*LayersUL)
		splits_info[coding][cpri_option][6]['bw'] = r6_UL
		splits_info[coding][cpri_option][6]['gops'] = gops_6
		#print "Split6 : %.3f Mbps	GOPS:%d |" % (r6_UL,gops_6)

		# SPLIT 7 -> SPLIT RRC-PDCP - SCF
		a7_UP = (IP_TTI_UL * IPpkt * nTBS_UL_TTI) / 125
		r7_UL = a7_UP * LayersUL
		# GOPS calculates the processing cost of a function, but function 7 doesn't exist for us
		# gops_7 = int(a7_UP * LayersUL) # legacy calcs for gops_7
		gops_7 = 0
		# BW in split 7 can also be approximated by TBS/1000.0
		splits_info[coding][cpri_option][7]['bw'] = r7_UL 
		splits_info[coding][cpri_option][7]['gops'] = gops_7
		#########

		# Total GOPS of coding and CPRI_option 
		GOPS_total= gops_1+gops_2+gops_3+gops_4+gops_5+gops_6+gops_7
		splits_info[coding][cpri_option]['gops_total']= GOPS_total
		#print "Split7 : %.3f Mbps	GOPS:%d 	GOPS TOTAL Split1:%d|\n" % (r7_UL,gops_7,GOPS_total)

		# measuring edge and metro gops for each split in each coding and cpri option
		for split in range(1,8):
			edge_gops=0
			if split > 1:
				for ec_split in range(1,split):
					edge_gops+= splits_info[coding][cpri_option][ec_split]['gops']
			splits_info[coding][cpri_option][split]['edge_gops']= edge_gops

			metro_gops=0
			if split < 7:
				for metro_split in range(split,7):
					metro_gops+= splits_info[coding][cpri_option][metro_split]['gops']
			splits_info[coding][cpri_option][split]['metro_gops']= metro_gops
		#-----------------END OF BW AND GOPS CALCS ----------------------- 

#print dict(splits_info)

# BETTER PRINT OF DEFAULT DICT \/
#import json
#data_as_dict = json.loads(json.dumps(splits_info, indent=4))
#print(data_as_dict)

# splits_info[MCS][CPRI option][Split]
#for cada in range(1,8):
#	print splits_info[28][1][cada] 

#-----------CLASSES AND SIMULATOR-------------

class Orchestrator(object):
	def __init__ (self,env,n_vBBUs,splitting_table,MID_port,edge_vBBU_pool_obj,\
				  fix_coding,interval=1002,thold_updt=0.03,multiplexing_rate=2,reduce_percnt=1.3):
		self.env=env
		self.interval = interval
		self.fix_coding = fix_coding
		self.n_vBBUs = n_vBBUs
		self.vBBU_splits={}
		for cada in range(n_vBBUs):
			self.vBBU_splits[cada] = 1
		self.splitting_table = splitting_table
		self.MID_port = MID_port
		self.thold_updt = thold_updt
		self.reduce_percnt = reduce_percnt

		# meanwhile: only a list with vbbus of 1 cell and direct obj updts
		self.edge_vBBU_pool_obj = edge_vBBU_pool_obj

		self.MID_max_bw = self.MID_port.max_bw
		self.max_bw_1_split = self.splitting_table[coding][7][1]['bw']
		self.min_bw_1_split = self.splitting_table[coding][1][7]['bw']
		#print self.max_bw_1_split
		self.bw_max_cost = self.max_bw_1_split * len(edge_vBBU_pool_obj)
		
		self.multiplexing_rate = 2 # baseline desired multiplexing rate #may not be used
		self.actual_multiplexing = 2
		self.last_multiplexing = 2
		self.bw_use = 0
		self.last_bw_use = 0
		
		# add UL entry at MID switch to enable edge_pools communication 
		# for now only one mid switch
		#for cada in MID_phi_ports: 
		self.MID_port.add_UL_entry('orchestrator',self)


		#self.metrics_dict = {}
		self.read_metrics = self.env.process(self.read_metrics())

	def splitting_updt(self,phi_metrics,vbbu_metrics):
		# total phi drops
		splitted = False
		phi_bytes_drops = phi_metrics['UL_bytes_drops']
		reduced_bw = 0
		print "----- ORCHESTRATOR ------"
		#aux_vBBU_splits = dict(self.vBBU_splits) # auxiliary split dict during splitting updt
		changed_vBBU_splits = {} # dict of changed vbbu splits key: vbbu_id and value: split 
		
		# ordered list of most drops on a vBBU
		dropper_list = []
		list_pos = ()
		#print "VBBU_metrics"
		#print vbbu_metrics
		for cada in vbbu_metrics:
			list_pos = (cada,vbbu_metrics[cada]['UL_bytes_drops'])
			dropper_list.append(list_pos)
		#list_pos = (1,1300)
		#dropper_list.append(list_pos)

		# ordered list
		dropper_list.sort(key=itemgetter(1))
		#print "Dropper list"
		#print dropper_list
		#TODO: Consider that CPRI_option changes. Now CPRI is fixed = 3
		cpri_option = 3
		#TODO: Consider energy in calcs

		# get most droppers and change their split until around 10% under maximum bw of MID
		for vbbu in vbbu_metrics:
			#print "Reduced bw: %f" % reduced_bw
			if reduced_bw < phi_bytes_drops*self.reduce_percnt:
				vbbu_tuple = dropper_list.pop()
				# get actual vbbu split
				vbbu_split = self.vBBU_splits[vbbu_tuple[0]]
				#print "vbbu_split: %d" % vbbu_split
				bw_vbbu_split = self.splitting_table[self.fix_coding][cpri_option][vbbu_split]['bw']
				
				# difference between actual split and split 7 to actual split
				for split in range(vbbu_split+1,7+1)[::-1]:
					bw_split = self.splitting_table[self.fix_coding][cpri_option][split]['bw']
					# difference between splits
					diff_bw = bw_vbbu_split - bw_split
					
					#print "Reduced bw: %f" % reduced_bw
					# add 
					changed_vBBU_splits[vbbu_tuple[0]] = split

					# check if applying split 7 we still have to change split of other vbbus
					if diff_bw < phi_bytes_drops*self.reduce_percnt:
						# update splitting of next vbbu
						reduced_bw += diff_bw
						print "Reduced bw: %f" % reduced_bw				
						break
			else:
				break

		if reduced_bw < phi_bytes_drops*self.reduce_percnt:
			print "WARNING: Demand higher than BW capacity after all splits done."
		
		# write changes to the EDGE VBBU POOL
		if len(changed_vBBU_splits) > 0:
			print changed_vBBU_splits
			for cada in changed_vBBU_splits:
				#create pkt
				str_vbbu = str(cada)
				cell_id = '0'
				split_updt = {'plane':'ctrl','src':'orchestrator', 'dst':'edge_pool_'+cell_id, 'vBBU_id':cada, 'split':changed_vBBU_splits[cada]}
				print split_updt
				self.MID_port.downstream.put(split_updt)
				#send to MID_port
				#self.MID_port.


	#def check_enough_reduction(self,diff_bw,total_drops):

	def read_metrics(self):
		# wait interval to gather metrics
		while True:
			yield self.env.timeout(self.interval)
			
			# read amount of bytes dropped in midhaul
			phi_metrics,vbbu_metrics = self.MID_port.get_metrics()
			self.MID_max_bw = phi_metrics['max_bw']
			bytes_drop = phi_metrics['UL_bytes_drops']
			#print bytes_drop
			# default thold_updt is a max of 3% losses in order to trigger splitting updt 
			if (bytes_drop/self.MID_max_bw) > self.thold_updt:
				self.splitting_updt(phi_metrics,vbbu_metrics)


class Edge_Cloud(object):
	def __init__ (self,env,n_vBBUs,FH_switch,MID_switch):
		self.env=env
		#self.baseline_energy=700	#from rodrigo's paper
		self.AC_energy= n_vBBUs * 100
		self.battery_energy= n_vBBUs *15
		self.base_pool_energy= n_vBBUs * 10
		self.baseline_energy= AC_energy + battery_energy + base_pool_energy

		self.FH_switch = FH_switch
		self.MID_switch = MID_switch
		
		self.FH_receiver = self.env.process(self.FH_receiver())
		self.MID_receiver = self.env.process(self.MID_receiver())
		self.MID_sender = self.env.process(self.MID_sender())

class Central_Cloud(object):
	def __init__(self,env,n_vBBUs):
		self.env=env
		self.AC_energy= n_vBBUs * 50
		self.battery_energy= n_vBBUs *10
		self.base_pool_energy= n_vBBUs * 5
		self.baseline_energy= AC_energy + battery_energy + base_pool_energy
		#self.baseline_energy=550	#from rodrigo's paper

class Edge_vBBU_Pool(object):
	def __init__(self,env,cell_id,n_vBBUs,edge_vBBU_dict,MID_phi_port):
		self.env = env
		self.cell_id = cell_id
		self.name = 'edge_pool_'+str(self.cell_id)
		self.MID_phi_port = MID_phi_port
		self.edge_vBBU_dict = edge_vBBU_dict
		
		self.DL_buffer = simpy.Store(env) # communication from metro & orch with this edge pool

		for vBBU in edge_vBBU_dict.values():
			vBBU.MID_port = self.MID_phi_port

		self.baseline_energy = 5 + (n_vBBUs * 5)

		# here run monitoring or something alike 
		# self.action = self.env.process(self.monitoring())
		self.action = self.env.process(self.listen_orchestrator())
		
		# add DL entry at MID switch to enable orchestrator communication 
		self.MID_phi_port.add_DL_entry(self.name,self)
		

	def listen_orchestrator(self):
		pkt = yield self.DL_buffer.get()
		print pkt
		if pkt['dst'] == self.name:
			self.set_vBBU_split(pkt['vBBU_id'],pkt['split'])


	def set_vBBU_split(self,vBBU_id,split):
		# orchestrator changing a split option of a vBBU
		edge_vBBU_dict[vBBU_id].set_split(split)


class Metro_vBBU_Pool(object):
	def __init__(self,env,cell_id,n_vBBUs,metro_vBBU_dict,MID_phi_port):
		self.env = env
		self.cell_id = str(cell_id)
		self.name = 'metro_pool_'+str(self.cell_id)
		self.MID_phi_port = MID_phi_port
		#self.MID_phi_port.set_metro_ctrl(self)

		self.metro_vBBU_dict = metro_vBBU_dict
		
		# writing output of metro_vBBU objs to the MID_phi_port'obj to enable downlink  
		for vBBU in metro_vBBU_dict.values():
			vBBU.MID_port = self.MID_phi_port

		self.baseline_energy = 5 + (n_vBBUs * 2)

		self.MID_phi_port.add_UL_entry(self.name,self)

		# here run monitoring or something alike 
		# self.action = self.env.process(self.monitoring())

	def set_vBBU_split(self,vBBU_id,split):
		# orchestrator changing a split option of a vBBU
		metro_vBBU_dict[vBBU_id].set_split(split)


class vBBU(object): # PARENT CLASS
	"""-------------- COMMENTS -------------
	  Packets from RRHs of a cell arrive at the vBBU (only UPLINK modelled)
	  Every BBU:
		has a max processing in giga operations per sec (GOPS) arbitrarily defined by us
		is able to run every function split
		has a fixed amount of cores
	
	 At the moment:
		1. The function chain of a RRH can only be processed in one core reserved exclusively for it in only one vBBU
		2. One core is considered enough for a RRH
	 	Meaning a few things:
			1. No multiplexing, no resource allocation schemes (No core sharing by multiple RRH)
			2. One core is reserved for one RRH and one core is enough for a RRH.
			3. Processing energy is calculated by the processing timeout the RRH traffic load requires for the core
			4. IF no core sharing and if one core is enough for a RRH, then there is no blocking in processing
			5. Intra-bbu processing model
	 				When implemmenting inter-bbu model (sending processes to other vBBU if theres high delay/blocking),
					refer to the article "Radio Base Stations in the Cloud - 2013"
	 We consider RRHs to not shutdown for energy saving, they're always UP
	-------------------------------------
	"""
	def __init__ (self,env,cell_id,vBBU_id,splitting_table,GOPS=8000):
		self.env= env
		self.cell_id = cell_id
		self.id = vBBU_id
		self.splitting_table= splitting_table 	# defaultdict with bw&gops per split
		self.GOPS= GOPS # every bbu is equal by default=8000GOPS

class Metro_vBBU(vBBU):
	def __init__(self,env,cell_id,vBBU_id,splitting_table,GOPS=8000,core_dyn_energy=15):
		vBBU.__init__(self,env,cell_id,vBBU_id,splitting_table,GOPS)
		self.core_dyn_energy = core_dyn_energy
		self.UL_buffer = simpy.Store(self.env)
		self.MID_port = None
		# TODO: DL functions 
		#self.DL_buffer = simpy.Store(self.env)

		self.MID_receiver = self.env.process(self.MID_receiver())
		#self.MID_sender = self.env.process(self.MID_sender(MID_port))

	def MID_receiver(self):
		"""process to get the packet from the FH_switch port buffer"""
		while True:
			#print "Time: %f. METRO vBBU%d waiting for pkt in UL_buffer" % (self.env.now, self.id)
			#pkt= yield self.FH_switch.upstream[self.id].get()
			#pkt= yield self.FH_port.buffer.get()
			pkt = yield self.UL_buffer.get()
			
			#print "Time: %f. METRO vBBU%d starting pkt%d processing (splitting) " % (self.env.now,self.id,pkt.id)
			yield self.env.process(self.splitting(pkt))


	def splitting(self,pkt):
		# print "METRO CLOUD"
		# print "Coding: %d" % pkt.coding,
		# print "CPRI_OPTION %d" % pkt.CPRI_option,
		# print "RRH ID %s " % pkt.dst,
		# print "Split: %d" % pkt.split
		# #pkt_split = table_rrh_id[pkt_rrh_id]['split'] # get split of pkt from table
		#pkt_split = splitting_table[pkt_coding][pkt_CPRI_option][split] # get split of pkt from table
		
		if pkt.split == 7: # if C-RAN split send everything to MetroCloud
			print "Split 7. Pkt%d already processed. Nothing to do" % pkt.id
			del pkt
			return

		#by the packets attributes (MCS, CPRI option) and its split, get GOPS and BW from table
		GOPS = self.splitting_table[pkt.coding][pkt.CPRI_option][pkt.split]['metro_gops']
		#print "METRO GOPS split: %d" % GOPS
		#print "Self gops %d" % self.GOPS
		#bw_split = (self.splitting_table[pkt.coding][pkt.CPRI_option][7]['bw'])/1000
		#print "BW pkt split: %f Mb" % bw_split
		#timeout proc
		proc_tout = float(GOPS)/self.GOPS
		#print "METRO Proc Timeout %f" % proc_tout
		energy = (float(proc_tout) * (self.core_dyn_energy))/1000 #== measured in 1ms instead of 1s
		#print "METRO Energy consumption: %f W" % energy
		yield self.env.timeout(proc_tout)
		#print "METRO After t_out. Time now= %f" % self.env.now
		
		# send pkt to core network (for us it means destroying it)
		del pkt

		# LOG the proc energy usage
		#energy_file.write( "{},{},{},{},{},{},{},{}\n".\
		#format("edge_", MAC_TABLE[ONU.oid],"02", time_stamp,counter, ONU.oid,self.env.now,grant_final_time) )


class Edge_vBBU(vBBU):
	def __init__(self,env,cell_id,vBBU_id,splitting_table,GOPS=8000,core_dyn_energy=30):
		vBBU.__init__(self,env,cell_id,vBBU_id,splitting_table,GOPS)
		self.core_dyn_energy = core_dyn_energy
		self.split=1 # variable not in metro_vbbu, because after edgesplit the pkt gets variable split
		self.UL_buffer = simpy.Store(self.env)

		self.MID_port = None
		
		# TODO: DL functions 
		#self.DL_buffer = simpy.Store(self.env)

		self.FH_receiver = self.env.process(self.FH_receiver()) # get from FH_switch buffer
		
		#self.MID_receiver = self.env.process(self.MID_receiver(MID_port))
		

	def set_split(self,split):
		self.split=split

	def FH_receiver(self):
		"""process to get the packet from the FH_switch port buffer"""
		while True:
			#print "Time: %f. BBU%d waiting for pkt in UL_buffer" % (self.env.now, self.id)
			#pkt= yield self.FH_switch.upstream[self.id].get()
			#pkt= yield self.FH_port.buffer.get()
			pkt = yield self.UL_buffer.get()
			
			#print "Time: %f. BBU%d starting pkt%d processing (splitting) " % (self.env.now,self.id,pkt.id)
			yield self.env.process(self.splitting(pkt))

	def splitting(self,pkt):
		#print "Coding: %d" % pkt.coding,
		#print "CPRI_OPTION %d" % pkt.CPRI_option,
		#print "RRH ID %s " % pkt.dst,
		#print "Split: %d" % self.split
		#pkt_split = table_rrh_id[pkt_rrh_id]['split'] # get split of pkt from table
		#pkt_split = splitting_table[pkt_coding][pkt_CPRI_option][split] # get split of pkt from table
		pkt.split=self.split
		if pkt.split == 1: # if C-RAN split send everything to MetroCloud
			#print "Split 1. Send pkt straight to metroCloud."
			# send pkt to Metro
			self.MID_port.upstream.put(pkt)
			return

		#by packets attributes (MCS, CPRI option) and its split, get GOPS and BW from table
		GOPS = self.splitting_table[pkt.coding][pkt.CPRI_option][self.split]['edge_gops']
		#print "GOPS: %d" % GOPS,
		#print "Self gops %d" % self.GOPS
		bw_split = (self.splitting_table[pkt.coding][pkt.CPRI_option][self.split]['bw'])/1000
		#print "BW pkt split: %f Mb" % bw_split,
		#timeout proc
		proc_tout = float(GOPS)/self.GOPS
		#print "Proc Timeout %f" % proc_tout,
		energy = (float(proc_tout) * (self.core_dyn_energy))/1000 # == measured in 1ms instead of 1s
		#print "Energy consumption: %f W" % energy
		yield self.env.timeout(proc_tout)
		#print "After t_out. Time now= %f" % self.env.now
		pkt.size= bw_split
		
		# send pkt to phi port (midhaul)
		#print "Sending to Phi Midhaul"
		self.MID_port.upstream.put(pkt)
		#print "Sent"

		# LOG the proc energy usage
		#energy_file.write( "{},{},{},{},{},{},{},{}\n".\
		#format("edge_", MAC_TABLE[ONU.oid],"02", time_stamp,counter, ONU.oid,self.env.now,grant_final_time) )


class Packet_CPRI(object):
    """ This class represents a network packet """
    def __init__(self, time,rrh_id, coding, CPRI_option, id, src="a", dst="z"):
		self.time = time# creation time
		self.rrh_id= rrh_id
		self.plane='data'
		self.coding = coding #same as MCS
		self.CPRI_option = CPRI_option
		self.size = (splits_info[coding][CPRI_option][1]['bw'])/1000
		#print self.size
		#self.PRB = PRB #not represented for now
		self.id = id # packet id
		self.src = src #packet source address
		self.dst = dst #packet destination address
		

    def __repr__(self):
        return "id: {}, src: {}, time: {}, CPRI_option: {}, coding: {}".\
            format(self.id, self.src, self.time, self.CPRI_option, self.coding)

class Packet_Generator(object):
    """This class represents the packet generation process """
    def __init__(self, env, rrh_id, adist, finish=float("inf")):
        self.rrh_id = rrh_id # packet id
        self.env = env # Simpy Environment
        self.arrivals_dist = adist #packet arrivals distribution

        self.coding = 28 #coding fixed for now
        self.CPRI_option = 3 #cpri_option fixed for now
        
        self.finish = finish # packet end time
        self.out = None # packet generator output
        self.packets_sent = 0 # packet counter
        self.action = env.process(self.run())  # starts the run() method as a SimPy process

    def run(self):
        """The generator function used in simulations.
        """
        while self.env.now < self.finish:
            # wait for next transmission
            yield self.env.timeout(self.arrivals_dist)
            self.packets_sent += 1
            #print "New packet generated at %f" % self.env.now
    
            p = Packet_CPRI(self.env.now, self.rrh_id, self.coding, self.CPRI_option, self.packets_sent, src=self.rrh_id, dst=self.rrh_id)
            #time,rrh_id, coding, CPRI_option, id, src="a"
            #print p
            #Logging
            #pkt_file.write("{}\n".format(self.fix_pkt_size))
            self.out.put(p)
            # call the function put() of RRHPort, inserting the generated packet into RRHPort' buffer


class Cell(object):
	"""Class representing a fixed set of RRHs connected to a fixed BBU"""
	def __init__(self,env,cell_id,edge_bbu_id,num_rrhs,exp,qlimit,fix_pkt_size):
		self.env = env
		self.cell_id = cell_id #Cell indentifier
		self.rrh_dict={}
		for rrh_id in range(1,num_rrhs+1):
			#creates the packet generator for each RRH and places it into dict
			self.rrh_dict[rrh_id] = RRH(self.env,cell_id,rrh_id,edge_bbu_id,exp,qlimit,fix_pkt_size)
        

class RRH(object):
	"""Class representing each RRH with its own packet generation (Uplink) to a fixed BBU"""
	#def __init__(self,env,cell_id,rrh_id,edge_bbu_id,qlimit,FH_switch):
	def __init__(self,env,cell_id,rrh_id,edge_bbu_id,qlimit,FH_phi_port):
		self.env = env
		self.cell_id = cell_id
		self.id = rrh_id
		arrivals_dist = 1 # fixed 1 ms
		self.str_id = str(self.id)
		self.pg = Packet_Generator(self.env, self.id, arrivals_dist)

		if qlimit == 0:# checks if the queue has a size limit
			queue_limit = None
		else:
			queue_limit = qlimit

		self.port = RRH_Port(self.env, qlimit=queue_limit) #create RRH PORT (FH) to bbu
		self.pg.out = self.port #forward packet generator output to RRH port
		#self.sender = self.env.process(self.RRH_sender(FH_switch))
		self.sender = self.env.process(self.RRH_sender(FH_phi_port))
		#self.receiver = self.env.process(self.ONU_receiver(odn))
		
		#self.action=self.env.process(self.run())

	def RRH_sender(self,Phi_port_pool):
		while True:
   			pkt = yield self.port.buffer.get()
   			self.port.byte_size -= pkt.size
   			if self.port.byte_size < 0:
   				print "Error: RRH %d port buffer sizer < 0" % self.id
   			#FH_switch.put_UL(self.id,pkt)
   			Phi_port_pool.upstream.put(pkt)
   			self.port.packets_tx +=1
		
	# def run(self):
	# 	while True:
	# 		adist=1
	# 		yield self.env.process(self.pg.run())

class RRH_Port(object):
    def __init__(self, env, qlimit=None):
        self.buffer = simpy.Store(env) #buffer
        self.env = env
        self.out = None # RRH port output # FH_switch[RRH_id] Store
        self.packets_rec = 0 #received pkt counter
        self.packets_tx = 0 #received pkt counter
        self.packets_drop = 0 #dropped pkt counter
        self.qlimit = qlimit #Buffer queue limit
        self.byte_size = 0  # Current size of the buffer in bytes
        self.action = env.process(self.run())  # starts the run() method as a SimPy process
        self.pkt = None #network packet obj

    def run(self): #run the port as a simpy process
        while True:
             yield self.env.timeout(5)

    def put(self, pkt):
        """receives a packet from the packet genarator and put it on the queue
            if the queue is not full, otherwise drop it.
        """

        self.packets_rec += 1
        #print "+1 RRHPort pkt. Packets received: %d" % self.packets_rec
        #pkt.size = (splits_info[pkt.coding][pkt.CPRI_option][1]['bw'])/1000
        #print "Pkt size: %f" % pkt.size
        tmp = self.byte_size + pkt.size
        #print "RRHPort buffer size: %f"  % tmp
        if self.qlimit is None: #checks if the queue size is unlimited
            self.byte_size = tmp
            return self.buffer.put(pkt)
        if tmp >= self.qlimit: # checks if the queue is full
            self.packets_drop += 1
            #return
        else:
            self.byte_size = tmp
            self.buffer.put(pkt)

class Phi_port_pool(object):
	def __init__(self,env,name,cell_id,NUMBER_OF_RRHs,UL_vBBU_obj_dict,DL_vBBU_obj_dict={},\
				 max_bw=None, bw_check_interval=1000):
		self.env = env
		self.name = name
		self.cell_id = cell_id
		self.DL_vBBU_obj_dict = DL_vBBU_obj_dict
		self.UL_vBBU_obj_dict = UL_vBBU_obj_dict
		self.num_RRHs=NUMBER_OF_RRHs

		self.UL_metrics = defaultdict(lambda : defaultdict(float))

		for UL_vBBU in self.UL_vBBU_obj_dict:
			#print "UL vBBU = %d" % UL_vBBU
			self.UL_metrics[UL_vBBU]['UL_pkt_rx'] = 0
			self.UL_metrics[UL_vBBU]['UL_bytes_rx'] = 0
			self.UL_metrics[UL_vBBU]['UL_pkt_tx'] = 0
			self.UL_metrics[UL_vBBU]['UL_bytes_tx'] = 0
			self.UL_metrics[UL_vBBU]['UL_pkt_drops'] = 0
			self.UL_metrics[UL_vBBU]['UL_bytes_drops'] = 0
			self.UL_metrics[UL_vBBU]['UL_pkt_errors'] = 0

			self.UL_metrics[UL_vBBU]['last_UL_bytes_rx'] = 0
			self.UL_metrics[UL_vBBU]['last_UL_bytes_tx'] = 0
			self.UL_metrics[UL_vBBU]['last_UL_pkt_rx'] = 0
			self.UL_metrics[UL_vBBU]['last_UL_pkt_tx'] = 0
			self.UL_metrics[UL_vBBU]['last_UL_pkt_drops'] = 0
			self.UL_metrics[UL_vBBU]['last_UL_bytes_drops'] = 0
			self.UL_metrics[UL_vBBU]['UL_usage'] = 0 # this metric is always outdated by concept
			self.UL_metrics[UL_vBBU]['last_UL_pkt_errors'] = 0
			#print self.UL_metrics[UL_vBBU]
		
		self.max_bw = max_bw
		if max_bw is not None:
			self.UL_bw_interval = 0
			self.DL_bw_interval = 0
		self.bw_check_interval = bw_check_interval # default 1 seg = 1000ms
		
		# metrics UL
		self.UL_pkt_rx = 0
		self.UL_bytes_rx = 0
		self.UL_pkt_tx = 0
		self.UL_bytes_tx = 0
		self.UL_pkt_errors = 0
		self.UL_pkt_drops = 0
		self.UL_bytes_drops = 0
		self.UL_usage = 0
		self.last_UL_bytes_rx = 0
		self.last_UL_bytes_tx = 0
		self.last_UL_pkt_rx = 0
		self.last_UL_pkt_tx = 0
		self.last_UL_pkt_errors = 0
		self.last_UL_pkt_drops = 0
		self.last_UL_bytes_drops = 0

		# TODO: increment and decrement buffer size on puts and gets
		self.UL_buffer_size = 0 # create function for put and get

		#------- DL -------
		self.DL_pkt_rx = 0
		self.DL_bytes_rx = 0
		self.DL_pkt_tx = 0
		self.DL_bytes_tx = 0
		self.DL_pkt_errors = 0
		self.DL_pkt_drops = 0
		
		# TODO: increment and decrement buffer size on puts and gets
		self.DL_buffer_size = 0
		#--------------

		# consider there's only a single phy port for vBBUpool at midhaul with UL and DL streams
		self.upstream = simpy.Store(env)
		self.downstream = simpy.Store(env)

		self.pkt_uplink = env.process(self.pkt_uplink())
		self.pkt_downlink = env.process(self.pkt_downlink())

		# measure metrics and check the BW in midhaul during interval
		if max_bw is not None:
			self.bw_check = env.process(self.bw_check())

	def add_UL_entry(self,key,obj):
		if key in self.UL_vBBU_obj_dict.keys():
			print "ERROR: key %d already exists in %s UL_table" % (key,self.name)
		else:
			self.UL_vBBU_obj_dict[key]=obj
			
	def del_UL_entry(self,key,obj):
		try: 
			del self.UL_vBBU_obj_dict[key]
		except:
			print "ERROR: key %d doesn't exist in %s UL_table" % (key,self.name)

	def add_DL_entry(self,key,obj):
		if key in self.DL_vBBU_obj_dict:
			print "ERROR: key %d already exists in %s DL_table" % (key,self.name)
		else:
			self.DL_vBBU_obj_dict[key]=obj

	def del_DL_entry(self,key,obj):
		try:
			del self.DL_vBBU_obj_dict[key]
		except:
			print "ERROR: key %d already exists in %s DL_table" % (key,self.name)

	def bw_check(self):
		while True:
			yield self.env.timeout(self.bw_check_interval)
			print "---MID CHECK ---"
			#self.last_UL_pkt_drops = self.UL_pkt_drops 
			#self.last_UL_byte_drops = self.UL_bw_interval - self.UL_byte_count
			
			#calc usage
			self.UL_tx_bytes_diff = self.UL_bytes_tx - self.last_UL_bytes_tx
			self.UL_tx_pkt_diff = self.UL_pkt_tx - self.last_UL_pkt_tx
			self.UL_rx_bytes_diff = self.UL_bytes_rx - self.last_UL_bytes_rx
			self.UL_rx_pkt_diff = self.UL_pkt_rx - self.last_UL_pkt_rx
			self.UL_pkt_drop_diff = self.UL_pkt_drops - self.last_UL_pkt_drops
			self.UL_bytes_drop_diff = self.UL_bytes_drops - self.last_UL_bytes_drops
			
			# update last values to actual values
			self.last_UL_bytes_rx = self.UL_bytes_rx
			self.last_UL_bytes_tx = self.UL_bytes_tx
			self.last_UL_pkt_rx = self.UL_pkt_rx
			self.last_UL_pkt_tx = self.UL_pkt_tx
			self.last_UL_pkt_errors = self.UL_pkt_errors
			self.last_UL_pkt_drops = self.UL_pkt_drops
			self.last_UL_bytes_drops = self.UL_bytes_drops

			print ""
			print "UL rx: %d pkts and %.3f Mbps" % (self.UL_rx_pkt_diff,self.UL_rx_bytes_diff)
			print "UL drops: %d pkts and %.3f Mbps " % (self.UL_pkt_drop_diff, self.UL_bytes_drop_diff)
			print "UL tx: %d pkts and %.3f Mbps" % (self.UL_tx_pkt_diff, self.UL_tx_bytes_diff)

			for vBBU in self.UL_metrics:
				#calc usage
				print vBBU
				self.UL_metrics[vBBU]['UL_tx_bytes_diff'] = self.UL_metrics[vBBU]['UL_bytes_tx'] - self.UL_metrics[vBBU]['last_UL_bytes_tx']
				self.UL_metrics[vBBU]['UL_tx_pkt_diff'] = self.UL_metrics[vBBU]['UL_pkt_tx'] - self.UL_metrics[vBBU]['last_UL_pkt_tx']
				self.UL_metrics[vBBU]['UL_rx_bytes_diff'] = self.UL_metrics[vBBU]['UL_bytes_rx'] - self.UL_metrics[vBBU]['last_UL_bytes_rx']
				self.UL_metrics[vBBU]['UL_rx_pkt_diff'] = self.UL_metrics[vBBU]['UL_pkt_rx'] - self.UL_metrics[vBBU]['last_UL_pkt_rx']
				self.UL_metrics[vBBU]['UL_pkt_drop_diff'] = self.UL_metrics[vBBU]['UL_pkt_drops'] - self.UL_metrics[vBBU]['last_UL_pkt_drops']
				self.UL_metrics[vBBU]['UL_bytes_drop_diff'] = self.UL_metrics[vBBU]['UL_bytes_drops'] - self.UL_metrics[vBBU]['last_UL_bytes_drops']
				
				# update last values to actual values
				self.UL_metrics[vBBU]['last_UL_bytes_rx'] = self.UL_metrics[vBBU]['UL_bytes_rx']
				self.UL_metrics[vBBU]['last_UL_bytes_tx'] = self.UL_metrics[vBBU]['UL_bytes_tx']
				self.UL_metrics[vBBU]['last_UL_pkt_rx'] = self.UL_metrics[vBBU]['UL_pkt_rx']
				self.UL_metrics[vBBU]['last_UL_pkt_tx'] = self.UL_metrics[vBBU]['UL_pkt_tx']
				self.UL_metrics[vBBU]['last_UL_pkt_errors'] = self.UL_metrics[vBBU]['UL_pkt_errors']
				self.UL_metrics[vBBU]['last_UL_pkt_drops'] = self.UL_metrics[vBBU]['UL_pkt_drops']
				self.UL_metrics[vBBU]['last_UL_bytes_drops'] = self.UL_metrics[vBBU]['UL_bytes_drops']

				print ""
				print "vBBU%d UL rx: %d pkts and %.3f Mbps" % (vBBU, self.UL_metrics[vBBU]['UL_rx_pkt_diff'],self.UL_metrics[vBBU]['UL_rx_bytes_diff'])
				print "vBBU%d UL drops: %d pkts and %.3f Mbps " % (vBBU, self.UL_metrics[vBBU]['UL_pkt_drop_diff'], self.UL_metrics[vBBU]['UL_bytes_drop_diff'])
				print "vBBU%d UL tx: %d pkts and %.3f Mbps" % (vBBU, self.UL_metrics[vBBU]['UL_tx_pkt_diff'], self.UL_metrics[vBBU]['UL_tx_bytes_diff'])


			# zeroing counters
			self.UL_bw_interval = 0
			self.DL_bw_interval = 0

			# for vBBU in self.UL_metrics:
			# 	self.UL_metrics[vBBU]['']
			print "------"

	def get_metrics(self):
		self.phi_UL_metrics={}
		self.phi_UL_metrics['UL_pkt_rx']= self.UL_pkt_rx
		self.phi_UL_metrics['UL_bytes_rx'] = self.UL_bytes_rx
		self.phi_UL_metrics['UL_pkt_tx'] = self.UL_pkt_tx
		self.phi_UL_metrics['UL_bytes_tx'] = self.UL_bytes_tx
		self.phi_UL_metrics['UL_pkt_errors'] = self.UL_pkt_errors
		self.phi_UL_metrics['UL_pkt_drops'] = self.UL_pkt_drops
		self.phi_UL_metrics['UL_bytes_drops'] = self.UL_bytes_drops
		self.phi_UL_metrics['UL_usage'] = self.UL_usage
		self.phi_UL_metrics['last_UL_bytes_rx'] = self.last_UL_bytes_rx
		self.phi_UL_metrics['last_UL_bytes_tx'] = self.last_UL_bytes_tx
		self.phi_UL_metrics['last_UL_pkt_rx'] = self.last_UL_pkt_rx
		self.phi_UL_metrics['last_UL_pkt_tx'] = self.last_UL_pkt_tx
		self.phi_UL_metrics['last_UL_pkt_errors'] = self.last_UL_pkt_errors	
		self.phi_UL_metrics['last_UL_pkt_drops'] = self.last_UL_pkt_drops
		self.phi_UL_metrics['last_UL_bytes_drops'] = self.last_UL_bytes_drops
		self.phi_UL_metrics['max_bw'] = self.max_bw
		return self.phi_UL_metrics,self.UL_metrics

	def set_vBBU_dict(self,vBBU_obj_dict):
		self.vBBU_obj_dict = vBBU_obj_dict

	def pkt_uplink(self):
		while True:
			pkt = yield self.upstream.get() # get pkt from a RRH of cell
			self.UL_pkt_rx +=1
			self.UL_metrics[pkt.dst]['UL_pkt_rx']+=1

			if self.max_bw is not None:
				self.UL_bw_interval += pkt.size
				self.UL_bytes_rx += pkt.size
				self.UL_metrics[pkt.dst]['UL_bytes_rx']+= pkt.size

			 	if self.UL_bw_interval > self.max_bw:
					#print "TIME: %f. WARNING: Pkt UL BLOCK!" % self.env.now
					# all UL
					self.UL_pkt_drops +=1
					self.UL_bytes_drops += pkt.size
					# individual vbbu counters
					self.UL_metrics[pkt.dst]['UL_pkt_drops']+= 1
					self.UL_metrics[pkt.dst]['UL_bytes_drops']+= pkt.size
					del pkt

				else:
					if pkt.plane == 'ctrl':
						self.UL_vBBU_obj_dict['metro_pool'].UL_buffer.put(pkt)
					else:
						self.UL_vBBU_obj_dict[pkt.dst].UL_buffer.put(pkt)
						# all UL
						self.UL_pkt_tx +=1
						self.UL_bytes_tx += pkt.size
						# individual vbbu counters
						self.UL_metrics[pkt.dst]['UL_pkt_tx']+= 1
						self.UL_metrics[pkt.dst]['UL_bytes_tx']+= pkt.size
					
			else:
				#print self.UL_vBBU_obj_dict[pkt.dst]
				self.UL_vBBU_obj_dict[pkt.dst].UL_buffer.put(pkt)
				# ALL UL
				self.UL_bytes_tx += pkt.size
				self.UL_pkt_tx +=1
				# individual vbbu counters
				self.UL_metrics[pkt.dst]['UL_pkt_tx']+= 1
				self.UL_metrics[pkt.dst]['UL_bytes_tx']+= pkt.size

	def pkt_downlink(self):
		while True:
			pkt = yield self.downstream.get() # get pkt from a RRH of cell
			# insert pkt in virtual vBBU port
			print pkt
			print "DL pkt dst: %s" % pkt['dst']
			print self.DL_vBBU_obj_dict 
			self.DL_vBBU_obj_dict[pkt['dst']].DL_buffer.put(pkt)


env = simpy.Environment()

#static variables
splitting_table=splits_info
adist = 1
num_cells = 1
num_RRHs=1

#instances
#x = PacketGenerator(env, rrh_id, adist)
#FH_SW = FH_switch(env,num_RRHs)
#MID_SW = MID_switch(env,num_RRHs)


for cell_id in range(num_cells):
	edge_vBBU_dict = {}
	metro_vBBU_dict = {}
	for vBBU_id in range(num_RRHs):
		metro_vBBU_dict[vBBU_id] = Metro_vBBU(env,cell_id,vBBU_id,splitting_table)
		edge_vBBU_dict[vBBU_id] = Edge_vBBU(env,cell_id,vBBU_id,splitting_table)

	FH_phi_port = Phi_port_pool(env,'FH',cell_id,num_RRHs,edge_vBBU_dict)
	MID_phi_port = Phi_port_pool(env,'MID',cell_id,num_RRHs,metro_vBBU_dict,edge_vBBU_dict,max_bw=1000)
	
	edge_pool = Edge_vBBU_Pool(env,cell_id,num_RRHs,edge_vBBU_dict,MID_phi_port)
	metro_pool = Metro_vBBU_Pool(env,cell_id,num_RRHs,metro_vBBU_dict,MID_phi_port)

	for id in range(num_RRHs):
		RRH(env,cell_id,id,id,0,FH_phi_port)
coding = 28
orch = Orchestrator(env,num_RRHs,splitting_table,MID_phi_port,edge_vBBU_dict, 28)
env.run(until=2001)