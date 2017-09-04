import simpy
import argparse
import logging
import pandas as pd
from collections import defaultdict
import os
import math
from operator import itemgetter
import time
from scipy.stats import truncnorm
import numpy as np


parser = argparse.ArgumentParser(description="Function Splitting - Hibryd RAN Simulator")
parser.add_argument("T", type=str, default='Hybrid',choices=["CRAN","DRAN","Hybrid"], help="Topology")
parser.add_argument("-D", "--duration", type=int, default=10, help="Duration of simulation, in seconds.")
parser.add_argument("-S", "--seed", type=int, default=10, help="Random number generator seed number.")
parser.add_argument("-C", "--cells", type=int, default=2, help="Cell clusters number.")
parser.add_argument("-R", "--rrhs", type=int, default=7, help="Remote radio heads number per cell cluster.")
parser.add_argument("-A", "--adist", type=int, default=10, help="Interval between CPRI packets arrival in ms.")
parser.add_argument("-L", "--lthold", type=int, default=60, help="Lower orchestrator's threshold in %, to trigger splitting.")
parser.add_argument("-H", "--hthold", type=int, default=90, help="Higher orchestrator's threshold in %, to trigger splitting.")
parser.add_argument("-B", "--bwmid", type=int, default=10, help="Midhaul ports bandwidth in Gbits.")
parser.add_argument("-I", "--interval", type=int, default=2.001, help="Interval in secs between orchestrator consulting the MID.")
#parser.add_argument("-Q", "--qlimit", type=int, default=None, help="The size of the FH and MID port queue in bytes.")

"""
---> TODO arguments <---
1.coding still fixed
2.pkt generation distribution's mean (CPRI_option) still fixed at 3.3
3.QLIMIT in MID and server. Look for common buffer size on switch and server port(vBBU), the latter receives the pause frames
"""
args = parser.parse_args()

#Arguments
TOPOLOGY = args.T
DURATION = args.duration
SEED = args.seed
N_CELLS = args.cells
N_RRHS = args.rrhs
ADIST = args.adist
LTHOLD = args.lthold
HTHOLD = args.hthold
BWMID = args.bwmid
INTERVAL = args.interval

print "TOPOLOGY == %s" % TOPOLOGY

logging.basicConfig(filename='func-sim.log',level=logging.DEBUG,format='%(asctime)s %(message)s')

dir_path = os.path.dirname(os.path.realpath(__file__))
timestamp = time.strftime("%Y-%m-%d_%H:%M:%S")
os.makedirs("csv/"+timestamp)
# os.makedirs("csv/"+timestamp+"/pkt")
# os.makedirs("csv/"+timestamp+"/proc")
# os.makedirs("csv/"+timestamp+"/bw")
# os.makedirs("csv/"+timestamp+"/energy")
pkts_file = open("csv/{}/{}-{}-{}-{}-{}-{}-{}-{}-pkts.csv".format(timestamp,N_CELLS,N_RRHS,ADIST,SEED,DURATION,LTHOLD,HTHOLD,BWMID),"w")
proc_pkt_file = open("csv/{}/{}-{}-{}-{}-{}-{}-{}-{}-proc-pkt.csv".format(timestamp,N_CELLS,N_RRHS,ADIST,SEED,DURATION,LTHOLD,HTHOLD,BWMID),"w")
bw_usage_file = open("csv/{}/{}-{}-{}-{}-{}-{}-{}-{}-bw-usage.csv".format(timestamp,N_CELLS,N_RRHS,ADIST,SEED,DURATION,LTHOLD,HTHOLD,BWMID),"w")
base_energy_file = open("csv/{}/{}-{}-{}-{}-{}-{}-{}-{}-base-energy.csv".format(timestamp,N_CELLS,N_RRHS,ADIST,SEED,DURATION,LTHOLD,HTHOLD,BWMID),"w")
proc_usage_file = open("csv/{}/{}-{}-{}-{}-{}-{}-{}-{}-proc-usage.csv".format(timestamp,N_CELLS,N_RRHS,ADIST,SEED,DURATION,LTHOLD,HTHOLD,BWMID),"w")


DURATION = DURATION * 1000 # transforming second to ms
BWMID = BWMID * 1000 # transforming GB to MB
INTERVAL = INTERVAL * 1000
LTHOLD = LTHOLD/100.0
HTHOLD = HTHOLD/100.0

pkts_file.write("timestamp,pkt_id,cell_id,prb,cpri_option,coding\n") #DONE
proc_pkt_file.write("cell_id,vbbu_id,cloud,pkt_id,split,gops_vbbu,gops_pkt,energy,time_start,time_end,proc_delay\n") #DONE
bw_usage_file.write("cell_id,vbbu_id,haul,pkt_id,bw,plane,type\n") #DONE
base_energy_file.write("entity,id,energy,timestamp\n") # DONE
proc_usage_file.write("cell_id,vbbu_id,cloud,gops,pcnt,timestamp\n") #DONE

#monitor 'justice' metrics after implementing joint (bw and energy) split algorithm

""" For cloudified C-RAN architectures see 'Radio Base Stations in the Cloud' paper """

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
# Regardless of System Bandwidth, PCFICH is always carried by 4 REGs (16 REs) at the first symbol of each subframe
# PCFICH_REs=16.0 
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
# for cada in range(1,8):
# 	print splits_info[28][3][cada]
# print "---"

#-----------CLASSES AND SIMULATOR-------------
# transform defaultdict into normal dict. remove garbage from print
def default_to_regular(d):
	if isinstance(d, defaultdict):
		d = {k: default_to_regular(v) for k, v in d.iteritems()}
	return d

class Orchestrator(object):
	""" Heuristic now: put everything in metro, limited by mid BW and proactivelly before losses appear (60%<x<90%)
	TODOs: 
	1. Consider energy in calcs
	2. Consider justice in calcs
	3. Consider maximum chain delay in calcs (QOS)
		3.1 Measure chain latency
			3.1.1 (all of them desired) Measure transmission + queing + processing delays

	"""
	def __init__ (self,env,splitting_table,fix_coding,interval=2001,high_thold=0.9,low_thold=0.6,topology='Hybrid'):
		self.env=env
		self.interval = interval
		self.fix_coding = fix_coding

		# multi layer default dict \/
		nested_dict = lambda: defaultdict(nested_dict)
		self.vBBUs_dict = nested_dict()

		self.splitting_table = splitting_table
		
		self.high_thold = high_thold
		self.low_thold = low_thold

		if topology == 'Hybrid':
			self.read_metrics = self.env.process(self.read_metrics())
	
	def add_cell(self, cell_id, cell_id_edge_vBBUs, MID_phi_port, cell_id_metro_vBBUs):
		# [cell_id][vbbu_id]['split']=split
		# [cell_id][vbbu_id]['metro_vbbu']=class_obj
		# [cell_id][vbbu_id]['edge_vbbu']=class_obj
		# [cell_id]['mid_port']= class_obj

		num_vBBUs = len(cell_id_edge_vBBUs)
		if num_vBBUs == len(cell_id_metro_vBBUs): # considering 1 to 1 edge and metro vBBUs
			self.vBBUs_dict[cell_id]['mid_port']= MID_phi_port

			MID_phi_port.add_UL_entry('orchestrator',self)

			for vBBU in range(num_vBBUs):
				edge_vBBU = cell_id_edge_vBBUs[str(vBBU)]
				self.vBBUs_dict[cell_id][str(vBBU)]['edge_vbbu'] = edge_vBBU

				metro_vBBU = cell_id_metro_vBBUs[vBBU]
				self.vBBUs_dict[cell_id][str(vBBU)]['metro_vbbu'] = metro_vBBU

				self.vBBUs_dict[cell_id][str(vBBU)]['split'] = 1
		else:
			logging.debug("Error adding cell in Orchestrator. Not 1 to 1 edges and metro vBBUs.")

	def high_splitting_updt(self,cell_id,phi_metrics,vbbu_metrics,MID_port):
		# total phi drops
		bytes_usage = phi_metrics['UL_bytes_rx_diff']
		reduced_bw = 0
		MID_max_bw = phi_metrics['max_bw']
		print "----- ORCHESTRATOR ------"
		#print "Total byte usage MID: %f " % bytes_usage
		
		changed_vBBU_splits = {} # dict of changed vbbu splits key: vbbu_id and value: split 
		
		# ordered list of most bw on a vBBU
		usage_list = []
		#print "VBBU_metrics"
		#print vbbu_metrics
		for cada in vbbu_metrics:
			list_pos = (cada,vbbu_metrics[cada]['UL_bytes_rx_diff'])
			usage_list.append(list_pos)

		# ordered list
		usage_list.sort(key=itemgetter(1),reverse=True)
		#print "usage list:"
		#print usage_list
		#print ""
		
		cpri_option = 3
		
		diff_bw = 0
		split = 0
		# get most usages and change their split until around 10% under maximum bw of MID
		for vbbu_tuple in usage_list:
			try :
				int(vbbu_tuple[0])
			except:
				continue

			#print "\nStart test vbbu: %s" % vbbu_tuple[0]
			if (bytes_usage) > (self.high_thold * MID_max_bw):
				last_bw_diff = 0
				#print "ENTER IF 1: Bytes usage %f > h_thold %f" % (bytes_usage,(self.high_thold * MID_max_bw))
				#print "Reduced bw: %f" % reduced_bw
				#vbbu_split = self.vBBU_splits[vbbu_tuple[0]]

				vbbu_split = self.vBBUs_dict[cell_id][vbbu_tuple[0]]['split']

				#print "BLA %s" % self.vBBUs_dict[(cell_id)][vbbu_tuple[0]]
				#print vbbu_tuple
				#print cell_id
				#print vbbu_split
				#print "vbbu_split: %d" % vbbu_split

				bw_vbbu_split = self.splitting_table[self.fix_coding][cpri_option][vbbu_split]['bw']
				last_bw_vbbu_split=bw_vbbu_split
				#print "Actual: vbbu split= %d | vBBU bw= %.3f | BW last split: %.3f" % \
				#(vbbu_split, bw_vbbu_split, last_bw_vbbu_split)

				for split in range(vbbu_split+1,7+1):
					#print "Test split %d" % split,
					bw_split = self.splitting_table[self.fix_coding][cpri_option][split]['bw']
					# difference between splits
					diff_bw = bw_vbbu_split - bw_split
					#print "Diff bw= %.3f" % diff_bw
					changed_vBBU_splits[vbbu_tuple[0]] = split
					
						#bytes_usage -= diff_bw
					# if what was reduced is lower than our threshold
					if (bytes_usage-diff_bw) <= (self.high_thold * MID_max_bw) or split==7:
						#print "ENTER IF 2: diff %f <= h_thold or split 7s" % (bytes_usage-diff_bw)
						if (bytes_usage-diff_bw) <= (self.low_thold * MID_max_bw):
							#print "Entrou low_thold"
							# test if last split bw is > h_thold
							diff_bw2 = last_bw_diff
							#print diff_bw2
							if (bytes_usage-diff_bw2) > (self.high_thold * MID_max_bw):
								#if one split > h_thold and the other < l_thold...
								# let the higher split and choose split of the next vbbu
								#print "Last split is higher than h_thold. %f" % (bytes_usage-diff_bw2)
								#print "Changing VBBU%d to split %d" % (vbbu_tuple[0],split)
								changed_vBBU_splits[vbbu_tuple[0]] = split
							else:
								#print "Changing VBBU%d to split %d" % (vbbu_tuple[0],split-1)
								changed_vBBU_splits[vbbu_tuple[0]] = split-1
							break
						bytes_usage -= diff_bw
						#print "Next vbbu. Bytes usage now: %f " % bytes_usage
						#set split
						break
					last_bw_diff=diff_bw
				

		# write changes to the EDGE VBBU POOL
		#print "CHANGED"
		print changed_vBBU_splits
		if len(changed_vBBU_splits) > 0:
			for cada in changed_vBBU_splits:
				#print "CADA %d" % cada
				#create pkt
				str_vbbu = str(cada)
				split_updt = {'plane':'ctrl','src':'orchestrator', 'dst':'edge_pool_'+str(cell_id), \
							  'vBBU_id':cada, 'split':changed_vBBU_splits[cada]}
				#print split_updt
				#send to MID_port
				
				# updt splits table of edge vbbus 
				#self.vBBU_splits[cell_id][cada]= changed_vBBU_splits[cada]
				self.vBBUs_dict[int(cell_id)][cada]['split'] = changed_vBBU_splits[cada]

				MID_port.downstream.put(split_updt)
		else:
			logging.debug("WARNING: No better splitting possible at cell%d. Actual MID BW: %.3f" %(cell_id,MID_max_bw))

	def low_splitting_updt(self,cell_id,phi_metrics,vbbu_metrics,MID_port):
		# total phi drops
		bytes_usage = phi_metrics['UL_bytes_rx_diff']
		reduced_bw = 0
		MID_max_bw = phi_metrics['max_bw']
		print "----- LOW ORCHESTRATOR ------"
		#print "Total byte usage MID: %f " % bytes_usage
		
		changed_vBBU_splits = {} # dict of changed vbbu splits key: vbbu_id and value: split 
		
		# ordered list of lowest bw on a vBBU
		usage_list = []
		#print "VBBU_metrics"
		#print vbbu_metrics
		for cada in vbbu_metrics:
			list_pos = (cada,vbbu_metrics[cada]['UL_bytes_rx_diff'])
			usage_list.append(list_pos)

		# ordered list
		usage_list.sort(key=itemgetter(1))
		#print "usage list"
		#print usage_list
		#TODO: Consider that CPRI_option changes. Now CPRI is fixed = 3
		cpri_option = 3
		#TODO: Consider energy in calcs
		diff_bw = 0
		split = 0
		# get most usages and change their split until around 10% under maximum bw of MID
		for vbbu_tuple in usage_list:
			try :
				int(vbbu_tuple[0])
			except:
				continue

			#print "\nStart test vbbu: %s" % vbbu_tuple[0]
			if (bytes_usage) <= (self.low_thold * MID_max_bw):
				#print "ENTER IF 1: Bytes usage %f <= l_thold %f" % (bytes_usage,(self.low_thold * MID_max_bw))
				#print self.vBBU_splits
				#print "Reduced bw: %f" % reduced_bw
				#vbbu_split = self.vBBU_splits[vbbu_tuple[0]]
				vbbu_split = self.vBBUs_dict[cell_id][vbbu_tuple[0]]['split']
				#print "vbbu_split: %d" % vbbu_split
				bw_vbbu_split = self.splitting_table[self.fix_coding][cpri_option][vbbu_split]['bw']
				last_bw_vbbu_split = bw_vbbu_split
				#print "Actual: vbbu split= %d | vBBU bw= %.3f | BW last split: %.3f" %\
				# (vbbu_split, bw_vbbu_split, last_bw_vbbu_split)

				#print range(1,vbbu_split)[::-1]
				for split in range(1,vbbu_split)[::-1]:
					#print "Test split %d" % split,
					bw_split = self.splitting_table[self.fix_coding][cpri_option][split]['bw']
					# difference between splits
					diff_bw = bw_split - bw_vbbu_split

					# initializing last_bw_diff at first bbbu
					# if split == vbbu_split-1:
					# 	last_bw_diff = diff_bw

					#print "DIFF BW: %.3f " % diff_bw
					changed_vBBU_splits[vbbu_tuple[0]] = split
					
					# if (bytes_usage+diff_bw) > (self.high_thold * self.MID_max_bw):
					# 	changed_vBBU_splits[vbbu_tuple[0]] = split+1
					# 	diff_bw = last_bw_diff
					# 	#print "MAIOR. Volta p/ split %d c/ diff %f" % (split+1, diff_bw)
					# 	break
					# if what was reduced is lower than our threshold
					if (bytes_usage+diff_bw) > (self.low_thold * MID_max_bw) or split == 1:
						#print "ENTER IF 2: diff %f > l_thold or split==1" % (bytes_usage+diff_bw)
						if (bytes_usage+diff_bw) > (self.high_thold * MID_max_bw):
							#print "Entrou low_thold"
							# test if last split bw is > h_thold
							diff_bw2 = last_bw_diff
							if (bytes_usage+diff_bw2) <= (self.low_thold * MID_max_bw):
								#if one split > h_thold and the other < l_thold...
								# let the lower split and choose split of the next vbbu
								print "Changing VBBU%d to split %d" % (vbbu_tuple[0],split+1)
								changed_vBBU_splits[vbbu_tuple[0]] = split+1
							else:
								print "Changing VBBU%d to split %d" % (vbbu_tuple[0],split)
								break
							
						#print "Entrou 4"
						
						bytes_usage += diff_bw
						#print bytes_usage
						#set split
						break
					last_bw_diff=diff_bw


		#print "CHANGED"
		print changed_vBBU_splits
		if len(changed_vBBU_splits) > 0: # write changes to the EDGE VBBU POOL
			for cada in changed_vBBU_splits:
				#print "CADA %s" % cada
				#create pkt
				str_vbbu = str(cada)
				split_updt = {'plane':'ctrl','src':'orchestrator', 'dst':'edge_pool_'+str(cell_id),\
							  'vBBU_id':cada, 'split':changed_vBBU_splits[cada]}
				#print split_updt
				
				#send to MID_port
				MID_port.downstream.put(split_updt)
				
				# change split in vbbu_dict 
				self.vBBUs_dict[int(cell_id)][cada]['split'] = changed_vBBU_splits[cada]
				

	def read_metrics(self):
		# wait interval to gather metrics
		while True:
			yield self.env.timeout(self.interval)
			#print "---> Time now: %d <--- " % self.env.now
			#print self.vBBUs_dict.keys()
			for cell_id in self.vBBUs_dict.keys():
			# read amount of bytes dropped in midhaul

				#phi_metrics,vbbu_metrics = self.MID_port.get_metrics()
				#print cell_id
				MID_port = self.vBBUs_dict[cell_id]['mid_port'] # get MID port
				phi_metrics,vbbu_metrics = MID_port.get_metrics() # get metrics of MID port
				MID_max_bw = phi_metrics['max_bw'] # get max BW of MID
				#print MID_max_bw
				#print self.high_thold
				bytes_usage = phi_metrics['UL_bytes_rx_diff'] # get last bytes usage of MID
				print "bytes_usage %.3f of MIDport from cell %d" % (bytes_usage,cell_id)
				#print "HIGH: %.3f" % (self.high_thold * MID_max_bw)
				#print "LOW: %.3f" % (self.low_thold * MID_max_bw)
				# default high_thold is a max of 90% in order to trigger splitting updt
				if (bytes_usage > (self.high_thold * MID_max_bw)):
					#print "HIGH: %.3f" % (float(self.high_thold * MID_max_bw))
					self.high_splitting_updt(cell_id, phi_metrics, vbbu_metrics, MID_port)
				elif (bytes_usage <= (self.low_thold * MID_max_bw)):
				 	self.low_splitting_updt(cell_id, phi_metrics, vbbu_metrics, MID_port)
				#print "\n\n"


class Edge_DC(object):
	""" 1 Edge DC supports only 1 RRH cell currently"""
	def __init__ (self,env,cell_id,n_vBBUs,FH_phi_port,MID_phi_port,topology="Hybrid"):
		self.env=env
		self.topology = topology
		self.name="edgeDC_"+str(cell_id)
		#self.baseline_energy=700	#from china mobile 2014 white paper
		self.AC_energy= 100 + n_vBBUs * 50
		self.battery_energy= 10 + n_vBBUs *10
		self.base_pool_energy= 10 + n_vBBUs * 10
		self.baseline_energy= self.AC_energy + self.battery_energy + self.base_pool_energy

		base_energy_file.write("{},{},{},{}\n".format(self.name,cell_id,self.baseline_energy,self.env.now))

		self.edge_pool= Edge_vBBU_Pool(env,cell_id,n_vBBUs,FH_phi_port,MID_phi_port,splitting_table,topology=topology)
		#print "Edge_pool %d = %s" % (cell_id,self.edge_pool)

		self.edge_vBBUs = self.edge_pool.edge_vBBU_dict

		self.FH_phi_port = FH_phi_port
		self.MID_phi_port = MID_phi_port
		

class Metro_DC(object):
	def __init__(self,env,splitting_table,H_THOLD,L_THOLD,interval,topology):
		self.env=env
		self.name="metroDC"
		self.metro_vBBUs= {}
		self.splitting_table = splitting_table
		self.n_vBBUs = 0
		print "Metro H_THOLD %f" % H_THOLD
		print "Metro L_THOLD %f" % L_THOLD
		# 1 metro pool per cell
		self.metro_pools={}
		self.fix_coding = 28
		self.orchestrator = Orchestrator(env,splitting_table,self.fix_coding,interval=interval,high_thold=H_THOLD,low_thold=L_THOLD,topology=topology)

		self.action=self.env.process(self.run())

	def run(self):
		yield self.env.timeout(0.001)
		base_energy_file.write("{},{},{},{}\n".format(self.name,0,self.baseline_energy,self.env.now))

	def add_cell_on_orch(self,cell_id, cell_id_edge_vBBUs, MID_phi_port, cell_id_metro_vBBUs):
		self.orchestrator.add_cell(cell_id, cell_id_edge_vBBUs, MID_phi_port, cell_id_metro_vBBUs)


	def add_metro_pool(self,cell_id,n_vBBUs,MID_phi_port):
		"""Create a new metro pool for a cell """
		#create metro pool
		self.n_vBBUs+=n_vBBUs
		self.metro_pools[cell_id] = Metro_vBBU_Pool(self.env,cell_id,n_vBBUs,MID_phi_port,self.splitting_table)

		# put metro poll's vBBUs in dict
		self.metro_vBBUs[cell_id] = self.metro_pools[cell_id].metro_vBBU_dict

		self.calc_energy()
		#self.baseline_energy=550	#from china mobile 2014 white paper

	def calc_energy(self):
		self.AC_energy = 50 + self.n_vBBUs * 10
		self.battery_energy = 5 + self.n_vBBUs *5
		self.base_pool_energy = 5 + self.n_vBBUs * 5
		self.baseline_energy = self.AC_energy + self.battery_energy + self.base_pool_energy

class Metro_vBBU_Pool(object):
	def __init__(self,env,cell_id,n_vBBUs,MID_phi_port,splitting_table):
		self.env = env
		self.splitting_table = splitting_table
		self.cell_id = cell_id
		self.n_vBBUs = n_vBBUs
		self.name = 'metro_pool_'+str(self.cell_id)
		self.MID_phi_port = MID_phi_port

		self.metro_vBBU_dict = {}
		
		for vBBU_id in range(n_vBBUs):
			self.metro_vBBU_dict[vBBU_id] = Metro_vBBU(env,cell_id,vBBU_id,MID_phi_port,splitting_table)

		self.baseline_energy = 5 + (n_vBBUs * 2)

		base_energy_file.write("{},{},{},{}\n".format(self.name,cell_id,self.baseline_energy,self.env.now))

		#add itself on MID phi UL
		self.MID_phi_port.add_UL_entry(self.name,self)

		# here run monitoring or something alike 
		# self.action = self.env.process(self.monitoring())

	def set_vBBUs_MID_port(self,MID_phi_port):
		for vBBU in self.metro_vBBU_dict.values():
			vBBU.MID_port = self.MID_phi_port

	def set_vBBU_split(self,vBBU_id,split):
		# orchestrator changing a split option of a vBBU
		metro_vBBU_dict[vBBU_id].set_split(split)


class Edge_vBBU_Pool(object):
	def __init__(self,env,cell_id,n_vBBUs,FH_phi_port,MID_phi_port,splitting_table,topology="Hybrid"):
		self.env = env
		self.cell_id = cell_id
		self.name = 'edge_pool_'+str(self.cell_id)
		self.FH_phi_port = FH_phi_port
		self.MID_phi_port = MID_phi_port
		self.n_vBBUs = n_vBBUs
		self.splitting_table = splitting_table
		#self.edge_vBBU_dict = edge_vBBU_dict
		self.edge_vBBU_dict = {}
		for vBBU in range(self.n_vBBUs):
			self.edge_vBBU_dict[str(vBBU)] = Edge_vBBU(env,cell_id,vBBU,FH_phi_port,MID_phi_port,splitting_table,topology=topology)
			
		self.DL_buffer = simpy.Store(env) # communication from metro & orch with this edge pool

		self.baseline_energy = 5 + (n_vBBUs * 5)
		base_energy_file.write("{},{},{},{}\n".format(self.name,cell_id,self.baseline_energy,self.env.now))

		# here run monitoring or something alike 
		# self.action = self.env.process(self.monitoring())
		self.action = self.env.process(self.listen_orchestrator())
		
		# add DL entry at MID switch to enable orchestrator communication 
		self.MID_phi_port.add_DL_entry(self.name,self)
		
	def set_vBBUs_MID_port(self,MID_phi_port):
		for vBBU in self.edge_vBBU_dict.values():
			vBBU.MID_port = self.MID_phi_port

	def listen_orchestrator(self):
		while True:
			pkt = yield self.DL_buffer.get()
			#print pkt
			if pkt['dst'] == self.name:
				self.set_vBBU_split(pkt['vBBU_id'],pkt['split'])


	def set_vBBU_split(self,vBBU_id,split):
		# orchestrator changing a split option of a vBBU
		#print "Setting split of vBBU%s to %d" % (vBBU_id,split)
		self.edge_vBBU_dict[vBBU_id].set_split(split)


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
base_energy_file.write("{},{},{},{}\n".format(self.name,cell_id,self.baseline_energy,self.env.now))
			2. One core is reserved for one RRH and one core is enough for a RRH.
			3. Processing energy is calculated by the processing timeout the RRH traffic load requires for the core
			4. IF no core sharing and if one core is enough for a RRH, then there is no blocking in processing
			5. Intra-bbu processing model
	 				When implemmenting inter-bbu model (sending processes to other vBBU if theres high delay/blocking),
					refer to the article "Radio Base Stations in the Cloud - 2013"
	 We consider RRHs to not shutdown for energy saving, they're always UP
	-------------------------------------
	"""
	def __init__ (self,env,cell_id,vBBU_id,splitting_table,GOPS=8000,topology="Hybrid"):
		self.env = env
		self.topology = topology
		self.cell_id = cell_id
		self.id = vBBU_id
		self.splitting_table = splitting_table 	# defaultdict with bw&gops per split
		self.GOPS = GOPS # every bbu is equal by default=8000GOPS

		self.used_GOPS_interval=0.
		self.utilization= self.env.process(self.proc_utilization(self.GOPS))

	def proc_utilization(self,GOPS):
		last_GOPS_interval=0.
		while True:
			yield self.env.timeout(1000) # wait 1s to measure percentage GOPS usage
			diff = self.used_GOPS_interval - last_GOPS_interval
			
			pcnt_used = float(diff)/GOPS
			
			last_GOPS_interval= self.used_GOPS_interval
			proc_usage_file.write("{},{},{},{},{},{}\n".\
				format(self.cell_id,self.id,"metro",8000,pcnt_used,self.env.now))


class Metro_vBBU(vBBU):
	def __init__(self,env,cell_id,vBBU_id,MID_phi_port,splitting_table,GOPS=8000,core_dyn_energy=15):
		vBBU.__init__(self,env,cell_id,vBBU_id,splitting_table,GOPS)
		self.core_dyn_energy = core_dyn_energy
		self.UL_buffer = simpy.Store(self.env)
		
		self.MID_phi_port = MID_phi_port # In future will be used for downlink
		# TODO: DL functions 
		#self.DL_buffer = simpy.Store(self.env)
		self.MID_phi_port.add_UL_entry(str(vBBU_id),self)

		self.MID_receiver = self.env.process(self.MID_receiver())
		#self.MID_sender = self.env.process(self.MID_sender(MID_port))


	def MID_receiver(self):
		"""process to get the packet from the FH_switch port buffer"""
		while True:
			#print "Time: %f. METRO vBBU%d waiting for pkt in UL_buffer" % (self.env.now, self.id)
			pkt = yield self.UL_buffer.get()
			
			#print "Time: %f. METRO vBBU%d starting pkt%d processing (splitting) " % (self.env.now,self.id,pkt.id)
			yield self.env.process(self.splitting(pkt))


	def splitting(self,pkt):
		# print "METRO CLOUD"
		# print "Coding: %d" % pkt.coding,
		# print "CPRI_OPTION %d" % pkt.CPRI_option,
		# print "RRH ID %s " % pkt.dst,
		# print "Split: %d" % pkt.split
		
		if pkt.split == 7: # if C-RAN split send everything to MetroCloud
			#print "Split 7. Pkt%d already processed. Nothing to do" % pkt.id
			del pkt
			return

		#by the packets attributes (MCS, CPRI option) and its split, get GOPS and BW from table
		GOPS = self.splitting_table[pkt.coding][pkt.CPRI_option][pkt.split]['metro_gops']
		#print "METRO GOPS split: %d" % GOPS
		#print "Self gops %d" % self.GOPS
		#print "BW pkt split: %f Mb" % bw_split
		
		#timeout proc
		proc_tout = float(GOPS)/self.GOPS
		#print "METRO Proc Timeout %f" % proc_tout
		energy = (float(proc_tout) * (self.core_dyn_energy))/100 #== measured in 10ms instead of 1s
		start = self.env.now
		#print "METRO Energy consumption: %f W" % energy
		self.used_GOPS_interval += GOPS
		yield self.env.timeout(proc_tout)

		#print "METRO After t_out. Time now= %f" % self.env.now
		proc_pkt_file.write("{},{},{},{},{},{},{},{},{},{}\n".format(self.cell_id,self.id,"metro",pkt.id,pkt.split,\
								self.GOPS,GOPS,energy,start,start+proc_tout,proc_tout))
		# send pkt to core network (for us it means destroying it)
		del pkt


class Edge_vBBU(vBBU):
	def __init__(self,env,cell_id,vBBU_id,FH_phi_port,MID_phi_port,splitting_table,GOPS=8000,core_dyn_energy=30,topology="Hybrid"):
		vBBU.__init__(self,env,cell_id,vBBU_id,splitting_table,GOPS,topology=topology)
		self.core_dyn_energy = core_dyn_energy
		self.split=1 # variable not in metro_vbbu, because after edgesplit the pkt gets variable split
		self.UL_buffer = simpy.Store(self.env)

		self.FH_phi_port = FH_phi_port
		self.MID_phi_port = MID_phi_port
		#print self.topology
		# TODO: DL functions 
		#self.DL_buffer = simpy.Store(self.env)

		self.FH_phi_port.add_UL_entry(str(vBBU_id),self)
		self.MID_phi_port.add_DL_entry(str(vBBU_id),self)

		self.FH_receiver = self.env.process(self.FH_receiver()) # get from FH_switch buffer
		#self.MID_receiver = self.env.process(self.MID_receiver()) # DL not done yet
		
	def set_split(self,split):
		self.split=split

	def FH_receiver(self):
		"""process to get the packet from the FH_switch port buffer"""
		while True:
			#print "Time: %f. BBU%d waiting for pkt in UL_buffer" % (self.env.now, self.id)
			pkt = yield self.UL_buffer.get()
			
			#print "Time: %f. BBU%d starting pkt%d processing (splitting) " % (self.env.now,self.id,pkt.id)
			yield self.env.process(self.splitting(pkt))

	def splitting(self,pkt):
		#print "Coding: %d" % pkt.coding,
		#print "CPRI_OPTION %d" % pkt.CPRI_option,
		#print "RRH ID %s " % pkt.dst,
		#print "Split: %d" % self.split

		if self.topology == "DRAN":
			#print "Entrou DRAN"
			pkt.split = 7
		else:
			pkt.split = self.split
			if pkt.split == 1 or self.topology == 'CRAN': # if C-RAN split send everything to MetroCloud
				#print "entrou C-RAN"
				#print "Split 1. Send pkt straight to metroCloud."
				# send pkt to Metro
				bw_split = (self.splitting_table[pkt.coding][pkt.CPRI_option][pkt.split]['bw'])/100
				pkt.size= bw_split
				energy = 0.001
				start = self.env.now
				proc_pkt_file.write("{},{},{},{},{},{},{},{},{},{}\n".format(self.cell_id,self.id,"edge",pkt.id,\
								pkt.split,self.GOPS,0,energy,start,start,0))
				self.MID_phi_port.upstream.put(pkt)
				return


		bw_split = (self.splitting_table[pkt.coding][pkt.CPRI_option][pkt.split]['bw'])/100	
		#by packets attributes (MCS, CPRI option) and its split, get GOPS and BW from table
		GOPS = self.splitting_table[pkt.coding][pkt.CPRI_option][pkt.split]['edge_gops']
		#print "GOPS: %d" % GOPS,
		#print "Self gops %d" % self.GOPS
		#print "BW pkt split: %f Mb" % bw_split,

		#timeout proc
		proc_tout = float(GOPS)/self.GOPS
		#print "Proc Timeout %f" % proc_tout,
		energy = (float(proc_tout) * (self.core_dyn_energy))/100 # == measured in 1ms instead of 1s
		#print "Energy consumption: %f W" % energy
		start = self.env.now
		yield self.env.timeout(proc_tout)
		#print "After t_out. Time now= %f" % self.env.now
		pkt.size= bw_split
		
		# LOG the proc delay and energy usage
		proc_pkt_file.write("{},{},{},{},{},{},{},{},{},{}\n".format(self.cell_id,self.id,"edge",pkt.id,\
							pkt.split,self.GOPS,GOPS,energy,start,start+proc_tout,proc_tout))

		# send pkt to phi port (midhaul)
		#print "Sending to Phi Midhaul"
		self.MID_phi_port.upstream.put(pkt)
		#print "Sent"



class Packet_CPRI(object):
	""" This class represents a network packet """
	def __init__(self, time, cell_id, rrh_id, CPRI_option, coding, id, src="a", dst="z"):
		self.time = time# creation time
		self.cell_id= cell_id
		self.rrh_id= rrh_id
		self.plane='data'
		self.coding = coding #same as MCS

		self.CPRI_option = CPRI_option

		self.size = (splits_info[coding][self.CPRI_option][1]['bw'])/100
		self.prb = CPRI[self.CPRI_option]['PRB']
		#print self.size
		self.id = id # packet id
		self.src = str(src) #packet source address
		self.dst = str(dst) #packet destination address
		pkts_file.write("{},{},{},{},{},{}\n".format(time,id,cell_id,self.prb,CPRI_option,coding))

	def __repr__(self):
		return "id: {}, src: {}, time: {}, CPRI_option: {}, coding: {}".\
			format(self.id, self.src, self.time, self.CPRI_option, self.coding)

class Packet_Generator(object):
	"""This class represents the packet generation process """
	def __init__(self, env, cell_id, rrh_id, adist, finish=float("inf"), interval=100):
		self.cell_id = cell_id
		self.rrh_id = rrh_id # packet id
		self.env = env # Simpy Environment
		self.adist = adist #packet arrivals distribution

		self.coding = 28 #coding fixed for now

		self.finish = finish # packet end time
		self.interval = interval

		self.out = None # packet generator output
		self.packets_sent = 0 # packet counter
		self.action = env.process(self.run())  # starts the run() method as a SimPy process
		self.change_cpri = env.process(self.change_cpri())

# In 'Analytical and Experimental Evaluation of CPRI over Ethernet Dynamic Rate Reconfiguration',
# its possible to change CPRI option in slightly under 1ms

	def change_cpri(self):
		# variables for cpri calcs (truncaded integer normal distribution)
		mean=3.3
		sd=1
		low=1
		upp=5
		# initial random timeout between RRHs
		#self.env.timeout(random.random())

		while True:
			# workaround to generate a normal distribution in a range of 1 to 5
			X = truncnorm((low - mean) / sd, (upp - mean) / sd, mean, sd).rvs().round().astype(int)
			if X == 4:
				self.CPRI_option = 5
			elif X == 5:
				self.CPRI_option = 7
			else:
				self.CPRI_option = X
			yield self.env.timeout(self.interval)

	def run(self):
		"""The generator function used in simulations. """
		while self.env.now < self.finish:
			# wait for next transmission
			yield self.env.timeout(self.adist)
			self.packets_sent += 1
			#print "New packet generated at %f" % self.env.now

			p = Packet_CPRI(self.env.now, self.cell_id, self.rrh_id, self.CPRI_option, self.coding,\
							self.packets_sent, src=self.rrh_id, dst=self.rrh_id)
			#time,rrh_id, coding, CPRI_option, id, src="a"
			#print p
			#Logging
			#pkt_file.write("{}\n".format(self.fix_pkt_size))
			self.out.put(p)
	# call the function put() of RRHPort, inserting the generated packet into RRHPort' buffer


class Cell(object):
	"""Class representing a fixed set of RRHs connected to a fixed BBU"""
	def __init__(self,env,cell_id,num_rrhs,FH_phi_port,adist,qlimit=0):
		self.env = env
		self.FH_phi_port = FH_phi_port
		self.cell_id = cell_id #Cell indentifier
		self.rrh_dict={}
		for rrh_id in range(0,num_rrhs):
			#creates the packet generator for each RRH and places it into dict
			self.rrh_dict[rrh_id] = RRH(self.env,cell_id,rrh_id,self.FH_phi_port,adist,qlimit)
        

class RRH(object):
	"""Class representing each RRH with its own packet generation (Uplink) to a fixed BBU"""
	def __init__(self,env,cell_id,rrh_id,FH_phi_port,adist,qlimit=0):
		self.env = env
		self.cell_id = cell_id
		self.id = rrh_id
		self.adist = adist # fixed 10 ms == interval between CPRI hyperframes and LTE frames
		self.str_id = str(self.id)
		self.pg = Packet_Generator(self.env, self.cell_id,self.id, self.adist)

		if qlimit == 0:# checks if the queue has a size limit
			queue_limit = None
		else:
			queue_limit = qlimit

		self.port = RRH_Port(self.env, qlimit=queue_limit) #create RRH PORT (FH) to bbu
		self.pg.out = self.port #forward packet generator output to RRH port
		self.sender = self.env.process(self.RRH_sender(FH_phi_port))

	def RRH_sender(self,Phi_port_pool):
		while True:
   			pkt = yield self.port.buffer.get()
   			self.port.byte_size -= pkt.size
   			if self.port.byte_size < 0:
   				print "Error: RRH %d port buffer sizer < 0" % self.id
   			# send to edge vbbu	
   			Phi_port_pool.upstream.put(pkt)
   			self.port.packets_tx +=1
		
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
        """receives a packet from the packet generator and put it on the queue
            if the queue is not full, otherwise drop it.
        """

        self.packets_rec += 1
        #print "+1 RRHPort pkt. Packets received: %d" % self.packets_rec
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
	# def __init__(self,env,name,cell_id,NUMBER_OF_RRHs,UL_vBBU_obj_dict,DL_vBBU_obj_dict={},\
	# 			 max_bw=None, bw_check_interval=1000):
	def __init__(self,env,name,cell_id,max_bw=None, bw_check_interval=1000):
		self.env = env
		self.name = name
		self.cell_id = cell_id

		#individual metrics for each vBBU
		self.DL_vBBU_obj_dict = {}
		self.UL_vBBU_obj_dict = {}

		self.UL_metrics = defaultdict(lambda : defaultdict(float))
		self.DL_metrics = defaultdict(lambda : defaultdict(float))
		
		self.baseline_energy = 10
		base_energy_file.write("{},{},{},{}\n".format(self.name,cell_id,self.baseline_energy,self.env.now))

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
		self.UL_pkt_error = 0
		self.UL_pkt_drop = 0
		self.UL_bytes_drop = 0
		self.UL_usage = 0
		self.last_UL_bytes_rx = 0
		self.last_UL_bytes_tx = 0
		self.last_UL_pkt_rx = 0
		self.last_UL_pkt_tx = 0
		self.last_UL_pkt_error = 0
		self.last_UL_pkt_drop = 0
		self.last_UL_bytes_drop = 0

		# TODO: increment and decrement buffer size on puts and gets
		self.UL_buffer_size = 0 # create function for put and get

		#------- DL -------
		self.DL_pkt_rx = 0
		self.DL_bytes_rx = 0
		self.DL_pkt_tx = 0
		self.DL_bytes_tx = 0
		self.DL_pkt_error = 0
		self.DL_pkt_drop = 0
		self.DL_bytes_drop = 0
		self.DL_usage = 0
		self.last_DL_bytes_rx = 0
		self.last_DL_bytes_tx = 0
		self.last_DL_pkt_rx = 0
		self.last_DL_pkt_tx = 0
		self.last_DL_pkt_error = 0
		self.last_DL_pkt_drop = 0
		self.last_DL_bytes_drop = 0
		
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
			print "Added key %s into UL of %s phi" % (key,self.name) 
			self.UL_vBBU_obj_dict[key]=obj
			
			self.UL_metrics[key]['UL_pkt_rx'] = 0
			self.UL_metrics[key]['UL_bytes_rx'] = 0
			self.UL_metrics[key]['UL_pkt_tx'] = 0
			self.UL_metrics[key]['UL_bytes_tx'] = 0
			self.UL_metrics[key]['UL_pkt_drop'] = 0
			self.UL_metrics[key]['UL_bytes_drop'] = 0
			self.UL_metrics[key]['UL_pkt_error'] = 0

			self.UL_metrics[key]['last_UL_bytes_rx'] = 0
			self.UL_metrics[key]['last_UL_bytes_tx'] = 0
			self.UL_metrics[key]['last_UL_pkt_rx'] = 0
			self.UL_metrics[key]['last_UL_pkt_tx'] = 0
			self.UL_metrics[key]['last_UL_pkt_drop'] = 0
			self.UL_metrics[key]['last_UL_bytes_drop'] = 0
			self.UL_metrics[key]['UL_usage'] = 0 # this metric is always outdated by concept
			self.UL_metrics[key]['last_UL_pkt_error'] = 0
			#print self.UL_metrics[UL_vBBU]
			
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

			self.DL_metrics[key]['DL_pkt_rx'] = 0
			self.DL_metrics[key]['DL_bytes_rx'] = 0
			self.DL_metrics[key]['DL_pkt_tx'] = 0
			self.DL_metrics[key]['DL_bytes_tx'] = 0
			self.DL_metrics[key]['DL_pkt_drop'] = 0
			self.DL_metrics[key]['DL_bytes_drop'] = 0
			self.DL_metrics[key]['DL_pkt_error'] = 0

			self.DL_metrics[key]['last_DL_bytes_rx'] = 0
			self.DL_metrics[key]['last_DL_bytes_tx'] = 0
			self.DL_metrics[key]['last_DL_pkt_rx'] = 0
			self.DL_metrics[key]['last_DL_pkt_tx'] = 0
			self.DL_metrics[key]['last_DL_pkt_drop'] = 0
			self.DL_metrics[key]['last_DL_bytes_drop'] = 0
			self.DL_metrics[key]['DL_usage'] = 0 # this metric is always outdated by concept
			self.DL_metrics[key]['last_DL_pkt_error'] = 0

	def del_DL_entry(self,key,obj):
		try:
			del self.DL_vBBU_obj_dict[key]
		except:
			print "ERROR: key %d already exists in %s DL_table" % (key,self.name)

	def bw_check(self):
		while True:
			yield self.env.timeout(self.bw_check_interval)
			#print "---MID CHECK ---"
			
			#calc usage
			self.UL_bytes_tx_diff = self.UL_bytes_tx - self.last_UL_bytes_tx
			self.UL_pkt_tx_diff = self.UL_pkt_tx - self.last_UL_pkt_tx
			self.UL_bytes_rx_diff = self.UL_bytes_rx - self.last_UL_bytes_rx
			self.UL_pkt_rx_diff = self.UL_pkt_rx - self.last_UL_pkt_rx
			self.UL_pkt_drop_diff = self.UL_pkt_drop - self.last_UL_pkt_drop
			self.UL_bytes_drop_diff = self.UL_bytes_drop - self.last_UL_bytes_drop
			self.UL_pkt_error_diff = self.UL_pkt_error - self.last_UL_pkt_error
			# update last values to actual values
			self.last_UL_bytes_rx = self.UL_bytes_rx
			self.last_UL_bytes_tx = self.UL_bytes_tx
			self.last_UL_pkt_rx = self.UL_pkt_rx
			self.last_UL_pkt_tx = self.UL_pkt_tx
			self.last_UL_pkt_error = self.UL_pkt_error
			self.last_UL_pkt_drop = self.UL_pkt_drop
			self.last_UL_bytes_drop = self.UL_bytes_drop

			#print ""
			#print "UL rx: %d pkts and %.3f Mbps" % (self.UL_pkt_rx_diff,self.UL_bytes_rx_diff)
			#print "UL drops: %d pkts and %.3f Mbps " % (self.UL_pkt_drop_diff, self.UL_bytes_drop_diff)
			#print "UL tx: %d pkts and %.3f Mbps" % (self.UL_pkt_tx_diff, self.UL_bytes_tx_diff)

			for vBBU in self.UL_metrics:
				#calc usage
				#print vBBU
				self.UL_metrics[vBBU]['UL_bytes_tx_diff'] = self.UL_metrics[vBBU]['UL_bytes_tx'] - self.UL_metrics[vBBU]['last_UL_bytes_tx']
				self.UL_metrics[vBBU]['UL_pkt_tx_diff'] = self.UL_metrics[vBBU]['UL_pkt_tx'] - self.UL_metrics[vBBU]['last_UL_pkt_tx']
				self.UL_metrics[vBBU]['UL_bytes_rx_diff'] = self.UL_metrics[vBBU]['UL_bytes_rx'] - self.UL_metrics[vBBU]['last_UL_bytes_rx']
				self.UL_metrics[vBBU]['UL_pkt_rx_diff'] = self.UL_metrics[vBBU]['UL_pkt_rx'] - self.UL_metrics[vBBU]['last_UL_pkt_rx']
				self.UL_metrics[vBBU]['UL_pkt_drop_diff'] = self.UL_metrics[vBBU]['UL_pkt_drop'] - self.UL_metrics[vBBU]['last_UL_pkt_drop']
				self.UL_metrics[vBBU]['UL_bytes_drop_diff'] = self.UL_metrics[vBBU]['UL_bytes_drop'] - self.UL_metrics[vBBU]['last_UL_bytes_drop']
				
				# update last values to actual values
				self.UL_metrics[vBBU]['last_UL_bytes_rx'] = self.UL_metrics[vBBU]['UL_bytes_rx']
				self.UL_metrics[vBBU]['last_UL_bytes_tx'] = self.UL_metrics[vBBU]['UL_bytes_tx']
				self.UL_metrics[vBBU]['last_UL_pkt_rx'] = self.UL_metrics[vBBU]['UL_pkt_rx']
				self.UL_metrics[vBBU]['last_UL_pkt_tx'] = self.UL_metrics[vBBU]['UL_pkt_tx']
				self.UL_metrics[vBBU]['last_UL_pkt_error'] = self.UL_metrics[vBBU]['UL_pkt_error']
				self.UL_metrics[vBBU]['last_UL_pkt_drop'] = self.UL_metrics[vBBU]['UL_pkt_drop']
				self.UL_metrics[vBBU]['last_UL_bytes_drop'] = self.UL_metrics[vBBU]['UL_bytes_drop']

				#print ""
				#print "vBBU%s UL rx: %d pkts and %.3f Mbps" % \
				#(vBBU, self.UL_metrics[vBBU]['UL_pkt_rx_diff'],self.UL_metrics[vBBU]['UL_bytes_rx_diff'])
				#print "vBBU%s UL drops: %d pkts and %.3f Mbps " % \
				#(vBBU, self.UL_metrics[vBBU]['UL_pkt_drop_diff'], self.UL_metrics[vBBU]['UL_bytes_drop_diff'])
				#print "vBBU%s UL tx: %d pkts and %.3f Mbps" % \
				#(vBBU, self.UL_metrics[vBBU]['UL_pkt_tx_diff'], self.UL_metrics[vBBU]['UL_bytes_tx_diff'])


			# zeroing counters
			self.UL_bw_interval = 0
			self.DL_bw_interval = 0
			#print "------"

	def get_metrics(self):
		self.phi_UL_metrics={} #dict of total PHI port metrics
		self.phi_UL_metrics['UL_pkt_rx']= self.UL_pkt_rx
		self.phi_UL_metrics['UL_bytes_rx'] = self.UL_bytes_rx
		self.phi_UL_metrics['UL_pkt_tx'] = self.UL_pkt_tx
		self.phi_UL_metrics['UL_bytes_tx'] = self.UL_bytes_tx
		self.phi_UL_metrics['UL_pkt_error'] = self.UL_pkt_error
		self.phi_UL_metrics['UL_pkt_drop'] = self.UL_pkt_drop
		self.phi_UL_metrics['UL_bytes_drop'] = self.UL_bytes_drop
		self.phi_UL_metrics['UL_usage'] = self.UL_usage
		self.phi_UL_metrics['UL_bytes_rx_diff'] = self.UL_bytes_rx_diff
		self.phi_UL_metrics['UL_bytes_tx_diff'] = self.UL_bytes_tx_diff
		self.phi_UL_metrics['UL_pkt_rx_diff'] = self.UL_pkt_rx_diff
		self.phi_UL_metrics['UL_bytes_tx_diff'] = self.UL_bytes_tx_diff
		self.phi_UL_metrics['UL_pkt_error_diff'] = self.UL_pkt_error_diff
		self.phi_UL_metrics['UL_pkt_drop_diff'] = self.UL_pkt_drop_diff
		self.phi_UL_metrics['UL_bytes_drop_diff'] = self.UL_bytes_drop_diff
		self.phi_UL_metrics['max_bw'] = self.max_bw

		return self.phi_UL_metrics,self.UL_metrics

	def set_vBBU_dict(self,vBBU_obj_dict):
		self.vBBU_obj_dict = vBBU_obj_dict

	def pkt_uplink(self):
		while True:
			pkt = yield self.upstream.get() # get pkt

			self.UL_pkt_rx +=1
			self.UL_metrics[pkt.dst]['UL_pkt_rx']+=1

			if self.max_bw is not None:
				self.UL_bw_interval += pkt.size
				self.UL_bytes_rx += pkt.size
				self.UL_metrics[pkt.dst]['UL_bytes_rx']+= pkt.size

			 	if self.UL_bw_interval > self.max_bw:
					#print "TIME: %f. WARNING: Pkt UL BLOCK!" % self.env.now
					# all UL
					self.UL_pkt_drop +=1
					self.UL_bytes_drop += pkt.size
					# individual vbbu counters
					self.UL_metrics[pkt.dst]['UL_pkt_drop']+= 1
					self.UL_metrics[pkt.dst]['UL_bytes_drop']+= pkt.size
					
					bw_usage_file.write("{},{},{},{},{},{},{}\n".\
					format(pkt.cell_id,pkt.rrh_id,self.name,pkt.id,pkt.size,pkt.plane,"drop"))
					del pkt

				else:
					bw_usage_file.write("{},{},{},{},{},{},{}\n".\
						format(pkt.cell_id,pkt.rrh_id,self.name,pkt.id,pkt.size,pkt.plane,"fw"))
					if pkt.plane == 'ctrl': # IF pkt from edge goes to metro (UL)
						self.UL_vBBU_obj_dict['metro_pool'].UL_buffer.put(pkt)
					else:
						# send pkt to UL end
						#print pkt
						#print self.UL_vBBU_obj_dict
						self.UL_vBBU_obj_dict[pkt.dst].UL_buffer.put(pkt)
						# all UL
						self.UL_pkt_tx +=1
						self.UL_bytes_tx += pkt.size
						# individual vbbu counters
						self.UL_metrics[pkt.dst]['UL_pkt_tx']+= 1
						self.UL_metrics[pkt.dst]['UL_bytes_tx']+= pkt.size

			else:
				#print self.UL_vBBU_obj_dict[pkt.dst]
				# send pkt to UL end
				bw_usage_file.write("{},{},{},{},{},{},{}\n".\
					format(pkt.cell_id,pkt.rrh_id,self.name,pkt.id,pkt.size,pkt.plane,"fw"))
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
			#print pkt
			#print "DL pkt dst: %s" % pkt['dst']
			#print self.DL_vBBU_obj_dict 
			#print "CELL: %d" % self.cell_id
			#print pkt
			#print pkt['dst']
			#print self.DL_vBBU_obj_dict
			#print self.DL_vBBU_obj_dict[pkt['dst']]
			self.DL_vBBU_obj_dict[pkt['dst']].DL_buffer.put(pkt)
			#print "----------"


env = simpy.Environment()
#static variables
splitting_table=splits_info

metro_DC = Metro_DC(env,splitting_table,HTHOLD,LTHOLD,INTERVAL,TOPOLOGY)
coding = 28
#orch = Orchestrator(env,N_RRHS,splitting_table,MID_phi_port,edge_vBBU_dict, coding)
edge_DCs = {}

metro_vBBUs={} #all metro vbbus
edge_vBBUs={} # all edge vbbus

FH_phi_ports_dict = {} # all FHs (1 for each cell)
MID_phi_ports_dict = {} # all MID ports (1 for each edge DC)

for cell_id in range(N_CELLS):
	FH_phi_port = Phi_port_pool(env,'FH',cell_id)
	FH_phi_ports_dict[cell_id] = FH_phi_port
	
	MID_phi_port = Phi_port_pool(env,'MID',cell_id,max_bw=BWMID)
	MID_phi_ports_dict[cell_id] = MID_phi_port

	# create edge DC
	edge_DCs[cell_id] = Edge_DC(env,cell_id,N_RRHS,FH_phi_port,MID_phi_port,topology=TOPOLOGY)

	# tell the Metro DC to create a pool and its metro_vBBUs
	metro_DC.add_metro_pool(cell_id,N_RRHS,MID_phi_port)

	# edge vBBUs of this cell_id 
	cell_id_edge_vBBUs = edge_DCs[cell_id].edge_vBBUs
	# adding them to all edge vbbus
	edge_vBBUs[cell_id]= cell_id_edge_vBBUs
	
	# metro vBBUs of this cell_id
	cell_id_metro_vBBUs = metro_DC.metro_vBBUs[cell_id]
	# adding them to all metro vbbus
	metro_vBBUs[cell_id] = cell_id_metro_vBBUs
	
	# letting the orch know the cell's vBBUs and mid port
	metro_DC.add_cell_on_orch(cell_id, cell_id_edge_vBBUs, MID_phi_port, cell_id_metro_vBBUs)

	Cell(env,cell_id,N_RRHS,FH_phi_port,ADIST)

# time
#duration= 30000 # 30s
#duration= 100000 # 100s
np.random.seed(SEED)

env.run(until=DURATION)
print "END SIMULATION: %d secs" % (int(env.now)/1000)
pkts_file.close()
proc_pkt_file.close()
bw_usage_file.close()
base_energy_file.close()
proc_usage_file.close()