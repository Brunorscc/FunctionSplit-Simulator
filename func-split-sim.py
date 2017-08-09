import simpy
import random
import functools
import sys
import argparse
import logging
import pandas as pd
from collections import defaultdict

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
tbs_table= pd.read_excel("/media/sktx/5E09-AFCB/Bruno/Simulador/tabelas/TBS-table.xlsx")


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
#
# #### L2/L3 layers parameters assumptions
# TBS_DL=75376 	# Transport block size (in bits) 
# IP_TTI_DL= (TBS_DL)/((IPpkt + HdrPDCP + HdrRLC + HdrMAC) *8) 
# nTBS_DL_TTI= 2 	# 2 Layers
# FAPI_DL=1.5
# ---------------------------

Fs=30.72	# Sampling rate at 20MHz
CPRI={\
1:{'Mhz':1.25,'Fs':1.92,'PRB':6},\
2:{'Mhz':2.5,'Fs':3.84,'PRB':12},\
3:{'Mhz':5.0,'Fs':7.68,'PRB':25},\
4:5,\
5:{'Mhz':10.0,'Fs':15.36,'PRB':50},\
6:10,\
7:{'Mhz':5,'Fs':30.72,'PRB':100}}
#nCPRI=32.0	# variavel do SCF, substituida pela tabela acima # CPRI type 4 -> dynamic, depending on FH
####

## PACKET INCOMING

#split=3
#CPRI_type=5

# CPRI option altera:
# 	PRB 
#	sampling frequency MHz
#	TBS
# 	IP pkt TTI

used_CPRIs=[1,2,3,5,7]

bw_splits = defaultdict( lambda: defaultdict(lambda: defaultdict( float )))

for coding in range(0,29):
	for cpri_option in used_CPRIs:
		# CPRI options 4 and 6 do not fit 2x2MIMO LTE traffic perfectly, its necessary 
		# to set custom Fs (Mhz) for each antenna in order achieve a better LTE<->CPRI fit 

		print "CPRI OPTION: %d" % cpri_option
		# print tbs_table[n_RB - PUCCH_RBs][mcs_2_tbs[MCS_UL]]
		# TBS_UL = tbs_table[CPRI[CPRI_type]['PRB'] - PUCCH_RBs][mcs_2_tbs[MCS_UL]]
		TBS_UL = tbs_table[CPRI[cpri_option]['PRB'] - PUCCH_RBs][mcs_2_tbs[MCS_UL]]
		#print "TBS: %.3f" % TBS_UL
		# TBS == MCS 23 UL -> mapped to IndexTBS -> multiplied by allocated PRB
		IP_TTI_UL= (TBS_UL)/((IPpkt + HdrPDCP + HdrRLC + HdrMAC) *8)
		#print "IP_TTI_UL: %.3f" % IP_TTI_UL
		####

		#def bw_split_UL(split,CPRI_type):
		#Upstream coefficients:
	#	if split==1:
		# PHY SPLIT IV - SCF
		a1_UL = nIQ * CPRI_coding
		r1_UL= CPRI[cpri_option]['Fs'] * nAnt * n_Sector * a1_UL
		# nIQ= (2*(15+1))= 32 -> facilita aproximacao nas contas
		# (2*IQ) * Fs * 16/15 * CPRIlinecoding * nAnt * nSectors
		# (2 * 15) * 1.92 * 16/15 * (10/8.0) * 2 * 4 = 614.39999
		split1=1
		bw_splits[coding][cpri_option][1] = r1_UL
		print "Split1 : %.3f Mbps	|" % r1_UL,
		#return r1_UL
	#	if split==2:
		# PHY SPLIT IIIb - SCF
		a2_UL= nIQ
		r2_UL= CPRI[cpri_option]['Fs'] * nAnt * n_Sector * a2_UL
		bw_splits[coding][cpri_option][2] = r2_UL
		print "Split2 : %.3f Mbps |" % r2_UL,
		#return r2_UL
	#	if split==3:
		# PHY SPLIT III - SCF
		a3_UL = n_RB_SC * n_Data_Sym * nIQ # <- *1000 / 1000000
		r3_UL = (a3_UL * nAnt * n_Sector * CPRI[cpri_option]['PRB'])/1000
		bw_splits[coding][cpri_option][3] = r3_UL
		print "Split3 : %.3f Mbps	" % r3_UL
		#return r3_UL
	#	if split==4:
		# PHY SPLIT II - SCF
		#a4_UL = n_RB_SC * n_Data_Sym * nIQ
		#b4_UL = n_RB_SC * n_Data_Sym * PUCCH_RBs * nIQ * nIQ
		r4_UL = (n_Data_Sym * n_RB_SC * (CPRI[cpri_option]['PRB'] - PUCCH_RBs) * nAnt * nIQ)/1000
		bw_splits[coding][cpri_option][4] = r4_UL
		print "Split4 : %.3f Mbps	|" % r4_UL,
		#return r4_UL

	#	if split==5:
		# PHY SPLIT I - SCF
		#a5_UL = (n_RB_SC * n_Data_Sym * QmPUSCH * LayersUL * nSIC * nLLR) / 1000
		#b5_UL = (PUCCH_RBs * n_RB_SC * n_Data_Sym * QmPUSCH * LayersUL * nSIC * nLLR ) / 1000
		#r5_UL = a5_UL * LayersUL * n_RB - b5_UL
		r5_UL = (n_Data_Sym * n_RB_SC * (CPRI[cpri_option]['PRB'] - PUCCH_RBs) * QmPUSCH * LayersUL * nSIC * nLLR) / 1000
		bw_splits[coding][cpri_option][5] = r5_UL
		print "Split5 : %.3f Mbps | " % r5_UL, 
		#return r5_UL

	#	if split==6:
		# SPLIT MAC-PHY - SCF
		a6_UP = (IP_TTI_UL* (IPpkt + HdrPDCP + HdrRLC + HdrMAC) * nTBS_UL_TTI)/ 125
		r6_UL = a6_UP * LayersUL + FAPI_UL
		bw_splits[coding][cpri_option][6] = r6_UL
		print "Split6 : %.3f Mbps" % r6_UL
		#return r6_UP

	#	if split==7:
		# SPLIT RRC-PDCP - SCF
		a7_UP = (IP_TTI_UL * IPpkt * nTBS_UL_TTI) / 125
		r7_UL = a7_UP * LayersUL
		bw_splits[coding][cpri_option][7] = r7_UL
		# OU apenas fazer TBS/1000.0 ...
		print "Split7 : %.3f Mbps\n" %r7_UL
		#return r7_UP
		#print bw_splits

#print dict(bw_splits)

# TO PRINT DEFAULT DICT \/
#import json
#data_as_dict = json.loads(json.dumps(bw_splits, indent=4))
#print(data_as_dict)

# bw_splits[MCS][CPRI option][Split]
print bw_splits[28][7][3]

#print({k: dict(v) for k, v in dict(group_ids).items()})


# campos do pacote CPRI
# list[antenaIDs], setor=2*qtd antenas, 
# 

class EdgeCloud(object):


class vBBU(object):
	#-------------- COMMENTS -------------
	# Packets from RRHs of a cell arrive at the vBBU (only UPLINK modelled)
	# Every BBU:
	#	has a max processing in giga operations per sec (GOPS) arbitrarily defined by us
	#	is able to run every function split
	#	has a fixed amount of CPUs
	#
	# At the moment:
	#	1. The function chain of a RRH can only be processed in one CPU reserved exclusively for it in only one vBBU
	#	2. One CPU is considered enough for a RRH
	# 	Meaning a few things:
	#		1. No multiplexing, no resource allocation schemes (No CPU sharing by multiple RRH)
	#		2. One CPU is reserved for one RRH and one CPU is enough for a RRH.
	#		3. Processing energy is calculated by the processing timeout the RRH traffic load requires for the CPU
	#		4. IF no CPU sharing and if one CPU is enough for a RRH, then there is no blocking in processing
	#		5. Intra-bbu processing model
	# 				When implemmenting inter-bbu model (sending processes to other vBBU if theres high delay/blocking),
	#				refer to the article "Radio Base Stations in the Cloud - 2013"
	# We consider RRHs to not shutdown for energy saving, they're always UP
	#-------------------------------------

	def __init__ (self,CPU_num,GOPS_per_CPU,cell_RRH_ids,exclusive=False,DC=False,initial_CPUid=1):
		# criar um objeto CPU e vinculá-lo a uma BBU
		# é basicamente isto que está sendo feito aqui
		self.env = env
		self.CPU_num=CPU_num
		self.CPUs=range(initial_CPUid,CPU_num+1) 	# BBU's CPUs List
		self.exclusive_CPUs=exclusive 
		
		if exclusive_CPUs: 					# IF CPUs are exclusive for a RRH
			if len(CPUs) == len(cell_RRH_ids):
				CPU_id=initial_CPUid
				self.CPU_RRH=defaultdict(int)
				for cada in cell_RRH_ids:	# RRH to CPU reserve mapping
					self.CPU_RRH[cada]=CPU_id
					CPU_id++
			else
				print "Error: Not enough exclusive CPUs (%d) for RRH amount (%d)" % (len(CPUs),len(cell_RRH_ids))
				return False
		
		self.GOPS_per_CPU= GOPS_per_CPU		# every CPU is equal
		self.GOPS_total= self.GOPS_per_CPU * CPU_num
		self.base_energy=30

	# Every processing function has its necessary GOPS to be done
	# Timeout calculated by GOPS_required / GOPS_BBU
	def timeout_calc(self,GOPS_function,GOPS_per_CPU):
		proc_timeout = float(GOPS_funcion/self.GOPS)
		yield self.env.timeout(proc_timeout)


	def split_1(CPRI_pkt):
		GOPS_SP1= 720
	def split_2(CPRI_pkt):
		GOPS_SP2=
		
	def split_3(CPRI_pkt):
		GOPS_SP3=
	def split_4(CPRI_pkt):
		GOPS_SP4=
	def split_5(CPRI_pkt):
		GOPS_SP5=
	def split_6(CPRI_pkt):
		GOPS_SP6=
	def split_7(CPRI_pkt):
		GOPS_SP7=

	def function_splitting(split_option,CPRI_pkt):
	# Total of 6 functions and 7 possible splits
	
	# Each split reads: e.g, CP1 and on stays at the central cloud
	# split_UP1 reads "UP1 and on stays at the central cloud"
	# split_UP4 reads "Every function is at Edge"
    switcher = {
        1: split_1(CPRI_pkt), # Split C-RAN - CP1
        2: split_2(CPRI_pkt), # Split CP1 - CP2
        3: split_3(CPRI_pkt), # Split CP2 - CP3 
        4: split_4(CPRI_pkt), # split CP3 - UP1
        5: split_5(CPRI_pkt), # split UP1 - UP2
        6: split_6(CPRI_pkt), # split UP2 - UP3
        7: split_7(CPRI_pkt), # Split UP3 - D-RAN
    }
    # Get the function from switcher dictionary
    # If argument is not mapped into switcher, returns False
    func = switcher.get(split_option, lambda: False)
    # Execute the function
    return func()