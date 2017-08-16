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
# >>> nested_dict = lambda: defaultdict(nested_dict)
# >>> nest = nested_dict()

for coding in range(0,29):
	for cpri_option in used_CPRIs:

		#----------- Changing CPRI option, we change:
		# 	PRB 
		#	sampling frequency MHz
		#	TBS
		# 	IP pkt TTI

		print "Coding: %d 	CPRI OPTION: %d" % (coding,cpri_option)
		# print tbs_table[n_RB - PUCCH_RBs][mcs_2_tbs[MCS_UL]]
		TBS_UL = tbs_table[CPRI[cpri_option]['PRB'] - PUCCH_RBs][mcs_2_tbs[MCS_UL]]
		#print "TBS: %.3f" % TBS_UL
		
		IP_TTI_UL= (TBS_UL)/((IPpkt + HdrPDCP + HdrRLC + HdrMAC) *8)
		#print "IP_TTI_UL: %.3f" % IP_TTI_UL
		#--------------------------------------------


		#--------------SPLITS BW & GOPS calcs--------
		# PHY SPLIT IV - SCF
		a1_UL = nIQ * CPRI_coding
		r1_UL= CPRI[cpri_option]['Fs'] * nAnt * n_Sector * a1_UL
		# nIQ= (2*(15+1))= 32 -> facilita aproximacao nas contas
		# (2*IQ) * Fs * 16/15 * CPRIlinecoding * nAnt * nSectors
		# (2 * 15) * 1.92 * 16/15 * (10/8.0) * 2 * 4 = 614.39999
		split1=1
		splits_info[coding][cpri_option][1]['bw'] = r1_UL
		
		gops_1 = int((cpri_option*nAnt*n_Sector*a1_UL)/10)
		splits_info[coding][cpri_option][1]['gops'] = gops_1
		print "Split1 : %.3f Mbps	GOPS:%d |" % (r1_UL, gops_1),
		#return r1_UL

		# PHY SPLIT IIIb - SCF
		a2_UL= nIQ
		r2_UL= CPRI[cpri_option]['Fs'] * nAnt * n_Sector * a2_UL
		splits_info[coding][cpri_option][2]['bw'] = r2_UL
		
		gops_2 = int((cpri_option*nAnt*n_Sector*a2_UL*nIQ)/100)
		splits_info[coding][cpri_option][2]['gops'] = gops_2
		print "Split2 : %.3f Mbps  GOPS:%d |" % (r2_UL,gops_2)
		#return r2_UL
	#	if split==3:
		
		# PHY SPLIT III - SCF
		a3_UL = n_RB_SC * n_Data_Sym * nIQ # <- *1000 / 1000000
		r3_UL = (a3_UL * nAnt * n_Sector * CPRI[cpri_option]['PRB'])/1000
		gops_3 = int(r3_UL/10)
		splits_info[coding][cpri_option][3]['bw'] = r3_UL
		splits_info[coding][cpri_option][3]['gops'] = gops_3
		print "Split3 : %.3f Mbps	GOPS:%d |" % (r3_UL,gops_3),
		#return r3_UL
	#	if split==4:
		# PHY SPLIT II - SCF
		#a4_UL = n_RB_SC * n_Data_Sym * nIQ
		#b4_UL = n_RB_SC * n_Data_Sym * PUCCH_RBs * nIQ * nIQ
		r4_UL = (n_Data_Sym * n_RB_SC * (CPRI[cpri_option]['PRB'] - PUCCH_RBs) * nAnt * nIQ)/1000
		splits_info[coding][cpri_option][4]['bw'] = r4_UL
		gops_4= int(2*gops_2) # Can be better represented.. -> insert every variable of r4_UL in the calculation (insert PUCCH_RBs)
		splits_info[coding][cpri_option][4]['gops'] = gops_4
		print "Split4 : %.3f Mbps   GOPS:%d |" % (r4_UL, gops_4)
		#return r4_UL

	#	if split==5:
		# PHY SPLIT I - SCF
		#a5_UL = (n_RB_SC * n_Data_Sym * QmPUSCH * LayersUL * nSIC * nLLR) / 1000
		#b5_UL = (PUCCH_RBs * n_RB_SC * n_Data_Sym * QmPUSCH * LayersUL * nSIC * nLLR ) / 1000
		#r5_UL = a5_UL * LayersUL * n_RB - b5_UL
		r5_UL = (n_Data_Sym * n_RB_SC * (CPRI[cpri_option]['PRB'] - PUCCH_RBs) * QmPUSCH * LayersUL * nSIC * nLLR) / 1000
		if coding > 3: 
			gops_5= int((gops_4) * (coding**2/(1+coding+(coding*math.sqrt(coding)))))
		else:
			gops_5= int(gops_4)
		splits_info[coding][cpri_option][5]['gops'] = gops_5
		splits_info[coding][cpri_option][5]['bw'] = r5_UL
		print "Split5 : %.3f Mbps 	GOPS:%d | " % (r5_UL, gops_5),
		#return r5_UL

	#	if split==6:
		# SPLIT MAC-PHY - SCF
		a6_UP = (IP_TTI_UL* (IPpkt + HdrPDCP + HdrRLC + HdrMAC) * nTBS_UL_TTI)/ 125
		r6_UL = a6_UP * LayersUL + FAPI_UL
		gops_6 = int(a6_UP*LayersUL)
		splits_info[coding][cpri_option][6]['bw'] = r6_UL
		splits_info[coding][cpri_option][6]['gops'] = gops_6
		print "Split6 : %.3f Mbps	GOPS:%d |" % (r6_UL,gops_6)
		#return r6_UP

	#	if split==7:
		# SPLIT RRC-PDCP - SCF
		a7_UP = (IP_TTI_UL * IPpkt * nTBS_UL_TTI) / 125
		r7_UL = a7_UP * LayersUL
		# GOPS calcula o custo de processar a funcao, mas a funcao virtual 7 nao existe para nos
		#gops_7 = int(a7_UP * LayersUL)
		gops_7 = 0
		splits_info[coding][cpri_option][7]['bw'] = r7_UL
		splits_info[coding][cpri_option][7]['gops'] = gops_7
		# OU apenas fazer TBS/1000.0 ...
		GOPS_total= gops_1+gops_2+gops_3+gops_4+gops_5+gops_6+gops_7
		splits_info[coding][cpri_option]['gops_total']= GOPS_total

		print "Split7 : %.3f Mbps	GOPS:%d 	GOPS TOTAL Split1:%d|\n" % (r7_UL,gops_7,GOPS_total)

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

		#return r7_UP
		#----------------------------------------

#print dict(splits_info)

# TO PRINT DEFAULT DICT \/
#import json
#data_as_dict = json.loads(json.dumps(splits_info, indent=4))
#print(data_as_dict)

# splits_info[MCS][CPRI option][Split]
for cada in range(1,8):
	print splits_info[28][7][cada]

#print({k: dict(v) for k, v in dict(group_ids).items()})


# campos do pacote CPRI
# list[antenaIDs], setor=2*qtd antenas, 
# 

class EdgeCloud(object):
	def __init__ (self,env,n_vBBUs,num_cores):
		self.env=env
		#self.baseline_energy=700	#from rodrigo's paper
		self.AC_energy= num_cores * 100
		self.battery_energy= num_cores *15
		self.base_cores_energy= num_cores * 10
		self.baseline_energy= AC_energy + battery_energy + base_cores_energy

		self.midhaul_receiver = self.env.process(self.EC_receiver())
		self.midhaul_sender =

class CentralCloud(object):
	def __init__(self,env,n_vBBUs,num_cores):
		self.env=env
		self.AC_energy= num_cores * 50
		self.battery_energy= num_cores *10
		self.base_cores_energy= num_cores * 5
		self.baseline_energy= AC_energy + battery_energy + base_cores_energy
		#self.baseline_energy=550	#from rodrigo's paper

class vBBU(object):
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
	def __init__ (self,env,vBBU_id,splitting_table,cell_RRH_ids,GOPS_per_core=8000,\
					exclusive=False,CentralCloud=False,initial_coreid=1):
		# future work to create a core (or CPU class) and put them into BBUs
		self.env= env
		self.vBBU_id = vBBU_id
		self.num_cores=_num_cores
		self.core_resource={} # creating dict of cores resource
		for core_id in range(initial_coreid,num_cores+1): # vBBU's cores from 1 to X
			self.core_resource[core_id]= simpy.Store(env)	

		self.exclusive_cores= exclusive
		if exclusive_cores: 					# IF cores are exclusive for a RRH
			if len(cores) == len(cell_RRH_ids):
				core_id=initial_coreid
				# dict table with RRHs info. Orchestrator alters RRH splits by modifying this table
				# Core Resource of a RRH is also in this dict   
				self.table_rrh_id=defaultdict(lambda : defaultdict(int))
				count=initial_coreid
				for rrh_id in cell_RRH_ids:		# RRH to core reserve mapping
					self.table_rrh_id[rrh_id]['split']=7
					self.table_rrh_id[rrh_id]['cpu']= count
					count+=1
			else:
	   			print "Error: Not enough exclusive cores (%d) for RRH amount (%d)" % (len(cores),len(cell_RRH_ids))
				return False
		
		self.splitting_table= splitting_table 	# defaultdict with bw&gops per split
		self.GOPS_per_core= GOPS_per_core # every core is equal and default=8000GOPS
		self.GOPS_total= self.GOPS_per_core * num_cores

		# edge/central cloud base Cores energy makes sense
		if CentralCloud:
			self.core_dyn_energy=15	# energy of CPU, memory, hdd, mboard, etc of 1 core under 100% usage
		else:
			self.core_dyn_energy=30

	def get_pkt(self):
		"""process to get the packet from the FH port buffer  """
		pass

	# Every processing function has its necessary GOPS to be done
	# Timeout calculated by GOPS_required / GOPS_BBU
	# Energy calculated by Watts * timeout
	def proc_timeout(self,GOPS_function,GOPS_per_core):
		
		yield proc_timeout
		# energy calc
		# energy_calc()
		#yield self.env.timeout(proc_timeout)

#	def proc_energy_calc(self,core_dyn_energy,time_proc):
		#simple way
#		energy = time_proc * ((core_dyn_energy)/1000)
#		return energy

	def calc_gops(self,pkt_MCS,pkt_CPRI_option,pkt_split):
		# CP1   CP2   CP3  |  UP1 	UP2   UP3
		#				 split here = split 4
		# sum GOPS of all VNF before split
		pass

	def split_1(self,pkt):
		# take gops from our big dict
		#codingpkt[]
		#coding=
		pkt_MCS = pkt['MCS']
		pkt_CPRI_option = pkt['CPRI_option']

		pkt_rrh_id = pkt['rrh_id']
		pkt_split = table_rrh_id[pkt_rrh_id]['split'] # get split of pkt from table
		

		#by packets attributes (MCS, CPRI option) and its split, get GOPS and BW from table
		GOPS = self.splitting_table[pkt_MCS][pkt_CPRI_option][pkt_split]['edge_gops']
		bw_split = self.splitting_table[pkt_MCS][pkt_CPRI_option][pkt_split]['bw']
		
		#timeout proc
		proc_tout = float(GOPS/self.GOPS_per_core)
		energy = proc_tout * ((self.core_dyn_energy)/1000) # /1000 == measured in 1ms instead of 1s
		proc_timeout(GOPS_SP1,self.GOPS_per_core)

		# LOG the proc energy usage
		#energy_file.write( "{},{},{},{},{},{},{},{}\n".format("edge_", MAC_TABLE[ONU.oid],"02", time_stamp,counter, ONU.oid,self.env.now,grant_final_time) )
		
		#timeout de processamento sempre < 1ms
		# algo similar a ..env.store.timeout(timeout_sp1) 
	def split_2(pkt):
		GOPS_SP2= 450
		
	def split_3(pkt):
		GOPS_SP3= 200
	def split_4(pkt):
		GOPS_SP4= 500
	def split_5(pkt):
		GOPS_SP5= 720
	def split_6(pkt):
		GOPS_SP6= 150
	def split_7(pkt):
		GOPS_SP7= 100


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

class PacketCPRI(object):
    """ This class represents a network packet """

    def __init__(self, time, CPRI_option, id, src="a", dst="z"):
        self.time = time# creation time
        self.id = id # packet id
        self.src = src #packet source address
        self.dst = dst #packet destination address
  		self.coding = coding
  		self.CPRI_option = CPRI_option

    def __repr__(self):
        return "id: {}, src: {}, time: {}, CPRI_option: {}, coding: {}".\
            format(self.id, self.src, self.time, self.CPRI_option, self.coding)

class PacketGenerator(object):
    """This class represents the packet generation process """
    def __init__(self, env, id,  adist, sdist, fix_pkt_size=None, finish=float("inf")):
        self.id = id # packet id
        self.env = env # Simpy Environment
        self.arrivals_dist = adist #packet arrivals distribution
        self.size_dist = sdist #packet size distribution

        self.fix_pkt_size = fix_pkt_size # Fixed packet size
        self.finish = finish # packet end time
        self.out = None # packet generator output
        self.packets_sent = 0 # packet counter
        self.action = env.process(self.run())  # starts the run() method as a SimPy process

    def run(self):
        """The generator function used in simulations.
        """
        while self.env.now < self.finish:
            # wait for next transmission
            yield self.env.timeout(self.arrivals_dist())
            self.packets_sent += 1

            if self.fix_pkt_size:
                p = PacketCPRI(self.env.now, self.fix_pkt_size, self.packets_sent, src=self.id)
                #Logging
                #pkt_file.write("{}\n".format(self.fix_pkt_size))
            else:
                size = self.size_dist()
                p = PacketCPRI(self.env.now, size, self.packets_sent, src=self.id)
                #Logging
                #pkt_file.write("{}\n".format(size))
            self.out.put(p) # put the packet in RRH port


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
	def __init__(self,env,cell_id,rrh_id,edge_bbu_id,exp,qlimit,fix_pkt_size):
		self.env = env
		self.cell_id = cell_id
		self.id = rrh_id

		arrivals_dist = functools.partial(random.expovariate, exp) #packet arrival distribuition
        size_dist = functools.partial(random.expovariate, 0.1)  # packet size distribuition, mean size 100 bytes
		self.pg = PacketGenerator(self.env, "rrh_"+rrh_id, arrivals_dist, size_dist,fix_pkt_size)

		if qlimit == 0:# checks if the queue has a size limit
            queue_limit = None
        else:
            queue_limit = qlimit

        self.port = RRHPort(self.env, qlimit=queue_limit) #create FH PORT
        self.pg.out = self.port #forward packet generator output to EdgeCloud port
        self.sender = self.env.process(self.RRH_sender(odn))
        #self.receiver = self.env.process(self.ONU_receiver(odn))

    def RRH_sender(self,env):
    	while True:
    		 = yield odn.get_grant(self.oid)


class RRHPort(object):
    def __init__(self, env, qlimit=None):
        self.buffer = simpy.Store(env) #buffer
        self.env = env
        self.out = None # RRH port output
        self.packets_rec = 0 #received pkt counter
        self.packets_tx = 0 #received pkt counter
        #self.packets_drop = 0 #dropped pkt counter
        self.qlimit = qlimit #Buffer queue limit
        self.action = env.process(self.run())  # starts the run() method as a SimPy process
        self.pkt = None #network packet obj

    def set_grant(self,grant): #setting grant byte size and its ending
        self.grant_size = grant['grant_size']
        self.grant_final_time = grant['grant_final_time']

    def update_last_buffer_size(self,requested_buffer): #update the size of the last buffer request
        self.last_buffer_size = requested_buffer

    def get_last_buffer_size(self): #return the size of the last buffer request
        return self.last_buffer_size

    def get_pkt(self):
        """ Process to get the packet from the buffer """

        try:
            pkt = (yield self.buffer.get() )#getting a packet from the buffer
            self.pkt = pkt

        except simpy.Interrupt as i:
            logging.debug("Error while getting a packet from the buffer ({})".format(i))

            pass

        if not self.grant_loop:#put the pkt back to the buffer if the grant time expired

            self.buffer.put(pkt)

    def send(self,ONU_id):
        """ process to send pkts
        """
        self.grant_loop = True #flag if grant time is being used
        start_grant_usage = None #grant timestamp
        end_grant_usage = 0 #grant timestamp

        while self.grant_final_time > self.env.now:

            get_pkt = self.env.process(self.get_pkt())#trying to get a package in the buffer
            grant_timeout = self.env.timeout(self.grant_final_time - self.env.now)
            yield get_pkt | grant_timeout#wait for a package to be sent or the grant timeout

            if (self.grant_final_time <= self.env.now):
                #The grant time has expired
                break
            if self.pkt is not None:
                pkt = self.pkt
                if not start_grant_usage:
                    start_grant_usage = self.env.now #initialized the real grant usage time
                start_pkt_usage = self.env.now ##initialized the pkt usage time

            else:
                #there is no pkt to be sent
                logging.debug("{}: there is no packet to be sent".format(self.env.now))
                break
            self.busy = 1
            self.byte_size -= pkt.size
            if self.byte_size < 0:#Prevent the buffer from being negative
                logging.debug("{}: Negative buffer".format(self.env.now))
                self.byte_size += pkt.size
                self.buffer.put(pkt)
                break

            bits = pkt.size * 8
            sending_time = 	bits/float(1000000000) # buffer transmission time

            #To avoid fragmentation by passing the Grant window
            if env.now + sending_time > self.grant_final_time + self.guard_interval:
                self.byte_size += pkt.size

                self.buffer.put(pkt)
                break

            #write the pkt transmission delay
            delay_file.write( "{},{}\n".format( ONU_id, (self.env.now - pkt.time) ) )
            yield self.env.timeout(sending_time)

            end_pkt_usage = self.env.now
            end_grant_usage += end_pkt_usage - start_pkt_usage

            self.pkt = None

        #ending of the grant
        self.grant_loop = False #flag if grant time is being used
        if start_grant_usage:# if any pkt has been sent
            #send the real grant usage
            self.grant_real_usage.put( [start_grant_usage , start_grant_usage + end_grant_usage] )
        else:
            #logging.debug("buffer_size:{}, grant duration:{}".format(b,grant_timeout))
            self.grant_real_usage.put([])# send a empty list


    def run(self): #run the port as a simpy process
        while True:
            yield self.env.timeout(5)


    def put(self, pkt):
        """receives a packet from the packet genarator and put it on the queue
            if the queue is not full, otherwise drop it.
        """

        self.packets_rec += 1
        tmp = self.byte_size + pkt.size
        if self.qlimit is None: #checks if the queue size is unlimited
            self.byte_size = tmp
            return self.buffer.put(pkt)
        if tmp >= self.qlimit: # checks if the queue is full
            self.packets_drop += 1
            #return
        else:
            self.byte_size = tmp
            self.buffer.put(pkt)



class MidhaulPort(object):
    def __init__(self, env, qlimit=None):
        self.buffer = simpy.Store(env)#buffer
        self.env = env
        self.out = None # ONU port output
        self.packets_rec = 0 #received pkt counter
        self.packets_drop = 0 #dropped pkt counter
        self.qlimit = qlimit #Buffer queue limit
        self.byte_size = 0  # Current size of the buffer in bytes
        self.last_buffer_size = 0 # size of the last buffer request
        self.busy = 0  # Used to track if a packet is currently being sent
        self.action = env.process(self.run())  # starts the run() method as a SimPy process
        self.pkt = None #network packet obj
        self.grant_loop = False #flag if grant time is being used

    def set_grant(self,grant): #setting grant byte size and its ending
        self.grant_size = grant['grant_size']
        self.grant_final_time = grant['grant_final_time']

    def update_last_buffer_size(self,requested_buffer): #update the size of the last buffer request
        self.last_buffer_size = requested_buffer

    def get_last_buffer_size(self): #return the size of the last buffer request
        return self.last_buffer_size

    def get_pkt(self):
        """process to get the packet from the buffer   """

        try:
            pkt = (yield self.buffer.get() )#getting a packet from the buffer
            self.pkt = pkt

        except simpy.Interrupt as i:
            logging.debug("Error while getting a packet from the buffer ({})".format(i))

            pass

        if not self.grant_loop:#put the pkt back to the buffer if the grant time expired

            self.buffer.put(pkt)



    def send(self,ONU_id):
        """ process to send pkts
        """
        self.grant_loop = True #flag if grant time is being used
        start_grant_usage = None #grant timestamp
        end_grant_usage = 0 #grant timestamp

        while self.grant_final_time > self.env.now:

            get_pkt = self.env.process(self.get_pkt())#trying to get a package in the buffer
            grant_timeout = self.env.timeout(self.grant_final_time - self.env.now)
            yield get_pkt | grant_timeout#wait for a package to be sent or the grant timeout

            if (self.grant_final_time <= self.env.now):
                #The grant time has expired
                break
            if self.pkt is not None:
                pkt = self.pkt
                if not start_grant_usage:
                    start_grant_usage = self.env.now #initialized the real grant usage time
                start_pkt_usage = self.env.now ##initialized the pkt usage time

            else:
                #there is no pkt to be sent
                logging.debug("{}: there is no packet to be sent".format(self.env.now))
                break
            self.busy = 1
            self.byte_size -= pkt.size
            if self.byte_size < 0:#Prevent the buffer from being negative
                logging.debug("{}: Negative buffer".format(self.env.now))
                self.byte_size += pkt.size
                self.buffer.put(pkt)
                break

            bits = pkt.size * 8
            sending_time = 	bits/float(1000000000) # buffer transmission time

            #To avoid fragmentation by passing the Grant window
            if env.now + sending_time > self.grant_final_time + self.guard_interval:
                self.byte_size += pkt.size

                self.buffer.put(pkt)
                break

            #write the pkt transmission delay
            delay_file.write( "{},{}\n".format( ONU_id, (self.env.now - pkt.time) ) )
            yield self.env.timeout(sending_time)

            end_pkt_usage = self.env.now
            end_grant_usage += end_pkt_usage - start_pkt_usage

            self.pkt = None

        #ending of the grant
        self.grant_loop = False #flag if grant time is being used
        if start_grant_usage:# if any pkt has been sent
            #send the real grant usage
            self.grant_real_usage.put( [start_grant_usage , start_grant_usage + end_grant_usage] )
        else:
            #logging.debug("buffer_size:{}, grant duration:{}".format(b,grant_timeout))
            self.grant_real_usage.put([])# send a empty list



    def run(self): #run the port as a simpy process
        while True:
            yield self.env.timeout(5)


    def put(self, pkt):
        """receives a packet from the packet genarator and put it on the queue
            if the queue is not full, otherwise drop it.
        """

        self.packets_rec += 1
        tmp = self.byte_size + pkt.size
        if self.qlimit is None: #checks if the queue size is unlimited
            self.byte_size = tmp
            return self.buffer.put(pkt)
        if tmp >= self.qlimit: # chcks if the queue is full
            self.packets_drop += 1
            #return
        else:
            self.byte_size = tmp
            self.buffer.put(pkt)