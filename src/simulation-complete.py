# Simulation 1
# ------------

from copy import deepcopy

from random import expovariate
#import simpy
#from SimComponents import PacketGenerator, PacketSink, SwitchPort, PortMonitor
#import random
import numpy as np
import matplotlib.pyplot as plt

import simpy
import random
import copy
from simpy.core import BoundClass
from simpy.resources import base
from heapq import heappush, heappop


# RED Parameters

Qmax = 1900
Qmin = 400
#p = 0.5
qavg = 0

avgqueue = []
# Other
bufSize = 2000
linerate = 100.0
# Weight calculation
# L + 1 + ((1 - w)^(L+1) - 1)/w < Qmin
#w = 0.05

# Congestion Windows
cwndsrc1 = 1
cwndsrc2 = 1
thresh1 = 30
thresh2 = 30



class Packet(object):
    """ A very simple class that represents a packet.
        This packet will run through a queue at a switch output port.
        We use a float to represent the size of the packet in bytes so that
        we can compare to ideal M/M/1 queues.

        Parameters
        ----------
        time : float
            the time the packet arrives at the output queue.
        size : float
            the size of the packet in bytes
        id : int
            an identifier for the packet
        src, dst : int
            identifiers for source and destination
        flow_id : int
            small integer that can be used to identify a flow
    """
    def __init__(self, time, size, id, src="a", dst="z", flow_id=0):
        self.time = time
        self.size = size
        self.id = id
        self.src = src
        self.dst = dst
        self.flow_id = flow_id

    def __repr__(self):
        return "id: {}, src: {}, time: {}, size: {}".\
            format(self.id, self.src, self.time, self.size)


class PacketGenerator(object):
    """ Generates packets with given inter-arrival time distribution.
        Set the "out" member variable to the entity to receive the packet.

        Parameters
        ----------
        env : simpy.Environment
            the simulation environment
        adist : function
            a no parameter function that returns the successive inter-arrival times of the packets
        sdist : function
            a no parameter function that returns the successive sizes of the packets
        initial_delay : number
            Starts generation after an initial delay. Default = 0
        finish : number
            Stops generation at the finish time. Default is infinite


    """
    def __init__(self, env, id,  adist, sdist, initial_delay=0, finish=float("inf"), flow_id=0):
        self.id = id
        self.env = env
        self.adist = adist
        self.sdist = sdist
        self.initial_delay = initial_delay
        self.finish = finish
        self.out = None
        self.packets_sent = 0
        self.action = env.process(self.run())  # starts the run() method as a SimPy process
        self.flow_id = flow_id

    def run(self):
        """The generator function used in simulations.
        """
        yield self.env.timeout(self.initial_delay)
        while self.env.now < self.finish:
            # wait for next transmission
            yield self.env.timeout(self.adist())
            self.packets_sent += 1
            p = Packet(self.env.now, self.sdist(), self.packets_sent, src=self.id, flow_id=self.flow_id)
            self.out.put(p)


class PacketSink(object):
    """ Receives packets and collects delay information into the
        waits list. You can then use this list to look at delay statistics.

        Parameters
        ----------
        env : simpy.Environment
            the simulation environment
        debug : boolean
            if true then the contents of each packet will be printed as it is received.
        rec_arrivals : boolean
            if true then arrivals will be recorded
        absolute_arrivals : boolean
            if true absolute arrival times will be recorded, otherwise the time between consecutive arrivals
            is recorded.
        rec_waits : boolean
            if true waiting time experienced by each packet is recorded
        selector: a function that takes a packet and returns a boolean
            used for selective statistics. Default none.

    """
    def __init__(self, env, rec_arrivals=False, absolute_arrivals=False, rec_waits=True, debug=False, selector=None):
        self.store = simpy.Store(env)
        self.env = env
        self.rec_waits = rec_waits
        self.rec_arrivals = rec_arrivals
        self.absolute_arrivals = absolute_arrivals
        self.waits = []
        self.arrivals = []
        self.debug = debug
        self.packets_rec = 0
        self.bytes_rec = 0
        self.selector = selector
        self.last_arrival = 0.0

    def put(self, pkt):
        if not self.selector or self.selector(pkt):
            now = self.env.now
            if self.rec_waits:
                self.waits.append(self.env.now - pkt.time)
            if self.rec_arrivals:
                if self.absolute_arrivals:
                    self.arrivals.append(now)
                else:
                    self.arrivals.append(now - self.last_arrival)
                self.last_arrival = now
            self.packets_rec += 1
            self.bytes_rec += pkt.size
            if self.debug:
                print(pkt)


class SwitchPort(object):

    
    """ Models a switch output port with a given rate and buffer size limit in bytes.
        Set the "out" member variable to the entity to receive the packet.

        Parameters
        ----------
        env : simpy.Environment
            the simulation environment
        rate : float
            the bit rate of the port
        qlimit : integer (or None)
            a buffer size limit in bytes or packets for the queue (including items
            in service).
        limit_bytes : If true, the queue limit will be based on bytes if false the
            queue limit will be based on packets.

    """
    def __init__(self, env, rate, qlimit=None, limit_bytes=True, debug=False):
        self.store = simpy.Store(env)
        self.rate = rate
        self.env = env
        self.out = None
        self.packets_rec = 0
        self.packets_drop = 0
        self.qlimit = qlimit
        self.limit_bytes = limit_bytes
        self.byte_size = 0  # Current size of the queue in bytes
        self.debug = debug
        self.busy = 0  # Used to track if a packet is currently being sent
        self.action = env.process(self.run())  # starts the run() method as a SimPy process
        self.droppacket = 0
        self.packet = None

    def run(self):
        while True:
            msg = (yield self.store.get())
            if(self.droppacket == 0):
                self.busy = 1
            self.byte_size -= msg.size
            yield self.env.timeout(msg.size*8.0/self.rate)
            # ADDITION - VIGNESH
            self.packet = msg

            self.out.put(msg)
            self.busy = 0
            #self.droppacket = 0
            if self.debug:
                print(msg)

    def put(self, pkt):
        global cwndsrc1
        global cwndsrc2
        global thresh1
        global thresh2	
        self.packets_rec += 1
        tmp_byte_count = self.byte_size + pkt.size

        if self.qlimit is None:
            self.byte_size = tmp_byte_count
            return self.store.put(pkt)
        if self.limit_bytes and tmp_byte_count >= self.qlimit:
            self.packets_drop += 1
            return
        
        elif not self.limit_bytes and len(self.store.items) >= self.qlimit-1:
            self.packets_drop += 1
        		
        	# CHANGED - VIGNESH

            # Find out whose packet has been dropped
            source = self.packet.src
			#print("Packet dropped: " + source)
			# Backoff CWND 	and thresh for this packet's source 
            if source == "S1":
                cwndsrc1 = 1
                thresh1 = thresh1/2
            elif source == "S2":
                cwndsrc2 = 1
                thresh2 = thresh2/2

        elif self.droppacket == 1:
            self.packets_drop += 1
            #self.droppacket = 0

            

        else:
            self.byte_size = tmp_byte_count
            return self.store.put(pkt)


class PortMonitor(object):
    """ A monitor for an SwitchPort. Looks at the number of items in the SwitchPort
        in service + in the queue and records that info in the sizes[] list. The
        monitor looks at the port at time intervals given by the distribution dist.

        Parameters
        ----------
        env : simpy.Environment
            the simulation environment
        port : SwitchPort
            the switch port object to be monitored.
        dist : function
            a no parameter function that returns the successive inter-arrival times of the
            packets
    """
    def __init__(self, env, port, dist, count_bytes=False):
        self.port = port
        self.env = env
        self.dist = dist
        self.count_bytes = count_bytes
        self.sizes = []
        self.action = env.process(self.run())

    def run(self):
        while True:
            yield self.env.timeout(self.dist())
            if self.count_bytes:
                total = self.port.byte_size
            else:
                total = len(self.port.store.items) + self.port.busy
            self.port.droppacket = 0
            self.sizes.append(total)



def constArrival():  # Constant arrival distribution for generator 1
	global cwndsrc1
	#print("TEST CWND 1: " + str(cwndsrc1))
	if cwndsrc1 < thresh1:
		cwndsrc1 = cwndsrc1 * 2
		if(cwndsrc1 > thresh1):
			cwndsrc1 = thresh1

	interval = 1/cwndsrc1
	return interval

def constArrival2():
	global cwndsrc2
	#print("TEST CWND 2: " + str(cwndsrc2))
	if cwndsrc2 < thresh2:
		cwndsrc2 = cwndsrc2 * 2
		if(cwndsrc2 > thresh2):
			cwndsrc2 = thresh2

	interval = 1/cwndsrc2
	return interval


def distSize():
    return 1

def distSize2():
	global pm
	global switch_port
	global qavg
	global avgqueue
	global bufSize
	global cwndsrc1
	global cwndsrc2

	global thresh1
	global thresh2
	#print("BUFFER SIZE")
	#print(pm.sizes)
	if(len(pm.sizes) != 0):
		# If greater than max threshold, always drop packet.
		# Queue length calculation method - EWMA.
		qavg = (1 - w)*qavg + w*pm.sizes[-1]
		avgqueue.append(qavg)
		#qavg = (1 - w)*qavg + w*len(switch_port.store.items)
		if(qavg > Qmax):
			# Then drop the packet
			switch_port.droppacket = 1
			
			# Find out whose packet has been dropped
			source = switch_port.packet.src
			# Backoff CWND for this packet's source 
			if source == "S1":
				thresh1 = cwndsrc1/2
				cwndsrc1 = 1
			elif source == "S2":
				thresh2 = cwndsrc2/2
				cwndsrc2 = 1
			#print("Packet dropped: " + source)
		# If greater than min threshold, drop with probability p = queuesize/buffersize
		elif(qavg > Qmin):
			p = qavg/(bufSize*10)
			if random.randint(0,100) < (p*100):
				switch_port.droppacket = 1
				# Find out whose packet has been dropped
				source = switch_port.packet.src
				#print("Packet dropped: " + source)
				# Backoff CWND 	and thresh for this packet's source 
				if source == "S1":
					cwndsrc1 = 1
					thresh1 = thresh1/2
				elif source == "S2":
					cwndsrc2 = 1
					thresh2 = thresh2/2
	return 0.25



env = simpy.Environment()  # Create the SimPy environment
# Create the packet generators and sink
ps = PacketSink(env, debug=False)  # debugging enable for simple output
pg = PacketGenerator(env, "S1", constArrival, distSize)
pg2 = PacketGenerator(env, "S2", constArrival2, distSize)

# Buffer output
switch_port = SwitchPort(env, rate=linerate, qlimit= bufSize, limit_bytes = False)

# Monitor queue size
pm = PortMonitor(env, switch_port, distSize2)

# Wire packet generators and sink together
pg.out = switch_port
pg2.out = switch_port
switch_port.out = ps

# ========================================================================================================================================
# ========================================================================================================================================

# MACHINE LEARNING PART

# ========================================================================================================================================
# ========================================================================================================================================

# STATE = [Qmax, Qmin]
def printState(state):
    print(state)


# Function which returns valid moves from a state, as a list.
def validMoves(state):
    # Initialize empty list to append valid moves to.
    # Moves: [+_, -_, _+, _-]
    valid = []
    # Qmax: 1 to bufferlen
    # Qmin: 1 to (Qmax-1)
    # state[0] - Qmax
    # state[1] - Qmin
    if state[0] == 1:
        valid.append("+_")
    elif state[0] == 1999:
        valid.append("-_")
    else:
        valid.append("+_")
        valid.append("-_")

    if state[1] == 1 and state[0] > 2:
        valid.append("_+")

    elif state[1] == 1999:
        valid.append("_-")
    
    elif (state[1] == state[0] - 1) or (state[1] == state[0]):
        valid.append("_-")

    else:
        valid.append("_+")
        valid.append("_-")


    return valid

# Function to apply a move to a state and return the new state.
def makeMove(state, move):
    if move != None:
        state2 = deepcopy(state)
        if(move == "+_"):
            state2[0] += 1
        elif(move == "-_"):
            state2[0] -= 1
        elif(move == "_+"):
            state2[1] += 1
        else:
            state2[1] -= 1

        return state2

# Function to check if the goal state has been reached.
def isGoalState(state):

    global Qmax
    global Qmin

    maxQ = state[0]
    minQ = state[1]

    Qmax = maxQ
    Qmin = minQ

    # Re-declare global variables here
    global qavg
    global avgqueue
    global bufSize
    global linerate
    global w
    global cwndsrc1
    global cwndsrc2
    global thresh1
    global thresh2
    global env
    global pg
    global ps
    global pg2
    global switch_port
    global pm

    #global env
    qavg = 0
    avgqueue = []
    bufSize = 2000
    linerate = 100.0
    w = 0.3
    cwndsrc1 = 1
    cwndsrc2 = 1
    thresh1 = 30
    thresh2 = 30

    env.run(until=40)

    #env = simpy.Environment()
    # Calculate performance
    ndrops = switch_port.packets_drop
    # Throughput - number of packets that made it through
    thput = (pg.packets_sent + pg2.packets_sent) - switch_port.packets_drop #ps.packets_rec
    #variance = np.var(pm.sizes)
    stddev = np.std(avgqueue)
    
    # Redeclare

    env2 = simpy.Environment()  # Create the SimPy environment
    env = env2
    # Create the packet generators and sink
    ps = PacketSink(env, debug=False)  # debugging enable for simple output
    pg = PacketGenerator(env, "S1", constArrival, distSize)
    pg2 = PacketGenerator(env, "S2", constArrival2, distSize)

    # Buffer output
    switch_port = SwitchPort(env, rate=linerate, qlimit= bufSize, limit_bytes = False)

    # Monitor queue size
    pm = PortMonitor(env, switch_port, distSize2)

    # Wire packet generators and sink together
    pg.out = switch_port
    pg2.out = switch_port
    switch_port.out = ps

    if ((stddev < 350) and (ndrops < 35) and ((cwndsrc1 + cwndsrc2) > 2)):
        return True
    else:
        return False


# Function which converts lists to tuples.
def stateMoveTuple(state, move):
    stateTuple = tuple(state)
    #moveTuple = tuple(move)
    stateMoveTuple = (stateTuple, move)
    return stateMoveTuple

import random
# Epsilon Greedy
def epsilonGreedy(epsilon, Q, state):
    # Get all the valid moves from state.
    valid = validMoves(state)
    qlist = []
    # If choosing random action.
    if np.random.uniform() < epsilon:
        # Random Move
        return random.choice(valid)
    # Greedy approach.
    else:
        # Greedy Move - get action for which the Q value is maximum.
        # Get Q values for all (state, move) pairs and store in a list.
        for move in valid:
            smt = stateMoveTuple(state, move)
            Qval = Q.get(smt, -1) # Assign -1 if not in tuple (default Q value)
            qlist.append((Qval, move))
        # Choose move with maximum value of Q.
        maxMove = max(qlist, key=lambda x:x[0])
        return maxMove[1]

# Function for training Q for Towers of Hanoi puzzle.

import numpy as np
def trainQ(nRepetitions, learningRate, epsilonDecayFactor, validMovesF, makeMoveF):
    # Initialize epsilon to 1, since we'll be decayign from 1.
    epsilon = 1.0
    # Initialize empty dictionary for the Q table. The Q table will be a dictionary with the as a tuple of the form:
    # (state, move). 'state' will be a tuple of tuples representing the state of the puzzle. For eg. ((1,2,3),(),())
    # would imply that all three disks are in the first peg. 'move' will be tuple with two elements, the source peg, and
    # the destination peg of the form (source, destination).
    Q = {}
    # List to keep track of steps taken to reach goal for each game played.
    stepsToGoal = []
    # Itearate nRepetitions times, to play as many games.
    for nGames in range(nRepetitions):
        # Decay epsilon to slowly change behaviour such that it stops selecting random moves and goes for a more greedy
        # approach. 
        epsilon *= epsilonDecayFactor
        # Initialize the start state (check if 3-disk puzzle or 4-disk)
        s = [1100,1000]
        done = False
        step = 0
        # Play a game till solution occurs.
        while not done:        
            step += 1
            # Choose a move using epsilonGreedy function.
            move = epsilonGreedy(epsilon, Q, s)
            # Apply the move on a copy of state.
            sNew = deepcopy(s)
            sNew = makeMoveF(sNew, move)
            # If this (state, move) is not yet updated in the Q table, the default value is -1.
            if stateMoveTuple(s, move) not in Q:
                Q[stateMoveTuple(s, move)] = -1 
            # If the goal state is reached, then update Q(s, move) = -1 and break game (inner loop).
            if isGoalState(sNew):
                Q[stateMoveTuple(s, move)] = -1
                print(sNew)
                done = True
            else:
                # If the current step isn't the first step, then update Q value. 
                if step > 1:
                    Q[stateMoveTuple(sOld,moveOld)] += learningRate * (-1 + Q[stateMoveTuple(s,move)] - Q[stateMoveTuple(sOld,moveOld)])
                # Update the current state and move to these variables so that we can use them in the next iteration.
                sOld, moveOld = s, move 
                # Update state to the new state, which we have obtained by performing the move.
                s = sNew
        # Append the number of steps taken to stepsToGoal list.
        stepsToGoal.append(step)
    return Q, stepsToGoal

# Function testQ - used to find the optimal path.
def testQ(Q, maxSteps, validMovesF, makeMoveF):
    path = []
    # Initialize the start state with all disks on peg 1 (if 4-disk puzzle, then 4).
    s = [1100,1000]
    # Append this start state to path list.
    path.append(s)
    steps = 0
    # Maintain a variable 'flag' to check if goal has been reached.
    flag = 0
    # Loop till maximu number of steps allowed.
    while steps < maxSteps:        
        # Choose greedy move (epsilonGreedy will always select the greedy move here since epsilon is specified as -1).
        move = epsilonGreedy(-1, Q, s)
        # Apply the move on the state.
        s = makeMoveF(s, move)
        # Append to path list.
        path.append(s)
        # If the goal state is reached, break out.
        if isGoalState(s):
            flag = 1
            break
        steps += 1
    # If no goal found within steps specified.
    if flag == 0:
        return "No goal found."
    return path

#print(stateMoveTuple([23,56], "_+"))
Q = trainQ(1000, 0.5, 1, validMoves, makeMove)
