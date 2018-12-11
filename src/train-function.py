# Function to represent the state (of the pegs)
from copy import deepcopy

# STATE = [Qmax, Qmin]
def printState(state):
    print(state)


# Function which returns valid moves from a state, as a list.
def validMoves(state, bufferlen):
    # Initialize empty list to append valid moves to.
    # Moves: [+_, -_, _+, _-]
    valid = []
    # Qmax: 1 to bufferlen
    # Qmin: 1 to (Qmax-1)
    # state[0] - Qmax
    # state[1] - Qmin
    if state[0] == 1:
        valid.append("+_")
    elif state[0] == bufferlen-1:
        valid.append("-_")
    else:
        valid.append("+_")
        valid.append("-_")

    if state[1] == 1 and state[0] > 2:
        valid.append("_+")

    elif state[1] == bufferlen-1:
        valid.append("_-")
    
    elif (state[1] = state[0] - 1) or (state[1] == state[0]):
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
    if state == [[], [], [1,2,3]]:
        return True
    else:
        return False


# Function which converts lists to tuples.
def stateMoveTuple(state, move):
    stateTuple = tuple(tuple(x) for x in state)
    moveTuple = tuple(move)
    stateMoveTuple = (stateTuple, moveTuple)
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
        s = [[1,2,3], [], []]
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
    s = [[1,2,3], [], []]
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