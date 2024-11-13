#!/usr/bin/python3.4

import sys
import random
import numpy
from mpi4py import MPI

# Function to randomly generate an adjacency matrix
def GenerateAdjacencyMatrix(numVertices):
    g = [[0 for i in range(numVertices)] for j in range(numVertices)]

    # Loop through half of the matrix (only half because it's not a directed graph)
    for i in range(numVertices):

        connectedToGraph = False

        for j in range(i + 1, numVertices):

            weight = float("inf")

            # Decide if there should be an edge (1 / 2 chance)
            if random.randint(0, 1) == 0:

                # Assign random weight to edge
                connectedToGraph = True
                weight = random.randint(1, 20)

            g[i][j] = weight
            g[j][i] = weight

        if not connectedToGraph:
            
            # There are no edges to this vertex, so add one
            otherVertex = (i + random.randint(1, numVertices - 1)) % numVertices
            g[i][otherVertex] = random.randint(1, 20)
            g[otherVertex][i] = g[i][otherVertex]

    return g

# Function to get the local closest vertex to the source through the cluster (also returns distance for mpi reduce)
def GetClosestVertexInSubgraph(unvisited, startIndex, endIndex, distances):
    minDist = float("inf")
    closestVertex = -1

    # For all the vertices in the subgraph
    for v in range(startIndex, endIndex):
        # And is not visited
        if v in unvisited and distances[v] <= minDist:
            minDist = distances[v]
            closestVertex = v

    return closestVertex, minDist

# Function to update the total distance of a vertex's unvisited neighbors
def RelaxNeighbors(graph, unvisited, distances, parents, vertex):
    for v in unvisited:
            edgeWeight = graph[vertex][v]
            if edgeWeight != 0 and distances[vertex] + edgeWeight < distances[v]:
                distances[v] = distances[vertex] + edgeWeight
                parents[v] = vertex

# Print out all the distances and parents (for debugging, not used in final code)
def PrintAllPaths(distances, parents):
    print("Vertex \t Distance from source \t Parent")
    for i in range(len(distances)):
        print(i, "\t", distances[i], "\t\t\t", parents[i])

# Start of main

# Input validation
if len(sys.argv) != 2:
    print("Invalid input: Wrong number of arguments")
    print("Usage: dijkstra_mpi.py [number of vertices]")
    sys.exit()

numVertices = int(sys.argv[1])

if numVertices < 2:
    numVertices = 2

# Debug graph
'''
graph = [   [0, 4, 0, 0, 0, 0, 0, 8, 0],
            [4, 0, 8, 0, 0, 0, 0, 11, 0],
            [0, 8, 0, 7, 0, 4, 0, 0, 2],

            [0, 0, 7, 0, 9, 14, 0, 0, 0],
            [0, 0, 0, 9, 0, 10, 0, 0, 0],
            [0, 0, 4, 14, 10, 0, 2, 0, 0],

            [0, 0, 0, 0, 0, 2, 0, 1, 6],
            [8, 11, 0, 0, 0, 0, 1, 0, 7],
            [0, 0, 2, 0, 0, 0, 6, 7, 0] ]
'''

# Create random graph
graph = GenerateAdjacencyMatrix(numVertices)

source = 0

# Set of unvisited vertices
unvisited = set()
[unvisited.add(i) for i in range(0, numVertices)]

# Distances array
distances = [float("inf") for i in range(numVertices)]
distances[source] = 0

# Parent array
parents = [0 for i in range(numVertices)]

comm = MPI.COMM_WORLD
myID = comm.Get_rank()
size = comm.Get_size()

# Print out matrix
if myID == 0:
    print("\nAdjacency matrix:")
    print(numpy.matrix(graph))

# Timer
if myID == 0:
    startTime = MPI.Wtime()

# Split graph into subgraphs for each processing node
verticesPerProcessor = numVertices // size # 1
remainingVertices = numVertices % size # 3

startIndex = 0
endIndex = 0

if myID < remainingVertices:
    startIndex = myID * (verticesPerProcessor + 1)
    endIndex = startIndex + (verticesPerProcessor + 1)
else:
    startIndex = (myID * verticesPerProcessor) + remainingVertices
    endIndex = startIndex + verticesPerProcessor

#print("Processor %d is working on vertices %d to %d. Total count: %d" %(myID, startIndex, endIndex - 1, endIndex - startIndex))

comm.Barrier()

# Start Dijkstra

# While there are unvisited vertices...
while len(unvisited) != 0:

    # Get the local closest unvisited vertex and distance within each subgraph
    closestVertex, minDist = GetClosestVertexInSubgraph(unvisited, startIndex, endIndex, distances)
    data = (minDist, closestVertex) # Put the distance first because mpi reduce needs to compare that
    #print("Processor", myID, "found vertex:", closestVertex, "with distance:", minDist, flush = True)

    # Get the global closest vertex and distance (distance is not needed after this)
    closestDist, globalClosest = comm.allreduce(data, op = MPI.MINLOC)

    # Remove the global closest from unvisited
    unvisited.remove(globalClosest)

    # Relax the neighbors of the global closest
    RelaxNeighbors(graph, unvisited, distances, parents, globalClosest)

# End Dijkstra

# Print output at the end
if myID == 0:
    # Timer
    endTime = MPI.Wtime()
    elapsedTime = endTime - startTime

    print("\nVertices:", numVertices)
    print("Total processors:", size)

    # Print outputs
    target = random.randint(1, numVertices) # Random target
    print("\nStart vertex:", source)
    print("End vertex:", target)
    print("Shortest path cost: ", distances[target])
    #PrintAllPaths(distances, parents)

    # Print time
    print("\nExecution time: %.8f seconds\n" %(elapsedTime))
