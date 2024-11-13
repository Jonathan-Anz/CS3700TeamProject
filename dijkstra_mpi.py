#!/usr/bin/python3.4

import sys
import random
import numpy
from mpi4py import MPI

# Function to randomly generate an adjacency matrix
def GenerateAdjacencyMatrix(numVertices):

    # Initialize the graph with all 0s
    g = [[0 for i in range(numVertices)] for j in range(numVertices)]

    # Loop through half of the matrix (only half because it's not a directed graph)
    for i in range(numVertices):

        # Bool to check if no edges were generated for this vertex
        connectedToGraph = False

        for j in range(i + 1, numVertices):

            weight = float("inf")

            # Decide if there should be an edge (1 / 2 chance)
            # If there's no edge, the weight will stay as infinity
            if random.randint(0, 1) == 0:

                # Assign random weight to edge and mark the vertex as connected
                connectedToGraph = True
                weight = random.randint(1, 20)

            g[i][j] = weight
            g[j][i] = weight

        # If no edges were generated for this vertex (it is not connected to the graph)
        if not connectedToGraph:
            
            # Find a random vertex that is not itself and assign an edge
            otherVertex = (i + random.randint(1, numVertices - 1)) % numVertices
            g[i][otherVertex] = random.randint(1, 20)
            g[otherVertex][i] = g[i][otherVertex]

    return g

# Function to get the local closest unvisited vertex to the source through the already visited vertices 
# (also returns distance for mpi reduce)
def GetClosestVertexInSubgraph(unvisited, startIndex, endIndex, distances):

    minDist = float("inf")
    closestVertex = -1

    # Loop through the vertices found in the subgraph
    for v in range(startIndex, endIndex):

        # If the vertex is unvisited and it has the shortest distance found so far
        if v in unvisited and distances[v] <= minDist:

            # Save it as the new closest vertex
            minDist = distances[v]
            closestVertex = v

    return closestVertex, minDist

# Function to update the distance value of a vertex's unvisited neighbors within each subgraph
def RelaxNeighborsInSubgraph(graph, unvisited, startIndex, endIndex, distances, parents, vertex):

    # Loop through the vertices found in the subgraph
    for v in range(startIndex, endIndex):

        # If the vertex is unvisited
        if v in unvisited:

            # If the vertex is a neighbor, 
            # and it's total distance is less than the currently stored distance
            edgeWeight = graph[vertex][v]
            if edgeWeight != 0 and distances[vertex] + edgeWeight < distances[v]:

                # Update the total distance and parent for that neighbor
                distances[v] = distances[vertex] + edgeWeight
                parents[v] = vertex

# Print out all the distances and parents (for debugging only)
def PrintAllPaths(distances, parents):

    print("Vertex \t Distance from source \t Parent")
    for i in range(len(distances)):
        print(i, "\t", int(distances[i]), "\t\t\t", parents[i])


# START OF MAIN()

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
numVertices = 9
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

# Start vertex (must be changed manually)
source = 0

# Set of unvisited vertices
unvisited = set()
[unvisited.add(i) for i in range(0, numVertices)]

# Distances array
distances = [float("inf") for i in range(numVertices)]
distances[source] = 0 # Distance of start vertex is 0

# Parent array
parents = [0 for i in range(numVertices)]

# MPI setup
comm = MPI.COMM_WORLD
myID = comm.Get_rank()
size = comm.Get_size()

# Generate and print out matrix
if myID == 0:
    # Create random graph
    graph = GenerateAdjacencyMatrix(numVertices)

    print("\nAdjacency matrix:")
    print(numpy.matrix(graph))
else:
    graph = []

# Give every node a copy of the adjacency matrix
graph = comm.bcast(graph, root = 0)

# Timer
if myID == 0:
    startTime = MPI.Wtime()

# Split graph into subgraphs for each processor (distribute remainder if there is any)
verticesPerProcessor = numVertices // size
remainder = numVertices % size

startIndex = 0
endIndex = 0

if myID < remainder:
    startIndex = myID * (verticesPerProcessor + 1)
    endIndex = startIndex + (verticesPerProcessor + 1)
else:
    startIndex = (myID * verticesPerProcessor) + remainder
    endIndex = startIndex + verticesPerProcessor

# Print out the vertex range each processor is working on (for debugging only)
#print("Processor %d is working on vertices %d to %d. Total count: %d" %(myID, startIndex, endIndex - 1, endIndex - startIndex))

# Synch up all the processors before starting
comm.Barrier()


# START DIJKSTRA

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

    # Relax the neighbors in each subgraph
    RelaxNeighborsInSubgraph(graph, unvisited, startIndex, endIndex, distances, parents, globalClosest)

    # Update the new distances and parents arrays for all processors
    subDistances = distances[startIndex : endIndex]
    subParents = parents[startIndex : endIndex]

    distances = comm.allgather(subDistances)
    parents = comm.allgather(subParents)

    # Allgather returns an array of arrays, so concatenate them into one big array
    distances = numpy.concatenate(distances)
    parents = numpy.concatenate(parents)

# END DIJKSTRA

# Print output at the end
if myID == 0:

    # Timer
    endTime = MPI.Wtime()
    elapsedTime = endTime - startTime

    print("\nVertices:", numVertices)
    print("Total processors:", size)

    # Random target, make sure it's not the start vertex
    target = random.randint(1, numVertices - 1)
    print("\nStart vertex:", source)
    print("End vertex:", target)
    print("Shortest path cost: ", int(distances[target]))
    #PrintAllPaths(distances, parents) (for debugging only)

    # Print time
    print("\nExecution time: %.8f seconds\n" %(elapsedTime))

# END OF MAIN()