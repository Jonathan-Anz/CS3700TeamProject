#include <iostream>
#include <limits.h>
#include <chrono>

using namespace std;

#define V 9

// Utility functions for initializing graph
void AddEdge(int graph[V][V], int v1, int v2, int cost)
{
    graph[v1][v2] = cost;
    graph[v2][v1] = cost;
}
void InitializeGraph(int graph[V][V])
{
    // Initialize all values to 0
    for (int i = 0; i < V; i++)
    {
        for (int j = 0; j < V; j++)
        {
            graph[i][j] = 0;
        }
    }

    AddEdge(graph, 0, 1, 4);
    AddEdge(graph, 0, 7, 8);
    AddEdge(graph, 1, 2, 8);
    AddEdge(graph, 1, 7, 11);
    AddEdge(graph, 2, 3, 7);
    AddEdge(graph, 2, 5, 4);
    AddEdge(graph, 2, 8, 2);
    AddEdge(graph, 3, 4, 9);
    AddEdge(graph, 3, 5, 14);
    AddEdge(graph, 4, 5, 10);
    AddEdge(graph, 5, 6, 2);
    AddEdge(graph, 6, 7, 1);
    AddEdge(graph, 6, 8, 6);
    AddEdge(graph, 7, 8, 7);
}

// Utility function to find the unvisited vertex with the smallest distance
int minDistance(int dist[], bool visited[])
{
    int minDist = INT_MAX;
    int minIndex = 0;

    for (int v = 0; v < V; v++)
    {
        if (visited[v] == false && dist[v] <= minDist)
        {
            minDist = dist[v];
            minIndex = v;
        }
    }

    return minIndex;
}

// Utility function to print out vertex array and the shortest distances
void PrintOutputs(int dist[], int parent[])
{
    cout << "Vertex \t Distance from Start \t Source" << endl;
    for (int i = 0; i < V; i++)
    {
        cout << i << "\t " << dist[i] << "\t\t\t " << parent[i] << endl;
    }
}

// Dijkstra implementation
void Dijkstra(int graph[V][V], int dist[V], int parent[V], int start)
{
    // Visited array
    bool visited[V];

    // Initialize all distances as infinity and all vertices as unvisited
    for (int i = 0; i < V; i++)
    {
        dist[i] = INT_MAX;
        visited[i] = false;
    }

    // Set the distance of start to 0
    dist[start] = 0;

    // Find shortest path
    for (int count = 0; count < V - 1; count++)
    {
        // Get the smallest distance vertex that is unvisited
        int current = minDistance(dist, visited);

        // Set the current vertex as visited
        visited[current] = true;

        // Update the distance values of current's neighboring vertices
        for (int v = 0; v < V; v++)
        {
            // Update distance if:
            // - v is unvisited
            // - v is a neighbor of current
            // - the path from start to v through current is less than the current distance
            if (visited[v] == false &&
                graph[current][v] != 0 &&
                dist[current] != INT_MAX &&
                dist[current] + graph[current][v] < dist[v])
            {
                // Set new smallest distance and parent
                dist[v] = dist[current] + graph[current][v];
                parent[v] = current;
            }
        }
    }
}

int main()
{
    // Create graph
    int graph[V][V];

    // Initialize graph
    InitializeGraph(graph);

    // Create output arrays
    int dist[V];
    int parent[V];

    // Start timer
    auto startTime = chrono::system_clock::now();

    // Run Dijkstra
    Dijkstra(graph, dist, parent, 0);

    // End timer
    chrono::duration<double> elapsedTime = chrono::system_clock::now() - startTime;

    // Print output arrays
    printf("Time: %.8f seconds\n", elapsedTime.count());
    PrintOutputs(dist, parent);

    return 0;
}