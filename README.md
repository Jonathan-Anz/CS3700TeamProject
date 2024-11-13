MPI Dijkstra

The number of processors and vertices are set in the batch file (so you don't have to update the original script everytime)

To change processors (number at the end):
#SBATCH --nodes=4

To change vertices (number at the end):
mpirun python3.4 dijkstra_mpi.py 64

As of now, the default is 64 vertices on 4 processors
