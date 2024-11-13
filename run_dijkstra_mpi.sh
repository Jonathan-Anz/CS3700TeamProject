#!/bin/bash
#SBATCH --job-name=DIJKSTRA_MPI
#SBATCH --output=DIJKSTRA_MPI.txt
#SBATCH --mem-per-cpu=1024
#SBATCH --partition=compute
#SBATCH --nodes=4
#SBATCH --time=00:02:00

. /etc/profile.d/modules.sh

module load openmpi/2.1.2
module load python/3/mpi4py/3.0.0

export MPI_HOME=/opt/openmpi-2.1.2
export PATH=${MPI_HOME}/bin:${PATH}
mpirun python3.4 dijkstra_mpi.py 64