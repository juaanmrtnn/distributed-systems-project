
## Time Series Prediction Algorithm Using Hybrid Computing

![Status](https://img.shields.io/badge/status-completed-success?style=for-the-badge)

## About The Project

Group project developed for the Distributed Systems course at Pablo de Olavide University.

## Built With

This project was developed using C an hybrid parallel programming standards as OpenMP and MPI

* <img src="https://img.shields.io/badge/C-00599C?style=for-the-badge&logo=c&logoColor=white" alt="C" />
* <img src="https://img.shields.io/badge/MPI-000000?style=for-the-badge&logoColor=white" alt="MPI" />
* <img src="https://img.shields.io/badge/OpenMP-314CB6?style=for-the-badge&logoColor=white" alt="OpenMP" />
* <img src="https://img.shields.io/badge/math.h-555555?style=for-the-badge" alt="math.h" />

## Getting Started

To run a local copy:

### Prerequisites

You need a C compiler that supports OpenMP (like 'gcc') and an MPI implementation installed on your system (OpenMPI/MPICH).

* Ubuntu/Debian:
  ```bash
  sudo apt-get install openmpi-bin openmpi-doc libopenmpi-dev
  ```

  ### Compilation
  1. Clone the repo
  ```bash
  git clone
  ```

  2. Compile the source code using mpicc, including -fopenmp for OpenMP and -lm to link the math.h library
  ```bash
  mpicc -fopenmp code.c -o code -lm
  ```

  ### Usage
  The program expects 4 arguments via the command line to execute properly.
  ```bash
  mpirun -np nP ./code <K> <file> <nP> <nH>
  ```
  #### Arguments Description
  * K: Number of nearest neighbors (k-NN) to use for the prediction. Determines the number of similar historical days considered.
  * file: Path to the input dataset containing the time series data. You can use datos_1X.txt (contains 3341 data rows) or datos_10X.txt (contains 33410 data rows)
  * nP: Number of processes
  * nH: Number of threads
 
  #### Execution Example
  To run the algorithm evaluating the 5 nearest neighbors, reading from datos_1X.txt, using 4 processes and 2 threads per process:
  ```bash
  mpirun -np 4 ./code 5 datos_1X.txt 4 2
  ```
  
  
  
