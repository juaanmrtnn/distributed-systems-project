
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
  To run the program in a distributed environment, use mpirun specifying the number of processes with the -np flag.
  ```bash
  mpirun -np 4 ./code
  ```
  
  
  
