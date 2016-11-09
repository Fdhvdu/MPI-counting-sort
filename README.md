# MPI-counting-sort
Use MV2_ENABLE_AFFINITY=0 when you are using MVAPICH2 and you want to enable threads.<br>
For example, `mpiexec -genv MV2_ENABLE_AFFINITY 0 ./a.out ...`<br><br>
For HW1_104062503_basic.cpp, only even odd sort is permitted. (Processes can communicate with all processes.)<br>
For HW1_104062503_advanced.cpp, processes can only communicate with (rank-1) process or (rank+1) process. (But you can do any sort inside process.)
