# Multy-Label Classification in Sparse Matrices #


### Enviroment and Setup ###
    * Python version -> 2.7.13
    * Dependencies ->
                        numba 0.34.0,
                        numpy 1.13.1,
                        scipy 0.19.1,
                        tqdm 4.15.0
                        scikit-learn 0.19.0,
                        
    * Profiling was done with line_profiler 2.0.0,


### Datasets and runtime ###
Preparation : Extract datasets in data folder, datasets found here[http://manikvarma.org/downloads/XC/XMLRepository.html]

|Name       | Feature Dimensionality|     Label Dimensionality         |  Number of Train Points    | Number of  Test Points  |   Avg. Points per Label|     Avg. Labels per Point| Runtime|
|--------------|:-------------------------:|:----------------------------------:|:--------------------------:|:-----------------------:|-----------------------:|:-----------------------:|:-------:|
| Delicious |500	                |    983	                       | 12920	                    |3185	                  |311.61	               |    19.03                | ~1,5min |
| RCV1-2K   |  47236                |     2456                         |623847                      |155962                   |1218.56                 |      4.79               | ~3h*    |
| eurlex-4K |5000	|3993	|15539|	3809	|25.73|	5.31| ~20min|
| AmazonCat-13k |203882	|13330	|1186239	|306782	|448.57	|5.04|~36h|
| AmazonCat-14k |597540|	14588	|4398050	|1099725|	1330.1	|3.53| ~109h|
| Delicious-200k| 782585 | 205443 | 196606 | 100095 |72.29 |75.54| ~400h|
|Amazon-3M	|337067	|2812281|	1717899	|742507	|31.64	|36.17| ~8829h|
