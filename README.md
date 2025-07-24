# User Manual
Single and multi-objective Integer Linear Programming approach for Automatic Software Cognitive Complexity Reduction.

![Overview of the proposed multiobjective ILP CC reducer tool](ILP_CC_reducer_tool_readme.png)




# Table of Contents
- [Table of Contents](#table-of-contents)
- [ILP Model](#ilp-model-engine)
  - [Requirements](#-requirements)
  - [Download and Installation](#%EF%B8%8F-download-and-installation)
  - [Overview](#-overview)
  - [Problem Context](#-problem-context)
  - [Getting Started](#-getting-started)
  - [Arguments](#-arguments)
  - [Output Types](#-output-types)
  - [Objectives (Cognitive Complexity Metrics)](#-objectives-cognitive-complexity-metrics)
  - [Additional Script](#%EF%B8%8F-additional-script)
  - [Examples](#-examples)
  - [Project Structure](#-project-structure)



  
# ILP Model Engine

This project provides a command-line engine for solving ILP (Integer Linear Programming) problems related to reducing cognitive complexity in Java code refactorings at the method level. The tool supports solving both single-objective and multi-objective ILP instances using various algorithms and configurations.


## 📦 Requirements
- [Python 3.9+](https://www.python.org/)
- [CPLEX](https://www.ibm.com/es-es/products/ilog-cplex-optimization-studio)

The library has been tested in Linux (Mint and Ubuntu) and Windows 11.


## ⬇️ Download and Installation
1. Install [Python 3.9+](https://www.python.org/)
2. Download/Clone this repository and enter into the main directory.
3. Create a virtual environment: `python -m venv env`
4. Activate the environment: 
   
   In Linux: `source env/bin/activate`

   In Windows: `.\env\Scripts\Activate`

   ** In case that you are running Ubuntu, please install the package python3-dev with the command `sudo apt update && sudo apt install python3-dev` and update wheel and setuptools with the command `pip  install --upgrade pip wheel setuptools` right after step 4.
   
5. Install the dependencies: `pip install -r requirements.txt`



## 💡 Overview

The main purpose of this application is to automate the generation and resolution of ILP models designed to **optimize code refactorings**. The models aim to **minimize**:

1. The number of extractions (refactorings).
2. The difference between the maximum and minimum **cognitive complexity** across all resulting sequences.
3. The difference between the maximum and minimum **lines of code** across all resulting sequences.

## 🧠 Problem Context

Given a Java method and its corresponding **refactoring cache**, the system builds ILP model instances to explore different refactoring strategies. These models are then solved with one of several available algorithms.

The results may vary:
- A **.csv file** and a corresponding **.lp file** for **single-objective** ILP.
- For **multi-objective** problems (2 or 3 objectives), a set of solutions using:
  - **Weighted sum** (single or multiple combinations).
  - **Augmented ε-constraint** (for two objectives).
  - A **hybrid algorithm** (for three objectives), exploring the entire objective space to generate an approximate Pareto front.

---

## 🚀 Getting Started

To run the main engine:

```bash
python main.py [OPTIONS]
```


## 🔧 Arguments

| Argument                    | Description                                                                                                                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `-f`, `--file`              | Path to a `.ini` file containing all parameters.                                                                                                                                                  |
| `-n`, `--num_of_objectives` | Number of objectives to be considered: 1, 2 or 3.                                                                                                                                                 |
| `-i`, `--instance`          | Path to the model instance. It can be the folder path with the three data files in CSV format for multiobjective or the general folder path with all instances for one objective.                 |
| `-a`, `--algorithm`         | Algorithm to use for solving single and multiobjective ILP problems. Must be one of: `['ObtainResultsAlgorithm', 'WeightedSumAlgorithm', 'EpsilonConstraintAlgorithm', 'HybridMethodAlgorithm']`. |
| `-t`, `--tau`               | Threshold (τ) used in optimization (e.g., for ε-constraint).                                                                                                                                      |
| `-s`, `--subdivisions`      | (Optional) Number of subdivisions for generating weighted combinations.                                                                                                                           |
| `-w`, `--weights`           | (Optional) Specific weights for weighted sum: `w1,w2` or `w1,w2,w3`.                                                                                                                              |
| `-o`, `--objectives`        | (Optional) Objectives to minimize: `obj1,obj2` or `obj1,obj2,obj3`.                                                                                                                               |
| `--plot`                    | (Optional) Plot the result of a specific experiment.                                                                                                                                              |
| `--3dPF`                    | (Optional) Plot the 3D PF of the given result.                                                                                                                                                    |
| `--all_plots`               | (Optional) Plot all results in the given directory.                                                                                                                                               |
| `--all_3dPF`                | (Optional) Plot all 3D PFs in a given directory.                                                                                                                                                  |
| `--statistics`              | (Optional) Generate a CSV file with statistics for all results in the given directory.                                                                                                            |
| `--input`                   | (Optional) Input directory for results (used for plotting/statistics). Defaults to `output/results`.                                                                                              |
| `--output`                  | (Optional) Output directory for plots/statistics. Defaults to `output/plots_and_statistics`.                                                                                                      |
| `--save`                    | (Optional) Save current configuration to a `.ini` file.                                                                                                                                           |



---

## 🧪 Output Types

Depending on the model and input configuration, the application can generate:

- For **single-objective ILP**:
  - A `.csv` file containing the solution.
  - An `.lp` file representing the ILP model for each instance.

- For **multi-objective ILP** (2 or 3 objectives):
  - A set of non-dominated solutions (Pareto front), either:
    - via **weighted sum**:
      - using a sweep over weight combinations (`--subdivisions`), or
      - with a specific combination (`--weights`).
    - via **augmented ε-constraint** algorithm (`--algorithm`), for 2 objectives.
      - CSV file with results.
      - Concrete model.
      - Output data with the solution for each Java method.
    - via a **hybrid objective-space exploration algorithm**, for 3 objectives:
      - CSV file with results.
      - Concrete model.
      - Output data with the solution for each Java method.
      - Nadir point.
      - Plot in case of requested.

---

## 🧠 Objectives (Cognitive Complexity Metrics)

The optimization process focuses on refactoring Java methods by minimizing the following cognitive complexity metrics:

1. **Number of Extractions**:  
   The total number of code extractions performed.  
   _Goal_: Minimize to keep changes limited.

2. **Complexity Range**:  
   Difference between the highest and lowest cognitive complexity values among extracted sequences.  
   _Goal_: Minimize to ensure balanced complexity across parts.

3. **Lines of Code Range**:  
   Difference between the largest and smallest number of lines of code among extracted parts.  
   _Goal_: Minimize to obtain balanced code lengths.

---

## 🗂️ Additional Script

A secondary `input_files_main.py` located in **ILP_data_from_refactoring_cache** folder is used to **generate ILP input files** from a refactoring cache. This cache must be generated beforehand by another tool that analyzes Java code.

The arguments for this file is the `input_folder` and `output_folder`.

---

## 📘 Examples

This command generates the input needed for the main module of the tool:
```bash
python input_files_main.py ./input_folder ./output_folder
```


This command generates the solution for two-objectives ILP problem with weighted sum algorithm for two objectives:
```bash
python main.py -m multiobjective -i ./instances/my_instance -a WeightedSumAlgorithm2obj -t 2 -s 6 -o seq,cc
```


This command generates the solution for three-objectives ILP problem with hybrid method algorithm for three objectives, and it also generates the parallel coordinates plot and the complete Pareto front in three dimensions:
```bash
python main.py -m multiobjective -i ./instances/my_instance -a HybirdMethodForThreeObj -t 15 -o seq,cc,loc --plot --3dPF
```





## 📂 Project Structure
    📁 M2I-TFM-Adriana  
    ├── 📁 ILP_CC_reducer  
    │   ├── 📁 algorithm  
    │   │   ├── __init__.py  
    │   │   └── algorithm.py  
    │   ├── 📁 algorithms  
    │   │   ├── __init__.py  
    │   │   ├── e_constraint_two_objs.py  
    │   │   ├── hybrid_method_three_objs.py  
    │   │   ├── obtain_results.py  
    │   │   ├── weighted_sum.py  
    │   │   └── weighted_sum_two_objs.py  
    │   ├── 📁 models  
    │   │   ├── __init__.py  
    │   │   ├── ILPmodelRsain.py  
    │   │   └── multiobjILPmodel.py  
    │   └── 📁 operations  
    │       ├── __init__.py  
    │       └── ILP_engine.py  
    ├── 📁 ILP_data_from_refactoring_cache  
    │   ├── 📁 dataset_refactoring_caches  
    │   ├── 📁 utils  
    │   │   ├── dataset.py  
    │   │   ├── offsets.py  
    │   │   └── refactoring_cache.py  
    │   ├── __init__.py  
    │   ├── input_files_main.py  
    │   └── README.md  
    ├── general_utils.py  
    ├── main.py  
    ├── README.md  
    └── requirements.txt  
