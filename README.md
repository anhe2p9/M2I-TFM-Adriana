# User Manual
Enhancement for Automatic Software Cognitive Complexity Reduction through Integer Linear Programming




# Table of Contents
- [Table of Contents](#table-of-contents)
- [ILP Model](#ilp-model)
  - [Requirements](#requirements)
  - [Download and installation](#download-and-installation)
  - [Execution of the ILP Model](#execution-of-the-ilp-model)
  - [Execution of the tests](#execution-of-the-tests)
  
# ILP Model

## Requirements
- [Python 3.9+](https://www.python.org/)
- [CPLEX](https://www.ibm.com/es-es/products/ilog-cplex-optimization-studio)

The library has been tested in Linux (Mint and Ubuntu) and Windows 11.


## Download and installation
1. Install [Python 3.9+](https://www.python.org/)
2. Download/Clone this repository and enter into the main directory.
3. Create a virtual environment: `python -m venv env`
4. Activate the environment: 
   
   In Linux: `source env/bin/activate`

   In Windows: `.\env\Scripts\Activate`

   ** In case that you are running Ubuntu, please install the package python3-dev with the command `sudo apt update && sudo apt install python3-dev` and update wheel and setuptools with the command `pip  install --upgrade pip wheel setuptools` right after step 4.
   
5. Install the dependencies: `pip install -r requirements.txt`


## Execution of the ILP Model
You can use any code in the [codes folder](codes/) to execute and test the ILP model.

- **Help:** Provide help to execute the refactoring engine.
    `python main.py -h`

- **Apply the CC reducer model to a method:** Apply the ILP model to the given method of the provided piece of code.
  
  - Execution: `python ilp-model-pfm.py FILE1 FILE2 FILE3 THRESHOLD`
  - Inputs: 
    - The `FILE1` parameter specifies the file path of the extraction opportunities, its lines of code (LOC) and its new method cognitive complexity ($NMCC_i = \iota_i + \nu_i$) in CSV format.
    - The `FILE2` parameter specifies the subsequences of each extraction opportunity, to specify the relation between each node of the conflict graph, and its cognitive complexity reduction ($CCR_{j \to i} = \iota_i + \nu_j + |\lambda_j - \lambda_i|\mu_j$)
    - The `FILE3` parameter specifies the conflict sequences and its cognitive complexity reduction (CCR defined above).
    - The `THRESHOLD` parameter specifies the maximum SSCC desired for that method.
  - Outputs:
    - An ILP model file in NO LO SÉ format with the given method refactored.
  - Example: `python ilp-model-pfm.py sequences.csv nested.csv conflict.csv 15`


## Execution of the tests
No sé si habrá.
