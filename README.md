# Adapting Convolutional Restricted Boltzmann Machines Through Evolutionary Optimization

*This repository holds all the necessary code to run the very-same experiments described in the paper "Adapting Convolutional Restricted Boltzmann Machines Through Evolutionary Optimization".*

---

## References

If you use our work to fulfill any of your needs, please cite us:

```
```

---

## Structure

 * `utils`
   * `loader.py`: Utility to load datasets and split them into training, validation, and testing sets;
   * `objects.py`: Wraps objects instantiation for command line usage;
   * `optimizer.py`: Wraps the optimization task into a single method;  
   * `target.py`: Implements the objective functions to be optimized.
   
---

## Package Guidelines

### Installation

Install all the pre-needed requirements using:

```Python
pip install -r requirements.txt
```

### Data configuration

In order to run the experiments, you can use `torchvision` and the internal downloader to load pre-implemented datasets. If necessary, one can also [download](https://www.recogna.tech/files/crbm_tuning) non-torchvision datasets and put them in the `data/` folder.

---

## Usage

### CRBM Optimization

The first step is to optimize a Convolutional RBM architecture. To accomplish such a step, one needs to use the following script:

```Python
python crbm_optimization.py -h
```

*Note that `-h` invokes the script helper, which assists users in employing the appropriate parameters.*

### CRBM Optimization using Genetic Programming

Alternatively, one can use Genetic Programming to optimize the architecture, as follows:

```Python
python crbm_tree_optimization.py -h
```

### CRBM Evaluation

After conducting the optimization task, one needs to evaluate the best parameters over the testing set. Please, use the following script to accomplish such a procedure:

```Python
python crbm_evaluation.py -h
```

*Note that this script evaluates the network on both reconstruction and classification tasks.*

### Bash Script

Instead of invoking every script to conduct the experiments, it is also possible to use the provided shell script, as follows:

```Bash
./pipeline.sh
```

Such a script will conduct every step needed to accomplish the experimentation used throughout this paper. Furthermore, one can change any input argument that is defined in the script.

---

## Support

We know that we do our best, but it is inevitable to acknowledge that we make mistakes. If you ever need to report a bug, report a problem, talk to us, please do so! We will be available at our bests at this repository or gustavo.rosa@unesp.br.

---
