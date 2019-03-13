# MADDPG-Unity-Env

In this project, I adopted a Multi-Agent Deep Deterministic Policy Gradien for creating two agents with are in charge of collaborate and compete for playing a tennis match. The environment is the similar to the Unity Tennis one. 

![Image](https://www.katnoria.com/static/tennis_play.debc77e3.gif)

##  How to Start

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

    - __Linux__ or __Mac__: 
    ```bash
    conda create --name drlnd python=3.6
    source activate drlnd
    ```
    - __Windows__: 
    ```bash
    conda create --name drlnd python=3.6 
    activate drlnd
    ```
    
2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
    - Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
    - Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
    
3. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.

```bash
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```
4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

##  PyTorch

The model has been developed using PyTorch library. The Pytorch library is available over the main page: https://pytorch.org/

Through the usage of Anaconda, you can download directly the pytorch and torchvision library. 

```bash
conda install pytorch torchvision -c pytorch
```

## Additional Libraries

In addition to PyTorch, in this repository has been used also Numpy. Numpy is already installed in Anaconda, otherwise you can use:

- **`UnityEnvironment`** 
- **`PyTorch`** 
- **`Numpy`** 
- **`Pandas`**
- **`Time`**
- **`Itertools`**
- **`Pandas`**
- **`Time`**
- **`Matplotlib`**

## Files inside repository

- **`report.md`**: it is a report file
- **`model.py`**: topology of the two networks
- **`Tennis.ipynb`**: 

## References

Deep Deterministic Policy Gradient - https://arxiv.org/abs/1509.02971
Reacher Challenge - https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher

## Author

Ivan Vigorito

## License
