# CRFL: Certifiably Robust Federated Learning against Backdoor Attacks (ICML 2021)

## Installation
1. Create a virtual environment via `conda`.

   ```shell
   conda create -n crfl python=3.6
   source activate crfl
   ```

2. Install `torch` and `torchvision` according to your CUDA Version and the instructions at [PyTorch](https://pytorch.org/). For example,

   ```shell
   conda install pytorch cudatoolkit=10.1 torchvision -c pytorch
   ```

3. Install requirements.

   ```shell
   pip install -r requirements.txt
   ```



## Dataset

1. MNIST and EMNIST:
MNIST and EMNIST datasets will be automatically downloaded into the dir `./data` during training or testing.

2. LOAN: Download the raw dataset `loan.csv` from [Google Drive](https://drive.google.com/file/d/14Fr32ujeuUvCDiTGBNKDLpWfYjpbKf7q/view?usp=sharing) into the dir `./data`.  
Run   
    ```shell
    python utils/loan_preprocess.py
    ```
    We will get 51 csv files in `./data/loan/`.

## Get Started

1. First, we training the FL models on the three datasets:

```python
python main.py --params configs/mnist_params.yaml
python main.py --params configs/emnist_params.yaml
python main.py --params configs/loan_params.yaml
```

Hyperparameters can be changed according to the comments in those yaml files (`configs/mnist_params.yaml`,`configs/emnist_params.yaml`, ` configs/loan_params.yaml`) to reproduce our experiments.

2. Second, we perform parameter smoothing for the global models on the three datasets:

```python
python smooth_mnist.py
python smooth_emnist.py
python smooth_loan.py
```

The filepaths of models can be changed in those yaml files (`configs/mnist_smooth_params.yaml`,`configs/emnist_smooth_params.yaml, ` `configs/loan_smooth_params.yaml`) .

3. Third, we plot the certified accuracy and certified rate for the three datasets:

```python
python certify_mnist.py
python certify_emnist.py
python certify_loan.py
```

