## Federated Learning: Sparsification
 
The communication efficiency of federated learning is improved by sparsifying the parameters uploaded by the clients, hence reducing the size of the upload.

Three approaches to sparsification are compared:  

* Random - Randomly selecting k% parameters  
* Top-k - Selecting the top k% parameters with the largest absolute differences before and after model training  
* Threshold - selecting parameters with absolute differences that are larger than a given threshold

## Experiments

**72 experiments were run for all combinations of:**

* Dataset = ``[femnist, cifar]``
* Approach = ``[random, topk, threshold]``
* Sparsify by:
	* Random & Top-k = ``[0.5, 0.3, 0.1, 0.05, 0.03, 0.01]``
	* Threshold = ``[0.001, 0.003, 0.005, 0.007, 0.0085, 0.01]``
* Keep first and last layer = ``[TRUE, FALSE]``

Results from the experiments: [data](results) and [figures](figures).

**Experimental set-up:**

* Number of clients = ``50``
* Number of epochs = ``1``
* Learning rate = ``0.1``
* Optimiser = ``SGD``
* Regularisation = ``0``
* Fraction of clients sampled each round:
	* FEMNIST = ``0.25``
	* CIFAR = ``0.3``
* Number of federated learning rounds:
	* FEMNIST = ``30``
	* CIFAR = ``180``
* Metric = Accuracy as measured on the test dataset
* Model = ``CNN500k`` as provided in [models.py](models.py)
* Data distribution = as provided in [data.py](data.py)

**Baseline:**

FedAvg [(https://arxiv.org/abs/1602.05629)](https://arxiv.org/abs/1602.05629) using the same set-up as above.


## Simulation

Example command:  
``python simulation.py --dataset_name="cifar" --approach="topk" --sparsify_by=0.1 --num_rounds=180``

| Parameter          | Description                                                                                                                                |
|--------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| --dataset_name     | Can be ``femnist`` or ``cifar``                                                                                                            |
| --femnist_location | Path to the location of the femnist data. Must be pre-downloaded.                                                                          |
| --approach         | Can be ``random`` ``topk``or ``threshold``                                                                                                 |
| --sparsify_by      | Float between 0 and 1 indicating the fraction of parameters to select. For the threshold approach this corresponds to the threshold value. |
| --num_rounds       | Number of federated learning rounds.                                                                                                       |
| --keep_first_last  | Boolean TRUE or FALSE. Indicates whether to ensure that the very first and last layers in the neural network are selected.                 |
| --epochs           | Number of epochs each client trains the local model for.                                                                                   |
| --learning_rate    | Learning rate for the local model training.                                                                                                |
| --regularisation   | Regularisation/weight decay parameter for the optimiser.                                                                                   |

