## Federated Learning: Sparsification
 
The communication efficiency of federated learning is improved by sparsifying the parameters uploaded by the clients, hence reducing the size of the upload.

Three approaches to sparsification are compared:  

* Random - Randomly selecting k% parameters  
* Top-k - Selecting the top k% parameters with the largest absolute differences before and after model training  
* Threshold - Selecting parameters with absolute differences that are larger than a given threshold

Two datasets ([data.py](data.py)) are used for the experiments:

* CIFAR-10 - Imbalanced, non-iid partition of data between the clients
* FEMNIST - Data distributed between the clients based on the author

The basic convolutional model ``CNN500k`` ([models.py](models.py)):

* Model for CIFAR-10 - 471,338 parameters (1.89 MB)
* Model for FEMNIST - 369,432 parameters (1.48 MB)

All experiments were implemented in PyTorch using Flower's virtual client engine ([https://github.com/adap/flower](https://github.com/adap/flower)).

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
* Metric = ``Accuracy`` as measured on the test dataset

**Baseline:**

Federated Averaging [(https://arxiv.org/abs/1602.05629)](https://arxiv.org/abs/1602.05629) using the same set-up as above.

## Simulation

Example command:  
``python simulation.py --dataset_name="cifar" --approach="topk" --sparsify_by=0.1 --num_rounds=180``

| Parameter          | Description                                                                                                                                |
|--------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| --dataset_name     | Can be ``femnist`` or ``cifar``.                                                                                                            |
| --femnist_location | Path to the location of the femnist data. Must be pre-downloaded.                                                                          |
| --approach         | Can be ``random`` ``topk``or ``threshold``.                                                                                                 |
| --sparsify_by      | Float between 0 and 1 indicating the fraction of parameters to select. For the threshold approach this corresponds to the threshold value. |
| --num_rounds       | Number of federated learning rounds.                                                                                                       |
| --keep_first_last  | Boolean ``TRUE`` or ``FALSE``. Indicates whether to force the selection of all parameters in the very first and last layers in the network.                 |
| --epochs           | Number of epochs each client trains the local model for.                                                                                   |
| --learning_rate    | Learning rate for the model training.                                                                                                |
| --regularisation   | Regularisation/weight decay parameter for the optimiser.                                                                                   |

