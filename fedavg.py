import models
import utils
import data
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import flwr as fl
import argparse
import os
from datetime import datetime
from collections import OrderedDict
    
def main(dataset_name,
         femnist_location,
         optimiser,
         learning_rate):
        
    if dataset_name=="femnist":
        model = models.create_model(dataset_name, "CNN500k")
        num_rounds = 30
        client_frac = 0.25
        min_clients = 12
        trainloaders, testloaders = data.femnist_data(femnist_location)

    else: # if dataset_name=="cifar":
        model = models.create_model(dataset_name, "CNN500k")
        num_rounds = 180
        client_frac = 0.3
        min_clients = 15
        trainloaders, testloaders = data.cifar_data()
    
    class FlowerClient(fl.client.NumPyClient):
        def __init__(self, cid, net, trainloader, testloader):
            self.cid = cid
            self.net = net
            self.trainloader = trainloader
            self.testloader = testloader

        def get_parameters(self, config):
            return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

        def set_parameters(self, parameters):
            params_dict = zip(self.net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            self.net.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            utils.train(self.net, self.trainloader, optimiser=optimiser, lr=learning_rate, epochs=1)
            return self.get_parameters({}), len(self.trainloader), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, accuracy = utils.test(self.net, self.testloader)
            return float(loss), len(self.testloader), {"accuracy": float(accuracy)}

    def client_fn(cid) -> FlowerClient:
        net = models.create_model(dataset_name, "CNN500k")
        trainloader = trainloaders[int(cid)]
        testloader = testloaders[int(cid)]
        return FlowerClient(cid, net, trainloader, testloader)

    def weighted_average(metrics):
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        return {"accuracy": sum(accuracies) / sum(examples)}
    
    strategy = fl.server.strategy.FedAvg(
    fraction_fit=client_frac,
    fraction_evaluate=client_frac,
    min_fit_clients=min_clients,
    min_evaluate_clients=min_clients,
    min_available_clients=50,
    evaluate_metrics_aggregation_fn=weighted_average,
    initial_parameters=fl.common.ndarrays_to_parameters([val.cpu().numpy() for _, val in model.state_dict().items()]),
    )

    start = datetime.now()
    history = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=50,
    config=fl.server.ServerConfig(num_rounds=num_rounds),
    strategy=strategy,
    client_resources={"num_gpus": 1, "num_cpus": 1},
    )
    end = datetime.now()
    time_taken = end-start
    
    results = pd.DataFrame({"time_taken": [time_taken],
                            "dataset": [dataset_name], 
                            "num_rounds": [num_rounds],
                            "approach": ["FedAvg"],
                            "optimiser": [optimiser],
                            "learning_rate": [learning_rate], 
                            "losses": [history.losses_distributed],
                            "accs": [history.metrics_distributed["accuracy"]],
                           })
    if os.path.isfile("results/fedavg_results.csv"): 
        results.to_csv("results/fedavg_results.csv", mode="a", index=False, header=False)
    else:
        results.to_csv("results/fedavg_results.csv", mode="a", index=False, header=True)
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Simulate FedAvg")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--femnist_location", default="femnist_data")
    parser.add_argument("--optimiser", default="SGD")
    parser.add_argument("--learning_rate", default=0.1, type=float)
    args = parser.parse_args()
    
    main(args.dataset_name,
         args.femnist_location,
         args.optimiser,
         args.learning_rate)