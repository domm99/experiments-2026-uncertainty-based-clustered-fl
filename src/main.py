import torch
import random
from Device import device
from phyelds.simulator import Simulator
from phyelds.simulator.deployments import deformed_lattice
from phyelds.simulator.neighborhood import radius_neighborhood
from phyelds.simulator.exporter import csv_exporter, ExporterConfig

def get_current_device():
    device: str = 'cpu'
    if torch.accelerator.is_available():
        current_accelerator = torch.accelerator.current_accelerator()
        if current_accelerator is not None:
            device = current_accelerator.type
    return device

def download_dataset(dataset_name: str):
    pass

def initialize_target_rnd():
    pass

def run_simulation(dataset_name: str, number_of_areas: int, device: str, random_seed: int) -> None:
    simulator = Simulator()
    simulator.environment.set_neighborhood_function(radius_neighborhood(1.12))
    deformed_lattice(simulator, 7, 7, 1, 0.01)

    devices = len(simulator.environment.nodes.values())
    mapping_devices_area = distribute_nodes_spatially(devices, number_of_areas)

    train_data, test_data = download_dataset(dataset_name)

    common_target_rnd = initialize_target_rnd() ## TODO implement this

    mapping = {} ### mapping node_id -> data
    for region_id, devices in mapping_devices_area.items():
        pass ## TODO - here we should split the data with the different noise

        # schedule the main function
        for node in simulator.environment.nodes.values():
            simulator.schedule_event(
                0.0,
                aggregate_program_runner,
                simulator,
                1.0,
                node,
                device,
            )

    RenderMonitor(
        simulator,
        RenderConfig(
            effects=[DrawEdges(), CustomDrawNodes(color_from="result")],
            mode=RenderMode.SAVE,
            save_as=f"export/seed-{seed}_areas-{number_of_areas}dataset-{dataset_name}.mp4",
            dt=1.0
        )
    )

    simulator.run(3)

if __name__ == '__main__':

    datasets = ['CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST', 'EMNIST']
    seeds = range(1)
    areas = [3, 5, 9]
    device = get_current_device()

    for seed in seeds:
        for dataset in datasets:
            for n_areas in areas:
                run_simulation(dataset, n_areas, device, seed)