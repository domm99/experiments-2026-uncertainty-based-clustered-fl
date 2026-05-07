from phyelds.libraries.time import local_time
from CustomLeaderElection import elect_leaders
from learning import (
    RND_UNCERTAINTY_REDUCTION,
    evaluate_rnd_on_dataset,
    load_rnd_from_weights,
    train_rnd_on_dataset,
)
from phyelds.libraries.collect import collect_with
from phyelds.libraries.device import local_id, store
from phyelds.libraries.distances import neighbors_distances
from phyelds.calculus import aggregate, neighbors, remember
from phyelds.libraries.spreading import broadcast, distance_to
from phyelds.data import NeighborhoodField

def check_cluster(self_uncertainty, neigh_uncertainty):

    alpha = 0.3

    lower = (1-alpha) * self_uncertainty
    upper = (1+alpha) * self_uncertainty
    if lower < neigh_uncertainty < upper:
        return 0.01
    else:
        return 1

@aggregate
def device(data, dataset_name, device, seed, target_rnd=None):

    set_value, stored_value = remember((0, None))
    tick, rnd_model = stored_value
    node_id = local_id()
    if tick == 0:
        rnd_model = train_rnd_on_dataset(
            dataset=data,
            target_network=target_rnd,
            batch_size=128,
            epochs=10,
            lr=1e-3,
            seed=seed + 10_000 + node_id,
            device=device,
        )

    rnd_eval = evaluate_rnd_on_dataset(
        rnd_model=rnd_model,
        dataset=data,
        batch_size=256,
        device=device,
        uncertainty_reduction=RND_UNCERTAINTY_REDUCTION,
    )

    my_mean_uncertainty = rnd_eval["mean_uncertainty"]
    print(f'----------------------------------------- node {node_id} -----------------------------------------')
    f = compute_distances(data, rnd_model, target_rnd, my_mean_uncertainty, device, seed)

    (leader, leader_id) = elect_leaders(0.3, f)  # If leader is true, then I'm an aggregator

    set_value((tick+1, rnd_model))

    # print(f'----------------------------------------- node {node_id} -----------------------------------------')
    print(f'tick {tick}')

    return leader_id

@aggregate
def compute_distances(data, rnd_model, target_rnd, my_mean_uncertainty, device, seed):
    models_weights = neighbors(rnd_model.state_dict())
    neighbors_models = NeighborhoodField(models_weights.exclude_self(), local_id())
    evaluations = neighbors_models.map(
        lambda m: evaluate_rnd_on_dataset(
            load_rnd_from_weights(m, data, target_rnd, seed, device=device),
            dataset=data,
            batch_size=256,
            device=device,
            uncertainty_reduction=RND_UNCERTAINTY_REDUCTION,
        )["mean_uncertainty"]
    )

    neighbor_field = evaluations.map(
        lambda e: check_cluster(my_mean_uncertainty, e)
    )
    print(f'Uncertainty: {my_mean_uncertainty}')
    print(evaluations)
    print(neighbor_field)
    return NeighborhoodField(neighbor_field.data | {local_id(): 0}, local_id())