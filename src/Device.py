from phyelds.libraries.time import local_time
from CustomLeaderElection import elect_leaders
from phyelds.libraries.collect import collect_with
from phyelds.libraries.device import local_id, store
from phyelds.libraries.distances import neighbors_distances
from phyelds.calculus import aggregate, neighbors, remember
from phyelds.libraries.spreading import broadcast, distance_to

@aggregate
def device():
    return 1