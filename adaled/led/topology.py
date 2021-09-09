from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union, TYPE_CHECKING

# Lazy load to be able to import adaled without having MPI.
if TYPE_CHECKING:
    from mpi4py import MPI
    Intracomm = MPI.Intracomm
    Intercomm = MPI.Intercomm
    del MPI
else:
    Intracomm = 'mpi4y.MPI.Intracomm'
    Intercomm = 'mpi4y.MPI.Intercomm'


@dataclass
class _Rank:
    is_server: bool

    # Root in the corresponding side communicator. Server ranks are always
    # their own roots.
    sidecomm_root: int

    # Ranks that operate with adaled. Includes all server ranks and, currently,
    # root ranks of the client components (individual simulations).
    is_active: bool

    # The remote ranks this rank communicates with. Only for active ranks.
    remote_activecomm_ranks: Optional[List[int]]

    @property
    def is_client(self):
        return not self.is_server


@dataclass
class Topology:
    """Topology of adaled server and client ranks, and non-adaled client
    (simulation) ranks.

    The "active" term below refers to all ranks that operate with adaled, which
    includes server ranks and root client ranks, one for each simulation.
    In other words, for multirank simulations, the user is supposed to
    implement micro solver intracommunication manually.

    Attributes:
        specification: (str) original string used to specify the topology
        world: communicator of all active and inactive ranks
        active_comm: communicator of active ranks, both server and clients
        active_side_comm: communicator of either server or active client ranks
        active_intercomm: intercommunicator over `active_side_comm`
        server_comm: alias of `active_side_comm` on server ranks, null otherwise
        remote_activecomm_ranks: (none or list of ints) List of server ranks
            this client rank sends its dataset to, or a list of (root) client
            ranks this server sends the networks to, with respect to
            `active_intercomm`. None for non-active client ranks.
    """
    specification: str
    is_server: bool
    world: Intracomm

    active_comm: Intracomm
    active_side_comm: Intracomm
    active_intercomm: Intracomm

    server_comm: Intracomm

    component_comm: Intracomm
    remote_activecomm_ranks: Optional[List[int]]

    @property
    def is_client(self):
        return not self.is_server

    @property
    def is_adaled_active(self):
        from mpi4py import MPI
        return self.active_comm != MPI.COMM_NULL


def _parse_topology(topology: str) -> List[_Rank]:
    parts = topology.split(',')
    components = [None] * len(parts)

    # Parse servers first.
    servers: Dict[str, _Rank] = {}
    for i, part in enumerate(parts):
        if part.startswith('S'):
            if part in servers:
                raise ValueError(f"identifier {part} used more than once: {topology}")
            # Remote ranks are filled later.
            servers[part] = _Rank(is_server=True, sidecomm_root=len(servers),
                                  is_active=True, remote_activecomm_ranks=[])
            components[i] = servers[part]

    # Then parse clients.
    clientcomm_roots: Dict[str, int] = {}
    for i, part in enumerate(parts):
        if part.startswith('S'):
            continue  # Already parsed.
        elif not part.startswith('C'):
            raise ValueError(f"unrecognized rank specified {part!r}")

        name, *servers_descr = part.split('->', 1)
        root = clientcomm_roots.get(name)
        if root is None:
            if len(servers_descr) != 1:
                raise ValueError(f"first rank of {name} is missing the server specification")
            clientcomm_roots[name] = root = len(clientcomm_roots)
            _servers = [servers[s] for s in servers_descr[0].split('+')]
            remote_ranks = [s.sidecomm_root for s in _servers]
            for s in _servers:
                s.remote_activecomm_ranks.append(root)
            components[i] = _Rank(
                    is_server=False, sidecomm_root=root,
                    is_active=True, remote_activecomm_ranks=remote_ranks)
        else:
            if len(servers_descr) == 1:
                raise ValueError(f"only the first rank of a client can "
                                 f"specify the server (client {name})")
            components[i] = _Rank(is_server=False, sidecomm_root=root,
                                  is_active=False, remote_activecomm_ranks=None)

    return components


def topology_split_comm(world: Intracomm, topology: str):
    """Split the communicator according to the given server-client rank topology.

    The topology is defined in groups of ranks of arbitrary size, and specifies
    how many servers ranks, client ranks and different simulations there are
    per group. The total world size must be a multiple of the group size.

    Examples:
        A group of 3 ranks with 1 server and 2 client ranks running two
        separate simulations. Useful for parallelizing 1 server + 2 simulation
        per node.
        +------+
        |  S0  |
        | ^ ^  |
        | | |  |
        | | |  |
        | |C0  |
        | |    |
        |  C1  |
        +------+
        >>> topology_split_comm(world, 'S0,C0->S0+S1,S1,C0')
        ...

        A group of 4 ranks, with 0th and 2nd ranks in the group are servers,
        and 1st and 3rd ranks are clients running a joint simulation. The
        second client is "inactive", i.e. does not communicate with the server
        or even operate with adaled. It is user's (micro solver's)
        responsibility to implement communication between the two client ranks
        (in the `advance()` and `get_state()` functions). This is useful for
        parallelizing a single simulation on two nodes.
        +------+   +------+
        |  S0  |   |  S1  |
        |   ^  |   |  ^   |
        |   |  |   |/     |
        |   |  |  /|      |
        |   |  |/  |      |
        |     /|   |      |
        |  C0  |   |  C0  |
        +------+   +------+
        >>> t = topology_split_comm(world, 'S0,C0->S0+S1,S1,C0')
        >>> t.component_comm.size
        ...  # 1 if server, 2 if client
    """
    from mpi4py import MPI
    components = _parse_topology(topology)

    if not any(c.is_client for c in components):
        raise ValueError(f"no clients specified in topology {topology!r}")
    if not any(c.is_server for c in components):
        raise ValueError(f"no servers specified in topology {topology!r}")

    group_size = len(components)
    if world.size % group_size != 0:
        raise RuntimeError(f"expected a multiple of {group_size}, "
                           f"got {world.size} ranks for topology {topology!r}")

    group_rank = world.rank % group_size
    group_id = world.rank // group_size
    me = components[group_rank]

    active_comm: Intracomm = world.Split(0 if me.is_active else MPI.UNDEFINED)
    if me.is_active:
        remote_leader = next(i for i, c in enumerate(components)
                             if c.is_server != me.is_server)
        active_side_comm: Intracomm = active_comm.Split(me.is_server)
        active_intercomm = active_side_comm.Create_intercomm(0, world, remote_leader)
    else:
        active_side_comm = MPI.COMM_NULL
        active_intercomm = MPI.COMM_NULL

    group_comm: Intracomm = world.Split(group_id)
    group_side_comm: Intracomm = group_comm.Split(me.is_server)
    component_comm: Intracomm = group_side_comm.Split(me.sidecomm_root)

    active_remote_per_group = \
            sum(c.is_active and c.is_server != me.is_server for c in components)
    offset = group_id * active_remote_per_group
    if me.remote_activecomm_ranks is not None:
        _remote_ranks = [offset + rank for rank in me.remote_activecomm_ranks]
    else:
        _remote_ranks = None

    return Topology(
            specification=topology,
            is_server=me.is_server,
            world=world,
            active_comm=active_comm,
            active_side_comm=active_side_comm,
            active_intercomm=active_intercomm,
            server_comm=(active_side_comm if me.is_server else MPI.COMM_NULL),
            component_comm=component_comm,
            remote_activecomm_ranks=_remote_ranks)
