import pytest
import importlib
import torch

from maenvs4vrp.core import parallel_env

ENVIRONMENT_LIST = ['cvrp', 'pcvrp', 'pcvrptw', 'top' , 'toptw']

DEFAULT_DEVICE = "cpu"
DEFAULT_BATCH_SIZE = 1
DEFAULT_NUM_AGENTS = [20, 30, 50]
DEFAULT_NUM_NODES = [51, 101, 501, 1001]

@pytest.fixture
def define_values_fixture(device, batch, num_agents, num_nodes):
    defined_device = torch.device(DEFAULT_DEVICE) if device == "cpu" else torch.device("cuda")
    defined_batch = DEFAULT_BATCH_SIZE if batch == None else batch
    defined_num_agents = DEFAULT_NUM_AGENTS if num_agents == None else num_agents
    defined_num_nodes = DEFAULT_NUM_NODES if num_nodes == None else num_nodes

    #Convert every list element from str to int
    defined_batch = int(defined_batch)
    defined_num_agents = list(map(int, defined_num_agents))
    defined_num_nodes = list(map(int, defined_num_nodes))

    return defined_device, defined_batch, defined_num_agents, defined_num_nodes

@pytest.fixture(params=ENVIRONMENT_LIST)
def environment_instances_fixture(request):

    observations_module_name = f'maenvs4vrp.parallel_environments.{request.param}.observations'
    observations = importlib.import_module(observations_module_name).Observations()

    generator_module_name = f'maenvs4vrp.parallel_environments.{request.param}.instances_generator'
    generator = importlib.import_module(generator_module_name).InstanceGenerator()

    environment_module_name = f'maenvs4vrp.parallel_environments.{request.param}.env'
    environment_module = importlib.import_module(environment_module_name)

    env_agent_reward_module_name = f'maenvs4vrp.parallel_environments.{request.param}.env_agent_reward'
    reward_evaluator = importlib.import_module(env_agent_reward_module_name).DenseReward()

    environment = environment_module.Environment(instance_generator_object=generator,
                                                 obs_builder_object=observations,
                                                 reward_evaluator=reward_evaluator,
                                                 )
    return environment


def test_print_defined_values(define_values_fixture):
    defined_device, defined_batch_size, defined_num_agents, defined_num_nodes = define_values_fixture
    print(f"""
Defined device: {defined_device}
Defined batch size: {defined_batch_size}
Defined num_agents: {defined_num_agents}
Defined num_nodes: {defined_num_nodes}
        """)

def test_instance_env_agent_iterator_gives_no_error(environment_instances_fixture, define_values_fixture):
    defined_device, defined_batch_size, defined_num_agents, defined_num_nodes = define_values_fixture
    env = environment_instances_fixture
    for nnodes in defined_num_nodes:
        for nagents in defined_num_agents:
            td = env.reset_observe(num_agents=nagents, num_nodes=nnodes, device=defined_device, batch_size=defined_batch_size)

            while not td['done'].all():
                # actions: [B, A] — one action per agent
                td = env.sample_actions_all(td)
                td = env.step_all_observe(td)
            env.check_solution_validity()

