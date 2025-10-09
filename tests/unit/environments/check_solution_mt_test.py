import pytest
import importlib

ENVIRONMENT_LIST = ['mtvrp', 'gmtvrp', 'mtdvrp', 'gmtdvrp']

VARIANT_PRESETS = [
    'cvrp', 'ovrp', 'ovrpb', 'ovrpbl', 'ovrpbltw', 'ovrpbtw',
    'ovrpl', 'ovrpltw', 'ovrpmb', 'ovrpmbl', 'ovrpmbltw', 'ovrpmbtw',
    'ovrptw', 'vrpb', 'vrpbl', 'vrpbltw', 'vrpbtw', 'vrpl',
    'vrpltw', 'vrpmb', 'vrpmbl', 'vrpmbltw', 'vrpmbtw', 'vrptw'
    ]


@pytest.fixture(params=ENVIRONMENT_LIST)
def environment_instances_fixture(request):
    env_agent_selector_module_name = f'maenvs4vrp.environments.{request.param}.env_agent_selector'
    env_agent_selector = importlib.import_module(env_agent_selector_module_name).AgentSelector()

    observations_module_name = f'maenvs4vrp.environments.{request.param}.observations'
    observations = importlib.import_module(observations_module_name).Observations()

    generator_module_name = f'maenvs4vrp.environments.{request.param}.instances_generator'
    generator = importlib.import_module(generator_module_name).InstanceGenerator()

    environment_module_name = f'maenvs4vrp.environments.{request.param}.env'
    environment_module = importlib.import_module(environment_module_name)

    env_agent_reward_module_name = f'maenvs4vrp.environments.{request.param}.env_agent_reward'
    reward_evaluator = importlib.import_module(env_agent_reward_module_name).DenseReward()

    environment = environment_module.Environment(instance_generator_object=generator,
                                                 obs_builder_object=observations,
                                                 agent_selector_object=env_agent_selector,
                                                 reward_evaluator=reward_evaluator,
                                                 )
    return environment

@pytest.fixture(params=ENVIRONMENT_LIST)
def environment_instances_fixture_st(request):
    env_agent_selector_module_name = f'maenvs4vrp.environments.{request.param}.env_agent_selector'
    env_agent_selector = importlib.import_module(env_agent_selector_module_name).SmallestTimeAgentSelector()

    observations_module_name = f'maenvs4vrp.environments.{request.param}.observations'
    observations = importlib.import_module(observations_module_name).Observations()

    generator_module_name = f'maenvs4vrp.environments.{request.param}.instances_generator'
    generator = importlib.import_module(generator_module_name).InstanceGenerator()

    environment_module_name = f'maenvs4vrp.environments.{request.param}.env'
    environment_module = importlib.import_module(environment_module_name)

    env_agent_reward_module_name = f'maenvs4vrp.environments.{request.param}.env_agent_reward'
    reward_evaluator = importlib.import_module(env_agent_reward_module_name).DenseReward()

    environment = environment_module.Environment(instance_generator_object=generator,
                                                 obs_builder_object=observations,
                                                 agent_selector_object=env_agent_selector,
                                                 reward_evaluator=reward_evaluator,
                                                 )
    return environment


@pytest.fixture(params=ENVIRONMENT_LIST)
def environment_instances_fixture_rand(request):
    env_agent_selector_module_name = f'maenvs4vrp.environments.{request.param}.env_agent_selector'
    env_agent_selector = importlib.import_module(env_agent_selector_module_name).RandomSelector()

    observations_module_name = f'maenvs4vrp.environments.{request.param}.observations'
    observations = importlib.import_module(observations_module_name).Observations()

    generator_module_name = f'maenvs4vrp.environments.{request.param}.instances_generator'
    generator = importlib.import_module(generator_module_name).InstanceGenerator()

    environment_module_name = f'maenvs4vrp.environments.{request.param}.env'
    environment_module = importlib.import_module(environment_module_name)

    env_agent_reward_module_name = f'maenvs4vrp.environments.{request.param}.env_agent_reward'
    reward_evaluator = importlib.import_module(env_agent_reward_module_name).DenseReward()

    environment = environment_module.Environment(instance_generator_object=generator,
                                                 obs_builder_object=observations,
                                                 agent_selector_object=env_agent_selector,
                                                 reward_evaluator=reward_evaluator,
                                                 )
    return environment


def test_instance_env_agent_iterator_gives_no_error(environment_instances_fixture):
    env = environment_instances_fixture
    for nnodes in [50, 100, 500, 1000]:
        for nagents in [20, 30, 50]:
            td = env.reset(num_agents=nagents, num_nodes=nnodes)
            while not td["done"].all():  
                td = env.sample_action(td)
                td = env.step(td)
            env.check_solution_validity()


def test_instance_env_agent_smallesttime_iterator_gives_no_error(environment_instances_fixture_st):
    env = environment_instances_fixture_st
    for nnodes in [50, 100, 500, 1000]:
        for nagents in [20, 30, 50]:
            td = env.reset(num_agents=nagents, num_nodes=nnodes)
            while not td["done"].all():  
                td = env.sample_action(td)
                td = env.step(td)
            env.check_solution_validity()

def test_instance_env_agent_rand_iterator_gives_no_error(environment_instances_fixture_rand):
    env = environment_instances_fixture_rand
    for nnodes in [50, 100, 500, 1000]:
        for nagents in [20, 30, 50]:
            td = env.reset(num_agents=nagents, num_nodes=nnodes)
            while not td["done"].all():  
                td = env.sample_action(td)
                td = env.step(td)
            env.check_solution_validity()


def test_instance_env_preset_agent_iterator_gives_no_error(environment_instances_fixture):
    for variant in VARIANT_PRESETS:
        env = environment_instances_fixture
        for nnodes in [50, 100, 500, 1000]:
            for nagents in [20, 30, 50]:
                td = env.reset(num_agents=nagents, num_nodes=nnodes, variant_preset=variant, batch_size=8)
                while not td["done"].all():  
                    td = env.sample_action(td)
                    td = env.step(td)
                env.check_solution_validity()


def test_instance_env_preset_agent_smallesttime_iterator_gives_no_error(environment_instances_fixture_st):
    for variant in VARIANT_PRESETS:
        env = environment_instances_fixture_st
        for nnodes in [50, 100, 500, 1000]:
            for nagents in [20, 30, 50]:
                td = env.reset(num_agents=nagents, num_nodes=nnodes, variant_preset=variant, batch_size=8)
                while not td["done"].all():  
                    td = env.sample_action(td)
                    td = env.step(td)
                env.check_solution_validity()

def test_instance_env_preset_agent_rand_iterator_gives_no_error(environment_instances_fixture_rand):
    for variant in VARIANT_PRESETS:
        env = environment_instances_fixture_rand
        for nnodes in [50, 100, 500, 1000]:
            for nagents in [20, 30, 50]:
                td = env.reset(num_agents=nagents, num_nodes=nnodes, variant_preset=variant, batch_size=8)
                while not td["done"].all():  
                    td = env.sample_action(td)
                    td = env.step(td)
                env.check_solution_validity()