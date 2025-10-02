import pytest
import importlib

ENVIRONMENT_LIST = ['cvrptw', 'toptw', 'cvrpstw', 'sdvrptw', 'pcvrptw', 'pdptw', 'mdvrptw', 'mtvrp', 'gmtvrp', 'mtdvrp', 'gmtdvrp']



@pytest.fixture(params=ENVIRONMENT_LIST)
def environment_benchmark_instance_fixture(request):
    env_agent_selector_module_name = f'maenvs4vrp.environments.{request.param}.env_agent_selector'
    env_agent_selector = importlib.import_module(env_agent_selector_module_name).AgentSelector()

    observations_module_name = f'maenvs4vrp.environments.{request.param}.observations'
    observations = importlib.import_module(observations_module_name).Observations()

    generator_module_name = f'maenvs4vrp.environments.{request.param}.benchmark_instances_generator'
    generator_module = importlib.import_module(generator_module_name)
    list_of_benchmark_instances = generator_module.BenchmarkInstanceGenerator.get_list_of_benchmark_instances()
    instance_types = list_of_benchmark_instances.keys()
    instance_type = list(instance_types)[0]
    set_of_instances = list_of_benchmark_instances.get(instance_type)
    generator = generator_module.BenchmarkInstanceGenerator(instance_type=instance_type,
                                                            set_of_instances=set_of_instances)

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
def environment_benchmark_instance_fixture_st(request):
    env_agent_selector_module_name = f'maenvs4vrp.environments.{request.param}.env_agent_selector'
    env_agent_selector = importlib.import_module(env_agent_selector_module_name).SmallestTimeAgentSelector()

    observations_module_name = f'maenvs4vrp.environments.{request.param}.observations'
    observations = importlib.import_module(observations_module_name).Observations()

    generator_module_name = f'maenvs4vrp.environments.{request.param}.benchmark_instances_generator'
    generator_module = importlib.import_module(generator_module_name)
    list_of_benchmark_instances = generator_module.BenchmarkInstanceGenerator.get_list_of_benchmark_instances()
    instance_types = list_of_benchmark_instances.keys()
    instance_type = list(instance_types)[0]
    set_of_instances = list_of_benchmark_instances.get(instance_type)
    generator = generator_module.BenchmarkInstanceGenerator(instance_type=instance_type,
                                                            set_of_instances=set_of_instances)

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
def environment_benchmark_instance_fixture_rand(request):
    env_agent_selector_module_name = f'maenvs4vrp.environments.{request.param}.env_agent_selector'
    env_agent_selector = importlib.import_module(env_agent_selector_module_name).RandomSelector()

    observations_module_name = f'maenvs4vrp.environments.{request.param}.observations'
    observations = importlib.import_module(observations_module_name).Observations()

    generator_module_name = f'maenvs4vrp.environments.{request.param}.benchmark_instances_generator'
    generator_module = importlib.import_module(generator_module_name)
    list_of_benchmark_instances = generator_module.BenchmarkInstanceGenerator.get_list_of_benchmark_instances()
    instance_types = list_of_benchmark_instances.keys()
    instance_type = list(instance_types)[0]
    set_of_instances = list_of_benchmark_instances.get(instance_type)
    generator = generator_module.BenchmarkInstanceGenerator(instance_type=instance_type,
                                                            set_of_instances=set_of_instances)

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

# reset tests
def test_benchmark_instance_env_reset_gives_no_error(environment_benchmark_instance_fixture):
    env = environment_benchmark_instance_fixture
    td = env.reset()


# observe
def test_benchmark_instance_env_observe_gives_no_error(environment_benchmark_instance_fixture):
    env = environment_benchmark_instance_fixture
    td = env.reset()
    td_observations = env.observe()


# agent iterator
def test_benchmark_instance_env_agent_iterator_gives_no_error(environment_benchmark_instance_fixture):
    env = environment_benchmark_instance_fixture
    td = env.reset()
    while not td["done"].all():  
        td = env.sample_action(td)
        td = env.step(td)


def test_benchmark_instance_env_smallesttime_agent_iterator_gives_no_error(environment_benchmark_instance_fixture_st):
    env = environment_benchmark_instance_fixture_st
    td = env.reset()
    while not td["done"].all():  
        td = env.sample_action(td)
        td = env.step(td)

def test_benchmark_instance_env_rand_agent_iterator_gives_no_error(environment_benchmark_instance_fixture_rand):
    env = environment_benchmark_instance_fixture_rand
    td = env.reset()
    while not td["done"].all():  
        td = env.sample_action(td)
        td = env.step(td)