import pytest
import importlib

ENVIRONMENT_LIST = ['cvrptw', 'toptw', 'cvrpstw', 'sdvrptw', 'pcvrptw', 'pdptw', 'mdvrptw']



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

@pytest.fixture(params=ENVIRONMENT_LIST)
def environment_instances_all_observations_fixture(request):
    env_agent_selector_module_name = f'maenvs4vrp.environments.{request.param}.env_agent_selector'
    env_agent_selector = importlib.import_module(env_agent_selector_module_name).RandomSelector()

    observations_module_name = f'maenvs4vrp.environments.{request.param}.observations'
    Observations = importlib.import_module(observations_module_name).Observations
    all_possible_features = {'nodes_static': dict([(c, {'feat': c, 'norm': None}) for c in Observations.POSSIBLE_NODES_STATIC_FEATURES]),
                             'nodes_dynamic': Observations.POSSIBLE_NODES_DYNAMIC_FEATURES,
                             'agent': Observations.POSSIBLE_AGENT_FEATURES,
                             'other_agents': Observations.POSSIBLE_OTHER_AGENTS_FEATURES,
                             'global': Observations.POSSIBLE_GLOBAL_FEATURES}
    observations = Observations(all_possible_features)

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

def test_instance_env_reset_gives_no_error(environment_instances_fixture):
    env = environment_instances_fixture
    td = env.reset()

# observe
def test_benchmark_instance_env_observe_gives_no_error(environment_benchmark_instance_fixture):
    env = environment_benchmark_instance_fixture
    td = env.reset()
    td_observations = env.observe()


def test_instance_env_observe_all_observations_gives_no_error(environment_instances_all_observations_fixture):
    env = environment_instances_all_observations_fixture
    td = env.reset()
    td_observations = env.observe()


# agent iterator
def test_benchmark_instance_env_agent_iterator_gives_no_error(environment_benchmark_instance_fixture):
    env = environment_benchmark_instance_fixture
    td = env.reset()
    while not td["done"].all():  
        td = env.sample_action(td)
        td = env.step(td)


def test_instance_env_agent_iterator_gives_no_error(environment_instances_fixture):
    env = environment_instances_fixture
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


def test_instance_env_agent_smallesttime_iterator_gives_no_error(environment_instances_fixture_st):
    env = environment_instances_fixture_st
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


def test_instance_env_agent_rand_iterator_gives_no_error(environment_instances_fixture_rand):
    env = environment_instances_fixture_rand
    td = env.reset()
    while not td["done"].all():  
        td = env.sample_action(td)
        td = env.step(td)