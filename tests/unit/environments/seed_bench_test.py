import pytest
import importlib
from maenvs4vrp.utils.utils import data_equivalence

ENVIRONMENT_LIST = ['cvrptw', 'toptw', 'cvrpstw', 'sdvrptw', 'pcvrptw', 'pdptw', 'mdvrptw', 'mtvrp', 'gmtvrp', 'mtdvrp', 'gmtdvrp']


@pytest.fixture(params=ENVIRONMENT_LIST)
def benchmark_instances_generator_module(request):
    module_name = f'maenvs4vrp.environments.{request.param}.benchmark_instances_generator'
    module = importlib.import_module(module_name)
    return module


@pytest.fixture
def benchmark_instance_generator_fixture(benchmark_instances_generator_module):
    list_of_benchmark_instances = benchmark_instances_generator_module.BenchmarkInstanceGenerator.get_list_of_instances()
    instance_names = list_of_benchmark_instances.keys()
    instance_name = list(instance_names)[0]
    list_of_instances = list_of_benchmark_instances.get(instance_name)
    if not list_of_instances:
        pytest.skip(f"No benchmark instances available for '{instance_name}'")
    generator = benchmark_instances_generator_module.BenchmarkInstanceGenerator(instance_name=instance_name, list_of_instances=list_of_instances)
    return generator


def test_different_seed_benchmark_instance_generator(benchmark_instance_generator_fixture):
    instance1 = benchmark_instance_generator_fixture.sample_instance(num_agents=20, num_nodes=51, seed=1)
    instance2 = benchmark_instance_generator_fixture.sample_instance(num_agents=20, num_nodes=51, seed=5)
    assert not data_equivalence(instance1, instance2)


def test_same_seed_benchmark_instance_generator(benchmark_instance_generator_fixture):
    instance1 = benchmark_instance_generator_fixture.sample_instance(num_agents=20, num_nodes=51, seed=1)
    instance2 = benchmark_instance_generator_fixture.sample_instance(num_agents=20, num_nodes=51, seed=1)
    assert data_equivalence(instance1, instance2)


