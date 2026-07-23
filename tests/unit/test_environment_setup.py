"""
Tests to verify the Python environment is correctly configured.
"""
import sys
import pytest


class TestPythonVersion:
    """Test that Python version meets requirements."""

    def test_python_version_minimum(self):
        """Verify Python version is >= 3.11."""
        assert sys.version_info >= (3, 11), (
            f"Python 3.11 or higher required, got {sys.version_info.major}.{sys.version_info.minor}"
        )

    def test_python_version_maximum(self):
        """Verify Python version is < 4.0."""
        assert sys.version_info < (4, 0), (
            f"Python version must be less than 4.0, got {sys.version_info.major}.{sys.version_info.minor}"
        )


class TestCoreImports:
    """Test that core package modules can be imported."""

    def test_import_maenvs4vrp(self):
        """Test that main package imports."""
        import maenvs4vrp
        assert maenvs4vrp is not None

    def test_import_core(self):
        """Test that core module imports."""
        import maenvs4vrp.core
        assert maenvs4vrp.core is not None

    def test_import_environments(self):
        """Test that environments module imports."""
        import maenvs4vrp.environments
        assert maenvs4vrp.environments is not None

    def test_import_neuro_solvers(self):
        """Test that neuro_solvers module imports."""
        import maenvs4vrp.neuro_solvers
        assert maenvs4vrp.neuro_solvers is not None

    def test_import_utils(self):
        """Test that utils module imports."""
        import maenvs4vrp.utils
        assert maenvs4vrp.utils is not None


class TestCriticalDependencies:
    """Test that critical dependencies are installed and meet version requirements."""

    def test_numpy_import(self):
        """Test that numpy can be imported."""
        import numpy as np
        assert np is not None

    def test_numpy_version(self):
        """Test that numpy version meets minimum requirement."""
        import numpy as np
        from packaging import version
        assert version.parse(np.__version__) >= version.parse("2.3.1"), (
            f"numpy >= 2.3.1 required, got {np.__version__}"
        )

    def test_torch_import(self):
        """Test that torch can be imported."""
        import torch
        assert torch is not None

    def test_torch_version(self):
        """Test that torch version meets minimum requirement."""
        import torch
        from packaging import version
        assert version.parse(torch.__version__.split('+')[0]) >= version.parse("2.7.0"), (
            f"torch >= 2.7.0 required, got {torch.__version__}"
        )

    def test_pandas_import(self):
        """Test that pandas can be imported."""
        import pandas as pd
        assert pd is not None

    def test_pandas_version(self):
        """Test that pandas version meets minimum requirement."""
        import pandas as pd
        from packaging import version
        assert version.parse(pd.__version__) >= version.parse("2.3.0"), (
            f"pandas >= 2.3.0 required, got {pd.__version__}"
        )

    def test_tensordict_import(self):
        """Test that tensordict can be imported."""
        import tensordict
        assert tensordict is not None

    def test_tensordict_version(self):
        """Test that tensordict version meets minimum requirement."""
        import tensordict
        from packaging import version
        assert version.parse(tensordict.__version__) >= version.parse("0.9.1"), (
            f"tensordict >= 0.9.1 required, got {tensordict.__version__}"
        )


class TestEnvironmentSamples:
    """Test that sample environments can be imported."""

    @pytest.mark.parametrize("env_name", [
        "cvrptw",
        "cvrpstw",
        "dvrptw",
        "dsvrptw",
        "mdvrptw",
        "pdptw",
        "sdvrptw",
        "toptw",
        "pcvrptw",
        "mtvrp",
        "mtdvrp",
        "gmtvrp",
        "gmtdvrp",
    ])
    def test_environment_module_imports(self, env_name):
        """Test that each environment module can be imported."""
        module_name = f"maenvs4vrp.environments.{env_name}"
        module = __import__(module_name, fromlist=[''])
        assert module is not None, f"Failed to import {module_name}"