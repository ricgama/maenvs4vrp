import os
from setuptools import find_packages, setup
import sys
import subprocess

requirements = ['numpy==1.24.2', 'tqdm==4.66.1', 'pandas==2.1.3', 'ipykernel', 'ipywidgets', 
                'matplotlib==3.7.1', 'jupyter==1.0.0', 'torch==2.2.0', 'tensorboard==2.13.0', 'scipy', 'wandb',
                'ml_collections', 'pytest==7.4.3', 'tensordict==0.3.0', 'sphinx-math-dollar==1.2.1', 'nbsphinx==0.9.5', 'furo==2024.8.6', 'sphinx-copybutton==0.5.2']
setup(
    name='maenvs4vrp',
    packages=find_packages(where=['maenvs4vrp']),
    python_requires='>=3.11, <4',
    install_requires=requirements,
    version='0.1.0',
    description='Multi Agent Environments for Vehicle Routing Problems',
    author='mustelideos',
    license='',
)

def install_pandoc():
    try:
        subprocess.run(['pandoc', '--version'], check=True)
        print("Pandoc já está instalado.")
    except (OSError, subprocess.CalledProcessError):
        print("A instalar o Pandoc...")
        if sys.platform == 'linux':
            subprocess.run(['sudo', 'apt-get', 'install', '-y', 'pandoc'], check=True)
        elif sys.platform == 'darwin': 
            subprocess.run(['brew', 'install', 'pandoc'], check=True)
        elif sys.platform == 'win32': 
            subprocess.run(['choco', 'install', 'pandoc', '-y'], check=True)
        else:
            print(f"Plataforma {sys.platform} não suportada para instalação automática.")

if __name__ == "__main__":
    install_pandoc()