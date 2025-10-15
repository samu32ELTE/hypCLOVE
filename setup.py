from setuptools import setup, find_packages

setup(
    name="hypCLOVE",
    version="1.0.0",
    description="Description of hypCLOVE",
    author="Sámuel G. Balogh",
    author_email="balogh@hal.elte.hu",
    url="https://github.com/samu32ELTE/hypCLOVE",
    packages=find_packages(),  # Automatically find the package folder
    install_requires=[
        "igraph>=0.10.4",
        "leidenalg",
        "networkx>=2.6.3",
        "numpy>=1.22.2",
        "python-louvain>=0.16",
        "scipy>=1.8.0"
    ],
)


