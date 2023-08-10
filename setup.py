# Define the contents of the setup.py script

setup_content = """
from setuptools import setup, find_packages

setup(
    name='suitability_model_pkg',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn'
    ],
    package_data={
        '': ['*.pkl']
    },
    include_package_data=True,
)

"""