from setuptools import setup, find_packages
# Read in the requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='robo-instruct',
    version='0.1',
    packages=find_packages(['roboeval', 'robo_instruct']),
    install_requires=requirements,
    author='Zichao Hu',
    author_email='zichao@utexas.edu',
    description='Adapted RoboEval Implementation',
    url='https://github.com/ut-amrl/codebotler',
    python_requires='>=3.10',
)