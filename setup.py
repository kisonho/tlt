from distutils.core import setup

from pathlib import Path
long_description = Path("README.md").read_text()

setup(name='transferring-lottery-ticket',
      version='0.9.1rc',
      description='Transferring Lottery Ticket (v0.9.1 Release Candidate)',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Kison Ho',
      author_email='unfit-gothic.0q@icloud.com',
      packages=[
            'torchmanager_dynamic_pruning',
            'torchmanager_dynamic_pruning.callbacks',
            'torchmanager_dynamic_pruning.schedulers'
      ],
      package_dir={
            'torchmanager_dynamic_pruning': 'torchmanager_dynamic_pruning',
            'torchmanager_dynamic_pruning.callbacks': 'torchmanager_dynamic_pruning/callbacks',
            'torchmanager_dynamic_pruning.schedulers': 'torchmanager_dynamic_pruning/schedulers'
      },
      requires=[
            "torchmanager"
      ],
      python_requires=">=3.8",
      )
