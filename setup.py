from distutils.core import setup

from pathlib import Path
long_description = Path("README.md").read_text()

setup(name='transferring-lottery-ticket',
      version='1.0',
      description='Transferring Lottery Ticket (v1.0)',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=[
            'tlt',
            'tlt.callbacks',
            'tlt.schedulers'
      ],
      package_dir={
            'tlt': 'tlt',
            'tlt.callbacks': 'tlt/callbacks',
            'tlt.schedulers': 'tlt/schedulers'
      },
      requires=[
            "torchmanager"
      ],
      python_requires=">=3.8",
      )
