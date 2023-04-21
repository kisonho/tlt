from distutils.core import setup

from pathlib import Path
long_description = Path("README.md").read_text()

setup(name='transferring-lottery-ticket',
      version='0.9.2rc',
      description='Transferring Lottery Ticket (v0.9.2 Release Candidate)',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Kison Ho',
      author_email='unfit-gothic.0q@icloud.com',
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
