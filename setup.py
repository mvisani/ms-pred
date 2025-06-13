import numpy as np
from setuptools import find_packages, setup

# Below is a setup.py call that is failing to correctly compile the cython when i run "python setup.py develop"
# This is because it tries to put the compiled code here 'src/massformer_code/', but that doesn't exist
setup(
    include_dirs=[np.get_include()],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    version="0.0.1",
)

# setup(name='ms_pred',
#      packages=find_packages(where="src"),
#      package_dir={'': "src"}
#      )
#
