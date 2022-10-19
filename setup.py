import setuptools, os
from Cython.Build import cythonize
# from setuptools_cythonize import get_cmdclass

with open("README.md", "r") as f:
    long_description = f.read()

def get_ext_paths(root_dir, exclude_files):
    """get filepaths for cython compilation"""
    paths = []

    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            
            if filename in exclude_files:
                continue

            if os.path.splitext(filename)[1] != '.py':
                continue

            file_path = os.path.join(root, filename)
            if file_path in exclude_files:
                continue

            paths.append(file_path)

    return paths

# exclude files from cython compilation
EXCLUDE_FILES = [
    '__init__.py'
]

from Cython.Compiler import Options

setuptools.setup(
    name="simuclustfactor",
    version='0.0.3',
    author="Prosper Ablordeppey, Adelaide Freitas, Giorgia Zaccaria",
    author_email="prablordeppey@gmail.com, adelaide@ua.pt, giorgia.zaccaria@unitelmasapienza.it",
    maintainer="Prosper Ablordeppey",
    maintainer_email="prablordeppey@gmail.com, ",
    Description="Implements two iterative techniques called T3Clus and 3Fkmeans, aimed at simultaneously clustering objects and a factorial dimensionality reduction of variables and occasions on three-mode datasets developed by Vichi et al. (2007) <doi:10.1007/s00357-007-0006-x>. Also, we provide a convex combination of these two simultaneous procedures called CT3Clus and based on a hyperparameter alpha (alpha in [0,1], with 3FKMeans for alpha=0 and T3Clus for alpha= 1) also developed by Vichi et al. (2007) <doi:10.1007/s00357-007-0006-x>. Furthermore, we implemented the traditional tandem procedures of T3Clus (TWCFTA) and 3FKMeans (TWFCTA) for sequential clustering-factorial decomposition (TWCFTA), and vice-versa (TWFCTA) proposed by P. Arabie and L. Hubert (1996) <doi:10.1007/978-3-642-79999-0_1>.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/prablordeppey/simuclustfactor-python",
    project_urls={
        "Bug Tracker": "https://github.com/prablordeppey/simuclustfactor-python/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(where="simuclustfactor"),
    python_requires=">=3.6",
    install_requires=['numpy>=1.19.2', 'tabulate>=0.8.9'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],

    # ext_modules=cythonize(
    #     get_ext_paths('simuclustfactor', EXCLUDE_FILES),
    #     compiler_directives={'language_level': 3}
    # ),

    license='MIT'
)


