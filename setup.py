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

Options.docstrings = False

setuptools.setup(
    name="simuclustfactor",
    version='0.0.1',
    author="Ablordeppey Prosper",
    author_email="prablordeppey@gmail.com",
    description="Simultaneous Component and Clustering Models for Three-way Data: Within and Between Approaches.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/prablordeppey/simuclustfactor",
    project_urls={
        "Bug Tracker": "https://github.com/prablordeppey/simuclustfactor/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(where="simuclustfactor"),
    # packages=['simuclustfactor'],
    # package_dir={"": ""},
    python_requires=">=3.6",
    install_requires=['numpy>=1.19.2', 'tabulate>=0.8.9'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],

    # ext_modules=cythonize(
    #     get_ext_paths('simuclustfactor', EXCLUDE_FILES),
    #     compiler_directives={'language_level': 3}
    # ),

    license='MIT',
)


