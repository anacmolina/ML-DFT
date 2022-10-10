import setuptools

long_description = """
Python package for sampling with real-nvp flows.
"""

setuptools.setup(
    name="flonacomldft",
    version="0.0.1",
    author="Ana Molina Taborda, Marylou Gabrié",
    author_email="marylou.gabrie@polytechnique.edu",
    description="python package for sampling ab-initio force fields potential with real-nvp flows",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
