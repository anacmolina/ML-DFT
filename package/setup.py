import setuptools

long_description = """
Python package for sampling with real-nvp flows.
"""

setuptools.setup(
    name="abflowmc",
    version="0.0.1",
    author="Ana Molina Taborda, Marylou Gabrié, Olga Lopez-Acevedo, Pilar Cossio",
    author_email="anac.molina@udea.edu.co",
    description="python package for learning machine learning potential for ab-initio force fields potential and sampling with real-nvp flows",
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
