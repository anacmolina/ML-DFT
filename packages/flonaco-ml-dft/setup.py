import setuptools

long_description = """
Python package for ab-flowMC simulations.
"""

setuptools.setup(
    name="flonacomldft",
    version="0.0.1",
    author="Ana Molina Taborda, Marylou Gabrié",
    author_email="anac.molina@udea.edu.co, marylou.gabrie@polytechnique.edu",
    description="Python package for ab-flowMC simulations",
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
