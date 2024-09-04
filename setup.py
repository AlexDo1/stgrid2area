from setuptools import setup, find_packages


def readme():
    with open("README.md") as f:
        return f.read().strip()


def version():
    with open("stgrid2area/__version__.py") as f:
        loc = dict()
        exec(f.read(), loc, loc)
        return loc["__version__"]


def requirements():
    with open("requirements.txt") as f:
        return f.read().strip().split("\n")


setup(name="stgrid2area",
      license="CC0-1.0 license",
      version=version(),
      author="Alexander Dolich",
      author_email="alexander.dolich@kit.edu",
      description="Extract and aggregate spatio-temporal data to a specified area.",
      long_description=readme(),
      long_description_content_type="text/markdown",
      install_requires=requirements(),
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False
      )
