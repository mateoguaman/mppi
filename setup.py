import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mppi-torch-ros-mateoguaman",
    version="0.0.1",
    author="Mateo Guaman Castro",
    author_email="mguamanc@andrew.cmu.edu",
    description="MPPI package with PyTorch and ROS interfaces",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mateoguaman/mppi",
    project_urls={
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)