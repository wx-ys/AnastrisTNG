import setuptools, os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setuptools.setup(
    name="AnastrisTNG",
    version="1.2.0",
    author="Shuai Lu",
    author_email="lushuai@stu.xmu.edu.cn",
    description="IllustrisTNG simulation data analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wx-ys/AnastrisTNG",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license='MIT',
    packages=setuptools.find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires=">=3.8",
    install_requires=install_requires,
    include_package_data=True,
)