import setuptools

setuptools.setup(
    name="cifar2png",
    version="0.0.4",
    author="Kenji Doi",
    author_email="knjcode@gmail.com",
    description="Convert CIFAR-10 and CIFAR-100 datasets into PNG images",
    long_description="Convert CIFAR-10 and CIFAR-100 datasets into PNG images",
    license="MIT",
    keywords="CIFAR-10 CIFAR-100 convert PNG",
    url="https://github.com/knjcode/cifar2png",
    packages=setuptools.find_packages(),
    scripts=['cifar2png'],
    install_requires=[
        'numpy',
        'pathlib',
        'Pillow>=6.2.0',
        'requests>=2.20.0',
        'six',
        'tqdm',
    ],
)
