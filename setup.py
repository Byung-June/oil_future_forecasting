from setuptools import setup, find_packages

setup(
    name='oil-future-forecasting',
    version='0.0.1',
    description='oil future forecasting',
    author='Taeyoon Kim, Byungjune Kim',
    author_email='lkjs112@postech.ac.kr, kbj219@postech.ac.kr',
    url='https://github.com/Byung-June/oil_future_forecasting',
    packages=find_packages(),
    install_requires=['pandas', 'numpy', 'pmdarima',
                      'tqdm', 'arch', 'scikit-learn',
                      'pandas-datareader', 'scikit-image', 'openpyxl', 'xlrd']
)
