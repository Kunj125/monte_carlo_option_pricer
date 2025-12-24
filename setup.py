from setuptools import setup, Extension

try:
    import pybind11
    pybind11_include = pybind11.get_include()
except ImportError:
    pybind11_include = ''

cpp_args = ['/std:c++14']

ext_modules = [
    Extension(
        'market_engine',
        ['Bindings.cpp', 'Engine.cpp'],
        include_dirs=[pybind11_include],
        language='c++',
        extra_compile_args=cpp_args,
    ),
]

setup(
    name='market_engine',
    version='2.0',
    author='Kunj',
    description='Heston monte marlo engine in C++',
    ext_modules=ext_modules,
)
