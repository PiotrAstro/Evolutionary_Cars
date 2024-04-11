from setuptools import setup, Extension
from Cython.Build import cythonize
import glob
import os
import sys


# To build and clean, copy this to terminal:
# python setup.py clean --all build_ext --inplace
# To just build, without clean:
# python setup.py build_ext --inplace

# Search for all .pyx files in subdirectories
pyx_files = glob.glob('**/*.pyx', recursive=True)

# Function to convert file path to module name
def filepath_to_modulename(file_path):
    base, _ = os.path.splitext(file_path)
    return base.replace(os.sep, '.')

# Create a list of Extension objects for each .pyx file
extensions = [
    Extension(
        name=filepath_to_modulename(pyx_file),
        sources=[pyx_file],
        extra_compile_args=["/Ox"],
    ) for pyx_file in pyx_files
]

# Setup configuration
setup(
    name="Stock_Agent",
    ext_modules=cythonize(
        extensions,
        annotate=True,
        compiler_directives={
            'language_level': "3",
        },
    ),
)