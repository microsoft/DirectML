import os
import re
import sys
import platform
import subprocess
import requests
import zipfile
import json
import shutil

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

dml_feed_url = 'https://api.nuget.org/v3/index.json'
dml_resource_id = 'microsoft.ai.directml'
dml_resource_version = '1.9.1'

dependency_dir = 'dependencies'
dml_bin_path = f'{dependency_dir}/{dml_resource_id}.{dml_resource_version}/bin/x64-win/'
lib_dir = '..\libraries'
base_path = os.path.dirname(os.path.realpath(__file__))
dependency_path = os.path.join(base_path, dependency_dir)

dml_resource_name = '.'.join([dml_resource_id, dml_resource_version])
dml_path = '%s\%s' % (dependency_path, dml_resource_name)

dmlx = 'DirectMLX'
dmlx_file = dmlx + '.h'
dmlx_source_path = '%s\%s' % (os.path.join(base_path, lib_dir), dmlx_file)
dmlx_path = os.path.join(dependency_path, dmlx)

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

            # gather dependencies
            if not os.path.exists(dependency_path):
                os.makedirs(dependency_path)

            self.download_nupkg(dml_feed_url, dml_resource_id, dml_resource_version, dml_path)

            if not os.path.exists(dmlx_path):
                os.makedirs(dmlx_path)

            shutil.copyfile(dmlx_source_path, os.path.join(dmlx_path, dmlx_file))

        for ext in self.extensions:
            self.build_extension(ext)

    def download_nupkg(self, feed_url, resource_id, resource_version, resource_path):
        if not os.path.exists(resource_path):
            url = self.get_resource_url(feed_url, resource_id, resource_version)
            if url:
                print('downloading ' + url)
                # download the package
                resource_file = resource_path + '.nupkg'
                with open(resource_file, 'wb') as file:
                    result = requests.get(url, stream=True)
                    for block in result.iter_content(1024):
                        file.write(block)

                if os.path.exists(resource_file):
                    # nupkg is just a zip, unzip it
                    with zipfile.ZipFile(resource_file, "r") as zip_ref:
                        zip_ref.extractall(resource_path)
                    os.remove(resource_file)

    def get_resource_url(self, feed_url, resource_id, resource_version):
        index = requests.get(feed_url)
        resources = json.loads(index.text)['resources']

        for resource in resources:
            if resource['@type'] == 'PackageBaseAddress/3.0.0':
                return resource['@id'] + '/'.join([resource_id, resource_version]) + '/' + '.'.join([resource_id, resource_version]) + '.nupkg'

        return ''                

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            cmake_args += ['-DDML_PATH={}'.format(dml_path)]
            cmake_args += ['-DDMLX_PATH={}'.format(dmlx_path)]
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

setup(
    name='pydirectml',
    version='1.0.0',
    author='Microsoft Corporation',
    author_email='askdirectml@microsoft.com',
    description='Python Binding for DirectML Samples',
    long_description='PyDirectML is a small Python binding library for DirectML written to facilitate DirectML sample authoring in Python. It simplifies DirectML graph authoring and execution with automatic resource management and binding support through NumPy arrays.',
    url="https://github.com/microsoft/directml",
    license='MIT',
    python_requires='>=3.6',
    ext_modules=[CMakeExtension('pydirectml')],
    cmdclass=dict(build_ext=CMakeBuild),
    keywords='DirectML Python samples',
    setup_requires=['cmake', 'requests'],
    data_files=[('', ['/'.join([dml_bin_path, 'directml.dll']), '/'.join([dml_bin_path, 'directml.debug.dll'])])],
    zip_safe=False
)
