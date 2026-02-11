from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import sys
from pathlib import Path
import shutil

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        build_dir = Path(self.build_temp) / f"py{sys.version_info.major}{sys.version_info.minor}"
        if build_dir.exists():
            shutil.rmtree(build_dir)

        build_dir.mkdir(parents=True, exist_ok=True)

        subprocess.check_call([
            "cmake",
            "-S", "py",
            "-B", str(build_dir),
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
        ])

        subprocess.check_call([
            "cmake",
            "--build", str(build_dir),
            "--config", "Release",
        ])

        # Copy built module to expected location
        built_so = next(build_dir.glob("_core*.so"))
        target = Path(self.get_ext_fullpath(ext.name))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(built_so, target)

setup(
    name="pyrefinery",
    version="0.1.0",
    packages=["pyrefinery"],
    package_dir={"": "src"},
    ext_modules=[Extension("pyrefinery._core", sources=[])],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
