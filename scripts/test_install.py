"""
Test that all of our modules can be imported
Thanks https://stackoverflow.com/a/25562415/10473080
and openscm-runner
"""
import importlib
import os.path
import pkgutil

import meteor

def import_submodules(package_name):
    package = importlib.import_module(package_name)

    for _, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + "." + name
        print(full_name)
        importlib.import_module(full_name)
        if is_pkg:
            import_submodules(full_name)


import_submodules("meteor")

# make sure input data etc. are included
meteor_root = os.path.dirname(meteor.__file__)
assert os.path.isfile(
    os.path.join(
        meteor_root, "default_scm_data", "gases_vupdate_2022_AR6.txt"
    )
)
assert os.path.isfile(
    os.path.join(
	meteor_root, "default_scm_data", "ssp245_conc_RCMIP.txt"
    )
)
assert os.path.isfile(
    os.path.join(
	meteor_root, "default_scm_data", "ssp245_em_RCMIP.txt"
    )
)

print(meteor.__version__)
