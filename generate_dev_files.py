"""A script to generate all development files necessary for the image filtering demo."""

import shutil
from common import AVAILABLE_FILTERS, FILTERS_PATH
from filters import Filter
from client_server_interface import FHEDev

print("Generating deployment files for all available filters")

for filter_name in AVAILABLE_FILTERS:
    print("Filter:", filter_name, "\n")

    filter = Filter(filter_name)
    filter.compile()

    deployment_path = FILTERS_PATH / (filter_name + "/deployment")

    if deployment_path.is_dir():
        shutil.rmtree(deployment_path)

    fhe_dev_filter = FHEDev(filter, deployment_path)
    fhe_dev_filter.save()

print("Done !")