# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from iqcc_calibration_tools.quam_config.components.quam_root import Quam
from qualibrate import NodeParameters
from qualibration_libs.parameters import get_qubits
from typing import Optional, List
from qualibration_libs.data import XarrayDataFetcher

from iqcc_calibration_tools.storage.iqcc_cloud_data_storage_utils.download_state_and_wiring import download_state_and_wiring

# %% {Node initialisation}
description = """Init and shuffle the QUAM state for the Qualibrate challenge. """

class Parameters(NodeParameters):
    qubits: Optional[List[str]] = ["qD1", "qD2"]
    shuffle: bool=True

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="00_init_challenge",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)

# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubits = ["q1", "q2"]
    pass


# Instantiate the QUAM class from the state file
# node.machine = Quam.load("C:/git/SQA-2025-Conference/iqcc-calibration/reference_state_qolab/")
download_state_and_wiring("arbel") 
node.machine = Quam.load()


# %% {Shuffle_state}
@node.run_action()
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    node.namespace["qubits"] = get_qubits(node)
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.parameters.shuffle:
                q.xy.operations["x180"].alpha *= 0.0
                q.xy.operations["x180"].amplitude *= (np.random.randint(5,15) / 10)
                q.z.joint_offset += (np.random.randint(-75,75) / 10000)


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
