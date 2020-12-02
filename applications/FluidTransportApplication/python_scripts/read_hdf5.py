import KratosMultiphysics
import KratosMultiphysics.FluidTransportApplication as KratosFluidTransport
import matplotlib.pyplot as plt
import h5py

from math import pi
from numpy import sin, cos, arctan, exp, arctan2
import numpy as np

def Factory(settings, Model):
    if(type(settings) != KratosMultiphysics.Parameters):
        raise Exception("expected input shall be a Parameters object, encapsulating a json string")
    return ReadHdf5(Model, settings["Parameters"])


class ReadHdf5(KratosMultiphysics.Process):
    def __init__(self, Model, settings ):
        KratosMultiphysics.Process.__init__(self)

        self.model_part = Model[settings["model_part_name"].GetString()]

        self.times = []

        self.f = h5py.File("data.hdf5", 'r')
        self.settings = settings

    def ExecuteInitializeSolutionStep(self):
        time = self.model_part.ProcessInfo[KratosMultiphysics.TIME]

        time_str = str(round(time,3))

        self.group_name = time_str

        i = 0
        # nodes = self.f[self.group_name + '/NODE'][:,]
        # temp = self.f[self.group_name + '/TEMP'][:,]
        vel_x = self.f[self.group_name + '/VEL_X'][:,]
        vel_y = self.f[self.group_name + '/VEL_Y'][:,]
        vel_z = self.f[self.group_name + '/VEL_Z'][:,]

        self.components_process_list = []

        for node in self.model_part.Nodes:
            if self.settings["active"][0].GetBool() == True:
                node.SetSolutionStepValue(KratosMultiphysics.VELOCITY_X,vel_x[i])
                node.Fix(KratosMultiphysics.VELOCITY_X)

            if self.settings["active"][1].GetBool() == True:
                node.SetSolutionStepValue(KratosMultiphysics.VELOCITY_Y,vel_y[i])
                node.Fix(KratosMultiphysics.VELOCITY_Y)

            if self.settings["active"][2].GetBool() == True:
                node.SetSolutionStepValue(KratosMultiphysics.VELOCITY_Z,vel_z[i])
                node.Fix(KratosMultiphysics.VELOCITY_Z)

            i += 1