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
    return WriteHdf5(Model, settings["Parameters"])


class WriteHdf5(KratosMultiphysics.Process):
    def __init__(self, Model, settings ):
        KratosMultiphysics.Process.__init__(self)

        self.model_part = Model[settings["model_part_name"].GetString()]

        self.times = []

        self.f = h5py.File("data.hdf5", 'w')

    def DeleteDataSet(self, file_or_group, dset_name):
        if dset_name in file_or_group:
            file_or_group.__delitem__(dset_name)

    def WriteDataToFile(self, file_or_group, names, data, dtype):
        self.sub_group = self.CreateGroup(file_or_group, self.group_name)
        for name, datum in zip(names, data):
            self.DeleteDataSet(file_or_group, name)
        for name, datum in zip(names, data):
            self.sub_group.create_dataset(name = name, data = datum, dtype = dtype)

    def CreateGroup(self, file_or_group, name, overwrite_previous = True):
        if name in file_or_group:
            if overwrite_previous:
                file_or_group['/'].__delitem__(name)
            else:
                return file_or_group['/' + name]
        return file_or_group.create_group(name)

    def ExecuteFinalizeSolutionStep(self):
        time = self.model_part.ProcessInfo[KratosMultiphysics.TIME]

        time_str = str(round(time,3))

        self.group_name = time_str

        temp = 0.0
        i = 0
        #self.temp_list = []
        self.vel_x_list = []
        self.vel_y_list = []
        self.vel_z_list = []
        self.node_id_list = []

        for node in self.model_part.Nodes:
            node_id = node.Id
            #temp = node.GetSolutionStepValue(KratosMultiphysics.TEMPERATURE)
            vel_x = node.GetSolutionStepValue(KratosMultiphysics.VELOCITY_X)
            vel_y = node.GetSolutionStepValue(KratosMultiphysics.VELOCITY_Y)
            vel_z = node.GetSolutionStepValue(KratosMultiphysics.VELOCITY_Z)

            self.node_id_list.append (node_id)
            #self.temp_list.append (temp)
            self.vel_x_list.append (vel_x)
            self.vel_y_list.append (vel_y)
            self.vel_z_list.append (vel_z)
            i += 1

        # self.WriteDataToFile(file_or_group = self.f,
        #                     names = ['NODE', 'TEMP', 'VEL_X', 'VEL_Y', 'VEL_Z'],
        #                     data = [self.node_id_list, self.temp_list, self.vel_x_list, self.vel_y_list, self.vel_z_list],
        #                     dtype = 'float16')
        self.WriteDataToFile(file_or_group = self.f,
                            names = ['NODE', 'VEL_X', 'VEL_Y', 'VEL_Z'],
                            data = [self.node_id_list, self.vel_x_list, self.vel_y_list, self.vel_z_list],
                            dtype = 'float32')