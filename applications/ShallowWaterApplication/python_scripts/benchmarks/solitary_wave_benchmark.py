import KratosMultiphysics as KM

from KratosMultiphysics.ShallowWaterApplication.benchmarks.base_benchmark_process import BaseBenchmarkProcess
from KratosMultiphysics.ShallowWaterApplication.utilities.wave_factory import SolitaryWaveFactory

def Factory(settings, model):
    if not isinstance(settings, KM.Parameters):
        raise Exception("expected input shall be a Parameters object, encapsulating a json string")
    return SolitaryWaveBenchmark(model, settings["Parameters"])

class SolitaryWaveBenchmark(BaseBenchmarkProcess):
    """Solitary wave benchmark.

    Propagation of a solitary wave along the x axis.
    """

    def ExecuteInitialize(self):
        # Costruction of the wave settings
        benchmark_settings = self.settings["benchmark_settings"]
        self.boundary_model_part = self.model.GetModelPart(benchmark_settings["boundary_model_part_name"].GetString())
        process_info = self.boundary_model_part.ProcessInfo
        self.use_depth_from_model_part = benchmark_settings["use_depth_from_model_part"].GetBool()
        if self.use_depth_from_model_part:
            self.wave = SolitaryWaveFactory(self.boundary_model_part, benchmark_settings, process_info)
        else:
            depth = benchmark_settings["depth"].GetDouble()
            self.wave = SolitaryWaveFactory(depth, benchmark_settings, process_info)
        self.x_shift = benchmark_settings["x_shift"].GetDouble()
        self.t_shift = benchmark_settings["t_shift"].GetDouble()

        # Here the base class set the topography and initial conditions
        super().ExecuteInitialize()


    @classmethod
    def _GetBenchmarkDefaultSettings(cls):
        return KM.Parameters("""
            {
                "boundary_model_part_name"  : "",
                "wave_theory"               : "boussinesq",
                "use_depth_from_model_part" : true,
                "depth"                     : 1.0,
                "amplitude"                 : 1.0,
                "x_shift"                   : 0.0,
                "t_shift"                   : 0.0
            }
            """
            )


    def _Topography(self, coordinates):
        if self.use_depth_from_model_part:
            return 0
        else:
            return -self.wave.depth


    def _FreeSurfaceElevation(self, coordinates, time):
        x = coordinates.X
        return self.wave.eta(x - self.x_shift, time - self.t_shift)


    def _Height(self, coordinates, time):
        return self._FreeSurfaceElevation(coordinates, time) + self.wave.depth


    def _Velocity(self, coordinates, time):
        x = coordinates.X
        u_x = self.wave.u(x - self.x_shift, time - self.t_shift)
        return [u_x, 0.0, 0.0]


    def ExecuteInitializeSolutionStep(self):
        time = self.boundary_model_part.ProcessInfo[KM.TIME]
        for node in self.boundary_model_part.Nodes:
            node.SetSolutionStepValue(KM.VELOCITY, self._Velocity(node, time))
        KM.VariableUtils().ApplyFixity(KM.VELOCITY_X, True, self.boundary_model_part.Nodes)


    def ExecuteFinalizeSolutionStep(self):
        KM.VariableUtils().ApplyFixity(KM.VELOCITY_X, False, self.boundary_model_part.Nodes)