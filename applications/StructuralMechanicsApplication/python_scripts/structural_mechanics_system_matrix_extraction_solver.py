# Importing the Kratos Library
import KratosMultiphysics

# Import applications
import KratosMultiphysics.StructuralMechanicsApplication as StructuralMechanicsApplication

# Import base class file
from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_solver import MechanicalSolver

# Import scipy modules
import KratosMultiphysics.scipy_conversion_tools
from scipy.sparse.linalg import eigsh

def CreateSolver(main_model_part, custom_settings):
    return SystemMatrixExtractionSolver(main_model_part, custom_settings)

class SystemMatrixExtractionSolver(MechanicalSolver):
    """The structural mechanics custom scipy base solver.

    This class creates the mechanical solvers to provide mass and stiffness matrices as scipy matrices.

    Derived class must override the function SolveSolutionStep. In there the Mass and Stiffness matrices
    can be obtained as scipy matrices.

    See structural_mechanics_solver.py for more information.
    """
    def __init__(self, main_model_part, custom_settings):
        # Construct the base solver.
        super().__init__(main_model_part, custom_settings)
        KratosMultiphysics.Logger.PrintInfo("::[CustomScipyBaseSolver]:: ", "Construction finished")

    @classmethod
    def GetDefaultParameters(cls):
        this_defaults = KratosMultiphysics.Parameters("""{
            "scheme_type"         : "dynamic"
        }""")
        this_defaults.AddMissingParameters(super().GetDefaultParameters())
        return this_defaults

    #### Private functions ####
    def _create_solution_scheme(self):
        """Create the scheme for the scipy solver.

        The scheme determines the mass and stiffness matrices
        """
        scheme_type = self.settings["scheme_type"].GetString()
        if scheme_type == "dynamic":
            solution_scheme = StructuralMechanicsApplication.EigensolverDynamicScheme()
        else: # here e.g. a stability scheme could be added
            err_msg =  "The requested scheme type \"" + scheme_type + "\" is not available!\n"
            err_msg += "Available options are: \"dynamic\""
            raise Exception(err_msg)

        return solution_scheme

    def _create_linear_solver(self):
        ''' Linear solver will not be used. But eventually the solution strategy calls the solver's clear function.
        To avoid crashing linear solver is provided here'''
        return KratosMultiphysics.LinearSolver()

    def _create_mechanical_solution_strategy(self):
        if self.settings["builder_and_solver_settings"]["use_block_builder"].GetBool():
            warn_msg = "In case an eigenvalue problem is computed an elimantion builder shall be used to ensure boundary conditions are applied correctly!"
            KratosMultiphysics.Logger.PrintWarning("CustomScipyBaseSolver", warn_msg)

        eigen_scheme = self.get_solution_scheme() # The scheme defines the matrices
        computing_model_part = self.GetComputingModelPart()
        builder_and_solver = self.get_builder_and_solver()

        return KratosMultiphysics.ResidualBasedLinearStrategy(computing_model_part,
                                                              eigen_scheme,
                                                              builder_and_solver,
                                                              False,
                                                              False,
                                                              False,
                                                              False )

    def _MatrixComputation(self, mat):
        if mat == "stiff":
            matID = 2
        elif mat == "mass":
            matID = 1
        space = KratosMultiphysics.UblasSparseSpace()
        self.GetComputingModelPart().ProcessInfo.SetValue(StructuralMechanicsApplication.BUILD_LEVEL, matID)
        scheme = self.get_mechanical_solution_strategy().GetScheme()

        aux = self.get_mechanical_solution_strategy().GetSystemMatrix()
        space.SetToZeroMatrix(aux)

        # Create dummy vectors
        b = space.CreateEmptyVectorPointer()
        space.ResizeVector( b, space.Size1(aux) )
        space.SetToZeroVector(b)

        xD = space.CreateEmptyVectorPointer()
        space.ResizeVector( xD, space.Size1(aux) )
        space.SetToZeroVector(xD)

        # Build matrix
        builder_and_solver = self.get_builder_and_solver()
        builder_and_solver.Build(scheme, self.GetComputingModelPart(), aux, b)
        # Apply constraints
        builder_and_solver.ApplyConstraints(scheme, self.GetComputingModelPart(), aux, b)
        # Apply boundary conditions
        builder_and_solver.ApplyDirichletConditions(scheme, self.GetComputingModelPart(), aux, xD, b)
        # Convert matrix to scipy
        return KratosMultiphysics.scipy_conversion_tools.to_csr(aux)


    def SolveSolutionStep(self):
        ## Obtain scipy matrices
        M = self._MatrixComputation("mass")
        K = self._MatrixComputation("stiff")
        #self.GetComputingModelPart().ProcessInfo.Set(StructuralMechanicsApplication.MODAL_MASS_MATRIX, M)     
        
        return True
