//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ `
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:         BSD License
//                   Kratos default license: kratos/license.txt
//
//  Main authors:    Ignasi de Pouplana
//

// Project includes
#include "central_differences_scheme.h"

namespace Kratos {

    void CentralDifferencesScheme::SetTranslationalIntegrationSchemeInProperties(Properties::Pointer pProp, bool verbose) const {
//         if(verbose) KRATOS_INFO("DEM") << "Assigning CentralDifferencesScheme to properties " << pProp->Id() << std::endl;
        pProp->SetValue(DEM_TRANSLATIONAL_INTEGRATION_SCHEME_POINTER, this->CloneShared());
    }

    void CentralDifferencesScheme::SetRotationalIntegrationSchemeInProperties(Properties::Pointer pProp, bool verbose) const {
//         if(verbose) KRATOS_INFO("DEM") << "Assigning CentralDifferencesScheme to properties " << pProp->Id() << std::endl;
        pProp->SetValue(DEM_ROTATIONAL_INTEGRATION_SCHEME_POINTER, this->CloneShared());
    }

    void CentralDifferencesScheme::UpdateTranslationalVariables(
            int StepFlag,
            Node < 3 >& i,
            array_1d<double, 3 >& coor,
            array_1d<double, 3 >& displ,
            array_1d<double, 3 >& delta_displ,
            array_1d<double, 3 >& vel,
            const array_1d<double, 3 >& initial_coor,
            const array_1d<double, 3 >& force,
            const double force_reduction_factor,
            const double mass,
            const double delta_t,
            const bool Fix_vel[3]) {

        const double& alpha = i.FastGetSolutionStepValue(RAYLEIGH_ALPHA);
        const double& beta = i.FastGetSolutionStepValue(RAYLEIGH_BETA);
        array_1d<double,3>& displ_old = i.FastGetSolutionStepValue(DISPLACEMENT_OLD);
        const array_1d<double,3>& internal_force = i.FastGetSolutionStepValue(INTERNAL_FORCE);
        const array_1d<double,3>& internal_force_old = i.FastGetSolutionStepValue(INTERNAL_FORCE_OLD);
        const array_1d<double,3>& external_force = i.FastGetSolutionStepValue(INTERNAL_FORCE);
        array_1d<double,3> displ_aux = ZeroVector(3);

        double mass_inv = 1.0 / mass;
        for (int k = 0; k < 3; k++) {
            if (Fix_vel[k] == false) {
                // TODO. Ignasi: seguir
                displ_aux[k] = (2.0-delta_t*alpha)*mass;
                vel[k] += delta_t * force_reduction_factor * force[k] * mass_inv;
                delta_displ[k] = delta_t * vel[k];
                displ[k] += delta_displ[k];
                coor[k] = initial_coor[k] + displ[k];
            } else {
                delta_displ[k] = delta_t * vel[k];
                displ[k] += delta_displ[k];
                coor[k] = initial_coor[k] + displ[k];
            }
        } // dimensions
    }

    void CentralDifferencesScheme::CalculateNewRotationalVariablesOfSpheres(
                int StepFlag,
                Node < 3 >& i,
                const double moment_of_inertia,
                array_1d<double, 3 >& angular_velocity,
                array_1d<double, 3 >& torque,
                const double moment_reduction_factor,
                array_1d<double, 3 >& rotated_angle,
                array_1d<double, 3 >& delta_rotation,
                const double delta_t,
                const bool Fix_Ang_vel[3]) {

        array_1d<double, 3 > angular_acceleration;
        CalculateLocalAngularAcceleration(moment_of_inertia, torque, moment_reduction_factor, angular_acceleration);

        UpdateRotationalVariables(StepFlag, i, rotated_angle, delta_rotation, angular_velocity, angular_acceleration, delta_t, Fix_Ang_vel);
    }

    void CentralDifferencesScheme::CalculateNewRotationalVariablesOfRigidBodyElements(
                int StepFlag,
                Node < 3 >& i,
                const array_1d<double, 3 > moments_of_inertia,
                array_1d<double, 3 >& angular_velocity,
                array_1d<double, 3 >& torque,
                const double moment_reduction_factor,
                array_1d<double, 3 >& rotated_angle,
                array_1d<double, 3 >& delta_rotation,
                Quaternion<double  >& Orientation,
                const double delta_t,
                const bool Fix_Ang_vel[3]) {

        array_1d<double, 3 >& local_angular_velocity = i.FastGetSolutionStepValue(LOCAL_ANGULAR_VELOCITY);

        array_1d<double, 3 > local_angular_acceleration, local_torque, angular_acceleration;

        GeometryFunctions::QuaternionVectorGlobal2Local(Orientation, torque, local_torque);
        GeometryFunctions::QuaternionVectorGlobal2Local(Orientation, angular_velocity, local_angular_velocity);
        CalculateLocalAngularAccelerationByEulerEquations(local_angular_velocity, moments_of_inertia, local_torque, moment_reduction_factor, local_angular_acceleration);
        GeometryFunctions::QuaternionVectorLocal2Global(Orientation, local_angular_acceleration, angular_acceleration);

        UpdateRotationalVariables(StepFlag, i, rotated_angle, delta_rotation, angular_velocity, angular_acceleration, delta_t, Fix_Ang_vel);

        double ang = DEM_INNER_PRODUCT_3(delta_rotation, delta_rotation);

        if (ang) {
            GeometryFunctions::UpdateOrientation(Orientation, delta_rotation);
        } //if ang
        GeometryFunctions::QuaternionVectorGlobal2Local(Orientation, angular_velocity, local_angular_velocity);
    }

    void CentralDifferencesScheme::UpdateRotationalVariables(
                int StepFlag,
                Node < 3 >& i,
                array_1d<double, 3 >& rotated_angle,
                array_1d<double, 3 >& delta_rotation,
                array_1d<double, 3 >& angular_velocity,
                array_1d<double, 3 >& angular_acceleration,
                const double delta_t,
                const bool Fix_Ang_vel[3]) {

        for (int k = 0; k < 3; k++) {
            if (Fix_Ang_vel[k] == false) {
                angular_velocity[k] += delta_t * angular_acceleration[k];
                delta_rotation[k] = angular_velocity[k] * delta_t;
                rotated_angle[k] += delta_rotation[k];
            } else {
                delta_rotation[k] = angular_velocity[k] * delta_t;
                rotated_angle[k] += delta_rotation[k];
            }
        }
    }

    void CentralDifferencesScheme::CalculateLocalAngularAcceleration(
                const double moment_of_inertia,
                const array_1d<double, 3 >& torque,
                const double moment_reduction_factor,
                array_1d<double, 3 >& angular_acceleration) {

        double moment_of_inertia_inv = 1.0 / moment_of_inertia;
        for (int j = 0; j < 3; j++) {
            angular_acceleration[j] = moment_reduction_factor * torque[j] * moment_of_inertia_inv;
        }
    }

    void CentralDifferencesScheme::CalculateLocalAngularAccelerationByEulerEquations(
                const array_1d<double, 3 >& local_angular_velocity,
                const array_1d<double, 3 >& moments_of_inertia,
                const array_1d<double, 3 >& local_torque,
                const double moment_reduction_factor,
                array_1d<double, 3 >& local_angular_acceleration) {

        for (int j = 0; j < 3; j++) {
            local_angular_acceleration[j] = (local_torque[j] - (local_angular_velocity[(j + 1) % 3] * moments_of_inertia[(j + 2) % 3] * local_angular_velocity[(j + 2) % 3] - local_angular_velocity[(j + 2) % 3] * moments_of_inertia[(j + 1) % 3] * local_angular_velocity[(j + 1) % 3])) / moments_of_inertia[j];
            local_angular_acceleration[j] = local_angular_acceleration[j] * moment_reduction_factor;
        }
    }
} //namespace Kratos
