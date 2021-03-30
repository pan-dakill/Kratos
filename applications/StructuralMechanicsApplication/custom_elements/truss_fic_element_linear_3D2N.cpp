// KRATOS  ___|  |                   |                   |
//       \___ \  __|  __| |   |  __| __| |   |  __| _` | |
//             | |   |    |   | (    |   |   | |   (   | |
//       _____/ \__|_|   \__,_|\___|\__|\__,_|_|  \__,_|_| MECHANICS
//
//  License:     BSD License
//           license: structural_mechanics_application/license.txt
//
//  Main authors: Ignasi de Pouplana
//
//
//

// System includes

// External includes

// Project includes
#include "custom_elements/truss_fic_element_linear_3D2N.hpp"
#include "includes/define.h"
#include "structural_mechanics_application_variables.h"
#include "custom_utilities/structural_mechanics_element_utilities.h"

namespace Kratos {
TrussFICElementLinear3D2N::TrussFICElementLinear3D2N(IndexType NewId,
        GeometryType::Pointer pGeometry)
    : TrussElementLinear3D2N(NewId, pGeometry) {}

TrussFICElementLinear3D2N::TrussFICElementLinear3D2N(
    IndexType NewId, GeometryType::Pointer pGeometry,
    PropertiesType::Pointer pProperties)
    : TrussElementLinear3D2N(NewId, pGeometry, pProperties) {}


TrussFICElementLinear3D2N::~TrussFICElementLinear3D2N() {}

Element::Pointer
TrussFICElementLinear3D2N::Create(IndexType NewId,
                               NodesArrayType const& rThisNodes,
                               PropertiesType::Pointer pProperties) const
{
    const GeometryType& rGeom = GetGeometry();
    return Kratos::make_intrusive<TrussFICElementLinear3D2N>(
               NewId, rGeom.Create(rThisNodes), pProperties);
}

Element::Pointer
TrussFICElementLinear3D2N::Create(IndexType NewId,
                               GeometryType::Pointer pGeom,
                               PropertiesType::Pointer pProperties) const
{
    return Kratos::make_intrusive<TrussFICElementLinear3D2N>(
               NewId, pGeom, pProperties);
}

void TrussFICElementLinear3D2N::AddExplicitContribution(
    const VectorType& rRHSVector,
    const Variable<VectorType>& rRHSVariable,
    const Variable<double >& rDestinationVariable,
    const ProcessInfo& rCurrentProcessInfo
)
{
    KRATOS_TRY;

    auto& r_geom = GetGeometry();

    if (rDestinationVariable == NODAL_MASS) {

        VectorType element_mass_vector(msLocalSize);
        CalculateLumpedMassVector(element_mass_vector);

        // VectorType element_stiffness_vector(msLocalSize);
        // CalculateLumpedStiffnessVector(element_stiffness_vector,rCurrentProcessInfo);

        // VectorType element_damping_vector(msLocalSize);
        // CalculateLumpedDampingVector(element_damping_vector, rCurrentProcessInfo);

        for (SizeType i = 0; i < msNumberOfNodes; ++i) {
            double& r_nodal_mass = r_geom[i].GetValue(NODAL_MASS);
            // array_1d<double, 3>& r_nodal_stiffness = r_geom[i].GetValue(NODAL_DIAGONAL_STIFFNESS);
            // array_1d<double, 3>& r_nodal_damping = r_geom[i].GetValue(NODAL_DIAGONAL_DAMPING);
            int index = i * msDimension;

            #pragma omp atomic
            r_nodal_mass += element_mass_vector[index];

            // for (SizeType j = 0; j < msDimension; ++j) {
            //     #pragma omp atomic
            //     r_nodal_stiffness[j] += element_stiffness_vector[index+j];

            //     #pragma omp atomic
            //     r_nodal_damping[j] += element_damping_vector[index+j];
            // }
        }
    }

    KRATOS_CATCH("")
}

void TrussFICElementLinear3D2N::AddExplicitContribution(
    const VectorType& rRHSVector, const Variable<VectorType>& rRHSVariable,
    const Variable<array_1d<double, 3>>& rDestinationVariable,
    const ProcessInfo& rCurrentProcessInfo
)
{
    KRATOS_TRY;

    if (rRHSVariable == RESIDUAL_VECTOR && rDestinationVariable == MIDDLE_VELOCITY) {

        // internal_forces = Ka
        BoundedVector<double, msLocalSize> element_internal_forces = ZeroVector(msLocalSize);
        UpdateInternalForces(element_internal_forces);

        Matrix damping_matrix(msLocalSize,msLocalSize);
        CalculateDampingMatrix(damping_matrix, rCurrentProcessInfo);
        Vector current_disp = ZeroVector(msLocalSize);
        GetValuesVector(current_disp);
        Vector damping_force = ZeroVector(msLocalSize);
        noalias(damping_force) = prod(damping_matrix,current_disp);

        Vector external_forces(msLocalSize);
        noalias(external_forces) = rRHSVector + element_internal_forces;

        // Lumped mass matrix
        VectorType mass_vector(msLocalSize);
        CalculateLumpedMassVector(mass_vector);
        Matrix MassMatrix(msLocalSize,msLocalSize);
        noalias(MassMatrix) = ZeroMatrix(msLocalSize,msLocalSize);
        for (IndexType i = 0; i < msLocalSize; ++i)
            MassMatrix(i, i) = mass_vector[i];

        Vector current_nodal_accelerations = ZeroVector(msLocalSize);
        GetSecondDerivativesVector(current_nodal_accelerations);

        Vector inertial_vector = ZeroVector(msLocalSize);
        noalias(inertial_vector) = prod(MassMatrix,current_nodal_accelerations);

        Vector current_nodal_velocities = ZeroVector(msLocalSize);
        GetFirstDerivativesVector(current_nodal_velocities);

        Vector damping_vector = ZeroVector(msLocalSize);
        noalias(damping_vector) = prod(damping_matrix,current_nodal_velocities);

        // TODO
        MatrixType stiffness_matrix = ZeroMatrix(msLocalSize,msLocalSize);
        noalias(stiffness_matrix) = CreateElementStiffnessMatrix(rCurrentProcessInfo);
        KRATOS_WATCH("BEFORE ASSEMBLING ANYTHING")
        KRATOS_WATCH(this->Id())
        KRATOS_WATCH(stiffness_matrix)
        KRATOS_WATCH(mass_vector)
        KRATOS_WATCH(MassMatrix)
        KRATOS_WATCH(damping_matrix)
        KRATOS_WATCH(current_disp)
        KRATOS_WATCH(element_internal_forces)
        KRATOS_WATCH(damping_force)
        KRATOS_WATCH(external_forces)

        for (size_t i = 0; i < msNumberOfNodes; ++i) {
            size_t index = msDimension * i;

            array_1d<double, 3>& r_force_residual = GetGeometry()[i].FastGetSolutionStepValue(FORCE_RESIDUAL);
            array_1d<double, 3>& r_internal_force = GetGeometry()[i].FastGetSolutionStepValue(NODAL_INERTIA);
            array_1d<double, 3>& r_damping_force = GetGeometry()[i].FastGetSolutionStepValue(NODAL_DISPLACEMENT_STIFFNESS);
            array_1d<double, 3>& r_external_forces = GetGeometry()[i].FastGetSolutionStepValue(EXTERNAL_FORCE);

            for (size_t j = 0; j < msDimension; ++j) {

                // rRHSVector = f-Ka
                #pragma omp atomic
                r_force_residual[j] += rRHSVector[index + j] - inertial_vector[index + j] - damping_vector[index + j];

                #pragma omp atomic
                r_internal_force[j] += element_internal_forces[index + j];

                #pragma omp atomic
                r_damping_force[j] += damping_force[index + j];

                #pragma omp atomic
                r_external_forces[j] += external_forces[index + j];
            }
        }

    } else if (rRHSVariable == RESIDUAL_VECTOR && rDestinationVariable == FORCE_RESIDUAL) {

        Vector internal_force = ZeroVector(msLocalSize);
        Vector damping_force = ZeroVector(msLocalSize);
        Vector external_forces = ZeroVector(msLocalSize);

        for (size_t i = 0; i < msNumberOfNodes; ++i) {
            size_t index = msDimension * i;

            const array_1d<double, 3>& r_internal_force = GetGeometry()[i].FastGetSolutionStepValue(NODAL_INERTIA);
            const array_1d<double, 3>& r_damping_force = GetGeometry()[i].FastGetSolutionStepValue(NODAL_DISPLACEMENT_STIFFNESS);
            const array_1d<double, 3>& r_external_forces = GetGeometry()[i].FastGetSolutionStepValue(EXTERNAL_FORCE);
            

            for (size_t j = 0; j < msDimension; ++j) {

                internal_force[index+j] = r_internal_force[j]; // Assembled internal_force

                damping_force[index+j] = r_damping_force[j]; // Assembled damping_force

                external_forces[index+j] = r_external_forces[j]; // External Forces coming from condition, elements and reactions
            }
        }

        MatrixType H1( msLocalSize, msLocalSize );
        this->CalculateFrequencyMatrix(H1, rCurrentProcessInfo);

        Vector delta_internal_force = ZeroVector(msLocalSize);
        noalias(delta_internal_force) = prod(H1,internal_force);

        Vector delta_damping_force = ZeroVector(msLocalSize);
        noalias(delta_damping_force) = prod(H1,damping_force);

        Vector delta_external_force = ZeroVector(msLocalSize);
        noalias(delta_external_force) = prod(H1,external_forces);

        // TODO
        KRATOS_WATCH("AFTER ASSEMBLING MID FORCES WITH REACTIONS, BEFORE ASSEMBLING DELTA FORCES")
        KRATOS_WATCH(this->Id())
        KRATOS_WATCH(internal_force)
        KRATOS_WATCH(damping_force)
        KRATOS_WATCH(external_forces)
        KRATOS_WATCH(H1)
        KRATOS_WATCH(delta_internal_force)
        KRATOS_WATCH(delta_damping_force)
        KRATOS_WATCH(delta_external_force)

        for (size_t i = 0; i < msNumberOfNodes; ++i) {
            size_t index = msDimension * i;
            
            array_1d<double, 3>& r_delta_internal_force = GetGeometry()[i].FastGetSolutionStepValue(MIDDLE_VELOCITY); // H1Ka
            array_1d<double, 3>& r_delta_damping_force = GetGeometry()[i].FastGetSolutionStepValue(MIDDLE_ANGULAR_VELOCITY); // H1Ca
            array_1d<double, 3>& r_delta_external_force = GetGeometry()[i].FastGetSolutionStepValue(FRACTIONAL_ACCELERATION); // H1f

            for (size_t j = 0; j < msDimension; ++j) {

                #pragma omp atomic
                r_delta_internal_force[j] += delta_internal_force[index + j];

                #pragma omp atomic
                r_delta_damping_force[j] += delta_damping_force[index + j];

                #pragma omp atomic
                r_delta_external_force[j] += delta_external_force[index + j];
            }
        }
    }

    KRATOS_CATCH("")
}

int TrussFICElementLinear3D2N::Check(const ProcessInfo& rCurrentProcessInfo) const
{
    KRATOS_TRY

    int ierr = TrussElement3D2N::Check(rCurrentProcessInfo);
    if(ierr != 0) return ierr;

    // double alpha = 0.0;
    // if( GetProperties().Has(RAYLEIGH_ALPHA) )
    //     alpha = GetProperties()[RAYLEIGH_ALPHA];
    // else if( rCurrentProcessInfo.Has(RAYLEIGH_ALPHA) )
    //     alpha = rCurrentProcessInfo[RAYLEIGH_ALPHA];

    // double beta  = 0.0;
    // if( GetProperties().Has(RAYLEIGH_BETA) )
    //     beta = GetProperties()[RAYLEIGH_BETA];
    // else if( rCurrentProcessInfo.Has(RAYLEIGH_BETA) )
    //     beta = rCurrentProcessInfo[RAYLEIGH_BETA];

    // if( std::abs(alpha) < std::numeric_limits<double>::epsilon() &&
    //     std::abs(beta) < std::numeric_limits<double>::epsilon() ) {
    //     KRATOS_ERROR << "Rayleigh Alpha and Rayleigh Beta are zero and this element needs the damping matrix (estimated with the rayleigh method) to be different from zero." << std::endl;
    // }

    return ierr;

    KRATOS_CATCH("")
}

void TrussFICElementLinear3D2N::CalculateLumpedMassVector(VectorType& rMassVector)
{
    KRATOS_TRY

    // Clear matrix
    if (rMassVector.size() != msLocalSize) {
        rMassVector.resize(msLocalSize, false);
    }

    const double A = GetProperties()[CROSS_AREA];
    const double L = StructuralMechanicsElementUtilities::CalculateReferenceLength3D2N(*this);
    const double rho = GetProperties()[DENSITY];

    const double total_mass = A * L * rho;

    for (int i = 0; i < msNumberOfNodes; ++i) {
        for (int j = 0; j < msDimension; ++j) {
            int index = i * msDimension + j;

            rMassVector[index] = total_mass * 0.50;
        }
    }

    KRATOS_CATCH("")
}

void TrussFICElementLinear3D2N::CalculateLumpedStiffnessVector(VectorType& rStiffnessVector,const ProcessInfo& rCurrentProcessInfo)
{
    KRATOS_TRY

    // Clear Vector
    if (rStiffnessVector.size() != msLocalSize) {
        rStiffnessVector.resize(msLocalSize, false);
    }

    MatrixType stiffness_matrix( msLocalSize, msLocalSize );
    noalias(stiffness_matrix) = ZeroMatrix(msLocalSize,msLocalSize);
    ProcessInfo temp_process_information = rCurrentProcessInfo;
    noalias(stiffness_matrix) = CreateElementStiffnessMatrix(temp_process_information);
    for (IndexType i = 0; i < msLocalSize; ++i)
        rStiffnessVector[i] = stiffness_matrix(i,i);

    KRATOS_CATCH("")
}

void TrussFICElementLinear3D2N::CalculateLumpedDampingVector(
    VectorType& rDampingVector,
    const ProcessInfo& rCurrentProcessInfo
    )
{
    KRATOS_TRY;

    // Clear Vector
    if (rDampingVector.size() != msLocalSize) {
        rDampingVector.resize(msLocalSize, false);
    }
    noalias(rDampingVector) = ZeroVector(msLocalSize);

    // Rayleigh Damping Vector (C= alpha*M + beta*K)

    // Get Damping Coefficients (RAYLEIGH_ALPHA, RAYLEIGH_BETA)
    double alpha = 0.0;
    if( GetProperties().Has(RAYLEIGH_ALPHA) )
        alpha = GetProperties()[RAYLEIGH_ALPHA];
    else if( rCurrentProcessInfo.Has(RAYLEIGH_ALPHA) )
        alpha = rCurrentProcessInfo[RAYLEIGH_ALPHA];
    double beta  = 0.0;
    if( GetProperties().Has(RAYLEIGH_BETA) )
        beta = GetProperties()[RAYLEIGH_BETA];
    else if( rCurrentProcessInfo.Has(RAYLEIGH_BETA) )
        beta = rCurrentProcessInfo[RAYLEIGH_BETA];

    // 1.-Calculate mass Vector:
    if (alpha > std::numeric_limits<double>::epsilon()) {
        VectorType mass_vector(msLocalSize);
        CalculateLumpedMassVector(mass_vector);
        for (IndexType i = 0; i < msLocalSize; ++i)
            rDampingVector[i] += alpha * mass_vector[i];
    }

    // 2.-Calculate Stiffness Vector:
    if (beta > std::numeric_limits<double>::epsilon()) {
        VectorType stiffness_vector(msLocalSize);
        CalculateLumpedStiffnessVector(stiffness_vector,rCurrentProcessInfo);
        for (IndexType i = 0; i < msLocalSize; ++i)
            rDampingVector[i] += beta * stiffness_vector[i];
    }

    KRATOS_CATCH( "" )
}

void TrussFICElementLinear3D2N::CalculateFrequencyMatrix(MatrixType& rMatrix, const ProcessInfo& rCurrentProcessInfo)
{
    KRATOS_TRY

    // Clear matrix
    if (rMatrix.size1() != msLocalSize || rMatrix.size2() != msLocalSize) {
        rMatrix.resize(msLocalSize, msLocalSize, false);
    }
    rMatrix = ZeroMatrix(msLocalSize, msLocalSize);

    // Damping matrix
    Matrix damping_matrix;
    CalculateDampingMatrix(damping_matrix, rCurrentProcessInfo);

    // Inverse of lumped mass matrix
    VectorType mass_vector(msLocalSize);
    CalculateLumpedMassVector(mass_vector);
    Matrix MassMatrixInverse(msLocalSize,msLocalSize);
    noalias(MassMatrixInverse) = ZeroMatrix(msLocalSize,msLocalSize);
    for (IndexType i = 0; i < msLocalSize; ++i)
        MassMatrixInverse(i, i) = 1.0/mass_vector[i];

    // Identity matrix
    MatrixType IdentityMatrix(msLocalSize,msLocalSize);
    noalias(IdentityMatrix) = ZeroMatrix(msLocalSize,msLocalSize);
    for(unsigned int i = 0; i<msLocalSize; i++){
        IdentityMatrix(i,i) = 1.0;
    }

    const double delta = rCurrentProcessInfo[LOAD_FACTOR];
    const double delta_time = rCurrentProcessInfo[DELTA_TIME];

    MatrixType aux_matrix(msLocalSize,msLocalSize);
    noalias(aux_matrix) = prod(damping_matrix,MassMatrixInverse);
    noalias(rMatrix) = (1.0+delta)*IdentityMatrix - delta*delta_time*aux_matrix;

    KRATOS_CATCH("")
}

/***********************************************************************************/
/***********************************************************************************/

// void TrussFICElementLinear3D2N::GetAuxiliaryVelocityVector(Vector& rValues, int Step) const
// {

//     KRATOS_TRY
//     if (rValues.size() != msLocalSize) {
//         rValues.resize(msLocalSize, false);
//     }

//     for (int i = 0; i < msNumberOfNodes; ++i) {
//         int index = i * msDimension;
//         const auto& aux_vel =
//             GetGeometry()[i].FastGetSolutionStepValue(NODAL_DISPLACEMENT_STIFFNESS, Step);

//         rValues[index] = aux_vel[0];
//         rValues[index + 1] = aux_vel[1];
//         rValues[index + 2] = aux_vel[2];
//     }
//     KRATOS_CATCH("")
// }

void TrussFICElementLinear3D2N::save(Serializer& rSerializer) const
{
    KRATOS_SERIALIZE_SAVE_BASE_CLASS(rSerializer, TrussElementLinear3D2N);
    rSerializer.save("mConstitutiveLaw", mpConstitutiveLaw);
}
void TrussFICElementLinear3D2N::load(Serializer& rSerializer)
{
    KRATOS_SERIALIZE_LOAD_BASE_CLASS(rSerializer, TrussElementLinear3D2N);
    rSerializer.load("mConstitutiveLaw", mpConstitutiveLaw);
}

} // namespace Kratos.
