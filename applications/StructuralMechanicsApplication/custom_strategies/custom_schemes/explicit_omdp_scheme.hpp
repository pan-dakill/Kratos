//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ `
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:		 BSD License
//					 Kratos default license:
//kratos/license.txt
//
//  Main authors:    Ignasi de Pouplana
//
//

#if !defined(KRATOS_EXPLICIT_OMDP_SCHEME_HPP_INCLUDED)
#define KRATOS_EXPLICIT_OMDP_SCHEME_HPP_INCLUDED

/* System includes */
#include <fstream>

/* External includes */

/* Project includes */
#include "custom_strategies/custom_schemes/explicit_cd_scheme.hpp"
#include "utilities/variable_utils.h"
#include "custom_utilities/explicit_integration_utilities.h"

namespace Kratos {

///@name Kratos Globals
///@{

///@}
///@name Type Definitions
///@{

///@}

///@name  Enum's
///@{

///@}
///@name  Functions
///@{

///@}
///@name Kratos Classes
///@{

/**
 * @class ExplicitOMDPScheme
 * @ingroup StructuralMechanicsApplciation
 * @brief An explicit forward euler scheme with a split of the inertial term
 * @author Ignasi de Pouplana
 */
template <class TSparseSpace,
          class TDenseSpace //= DenseSpace<double>
          >
class ExplicitOMDPScheme
    : public ExplicitCDScheme<TSparseSpace, TDenseSpace> {

public:
    ///@name Type Definitions
    ///@{

    /// The definition of the base type
    typedef ExplicitCDScheme<TSparseSpace, TDenseSpace> BaseType;

    /// Some definitions related with the base class
    typedef typename BaseType::DofsArrayType DofsArrayType;
    typedef typename BaseType::TSystemMatrixType TSystemMatrixType;
    typedef typename BaseType::TSystemVectorType TSystemVectorType;
    typedef typename BaseType::LocalSystemVectorType LocalSystemVectorType;

    /// The arrays of elements and nodes
    typedef ModelPart::ElementsContainerType ElementsArrayType;
    typedef ModelPart::NodesContainerType NodesArrayType;

    /// Definition of the size type
    typedef std::size_t SizeType;

    /// Definition of the index type
    typedef std::size_t IndexType;

    /// Definition fo the node iterator
    typedef typename ModelPart::NodeIterator NodeIterator;

    /// The definition of the numerical limit
    static constexpr double numerical_limit = std::numeric_limits<double>::epsilon();

    using BaseType::mDeltaTime;
    using BaseType::mAlpha;
    using BaseType::mBeta;
    using BaseType::mTheta1;

    /// Counted pointer of ExplicitOMDPScheme
    KRATOS_CLASS_POINTER_DEFINITION(ExplicitOMDPScheme);

    ///@}
    ///@name Life Cycle
    ///@{

    /**
     * @brief Default constructor.
     * @details The ExplicitOMDPScheme method
     */
    ExplicitOMDPScheme()
        : ExplicitCDScheme<TSparseSpace, TDenseSpace>() {}

    /** Destructor.
    */
    virtual ~ExplicitOMDPScheme() {}

    ///@}
    ///@name Operators
    ///@{

    /**
     * @brief This function is designed to be called once to perform all the checks needed
     * on the input provided.
     * @details Checks can be "expensive" as the function is designed
     * to catch user's errors.
     * @param rModelPart The model of the problem to solve
     * @return Zero means  all ok
     */
    int Check(const ModelPart& rModelPart) const override
    {
        KRATOS_TRY;

        BaseType::Check(rModelPart);

        KRATOS_ERROR_IF(rModelPart.GetBufferSize() < 2) << "Insufficient buffer size for OMDP Scheme. It has to be >= 2" << std::endl;

        KRATOS_ERROR_IF_NOT(rModelPart.GetProcessInfo().Has(DOMAIN_SIZE)) << "DOMAIN_SIZE not defined on ProcessInfo. Please define" << std::endl;

        return 0;

        KRATOS_CATCH("");
    }

    /**
     * @brief This method initializes the residual in the nodes of the model part
     * @param rModelPart The model of the problem to solve
     */
    void InitializeResidual(ModelPart& rModelPart) override
    {
        KRATOS_TRY

        // The array of nodes
        NodesArrayType& r_nodes = rModelPart.Nodes();

        // Auxiliar values
        const array_1d<double, 3> zero_array = ZeroVector(3);
        // Initializing the variables
        VariableUtils().SetVariable(FORCE_RESIDUAL, zero_array,r_nodes); // f (external_forces)
        VariableUtils().SetVariable(NODAL_INERTIA, zero_array,r_nodes); // K*a (internal_forces)
        VariableUtils().SetVariable(MIDDLE_ANGULAR_VELOCITY, zero_array,r_nodes); // K*K*a
        VariableUtils().SetVariable(FRACTIONAL_ACCELERATION, zero_array,r_nodes); // K*b

        KRATOS_CATCH("")
    }

    /**
     * @brief This method initializes some rutines related with the explicit scheme
     * @param rModelPart The model of the problem to solve
     * @param DomainSize The current dimention of the problem
     */
    void InitializeExplicitScheme(
        ModelPart& rModelPart,
        const SizeType DomainSize = 3
        ) override
    {
        KRATOS_TRY

        /// The array of ndoes
        NodesArrayType& r_nodes = rModelPart.Nodes();

        // The first iterator of the array of nodes
        const auto it_node_begin = rModelPart.NodesBegin();

        /// Initialise the database of the nodes
        #pragma omp parallel for schedule(guided,512)
        for (int i = 0; i < static_cast<int>(r_nodes.size()); ++i) {
            auto it_node = (it_node_begin + i);
            it_node->SetValue(NODAL_MASS, 0.0);
            array_1d<double, 3>& r_current_impulse = it_node->FastGetSolutionStepValue(NODAL_DISPLACEMENT_STIFFNESS);
            array_1d<double, 3>& r_external_forces = it_node->FastGetSolutionStepValue(FORCE_RESIDUAL);
            array_1d<double, 3>& r_current_internal_force = it_node->FastGetSolutionStepValue(NODAL_INERTIA);
            array_1d<double, 3>& r_k_k_a = it_node->FastGetSolutionStepValue(MIDDLE_ANGULAR_VELOCITY);
            array_1d<double, 3>& r_k_b = it_node->FastGetSolutionStepValue(FRACTIONAL_ACCELERATION);
            noalias(r_current_impulse) = ZeroVector(3);
            noalias(r_external_forces) = ZeroVector(3);
            noalias(r_current_internal_force) = ZeroVector(3);
            noalias(r_k_k_a) = ZeroVector(3);
            noalias(r_k_b) = ZeroVector(3);
        }

        KRATOS_CATCH("")
    }

    /**
     * @brief This method updates the translation DoF
     * @param itCurrentNode The iterator of the current node
     * @param DisplacementPosition The position of the displacement dof on the database
     * @param DomainSize The current dimention of the problem
     */
    void UpdateTranslationalDegreesOfFreedom(
        NodeIterator itCurrentNode,
        const IndexType DisplacementPosition,
        const SizeType DomainSize = 3
        ) override
    {
        array_1d<double, 3>& r_current_impulse = itCurrentNode->FastGetSolutionStepValue(NODAL_DISPLACEMENT_STIFFNESS);
        array_1d<double, 3>& r_current_displacement = itCurrentNode->FastGetSolutionStepValue(DISPLACEMENT);
        // const array_1d<double, 3>& r_previous_displacement = itCurrentNode->FastGetSolutionStepValue(DISPLACEMENT,1);
        // const array_1d<double, 3>& r_actual_previous_displacement = itCurrentNode->FastGetSolutionStepValue(DISPLACEMENT,2);
        const double nodal_mass = itCurrentNode->GetValue(NODAL_MASS);

        const array_1d<double, 3>& r_external_forces = itCurrentNode->FastGetSolutionStepValue(FORCE_RESIDUAL);
        // const array_1d<double, 3>& r_previous_external_forces = itCurrentNode->FastGetSolutionStepValue(FORCE_RESIDUAL,1);
        // const array_1d<double, 3>& r_actual_previous_external_forces = itCurrentNode->FastGetSolutionStepValue(FORCE_RESIDUAL,2);
        const array_1d<double, 3>& r_current_internal_force = itCurrentNode->FastGetSolutionStepValue(NODAL_INERTIA);
        // const array_1d<double, 3>& r_previous_internal_force = itCurrentNode->FastGetSolutionStepValue(NODAL_INERTIA,1);
        // const array_1d<double, 3>& r_actual_previous_internal_force = itCurrentNode->FastGetSolutionStepValue(NODAL_INERTIA,2);
        const array_1d<double, 3>& r_k_k_a = itCurrentNode->FastGetSolutionStepValue(MIDDLE_ANGULAR_VELOCITY);
        const array_1d<double, 3>& r_k_b = itCurrentNode->FastGetSolutionStepValue(FRACTIONAL_ACCELERATION);

        if ( nodal_mass > numerical_limit ){
            for (IndexType j = 0; j < DomainSize; j++) {
                r_current_impulse[j] = mBeta*mDeltaTime*mTheta1/nodal_mass*(mDeltaTime*mTheta1-mBeta*(mTheta1+1.0))*r_k_k_a[j]
                                     + mDeltaTime*(mAlpha*mTheta1*(mTheta1*mDeltaTime-2.0*mBeta*(mTheta1+1.0))-1.0)*r_current_internal_force[j]
                                     + mDeltaTime/nodal_mass*(mTheta1*mDeltaTime-mBeta*(mTheta1+1.0))*r_k_b[j]
                                     + (1.0-mAlpha*mDeltaTime*(mTheta1+1.0))*r_current_impulse[j]
                                     - mAlpha*mAlpha*mDeltaTime*mTheta1*(mTheta1+1.0)*nodal_mass*r_current_displacement[j]
                                     + mDeltaTime*r_external_forces[j];
            }
        } else {
            for (IndexType j = 0; j < DomainSize; j++) {
                r_current_impulse[j] = 0.0;
            }
        }

        const array_1d<double, 3>& r_previous_impulse = itCurrentNode->FastGetSolutionStepValue(NODAL_DISPLACEMENT_STIFFNESS,1);

        std::array<bool, 3> fix_displacements = {false, false, false};
        fix_displacements[0] = (itCurrentNode->GetDof(DISPLACEMENT_X, DisplacementPosition).IsFixed());
        fix_displacements[1] = (itCurrentNode->GetDof(DISPLACEMENT_Y, DisplacementPosition + 1).IsFixed());
        if (DomainSize == 3)
            fix_displacements[2] = (itCurrentNode->GetDof(DISPLACEMENT_Z, DisplacementPosition + 2).IsFixed());

        // Solution of the explicit equation:
        if ( nodal_mass > numerical_limit ){
            for (IndexType j = 0; j < DomainSize; j++) {
                if (fix_displacements[j] == false) {
                    r_current_displacement[j] = mBeta*mTheta1*mDeltaTime/nodal_mass*r_current_internal_force[j]
                                              + mTheta1*mDeltaTime/nodal_mass*r_current_impulse[j]
                                              + mDeltaTime/nodal_mass*(1.0-mTheta1)*r_previous_impulse[j]
                                              + (mAlpha*mDeltaTime*mTheta1+1.0)*r_current_displacement[j];
                }
            }
        } else{
            for (IndexType j = 0; j < DomainSize; j++) {
                if (fix_displacements[j] == false) {
                    r_current_displacement[j] = 0.0;
                }
            }
        }

        const array_1d<double, 3>& r_previous_displacement = itCurrentNode->FastGetSolutionStepValue(DISPLACEMENT,1);
        const array_1d<double, 3>& r_previous_velocity = itCurrentNode->FastGetSolutionStepValue(VELOCITY,1);
        array_1d<double, 3>& r_current_velocity = itCurrentNode->FastGetSolutionStepValue(VELOCITY);
        array_1d<double, 3>& r_current_acceleration = itCurrentNode->FastGetSolutionStepValue(ACCELERATION);

        noalias(r_current_velocity) = (1.0/mDeltaTime) * (r_current_displacement - r_previous_displacement);
        noalias(r_current_acceleration) = (1.0/mDeltaTime) * (r_current_velocity - r_previous_velocity);
    }

    ///@}
    ///@name Operations
    ///@{

    ///@}
    ///@name Access
    ///@{

    ///@}
    ///@name Inquiry
    ///@{

    ///@}
    ///@name Friends
    ///@{

    ///@}

protected:

    ///@}
    ///@name Protected Structs
    ///@{

    /**
     * @brief This struct contains the information related with the increment od time step
     */
    // struct DeltaTimeParameters {
    //     double Maximum;         // Maximum delta time
    //     double Fraction;        // Fraction of the delta time
    // };

    /**
     * @brief This struct contains the details of the time variables
     */
    // struct TimeVariables {
    //     double Current;        // n+1

    //     double Delta;          // Time step
    // };

    ///@name Protected static Member Variables
    ///@{

    // TimeVariables mTime;            /// This struct contains the details of the time variables
    // DeltaTimeParameters mDeltaTime; /// This struct contains the information related with the increment od time step

    ///@}
    ///@name Protected member Variables
    ///@{

    ///@}
    ///@name Protected Operators
    ///@{

    ///@}
    ///@name Protected Operations
    ///@{

    ///@}
    ///@name Protected  Access
    ///@{

    ///@}
    ///@name Protected Inquiry
    ///@{

    ///@}
    ///@name Protected LifeCycle
    ///@{

    ///@}

private:
    ///@name Static Member Variables
    ///@{

    ///@}
    ///@name Member Variables
    ///@{

    ///@}
    ///@name Private Operators
    ///@{

    ///@}
    ///@name Private Operations
    ///@{

    ///@}
    ///@name Private  Access
    ///@{

    ///@}
    ///@name Private Inquiry
    ///@{

    ///@}
    ///@name Un accessible methods
    ///@{

    ///@}

}; /* Class ExplicitOMDPScheme */

///@}

///@name Type Definitions
///@{

///@}

} /* namespace Kratos.*/

#endif /* KRATOS_EXPLICIT_OMDP_SCHEME_HPP_INCLUDED  defined */
