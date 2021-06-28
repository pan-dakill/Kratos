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

#if !defined(KRATOS_EXPLICIT_CD_FIC_SCHEME_HPP_INCLUDED)
#define KRATOS_EXPLICIT_CD_FIC_SCHEME_HPP_INCLUDED

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
 * @class ExplicitCDFICScheme
 * @ingroup StructuralMechanicsApplciation
 * @brief An explicit forward euler scheme with a split of the inertial term
 * @author Ignasi de Pouplana
 */
template <class TSparseSpace,
          class TDenseSpace //= DenseSpace<double>
          >
class ExplicitCDFICScheme
    : public ExplicitCDScheme<TSparseSpace, TDenseSpace> {

public:
    ///@name Type Definitions
    ///@{

    /// The definition of the base type
    typedef Scheme<TSparseSpace, TDenseSpace> BaseofBaseType;
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

    /// Counted pointer of ExplicitCDFICScheme
    KRATOS_CLASS_POINTER_DEFINITION(ExplicitCDFICScheme);

    ///@}
    ///@name Life Cycle
    ///@{

    /**
     * @brief Default constructor.
     * @details The ExplicitCDFICScheme method
     */
    ExplicitCDFICScheme()
        : ExplicitCDScheme<TSparseSpace, TDenseSpace>() {}

    /** Destructor.
    */
    virtual ~ExplicitCDFICScheme() {}

    ///@}
    ///@name Operators
    ///@{

    /**
     * @brief This is the place to initialize the Scheme. This is intended to be called just once when the strategy is initialized
     * @param rModelPart The model of the problem to solve
     */
    void Initialize(ModelPart& rModelPart) override
    {
        KRATOS_TRY

        const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();

        mDelta1 = r_current_process_info[LOAD_FACTOR];
        mGamma = r_current_process_info[GAMMA];
        mDelta2 = r_current_process_info[DELTA_2];
        // mDelta2 = (1.0-mGamma)*0.25;

        BaseType::Initialize(rModelPart);

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
            it_node->SetValue(NUMBER_OF_NEIGHBOUR_ELEMENTS, 0.0);
            array_1d<double, 3>& r_force_residual = it_node->FastGetSolutionStepValue(FORCE_RESIDUAL);
            array_1d<double, 3>& r_external_forces = it_node->FastGetSolutionStepValue(EXTERNAL_FORCE);
            array_1d<double, 3>& r_current_internal_force = it_node->FastGetSolutionStepValue(NODAL_INERTIA); // K*d (internal_forces)
            noalias(r_force_residual) = ZeroVector(3);
            noalias(r_external_forces) = ZeroVector(3);
            noalias(r_current_internal_force) = ZeroVector(3);

            array_1d<double, 3>& r_delta_internal_force = it_node->FastGetSolutionStepValue(MIDDLE_VELOCITY); // H1Kd
            array_1d<double, 3>& r_delta_damping_force = it_node->FastGetSolutionStepValue(MIDDLE_ANGULAR_VELOCITY); // H1Cd
            array_1d<double, 3>& r_delta_external_force = it_node->FastGetSolutionStepValue(FRACTIONAL_ACCELERATION); // H1f
            array_1d<double, 3>& r_damping_force = it_node->FastGetSolutionStepValue(NODAL_DISPLACEMENT_STIFFNESS); // C*d
            noalias(r_delta_internal_force) = ZeroVector(3);
            noalias(r_delta_damping_force) = ZeroVector(3);
            noalias(r_delta_external_force) = ZeroVector(3);
            noalias(r_damping_force) = ZeroVector(3);
        }

        KRATOS_CATCH("")
    }

    /**
     * @brief This method initializes the residual in the nodes of the model part
     * @param rModelPart The model of the problem to solve
     */
    void InitializeResidual(ModelPart& rModelPart) override
    {
        KRATOS_TRY

        BaseType::InitializeResidual(rModelPart);

        // The array of nodes
        NodesArrayType& r_nodes = rModelPart.Nodes();

        // Auxiliar values
        const array_1d<double, 3> zero_array = ZeroVector(3);
        // Initializing the variables
        VariableUtils().SetVariable(MIDDLE_VELOCITY, zero_array,r_nodes);
        VariableUtils().SetVariable(MIDDLE_ANGULAR_VELOCITY, zero_array,r_nodes);
        VariableUtils().SetVariable(FRACTIONAL_ACCELERATION, zero_array,r_nodes);
        VariableUtils().SetVariable(NODAL_DISPLACEMENT_STIFFNESS, zero_array,r_nodes);

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
        array_1d<double, 3>& r_current_displacement = itCurrentNode->FastGetSolutionStepValue(DISPLACEMENT);
        // const array_1d<double, 3>& r_previous_displacement = itCurrentNode->FastGetSolutionStepValue(DISPLACEMENT,1);
        const array_1d<double, 3>& r_actual_previous_displacement = itCurrentNode->FastGetSolutionStepValue(DISPLACEMENT,2);

        const double nodal_mass = itCurrentNode->GetValue(NODAL_MASS);

        const array_1d<double, 3>& r_external_force = itCurrentNode->FastGetSolutionStepValue(EXTERNAL_FORCE);
        // const array_1d<double, 3>& r_previous_external_force = itCurrentNode->FastGetSolutionStepValue(EXTERNAL_FORCE,1);
        // const array_1d<double, 3>& r_actual_previous_external_forces = itCurrentNode->FastGetSolutionStepValue(EXTERNAL_FORCE,2);
        const array_1d<double, 3>& r_current_internal_force = itCurrentNode->FastGetSolutionStepValue(NODAL_INERTIA);
        // const array_1d<double, 3>& r_previous_internal_force = itCurrentNode->FastGetSolutionStepValue(NODAL_INERTIA,1);
        // const array_1d<double, 3>& r_actual_previous_internal_force = itCurrentNode->FastGetSolutionStepValue(NODAL_INERTIA,2);
        const array_1d<double, 3>& r_current_damping_force = itCurrentNode->FastGetSolutionStepValue(NODAL_DISPLACEMENT_STIFFNESS);
        const array_1d<double, 3>& r_previous_damping_force = itCurrentNode->FastGetSolutionStepValue(NODAL_DISPLACEMENT_STIFFNESS,1);

        const array_1d<double, 3>& r_delta_external_force = itCurrentNode->FastGetSolutionStepValue(FRACTIONAL_ACCELERATION);
        const array_1d<double, 3>& r_delta_internal_force = itCurrentNode->FastGetSolutionStepValue(MIDDLE_VELOCITY);
        const array_1d<double, 3>& r_current_delta_damping_force = itCurrentNode->FastGetSolutionStepValue(MIDDLE_ANGULAR_VELOCITY);
        const array_1d<double, 3>& r_previous_delta_damping_force = itCurrentNode->FastGetSolutionStepValue(MIDDLE_ANGULAR_VELOCITY,1);
        // const array_1d<double, 3>& r_actual_previous_delta_damping_force = itCurrentNode->FastGetSolutionStepValue(MIDDLE_ANGULAR_VELOCITY,2);

        std::array<bool, 3> fix_displacements = {false, false, false};
        fix_displacements[0] = (itCurrentNode->GetDof(DISPLACEMENT_X, DisplacementPosition).IsFixed());
        fix_displacements[1] = (itCurrentNode->GetDof(DISPLACEMENT_Y, DisplacementPosition + 1).IsFixed());
        if (DomainSize == 3)
            fix_displacements[2] = (itCurrentNode->GetDof(DISPLACEMENT_Z, DisplacementPosition + 2).IsFixed());

        // TODO
        // KRATOS_WATCH(itCurrentNode->Id())
        // KRATOS_WATCH(mDeltaTime)
        // KRATOS_WATCH(mAlpha)
        // KRATOS_WATCH(mBeta)
        // KRATOS_WATCH(nodal_mass)
        // KRATOS_WATCH(r_current_displacement)
        // KRATOS_WATCH(r_actual_previous_displacement)
        // KRATOS_WATCH(r_current_internal_force)
        // KRATOS_WATCH(r_previous_internal_force)
        // KRATOS_WATCH(r_external_force)

        //TODO
        // KRATOS_WATCH("Before solving")
        // KRATOS_WATCH(mDeltaTime)
        // KRATOS_WATCH(mDelta1)
        // KRATOS_WATCH(mDelta2)
        // KRATOS_WATCH(mAlpha)
        // KRATOS_WATCH(mBeta)
        // KRATOS_WATCH(nodal_mass)
        // KRATOS_WATCH(r_current_displacement)
        // KRATOS_WATCH(r_actual_previous_displacement)
        // KRATOS_WATCH(r_current_internal_force)
        // KRATOS_WATCH(r_damping_force)
        // KRATOS_WATCH(r_delta_internal_force)
        // KRATOS_WATCH(r_current_delta_damping_force)
        // KRATOS_WATCH(r_previous_internal_force)
        // KRATOS_WATCH(r_previous_delta_damping_force)
        // KRATOS_WATCH(r_delta_external_force)
        // KRATOS_WATCH(r_external_force)
        // KRATOS_WATCH(r_previous_external_force)
        // KRATOS_WATCH(r_delta2_internal_force)
        // KRATOS_WATCH(r_delta2_damping_force)
        // KRATOS_WATCH(r_previous_delta2_damping_force)
        // KRATOS_WATCH(r_delta2_external_force)

        for (IndexType j = 0; j < DomainSize; j++) {
            if (fix_displacements[j] == false) {
                r_current_displacement[j] = ( 2.0*(1.0+2.0*mDelta1)*nodal_mass*r_current_displacement[j]
                                            - (1.0+2.0*mDelta1)*mDeltaTime*mDeltaTime*r_current_internal_force[j]
                                            - (1.0+2.0*mDelta1)*mDeltaTime*r_current_damping_force[j]
                                            + mDeltaTime*mDeltaTime*r_delta_internal_force[j]
                                            + mDeltaTime*r_current_delta_damping_force[j]
                                            - (1.0+2.0*mDelta1)*nodal_mass*r_actual_previous_displacement[j]
                                            + (1.0+2.0*mDelta1)*mDeltaTime*r_previous_damping_force[j]
                                            - mDeltaTime*r_previous_delta_damping_force[j]
                                            + (1.0+2.0*mDelta1)*mDeltaTime*mDeltaTime*r_external_force[j]
                                            - mDeltaTime*mDeltaTime*r_delta_external_force[j]
                                            + (1.0+2.0*mDelta1)*mDelta2*mDelta2*(
                                                mDeltaTime*mDeltaTime*r_delta2_internal_force[j]
                                                + mDeltaTime*r_delta2_damping_force[j]
                                                - mDeltaTime*mDeltaTime*r_delta2_external_force[j]
                                                - mDeltaTime*r_previous_delta2_damping_force[j]
                                                )
                                            + (1.0+2.0*mDelta1)*mGamma*mDeltaTime*mDeltaTime*(
                                                r_current_internal_force[j]
                                                - r_external_force[j]
                                                )
                                            ) / ( nodal_mass*(1.0+2.0*mDelta1) );
            }
        }

        const array_1d<double, 3>& r_previous_displacement = itCurrentNode->FastGetSolutionStepValue(DISPLACEMENT,1);
        const array_1d<double, 3>& r_previous_velocity = itCurrentNode->FastGetSolutionStepValue(VELOCITY,1);
        array_1d<double, 3>& r_current_velocity = itCurrentNode->FastGetSolutionStepValue(VELOCITY);
        array_1d<double, 3>& r_current_acceleration = itCurrentNode->FastGetSolutionStepValue(ACCELERATION);

        noalias(r_current_velocity) = (1.0/mDeltaTime) * (r_current_displacement - r_previous_displacement);
        noalias(r_current_acceleration) = (1.0/mDeltaTime) * (r_current_velocity - r_previous_velocity);

        //TODO
        // KRATOS_WATCH("After solving")
        // KRATOS_WATCH(r_current_displacement)
        // KRATOS_WATCH(r_previous_displacement)
        // KRATOS_WATCH(r_actual_previous_displacement)
        // KRATOS_WATCH(r_current_velocity)
        // KRATOS_WATCH(r_previous_velocity)
        // KRATOS_WATCH(r_current_acceleration)

    }

    void CalculateAndAddRHS(ModelPart& rModelPart) override
    {
        KRATOS_TRY

        const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();
        ConditionsArrayType& r_conditions = rModelPart.Conditions();
        ElementsArrayType& r_elements = rModelPart.Elements();

        LocalSystemVectorType RHS_Contribution = LocalSystemVectorType(0);
        Element::EquationIdVectorType equation_id_vector_dummy; // Dummy

        #pragma omp parallel for firstprivate(RHS_Contribution, equation_id_vector_dummy), schedule(guided,512)
        for (int i = 0; i < static_cast<int>(r_conditions.size()); ++i) {
            auto it_cond = r_conditions.begin() + i;
            CalculateRHSContribution(*it_cond, RHS_Contribution, equation_id_vector_dummy, r_current_process_info);
        }

        #pragma omp parallel for firstprivate(RHS_Contribution, equation_id_vector_dummy), schedule(guided,512)
        for (int i = 0; i < static_cast<int>(r_elements.size()); ++i) {
            auto it_elem = r_elements.begin() + i;
            CalculateRHSContribution(*it_elem, RHS_Contribution, equation_id_vector_dummy, r_current_process_info);
        }

        // Adding previous reactions to external forces
        auto& r_nodes = rModelPart.Nodes();
        const auto it_node_begin = r_nodes.begin();
        #pragma omp parallel for schedule(guided,512)
        for(int i=0; i<static_cast<int>(r_nodes.size()); ++i) {
            auto it_node = it_node_begin + i;
            array_1d<double, 3>& r_external_force = it_node->FastGetSolutionStepValue(EXTERNAL_FORCE);
            const array_1d<double, 3>& r_reaction = it_node->FastGetSolutionStepValue(REACTION,0); // It doesn't matter whether this is ,0 or ,1 (it's the same at this point)
            // const array_1d<double, 3>& r_previous_reaction = it_node->FastGetSolutionStepValue(REACTION,1);
            // KRATOS_WATCH(r_reaction)
            // KRATOS_WATCH(r_previous_reaction)
            // NOTE: comment this to recover CD
            noalias(r_external_force) += r_reaction;
            // KRATOS_WATCH("reactions:")
            // KRATOS_WATCH(it_node->Id())
            // KRATOS_WATCH(r_external_force)
        }

        #pragma omp parallel for firstprivate(RHS_Contribution, equation_id_vector_dummy), schedule(guided,512)
        for (int i = 0; i < static_cast<int>(r_elements.size()); ++i) {
            auto it_elem = r_elements.begin() + i;
            CalculateRHSContributionFinal(*it_elem, RHS_Contribution, equation_id_vector_dummy, r_current_process_info);
        }

        KRATOS_CATCH("")
    }

    void CalculateAndAddRHSFinal(ModelPart& rModelPart) override
    {
        KRATOS_TRY

        // The array of nodes
        NodesArrayType& r_nodes = rModelPart.Nodes();

        // Auxiliar values
        const array_1d<double, 3> zero_array = ZeroVector(3);
        // Initializing the variables
        VariableUtils().SetVariable(FORCE_RESIDUAL, zero_array,r_nodes);

        const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();
        ConditionsArrayType& r_conditions = rModelPart.Conditions();
        ElementsArrayType& r_elements = rModelPart.Elements();

        LocalSystemVectorType RHS_Contribution = LocalSystemVectorType(0);
        Element::EquationIdVectorType equation_id_vector_dummy; // Dummy

        #pragma omp parallel for firstprivate(RHS_Contribution, equation_id_vector_dummy), schedule(guided,512)
        for (int i = 0; i < static_cast<int>(r_conditions.size()); ++i) {
            auto it_cond = r_conditions.begin() + i;
            CalculateRHSContributionResidual(*it_cond, RHS_Contribution, equation_id_vector_dummy, r_current_process_info);
        }

        #pragma omp parallel for firstprivate(RHS_Contribution, equation_id_vector_dummy), schedule(guided,512)
        for (int i = 0; i < static_cast<int>(r_elements.size()); ++i) {
            auto it_elem = r_elements.begin() + i;
            CalculateRHSContributionResidual(*it_elem, RHS_Contribution, equation_id_vector_dummy, r_current_process_info);
        }

        KRATOS_CATCH("")
    }

    /**
     * @brief Functions that calculates the RHS of a "condition" object
     * @param rCurrentCondition The condition to compute
     * @param RHS_Contribution The RHS vector contribution
     * @param EquationId The ID's of the condition degrees of freedom
     * @param rCurrentProcessInfo The current process info instance
     */
    void CalculateRHSContribution(
        Condition& rCurrentCondition,
        LocalSystemVectorType& RHS_Contribution,
        Element::EquationIdVectorType& EquationId,
        const ProcessInfo& rCurrentProcessInfo
        ) override
    {
        KRATOS_TRY

        rCurrentCondition.CalculateRightHandSide(RHS_Contribution, rCurrentProcessInfo);

        rCurrentCondition.AddExplicitContribution(RHS_Contribution, RESIDUAL_VECTOR, FORCE_RESIDUAL, rCurrentProcessInfo);

        KRATOS_CATCH("")
    }

    /**
     * @brief This function is designed to calculate just the RHS contribution
     * @param rCurrentElement The element to compute
     * @param RHS_Contribution The RHS vector contribution
     * @param EquationId The ID's of the element degrees of freedom
     * @param rCurrentProcessInfo The current process info instance
     */
    void CalculateRHSContribution(
        Element& rCurrentElement,
        LocalSystemVectorType& RHS_Contribution,
        Element::EquationIdVectorType& EquationId,
        const ProcessInfo& rCurrentProcessInfo
        ) override
    {
        KRATOS_TRY

        rCurrentElement.CalculateRightHandSide(RHS_Contribution, rCurrentProcessInfo);

        rCurrentElement.AddExplicitContribution(RHS_Contribution, RESIDUAL_VECTOR, FORCE_RESIDUAL , rCurrentProcessInfo);
        KRATOS_CATCH("")
    }

    /**
     * @brief This function is designed to calculate just the RHS contribution
     * @param rCurrentElement The element to compute
     * @param RHS_Contribution The RHS vector contribution
     * @param EquationId The ID's of the element degrees of freedom
     * @param rCurrentProcessInfo The current process info instance
     */
    void CalculateRHSContributionFinal(
        Element& rCurrentElement,
        LocalSystemVectorType& RHS_Contribution,
        Element::EquationIdVectorType& EquationId,
        const ProcessInfo& rCurrentProcessInfo
        )
    {
        KRATOS_TRY

        rCurrentElement.CalculateRightHandSide(RHS_Contribution, rCurrentProcessInfo);

        rCurrentElement.AddExplicitContribution(RHS_Contribution, RESIDUAL_VECTOR, FORCE_RESIDUAL, rCurrentProcessInfo);
        KRATOS_CATCH("")
    }

    /**
     * @brief Functions that calculates the RHS of a "condition" object
     * @param rCurrentCondition The condition to compute
     * @param RHS_Contribution The RHS vector contribution
     * @param EquationId The ID's of the condition degrees of freedom
     * @param rCurrentProcessInfo The current process info instance
     */
    void CalculateRHSContributionResidual(
        Condition& rCurrentCondition,
        LocalSystemVectorType& RHS_Contribution,
        Element::EquationIdVectorType& EquationId,
        const ProcessInfo& rCurrentProcessInfo
        ) override
    {
        KRATOS_TRY

        rCurrentCondition.CalculateRightHandSide(RHS_Contribution, rCurrentProcessInfo);

        rCurrentCondition.AddExplicitContribution(RHS_Contribution, RESIDUAL_VECTOR, REACTION, rCurrentProcessInfo);

        KRATOS_CATCH("")
    }

    /**
     * @brief This function is designed to calculate just the RHS contribution
     * @param rCurrentElement The element to compute
     * @param RHS_Contribution The RHS vector contribution
     * @param EquationId The ID's of the element degrees of freedom
     * @param rCurrentProcessInfo The current process info instance
     */
    void CalculateRHSContributionResidual(
        Element& rCurrentElement,
        LocalSystemVectorType& RHS_Contribution,
        Element::EquationIdVectorType& EquationId,
        const ProcessInfo& rCurrentProcessInfo
        ) override
    {
        KRATOS_TRY

        rCurrentElement.CalculateRightHandSide(RHS_Contribution, rCurrentProcessInfo);

        rCurrentElement.AddExplicitContribution(RHS_Contribution, RESIDUAL_VECTOR, REACTION, rCurrentProcessInfo);
        KRATOS_CATCH("")
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

    /**
     * @brief This struct contains the details of the time variables
     */

    ///@name Protected static Member Variables
    ///@{

    ///@}
    ///@name Protected member Variables
    ///@{

    double mDelta1;
    double mDelta2;
    double mGamma;

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

}; /* Class ExplicitCDFICScheme */

///@}

///@name Type Definitions
///@{

///@}

} /* namespace Kratos.*/

#endif /* KRATOS_EXPLICIT_CD_FIC_SCHEME_HPP_INCLUDED  defined */
