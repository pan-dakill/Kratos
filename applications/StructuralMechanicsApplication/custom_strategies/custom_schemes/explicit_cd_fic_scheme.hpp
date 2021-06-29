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
    using BaseType::mTheta;
    using BaseType::mGCoefficient;

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

        mDelta = r_current_process_info[DELTA_S];
        mAlpha1 = r_current_process_info[RAYLEIGH_ALPHA_1_S];
        mBeta1 = r_current_process_info[RAYLEIGH_BETA_1_S];
        mEpsilon = r_current_process_info[EPSILON_S];

        BaseType::Initialize(rModelPart);

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
        array_1d<double, 3>& r_displacement = itCurrentNode->FastGetSolutionStepValue(DISPLACEMENT);
        array_1d<double, 3> displacement_aux;
        noalias(displacement_aux) = r_displacement;
        array_1d<double, 3>& r_displacement_old = itCurrentNode->FastGetSolutionStepValue(DISPLACEMENT_OLD_S);
        array_1d<double, 3>& r_displacement_older = itCurrentNode->FastGetSolutionStepValue(DISPLACEMENT_OLDER_S);
        const double nodal_mass = itCurrentNode->GetValue(NODAL_MASS);

        const array_1d<double, 3>& r_external_force = itCurrentNode->FastGetSolutionStepValue(EXTERNAL_FORCE);
        const array_1d<double, 3>& r_external_force_old = itCurrentNode->FastGetSolutionStepValue(EXTERNAL_FORCE,1);
        const array_1d<double, 3>& r_internal_force = itCurrentNode->FastGetSolutionStepValue(NODAL_INERTIA);
        const array_1d<double, 3>& r_internal_force_old = itCurrentNode->FastGetSolutionStepValue(NODAL_INERTIA,1);

        std::array<bool, 3> fix_displacements = {false, false, false};
        fix_displacements[0] = (itCurrentNode->GetDof(DISPLACEMENT_X, DisplacementPosition).IsFixed());
        fix_displacements[1] = (itCurrentNode->GetDof(DISPLACEMENT_Y, DisplacementPosition + 1).IsFixed());
        if (DomainSize == 3)
            fix_displacements[2] = (itCurrentNode->GetDof(DISPLACEMENT_Z, DisplacementPosition + 2).IsFixed());

        // KRATOS_WATCH("Before Solving")
        // KRATOS_WATCH(mDeltaTime)
        // KRATOS_WATCH(nodal_mass)
        // KRATOS_WATCH(mEpsilon)
        // KRATOS_WATCH(mDelta)
        // KRATOS_WATCH(mGCoefficient)
        // KRATOS_WATCH(mAlpha)
        // KRATOS_WATCH(mBeta)
        // KRATOS_WATCH(mAlpha1)
        // KRATOS_WATCH(mBeta1)
        // KRATOS_WATCH(r_displacement)
        // KRATOS_WATCH(r_displacement_old)
        // KRATOS_WATCH(r_displacement_older)
        // KRATOS_WATCH(r_internal_force)
        // KRATOS_WATCH(r_internal_force_old)
        // KRATOS_WATCH(r_external_force)
        // KRATOS_WATCH(r_external_force_old)

        // Solution of the explicit equation:
        if ( nodal_mass > numerical_limit ){
            for (IndexType j = 0; j < DomainSize; j++) {
                if (fix_displacements[j] == false) {
                    r_displacement[j] = ( (2.0+mDelta-mDeltaTime*(mAlpha+mDelta*mAlpha1))*nodal_mass*r_displacement[j]
                                          + (mDelta*(1.0+mDelta*mGCoefficient)-1.0+mDeltaTime*(mAlpha+mDelta*mAlpha1))*nodal_mass*r_displacement_old[j]
                                          - mDelta*nodal_mass*r_displacement_older[j]
                                          - (0.5*mDeltaTime*mDeltaTime*(1.0+mDelta*mEpsilon)+mDeltaTime*(mBeta+mDelta*mBeta1))*r_internal_force[j]
                                          - (0.5*mDeltaTime*mDeltaTime*(1.0+mDelta*mEpsilon)-mDeltaTime*(mBeta+mDelta*mBeta1))*r_internal_force_old[j]
                                          + 0.5*mDeltaTime*mDeltaTime*(1.0+mDelta*mEpsilon)*(r_external_force[j]+r_external_force_old[j])
                                        ) / ( (1.0+mDelta*(1.0+mDelta*mGCoefficient))*nodal_mass );
                }
            }
        } else{
            for (IndexType j = 0; j < DomainSize; j++) {
                if (fix_displacements[j] == false) {
                    r_displacement[j] = 0.0;
                }
            }
        }

        // KRATOS_WATCH("After Solving")
        // KRATOS_WATCH(r_displacement)

        noalias(r_displacement_older) = r_displacement_old;
        noalias(r_displacement_old) = displacement_aux;
        const array_1d<double, 3>& r_velocity_old = itCurrentNode->FastGetSolutionStepValue(VELOCITY,1);
        array_1d<double, 3>& r_velocity = itCurrentNode->FastGetSolutionStepValue(VELOCITY);
        array_1d<double, 3>& r_acceleration = itCurrentNode->FastGetSolutionStepValue(ACCELERATION);

        noalias(r_velocity) = (1.0/mDeltaTime) * (r_displacement - r_displacement_old);
        noalias(r_acceleration) = (1.0/mDeltaTime) * (r_velocity - r_velocity_old);
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

    double mDelta;
    double mAlpha1;
    double mBeta1;
    double mEpsilon;

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
