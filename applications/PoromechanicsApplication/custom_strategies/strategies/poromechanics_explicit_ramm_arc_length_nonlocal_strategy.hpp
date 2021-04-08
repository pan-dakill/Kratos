
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


#if !defined(KRATOS_POROMECHANICS_EXPLICIT_RAMM_ARC_LENGTH_NONLOCAL_STRATEGY)
#define KRATOS_POROMECHANICS_EXPLICIT_RAMM_ARC_LENGTH_NONLOCAL_STRATEGY

// Project includes
#include "custom_strategies/strategies/poromechanics_explicit_ramm_arc_length_strategy.hpp"
#include "custom_utilities/nonlocal_damage_utilities.hpp"
#include "custom_utilities/nonlocal_damage_2D_utilities.hpp"
#include "custom_utilities/nonlocal_damage_3D_utilities.hpp"

// Application includes
#include "poromechanics_application_variables.h"

namespace Kratos {

template <class TSparseSpace,
          class TDenseSpace,
          class TLinearSolver
          >
class PoromechanicsExplicitRammArcLengthNonlocalStrategy
    : public PoromechanicsExplicitRammArcLengthStrategy<TSparseSpace, TDenseSpace, TLinearSolver> {
public:

    KRATOS_CLASS_POINTER_DEFINITION(PoromechanicsExplicitRammArcLengthNonlocalStrategy);

    typedef SolvingStrategy<TSparseSpace, TDenseSpace, TLinearSolver> BaseType;
    typedef MechanicalExplicitStrategy<TSparseSpace, TDenseSpace, TLinearSolver> Grandx2MotherType;
    typedef PoromechanicsExplicitStrategy<TSparseSpace, TDenseSpace, TLinearSolver> GrandMotherType;
    typedef PoromechanicsExplicitRammArcLengthStrategy<TSparseSpace, TDenseSpace, TLinearSolver> MotherType;
    // typedef typename BaseType::TBuilderAndSolverType TBuilderAndSolverType;
    typedef typename BaseType::TSchemeType TSchemeType;
    typedef typename BaseType::DofsArrayType DofsArrayType;
    typedef typename BaseType::TSystemMatrixType TSystemMatrixType;
    typedef typename BaseType::TSystemVectorType TSystemVectorType;
    typedef typename BaseType::TSystemMatrixPointerType TSystemMatrixPointerType;
    typedef typename BaseType::TSystemVectorPointerType TSystemVectorPointerType;
    typedef typename BaseType::NodesArrayType NodesArrayType;
    typedef typename BaseType::ElementsArrayType ElementsArrayType;
    typedef typename BaseType::ConditionsArrayType ConditionsArrayType;
    typedef typename BaseType::LocalSystemVectorType LocalSystemVectorType;

    /// DoF types definition
    typedef typename Node<3>::DofType DofType;
    typedef typename DofType::Pointer DofPointerType;

    using Grandx2MotherType::mpScheme;
    using Grandx2MotherType::mCalculateReactionsFlag;
    using Grandx2MotherType::mInitializeWasPerformed;
    using GrandMotherType::mpParameters;
    using GrandMotherType::mSubModelPartList;
    using GrandMotherType::mVariableNames;
    using MotherType::mInitializeArcLengthWasPerformed;
    using MotherType::mSolutionStepIsInitialized;
    using MotherType::mLambda;
    using MotherType::mLambda_old;
    using MotherType::mRadius;
    using MotherType::mRadius_0;
    using MotherType::mMaxRadiusFactor;
    using MotherType::mMinRadiusFactor;

//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    ///Constructor
    PoromechanicsExplicitRammArcLengthNonlocalStrategy(
        ModelPart& model_part,
        typename TSchemeType::Pointer pScheme,
        Parameters& rParameters,
        bool CalculateReactions = false,
        bool ReformDofSetAtEachStep = false,
        bool MoveMeshFlag = false
        ) : PoromechanicsExplicitRammArcLengthStrategy<TSparseSpace, TDenseSpace, TLinearSolver>(model_part, pScheme, rParameters,
                CalculateReactions, ReformDofSetAtEachStep, MoveMeshFlag)
        {
            mNonlocalDamageIsInitialized = false;
            mSearchNeighboursAtEachStep = rParameters["search_neighbours_step"].GetBool();
        }

    //------------------------------------------------------------------------------------

    ///Destructor
    ~PoromechanicsExplicitRammArcLengthNonlocalStrategy() override {}

//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    /**
     * @brief Initialization of member variables and prior operations
     */
    void Initialize() override
    {
        KRATOS_TRY

        if (mInitializeWasPerformed == false)
		{
            MotherType::Initialize();

            if(mNonlocalDamageIsInitialized == false)
            {
                if(BaseType::GetModelPart().GetProcessInfo()[DOMAIN_SIZE]==2)
                {
                    mpNonlocalDamageUtility = new NonlocalDamage2DUtilities();
                }
                else
                {
                    mpNonlocalDamageUtility = new NonlocalDamage3DUtilities();
                }
                mpNonlocalDamageUtility->SearchGaussPointsNeighbours(mpParameters,BaseType::GetModelPart());

                mNonlocalDamageIsInitialized = true;
            }
        }

        KRATOS_CATCH( "" )
    }

//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    /**
     * @brief Performs all the required operations that should be done (for each step) before solving the solution step.
     * @details A member variable should be used as a flag to make sure this function is called only once per step.
     */
    void InitializeSolutionStep() override {
        KRATOS_TRY

        if (mSolutionStepIsInitialized == false) {

            MotherType::InitializeSolutionStep();

            if(mNonlocalDamageIsInitialized == false)
            {
                if(BaseType::GetModelPart().GetProcessInfo()[DOMAIN_SIZE]==2)
                {
                    mpNonlocalDamageUtility = new NonlocalDamage2DUtilities();
                }
                else
                {
                    mpNonlocalDamageUtility = new NonlocalDamage3DUtilities();
                }
                mpNonlocalDamageUtility->SearchGaussPointsNeighbours(mpParameters,BaseType::GetModelPart());

                mNonlocalDamageIsInitialized = true;
            }
        }

        KRATOS_CATCH("")
    }

//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	void FinalizeSolutionStep() override
	{
		KRATOS_TRY

        MotherType::FinalizeSolutionStep();

        if(mSearchNeighboursAtEachStep == true)
        {
            delete mpNonlocalDamageUtility;
            mNonlocalDamageIsInitialized = false;
        }

		KRATOS_CATCH("")
	}

//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    void Clear() override
    {
        KRATOS_TRY

        Grandx2MotherType::Clear();

        if(mSearchNeighboursAtEachStep == false) {
            delete mpNonlocalDamageUtility;
            mNonlocalDamageIsInitialized = false;
        }

        KRATOS_CATCH( "" )
    }

//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

protected:

    /// Member Variables
    NonlocalDamageUtilities* mpNonlocalDamageUtility;
    bool mNonlocalDamageIsInitialized;
    bool mSearchNeighboursAtEachStep;

//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    void CalculateNewX() override {

        KRATOS_TRY

        ModelPart& r_model_part = BaseType::GetModelPart();

        // Some dummy sets and matrices
        DofsArrayType dof_set_dummy;
        TSystemMatrixType rA = TSystemMatrixType();
        TSystemVectorType rDx = TSystemVectorType();
        TSystemVectorType rb = TSystemVectorType();

        // Initialize the non linear iteration
        mpScheme->InitializeNonLinIteration(BaseType::GetModelPart(), rA, rDx, rb);
        mpNonlocalDamageUtility->CalculateNonlocalEquivalentStrain(mpParameters,BaseType::GetModelPart().GetProcessInfo());

        mpScheme->Predict(r_model_part, dof_set_dummy, rA, rDx, rb);

        // Move the mesh if needed
        if (BaseType::MoveMeshFlag())
            BaseType::MoveMesh();

        mpScheme->InitializeNonLinIteration(BaseType::GetModelPart(), rA, rDx, rb);
        mpNonlocalDamageUtility->CalculateNonlocalEquivalentStrain(mpParameters,BaseType::GetModelPart().GetProcessInfo());

        // Explicitly integrates the equation of motion.
        mpScheme->Update(r_model_part, dof_set_dummy, rA, rDx, rb);

        // Move the mesh if needed
        if (BaseType::MoveMeshFlag())
            BaseType::MoveMesh();

        // Finalize the non linear iteration
        mpScheme->FinalizeNonLinIteration(BaseType::GetModelPart(), rA, rDx, rb);
        mpNonlocalDamageUtility->CalculateNonlocalEquivalentStrain(mpParameters,BaseType::GetModelPart().GetProcessInfo());

        KRATOS_CATCH("")
    }

}; // Class PoromechanicsExplicitRammArcLengthNonlocalStrategy

} // namespace Kratos

#endif // KRATOS_POROMECHANICS_EXPLICIT_RAMM_ARC_LENGTH_NONLOCAL_STRATEGY  defined
