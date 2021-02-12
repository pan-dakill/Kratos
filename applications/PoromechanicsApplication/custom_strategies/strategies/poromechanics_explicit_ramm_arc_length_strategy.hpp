
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


#if !defined(KRATOS_POROMECHANICS_EXPLICIT_RAMM_ARC_LENGTH_STRATEGY)
#define KRATOS_POROMECHANICS_EXPLICIT_RAMM_ARC_LENGTH_STRATEGY

// Project includes
#include "custom_strategies/strategies/poromechanics_explicit_strategy.hpp"

// Application includes
#include "poromechanics_application_variables.h"

namespace Kratos {

template <class TSparseSpace,
          class TDenseSpace,
          class TLinearSolver
          >
class PoromechanicsExplicitRammArcLengthStrategy
    : public PoromechanicsExplicitStrategy<TSparseSpace, TDenseSpace, TLinearSolver> {
public:

    KRATOS_CLASS_POINTER_DEFINITION(PoromechanicsExplicitRammArcLengthStrategy);

    typedef SolvingStrategy<TSparseSpace, TDenseSpace, TLinearSolver> BaseType;
    typedef MechanicalExplicitStrategy<TSparseSpace, TDenseSpace, TLinearSolver> GrandMotherType;
    typedef PoromechanicsExplicitStrategy<TSparseSpace, TDenseSpace, TLinearSolver> MotherType;
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

    using GrandMotherType::mInitializeWasPerformed;
    using GrandMotherType::mCalculateReactionsFlag;
    using GrandMotherType::mpScheme;
    using MotherType::mpParameters;
    using MotherType::mSubModelPartList;
    using MotherType::mVariableNames;

//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    ///Constructor
    PoromechanicsExplicitRammArcLengthStrategy(
        ModelPart& model_part,
        typename TSchemeType::Pointer pScheme,
        Parameters& rParameters,
        bool CalculateReactions = false,
        bool ReformDofSetAtEachStep = false,
        bool MoveMeshFlag = false
        ) : PoromechanicsExplicitStrategy<TSparseSpace, TDenseSpace, TLinearSolver>(model_part, pScheme, rParameters,
                CalculateReactions, ReformDofSetAtEachStep, MoveMeshFlag)
        {
            mMaxRadiusFactor = rParameters["max_radius_factor"].GetDouble();
            mMinRadiusFactor = rParameters["min_radius_factor"].GetDouble();

            mInitializeArcLengthWasPerformed = false;
            mSolutionStepIsInitialized = false;
        }

    //------------------------------------------------------------------------------------

    ///Destructor
    ~PoromechanicsExplicitRammArcLengthStrategy() override {}

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

            if (mInitializeArcLengthWasPerformed == false)
            {
                ModelPart& r_model_part = BaseType::GetModelPart();

                this->InitializeSolutionStep();

                // Compute initial radius (mRadius_0)
                this->CalculateNewX();
                this->SaveDxF(r_model_part);
                this->RestoreX(r_model_part);

                mRadius_0 = this->CalculateDxFNorm(r_model_part);
                mRadius = mRadius_0;

                //Initialize the loading factor Lambda
                mLambda = 0.0;
                mLambda_old = 1.0;

                // Initialize Norm of solution
                // mNormxEquilibrium = 0.0;

                mInitializeArcLengthWasPerformed = true;

                KRATOS_INFO("Ramm's Arc Length Explicit Strategy") << "Strategy Initialized" << std::endl;
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
            mSolutionStepIsInitialized = true;
        }

        KRATOS_CATCH("")
    }

//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    /**
     * @brief Solves the current step. This function returns true if a solution has been found, false otherwise.
     */
    bool SolveSolutionStep() override
    {
        // ********** Prediction phase **********

        KRATOS_INFO("Ramm's Arc Length Explicit Strategy") << "ARC-LENGTH RADIUS: " << mRadius/mRadius_0 << " X initial radius" << std::endl;

        // Get DxF
        this->RestoreF();
        this->CalculateNewX();
        this->SaveDxF(r_model_part);

        // Predict Lambda
        double DLambda = mRadius/this->CalculateDxFNorm(r_model_part);
        mLambda += DLambda;
        this->RestoreX(r_model_part);
        // Predict X
        this->CalculateDxPred(r_model_part,DLambda);

        // Update X with prediction
        this->UpdateXWithPrediction(r_model_part);

        // ********** Correction phase **********

        // Get Correction DxF
        this->CalculateNewX();
        this->SaveCorrectionDxF(r_model_part);
        this->RestoreX(r_model_part);
        this->UpdateXWithPrediction(r_model_part);

        // Get Correction DxLF
        this->UpdateF()
        this->CalculateNewX();
        this->SaveCorrectionDxLF(r_model_part);
        this->RestoreX(r_model_part);
        this->UpdateXWithPrediction(r_model_part);

        // Correct Lambda
        DLambda = - this->CalculateDxPDotDxLF(r_model_part)/this->CalculateDxPDotDxF(r_model_part);
        mLambda += DLambda;

        // Update X with correction
        this->UpdateXWithCorrection(r_model_part,DLambda);

        // Update Load
        this->RestoreF();
        this->UpdateF()

        // Calculate reactions if required
        if (mCalculateReactionsFlag) {
            this->CalculateReactions(mpScheme, r_model_part);
        }

        return true;
    }

//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	void FinalizeSolutionStep() override
	{
		KRATOS_TRY

        // Update the radius
        //mRadius = mRadius*sqrt(double(mDesiredIterations)/double(iteration_number));
        mRadius = mRadius*1.0; // TODO
        if (mRadius > mMaxRadiusFactor*mRadius_0)
            mRadius = mMaxRadiusFactor*mRadius_0;
        else if(mRadius < mMinRadiusFactor*mRadius_0)
            mRadius = mMinRadiusFactor*mRadius_0;

        // if (BaseType::GetModelPart().GetProcessInfo()[IS_CONVERGED] == true)
        // {
        //     // Modify the radius to advance faster when convergence is achieved
        //     if (mRadius > mMaxRadiusFactor*mRadius_0)
        //         mRadius = mMaxRadiusFactor*mRadius_0;
        //     else if(mRadius < mMinRadiusFactor*mRadius_0)
        //         mRadius = mMinRadiusFactor*mRadius_0;

        //     // Update Norm of x
        //     mNormxEquilibrium = this->CalculateReferenceDofsNorm(rDofSet);
        // }
        // else
        // {
        //     std::cout << "************ NO CONVERGENCE: restoring equilibrium path ************" << std::endl;

        //     TSystemVectorType& mDxStep = *mpDxStep;

        //     //update results
        //     mLambda -= mDLambdaStep;
        //     noalias(mDx) = -mDxStep;
        //     this->Update(rDofSet, mA, mDx, mb);

        //     //move the mesh if needed
        //     if(BaseType::MoveMeshFlag() == true) BaseType::MoveMesh();
        // }

        BaseType::GetModelPart().GetProcessInfo()[ARC_LENGTH_LAMBDA] = mLambda;
        BaseType::GetModelPart().GetProcessInfo()[ARC_LENGTH_RADIUS_FACTOR] = mRadius/mRadius_0;

        MotherType::FinalizeSolutionStep();

        //reset flags for next step
        mSolutionStepIsInitialized = false;

		KRATOS_CATCH("")
	}

//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

protected:

    /// Member Variables
    bool mSolutionStepIsInitialized; /// Flag to set as initialized the solution step

    bool mInitializeArcLengthWasPerformed;

    double mMaxRadiusFactor, mMinRadiusFactor; /// Used to limit the radius of the arc length strategy
    double mRadius, mRadius_0; /// Radius of the arc length strategy
    double mLambda, mLambda_old; /// Loading factor

//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    virtual void CalculateNewX() {

        KRATOS_TRY

        ModelPart& r_model_part = BaseType::GetModelPart();

        // Some dummy sets and matrices
        DofsArrayType dof_set_dummy;
        TSystemMatrixType rA = TSystemMatrixType();
        TSystemVectorType rDx = TSystemVectorType();
        TSystemVectorType rb = TSystemVectorType();

        // Initialize the non linear iteration
        mpScheme->InitializeNonLinIteration(BaseType::GetModelPart(), rA, rDx, rb);

        mpScheme->Predict(r_model_part, dof_set_dummy, rA, rDx, rb);

        // Move the mesh if needed
        if (BaseType::MoveMeshFlag())
            BaseType::MoveMesh();

        // Explicitly integrates the equation of motion.
        mpScheme->Update(r_model_part, dof_set_dummy, rA, rDx, rb);

        // Move the mesh if needed
        if (BaseType::MoveMeshFlag())
            BaseType::MoveMesh();

        // Finalize the non linear iteration
        mpScheme->FinalizeNonLinIteration(BaseType::GetModelPart(), rA, rDx, rb);

        KRATOS_CATCH("")
    }

//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    virtual void SaveDxF(ModelPart& rModelPart)
    {
        KRATOS_TRY

        // The array of nodes
        NodesArrayType& r_nodes = rModelPart.Nodes();

        // The iterator of the first node
        const auto it_node_begin = rModelPart.NodesBegin();

        #pragma omp parallel for schedule(guided,512)
        for (int i = 0; i < static_cast<int>(r_nodes.size()); ++i) {

            ModelPart::NodeIterator itCurrentNode = it_node_begin + i;

            array_1d<double, 3>& r_delta_displacement_f = itCurrentNode->FastGetSolutionStepValue(DELTA_DISPLACEMENT_F);
            const array_1d<double, 3>& r_current_displacement = itCurrentNode->FastGetSolutionStepValue(DISPLACEMENT);
            const array_1d<double, 3>& r_previous_displacement = itCurrentNode->FastGetSolutionStepValue(DISPLACEMENT,1);
            noalias(r_delta_displacement_f) = r_current_displacement - r_previous_displacement;

            array_1d<double, 3>& r_delta_velocity_f = itCurrentNode->FastGetSolutionStepValue(DELTA_VELOCITY_F);
            const array_1d<double, 3>& r_current_velocity = itCurrentNode->FastGetSolutionStepValue(VELOCITY);
            const array_1d<double, 3>& r_previous_velocity = itCurrentNode->FastGetSolutionStepValue(VELOCITY,1);
            noalias(r_delta_velocity_f) = r_current_velocity - r_previous_velocity;

            array_1d<double, 3>& r_delta_acceleration_f = itCurrentNode->FastGetSolutionStepValue(DELTA_ACCELERATION_F);
            const array_1d<double, 3>& r_current_acceleration = itCurrentNode->FastGetSolutionStepValue(ACCELERATION);
            const array_1d<double, 3>& r_previous_acceleration = itCurrentNode->FastGetSolutionStepValue(ACCELERATION,1);
            noalias(r_delta_acceleration_f) = r_current_acceleration - r_previous_acceleration;

            double& r_delta_water_pressure_f = itCurrentNode->FastGetSolutionStepValue(DELTA_WATER_PRESSURE_F);
            const double& r_current_water_pressure = itCurrentNode->FastGetSolutionStepValue(WATER_PRESSURE);
            const double& r_previous_water_pressure = itCurrentNode->FastGetSolutionStepValue(WATER_PRESSURE,1);
            r_delta_water_pressure_f = r_current_water_pressure - r_previous_water_pressure;

            double& r_delta_dt_water_pressure_f = itCurrentNode->FastGetSolutionStepValue(DELTA_DT_WATER_PRESSURE_F);
            const double& r_current_dt_water_pressure = itCurrentNode->FastGetSolutionStepValue(DT_WATER_PRESSURE);
            const double& r_previous_dt_water_pressure = itCurrentNode->FastGetSolutionStepValue(DT_WATER_PRESSURE,1);
            r_delta_dt_water_pressure_f = r_current_dt_water_pressure - r_previous_dt_water_pressure;

        } // for Node parallel

        KRATOS_CATCH( "" )
    }

//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    virtual void RestoreX(ModelPart& rModelPart)
    {
        KRATOS_TRY

        // The array of nodes
        NodesArrayType& r_nodes = rModelPart.Nodes();

        // The iterator of the first node
        const auto it_node_begin = rModelPart.NodesBegin();

        #pragma omp parallel for schedule(guided,512)
        for (int i = 0; i < static_cast<int>(r_nodes.size()); ++i) {

            ModelPart::NodeIterator itCurrentNode = it_node_begin + i;

            array_1d<double, 3>& r_current_displacement = itCurrentNode->FastGetSolutionStepValue(DISPLACEMENT);
            const array_1d<double, 3>& r_previous_displacement = itCurrentNode->FastGetSolutionStepValue(DISPLACEMENT,1);
            noalias(r_current_displacement) = r_previous_displacement;

            array_1d<double, 3>& r_current_velocity = itCurrentNode->FastGetSolutionStepValue(VELOCITY);
            const array_1d<double, 3>& r_previous_velocity = itCurrentNode->FastGetSolutionStepValue(VELOCITY,1);
            noalias(r_current_velocity) = r_previous_velocity;

            array_1d<double, 3>& r_current_acceleration = itCurrentNode->FastGetSolutionStepValue(ACCELERATION);
            const array_1d<double, 3>& r_previous_acceleration = itCurrentNode->FastGetSolutionStepValue(ACCELERATION,1);
            noalias(r_current_acceleration) = r_previous_acceleration;

            double& r_current_water_pressure = itCurrentNode->FastGetSolutionStepValue(WATER_PRESSURE);
            const double& r_previous_water_pressure = itCurrentNode->FastGetSolutionStepValue(WATER_PRESSURE,1);
            r_current_water_pressure = r_previous_water_pressure;

            double& r_current_dt_water_pressure = itCurrentNode->FastGetSolutionStepValue(DT_WATER_PRESSURE);
            const double& r_previous_dt_water_pressure = itCurrentNode->FastGetSolutionStepValue(DT_WATER_PRESSURE,1);
            r_current_dt_water_pressure = r_previous_dt_water_pressure;

        } // for Node parallel

        // Move the mesh if needed
        if (BaseType::MoveMeshFlag())
            BaseType::MoveMesh();

        KRATOS_CATCH( "" )
    }

//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    virtual double CalculateDxFNorm(ModelPart& rModelPart)
    {
        KRATOS_TRY

        double DxF_norm = 0.0;

        // The array of nodes
        NodesArrayType& r_nodes = rModelPart.Nodes();

        // The iterator of the first node
        const auto it_node_begin = rModelPart.NodesBegin();

        #pragma omp parallel for reduction(+:DxF_norm)
        for (int i = 0; i < static_cast<int>(r_nodes.size()); ++i) {
            ModelPart::NodeIterator itCurrentNode = it_node_begin + i;

            // TODO: should I calculate the norm taking into account the time derivatives of the unknowns ?
            const array_1d<double, 3>& r_delta_displacement_f = itCurrentNode->FastGetSolutionStepValue(DELTA_DISPLACEMENT_F);
            const array_1d<double, 3>& r_delta_velocity_f = itCurrentNode->FastGetSolutionStepValue(DELTA_VELOCITY_F);
            const array_1d<double, 3>& r_delta_acceleration_f = itCurrentNode->FastGetSolutionStepValue(DELTA_ACCELERATION_F);
            const double& r_delta_water_pressure_f = itCurrentNode->FastGetSolutionStepValue(DELTA_WATER_PRESSURE_F);
            const double& r_delta_dt_water_pressure_f = itCurrentNode->FastGetSolutionStepValue(DELTA_DT_WATER_PRESSURE_F);

            DxF_norm += r_delta_displacement_f[0]*r_delta_displacement_f[0]
                        + r_delta_displacement_f[1]*r_delta_displacement_f[1]
                        + r_delta_displacement_f[2]*r_delta_displacement_f[2]
                        + r_delta_velocity_f[0]*r_delta_velocity_f[0]
                        + r_delta_velocity_f[1]*r_delta_velocity_f[1]
                        + r_delta_velocity_f[2]*r_delta_velocity_f[2]
                        + r_delta_acceleration_f[0]*r_delta_acceleration_f[0]
                        + r_delta_acceleration_f[1]*r_delta_acceleration_f[1]
                        + r_delta_acceleration_f[2]*r_delta_acceleration_f[2]
                        + r_delta_water_pressure_f*r_delta_water_pressure_f
                        + r_delta_dt_water_pressure_f*r_delta_dt_water_pressure_f;

        }
        return std::sqrt(DxF_norm);

        KRATOS_CATCH( "" )
    }

//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    virtual void CalculateDxPred(ModelPart& rModelPart, const double& DLambda)
    {
        KRATOS_TRY

        // The array of nodes
        NodesArrayType& r_nodes = rModelPart.Nodes();

        // The iterator of the first node
        const auto it_node_begin = rModelPart.NodesBegin();

        #pragma omp parallel for schedule(guided,512)
        for (int i = 0; i < static_cast<int>(r_nodes.size()); ++i) {

            ModelPart::NodeIterator itCurrentNode = it_node_begin + i;

            array_1d<double, 3>& r_delta_displacement_p = itCurrentNode->FastGetSolutionStepValue(DELTA_DISPLACEMENT_P);
            const array_1d<double, 3>& r_delta_displacement_f = itCurrentNode->FastGetSolutionStepValue(DELTA_DISPLACEMENT_F);
            noalias(r_delta_displacement_p) = DLambda*r_delta_displacement_f;

            array_1d<double, 3>& r_delta_velocity_p = itCurrentNode->FastGetSolutionStepValue(DELTA_VELOCITY_P);
            const array_1d<double, 3>& r_delta_velocity_f = itCurrentNode->FastGetSolutionStepValue(DELTA_VELOCITY_F);
            noalias(r_delta_velocity_p) = DLambda*r_delta_velocity_f;

            array_1d<double, 3>& r_delta_acceleration_p = itCurrentNode->FastGetSolutionStepValue(DELTA_ACCELERATION_P);
            const array_1d<double, 3>& r_delta_acceleration_f = itCurrentNode->FastGetSolutionStepValue(DELTA_ACCELERATION_F);
            noalias(r_delta_acceleration_p) = DLambda*r_delta_acceleration_f;

            double& r_delta_water_pressure_p = itCurrentNode->FastGetSolutionStepValue(DELTA_WATER_PRESSURE_P);
            const double& r_delta_water_pressure_f = itCurrentNode->FastGetSolutionStepValue(DELTA_WATER_PRESSURE_F);
            r_delta_water_pressure_p = DLambda*r_delta_water_pressure_f;

            double& r_delta_dt_water_pressure_p = itCurrentNode->FastGetSolutionStepValue(DELTA_DT_WATER_PRESSURE_P);
            const double& r_delta_dt_water_pressure_f = itCurrentNode->FastGetSolutionStepValue(DELTA_DT_WATER_PRESSURE_F);
            r_delta_dt_water_pressure_p = DLambda*r_delta_dt_water_pressure_f;

        } // for Node parallel

        KRATOS_CATCH( "" )
    }

//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    virtual void UpdateXWithPrediction(ModelPart& rModelPart)
    {
        KRATOS_TRY

        // Update unknowns from prediction

        // The array of nodes
        NodesArrayType& r_nodes = rModelPart.Nodes();

        // The iterator of the first node
        const auto it_node_begin = rModelPart.NodesBegin();

        #pragma omp parallel for schedule(guided,512)
        for (int i = 0; i < static_cast<int>(r_nodes.size()); ++i) {

            ModelPart::NodeIterator itCurrentNode = it_node_begin + i;

            array_1d<double, 3>& r_current_displacement = itCurrentNode->FastGetSolutionStepValue(DISPLACEMENT);
            const array_1d<double, 3>& r_delta_displacement_p = itCurrentNode->FastGetSolutionStepValue(DELTA_DISPLACEMENT_P);
            noalias(r_current_displacement) += r_delta_displacement_p;

            array_1d<double, 3>& r_current_velocity = itCurrentNode->FastGetSolutionStepValue(VELOCITY);
            const array_1d<double, 3>& r_delta_velocity_p = itCurrentNode->FastGetSolutionStepValue(DELTA_VELOCITY_P);
            noalias(r_current_velocity) += r_delta_velocity_p;

            array_1d<double, 3>& r_current_acceleration = itCurrentNode->FastGetSolutionStepValue(ACCELERATION);
            const array_1d<double, 3>& r_delta_acceleration_p = itCurrentNode->FastGetSolutionStepValue(DELTA_ACCELERATION_P);
            noalias(r_current_acceleration) += r_delta_acceleration_p;

            double& r_current_water_pressure = itCurrentNode->FastGetSolutionStepValue(WATER_PRESSURE);
            const double& r_delta_water_pressure_p = itCurrentNode->FastGetSolutionStepValue(DELTA_WATER_PRESSURE_P);
            r_current_water_pressure += r_delta_water_pressure_p;

            double& r_current_dt_water_pressure = itCurrentNode->FastGetSolutionStepValue(DT_WATER_PRESSURE);
            const double& r_delta_dt_water_pressure_p = itCurrentNode->FastGetSolutionStepValue(DELTA_DT_WATER_PRESSURE_P);
            r_current_dt_water_pressure += r_delta_dt_water_pressure_p;

        } // for Node parallel

        // Move the mesh if needed
        if (BaseType::MoveMeshFlag())
            BaseType::MoveMesh();

        KRATOS_CATCH( "" )
    }

//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    virtual void SaveCorrectionDxF(ModelPart& rModelPart)
    {
        KRATOS_TRY

        // The array of nodes
        NodesArrayType& r_nodes = rModelPart.Nodes();

        // The iterator of the first node
        const auto it_node_begin = rModelPart.NodesBegin();

        #pragma omp parallel for schedule(guided,512)
        for (int i = 0; i < static_cast<int>(r_nodes.size()); ++i) {

            ModelPart::NodeIterator itCurrentNode = it_node_begin + i;

            array_1d<double, 3>& r_delta_displacement_f = itCurrentNode->FastGetSolutionStepValue(DELTA_DISPLACEMENT_F);
            const array_1d<double, 3>& r_current_displacement = itCurrentNode->FastGetSolutionStepValue(DISPLACEMENT);
            const array_1d<double, 3>& r_previous_displacement = itCurrentNode->FastGetSolutionStepValue(DISPLACEMENT,1);
            const array_1d<double, 3>& r_delta_displacement_p = itCurrentNode->FastGetSolutionStepValue(DELTA_DISPLACEMENT_P);
            noalias(r_delta_displacement_f) = r_current_displacement - r_previous_displacement - r_delta_displacement_p;

            // TODO: seguir
            array_1d<double, 3>& r_delta_velocity_f = itCurrentNode->FastGetSolutionStepValue(DELTA_VELOCITY_F);
            const array_1d<double, 3>& r_current_velocity = itCurrentNode->FastGetSolutionStepValue(VELOCITY);
            const array_1d<double, 3>& r_previous_velocity = itCurrentNode->FastGetSolutionStepValue(VELOCITY,1);
            noalias(r_delta_velocity_f) = r_current_velocity - r_previous_velocity;

            array_1d<double, 3>& r_delta_acceleration_f = itCurrentNode->FastGetSolutionStepValue(DELTA_ACCELERATION_F);
            const array_1d<double, 3>& r_current_acceleration = itCurrentNode->FastGetSolutionStepValue(ACCELERATION);
            const array_1d<double, 3>& r_previous_acceleration = itCurrentNode->FastGetSolutionStepValue(ACCELERATION,1);
            noalias(r_delta_acceleration_f) = r_current_acceleration - r_previous_acceleration;

            double& r_delta_water_pressure_f = itCurrentNode->FastGetSolutionStepValue(DELTA_WATER_PRESSURE_F);
            const double& r_current_water_pressure = itCurrentNode->FastGetSolutionStepValue(WATER_PRESSURE);
            const double& r_previous_water_pressure = itCurrentNode->FastGetSolutionStepValue(WATER_PRESSURE,1);
            r_delta_water_pressure_f = r_current_water_pressure - r_previous_water_pressure;

            double& r_delta_dt_water_pressure_f = itCurrentNode->FastGetSolutionStepValue(DELTA_DT_WATER_PRESSURE_F);
            const double& r_current_dt_water_pressure = itCurrentNode->FastGetSolutionStepValue(DT_WATER_PRESSURE);
            const double& r_previous_dt_water_pressure = itCurrentNode->FastGetSolutionStepValue(DT_WATER_PRESSURE,1);
            r_delta_dt_water_pressure_f = r_current_dt_water_pressure - r_previous_dt_water_pressure;

        } // for Node parallel

        KRATOS_CATCH( "" )
    }

//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    virtual void SaveCorrectionDxLF(ModelPart& rModelPart)
    {
        KRATOS_TRY

        // The array of nodes
        NodesArrayType& r_nodes = rModelPart.Nodes();

        // The iterator of the first node
        const auto it_node_begin = rModelPart.NodesBegin();

        #pragma omp parallel for schedule(guided,512)
        for (int i = 0; i < static_cast<int>(r_nodes.size()); ++i) {

            ModelPart::NodeIterator itCurrentNode = it_node_begin + i;

            // TODO:


        } // for Node parallel

        KRATOS_CATCH( "" )
    }

//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    virtual double CalculateDxPDotDxLF(ModelPart& rModelPart)
    {
        KRATOS_TRY

        double DxP_dot_DxLF = 0.0;

        // The array of nodes
        NodesArrayType& r_nodes = rModelPart.Nodes();

        // The iterator of the first node
        const auto it_node_begin = rModelPart.NodesBegin();

        // TODO:

        #pragma omp parallel for reduction(+:DxP_dot_DxLF)
        for (int i = 0; i < static_cast<int>(r_nodes.size()); ++i) {
            ModelPart::NodeIterator itCurrentNode = it_node_begin + i;

            // TODO: should I calculate the norm taking into account the time derivatives of the unknowns ?
            const array_1d<double, 3>& r_delta_displacement_f = itCurrentNode->FastGetSolutionStepValue(DELTA_DISPLACEMENT_F);
            const array_1d<double, 3>& r_delta_velocity_f = itCurrentNode->FastGetSolutionStepValue(DELTA_VELOCITY_F);
            const array_1d<double, 3>& r_delta_acceleration_f = itCurrentNode->FastGetSolutionStepValue(DELTA_ACCELERATION_F);
            const double& r_delta_water_pressure_f = itCurrentNode->FastGetSolutionStepValue(DELTA_WATER_PRESSURE_F);
            const double& r_delta_dt_water_pressure_f = itCurrentNode->FastGetSolutionStepValue(DELTA_DT_WATER_PRESSURE_F);

            DxP_dot_DxLF += r_delta_displacement_f[0]*r_delta_displacement_f[0]
                        + r_delta_displacement_f[1]*r_delta_displacement_f[1]
                        + r_delta_displacement_f[2]*r_delta_displacement_f[2]
                        + r_delta_velocity_f[0]*r_delta_velocity_f[0]
                        + r_delta_velocity_f[1]*r_delta_velocity_f[1]
                        + r_delta_velocity_f[2]*r_delta_velocity_f[2]
                        + r_delta_acceleration_f[0]*r_delta_acceleration_f[0]
                        + r_delta_acceleration_f[1]*r_delta_acceleration_f[1]
                        + r_delta_acceleration_f[2]*r_delta_acceleration_f[2]
                        + r_delta_water_pressure_f*r_delta_water_pressure_f
                        + r_delta_dt_water_pressure_f*r_delta_dt_water_pressure_f;

        }
        return DxP_dot_DxLF;

        KRATOS_CATCH( "" )
    }

//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    virtual double CalculateDxPDotDxF(ModelPart& rModelPart)
    {
        KRATOS_TRY

        double DxP_dot_DxF = 0.0;

        // The array of nodes
        NodesArrayType& r_nodes = rModelPart.Nodes();

        // The iterator of the first node
        const auto it_node_begin = rModelPart.NodesBegin();

        // TODO:

        #pragma omp parallel for reduction(+:DxP_dot_DxF)
        for (int i = 0; i < static_cast<int>(r_nodes.size()); ++i) {
            ModelPart::NodeIterator itCurrentNode = it_node_begin + i;

            // TODO: should I calculate the norm taking into account the time derivatives of the unknowns ?
            const array_1d<double, 3>& r_delta_displacement_f = itCurrentNode->FastGetSolutionStepValue(DELTA_DISPLACEMENT_F);
            const array_1d<double, 3>& r_delta_velocity_f = itCurrentNode->FastGetSolutionStepValue(DELTA_VELOCITY_F);
            const array_1d<double, 3>& r_delta_acceleration_f = itCurrentNode->FastGetSolutionStepValue(DELTA_ACCELERATION_F);
            const double& r_delta_water_pressure_f = itCurrentNode->FastGetSolutionStepValue(DELTA_WATER_PRESSURE_F);
            const double& r_delta_dt_water_pressure_f = itCurrentNode->FastGetSolutionStepValue(DELTA_DT_WATER_PRESSURE_F);

            DxP_dot_DxF += r_delta_displacement_f[0]*r_delta_displacement_f[0]
                        + r_delta_displacement_f[1]*r_delta_displacement_f[1]
                        + r_delta_displacement_f[2]*r_delta_displacement_f[2]
                        + r_delta_velocity_f[0]*r_delta_velocity_f[0]
                        + r_delta_velocity_f[1]*r_delta_velocity_f[1]
                        + r_delta_velocity_f[2]*r_delta_velocity_f[2]
                        + r_delta_acceleration_f[0]*r_delta_acceleration_f[0]
                        + r_delta_acceleration_f[1]*r_delta_acceleration_f[1]
                        + r_delta_acceleration_f[2]*r_delta_acceleration_f[2]
                        + r_delta_water_pressure_f*r_delta_water_pressure_f
                        + r_delta_dt_water_pressure_f*r_delta_dt_water_pressure_f;

        }
        return DxP_dot_DxF;

        KRATOS_CATCH( "" )
    }

//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    virtual void UpdateXWithCorrection(ModelPart& rModelPart, const double& DLambda)
    {
        KRATOS_TRY

        // Update unknowns from prediction

        // The array of nodes
        NodesArrayType& r_nodes = rModelPart.Nodes();

        // The iterator of the first node
        const auto it_node_begin = rModelPart.NodesBegin();

        // TODO: noalias(X) += mDxb + DLambda*mDxf;

        #pragma omp parallel for schedule(guided,512)
        for (int i = 0; i < static_cast<int>(r_nodes.size()); ++i) {

            ModelPart::NodeIterator itCurrentNode = it_node_begin + i;

            array_1d<double, 3>& r_current_displacement = itCurrentNode->FastGetSolutionStepValue(DISPLACEMENT);
            const array_1d<double, 3>& r_delta_displacement_p = itCurrentNode->FastGetSolutionStepValue(DELTA_DISPLACEMENT_P);
            noalias(r_current_displacement) += r_delta_displacement_p;

            array_1d<double, 3>& r_current_velocity = itCurrentNode->FastGetSolutionStepValue(VELOCITY);
            const array_1d<double, 3>& r_delta_velocity_p = itCurrentNode->FastGetSolutionStepValue(DELTA_VELOCITY_P);
            noalias(r_current_velocity) += r_delta_velocity_p;

            array_1d<double, 3>& r_current_acceleration = itCurrentNode->FastGetSolutionStepValue(ACCELERATION);
            const array_1d<double, 3>& r_delta_acceleration_p = itCurrentNode->FastGetSolutionStepValue(DELTA_ACCELERATION_P);
            noalias(r_current_acceleration) += r_delta_acceleration_p;

            double& r_current_water_pressure = itCurrentNode->FastGetSolutionStepValue(WATER_PRESSURE);
            const double& r_delta_water_pressure_p = itCurrentNode->FastGetSolutionStepValue(DELTA_WATER_PRESSURE_P);
            r_current_water_pressure += r_delta_water_pressure_p;

            double& r_current_dt_water_pressure = itCurrentNode->FastGetSolutionStepValue(DT_WATER_PRESSURE);
            const double& r_delta_dt_water_pressure_p = itCurrentNode->FastGetSolutionStepValue(DELTA_DT_WATER_PRESSURE_P);
            r_current_dt_water_pressure += r_delta_dt_water_pressure_p;

        } // for Node parallel

        // Move the mesh if needed
        if (BaseType::MoveMeshFlag())
            BaseType::MoveMesh();

        KRATOS_CATCH( "" )
    }

//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    virtual void RestoreF()
    {
        KRATOS_TRY

        // Restore External Loads
        for(unsigned int i = 0; i < mVariableNames.size(); i++)
        {
            ModelPart& rSubModelPart = *(mSubModelPartList[i]);
            const std::string& VariableName = mVariableNames[i];

            if( KratosComponents< Variable<double> >::Has( VariableName ) )
            {
                const Variable<double>& var = KratosComponents< Variable<double> >::Get( VariableName );

                #pragma omp parallel
                {
                    ModelPart::NodeIterator NodesBegin;
                    ModelPart::NodeIterator NodesEnd;
                    OpenMPUtils::PartitionedIterators(rSubModelPart.Nodes(),NodesBegin,NodesEnd);

                    for (ModelPart::NodeIterator itNode = NodesBegin; itNode != NodesEnd; ++itNode)
                    {
                        double& rvalue = itNode->FastGetSolutionStepValue(var);
                        rvalue *= (1.0/mLambda_old);
                    }
                }
            }
            else if( KratosComponents< Variable<array_1d<double,3> > >::Has(VariableName) )
            {
                typedef Variable<double> component_type;
                const component_type& varx = KratosComponents< component_type >::Get(VariableName+std::string("_X"));
                const component_type& vary = KratosComponents< component_type >::Get(VariableName+std::string("_Y"));
                const component_type& varz = KratosComponents< component_type >::Get(VariableName+std::string("_Z"));

                #pragma omp parallel
                {
                    ModelPart::NodeIterator NodesBegin;
                    ModelPart::NodeIterator NodesEnd;
                    OpenMPUtils::PartitionedIterators(rSubModelPart.Nodes(),NodesBegin,NodesEnd);

                    for (ModelPart::NodeIterator itNode = NodesBegin; itNode != NodesEnd; ++itNode)
                    {
                        double& rvaluex = itNode->FastGetSolutionStepValue(varx);
                        rvaluex *= (1.0/mLambda_old);
                        double& rvaluey = itNode->FastGetSolutionStepValue(vary);
                        rvaluey *= (1.0/mLambda_old);
                        double& rvaluez = itNode->FastGetSolutionStepValue(varz);
                        rvaluez *= (1.0/mLambda_old);
                    }
                }
            }
            else
            {
                KRATOS_THROW_ERROR( std::logic_error, "One variable of the applied loads has a non supported type. Variable: ", VariableName )
            }
        }

        KRATOS_CATCH( "" )
    }

//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    virtual void UpdateF()
    {
        KRATOS_TRY

        // Update External Loads
        for(unsigned int i = 0; i < mVariableNames.size(); i++)
        {
            ModelPart& rSubModelPart = *(mSubModelPartList[i]);
            const std::string& VariableName = mVariableNames[i];

            if( KratosComponents< Variable<double> >::Has( VariableName ) )
            {
                const Variable<double>& var = KratosComponents< Variable<double> >::Get( VariableName );

                #pragma omp parallel
                {
                    ModelPart::NodeIterator NodesBegin;
                    ModelPart::NodeIterator NodesEnd;
                    OpenMPUtils::PartitionedIterators(rSubModelPart.Nodes(),NodesBegin,NodesEnd);

                    for (ModelPart::NodeIterator itNode = NodesBegin; itNode != NodesEnd; ++itNode)
                    {
                        double& rvalue = itNode->FastGetSolutionStepValue(var);
                        rvalue *= mLambda;
                    }
                }
            }
            else if( KratosComponents< Variable<array_1d<double,3> > >::Has(VariableName) )
            {
                typedef Variable<double> component_type;
                const component_type& varx = KratosComponents< component_type >::Get(VariableName+std::string("_X"));
                const component_type& vary = KratosComponents< component_type >::Get(VariableName+std::string("_Y"));
                const component_type& varz = KratosComponents< component_type >::Get(VariableName+std::string("_Z"));

                #pragma omp parallel
                {
                    ModelPart::NodeIterator NodesBegin;
                    ModelPart::NodeIterator NodesEnd;
                    OpenMPUtils::PartitionedIterators(rSubModelPart.Nodes(),NodesBegin,NodesEnd);

                    for (ModelPart::NodeIterator itNode = NodesBegin; itNode != NodesEnd; ++itNode)
                    {
                        double& rvaluex = itNode->FastGetSolutionStepValue(varx);
                        rvaluex *= mLambda;
                        double& rvaluey = itNode->FastGetSolutionStepValue(vary);
                        rvaluey *= mLambda;
                        double& rvaluez = itNode->FastGetSolutionStepValue(varz);
                        rvaluez *= mLambda;
                    }
                }
            }
            else
            {
                KRATOS_THROW_ERROR( std::logic_error, "One variable of the applied loads has a non supported type. Variable: ", VariableName )
            }
        }

        // Save the applied Lambda factor
        mLambda_old = mLambda;

        KRATOS_CATCH( "" )
    }

}; // Class PoromechanicsExplicitRammArcLengthStrategy

} // namespace Kratos

#endif // KRATOS_POROMECHANICS_EXPLICIT_RAMM_ARC_LENGTH_STRATEGY  defined
