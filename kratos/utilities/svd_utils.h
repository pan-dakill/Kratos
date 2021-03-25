//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ `
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:		 BSD License
//					 Kratos default license: kratos/license.txt
//
//  Main authors:    Vicente Mataix Ferrandiz
//

#if !defined(KRATOS_SVD_UTILS )
#define  KRATOS_SVD_UTILS


/* System includes */


/* External includes */

/* Project includes */
#include "utilities/math_utils.h"
#include "spaces/ublas_space.h"


namespace Kratos
{

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
 * @class SVDUtils
 * @ingroup KratosCore
 * @brief Various mathematical utilities to compute SVD and the condition number of a matrix
 * @details Defines several utility functions
 * @author Vicente Mataix Ferrandiz
 */
template<class TDataType>
class SVDUtils
{
public:

    ///@name Type Definitions
    ///@{

    /// Definition of the matrix type
    typedef Matrix MatrixType;

    /// Definition of the vector type
    typedef Vector VectorType;

    /// Definition of the size type
    typedef std::size_t SizeType;

    /// Definition of index type
    typedef std::size_t IndexType;

    /// Definition of local space
    typedef UblasSpace<TDataType, Matrix, Vector> LocalSpaceType;

    /// Definition of epsilon zero tolerance
    constexpr static TDataType ZeroTolerance = std::numeric_limits<double>::epsilon();

    ///@}
    ///@name Life Cycle
    ///@{

    /* Constructor */


    /** Destructor */

    ///@}
    ///@name Operators
    ///@{


    ///@}
    ///@name Operations
    ///@{

    /**
     * @brief This function gives the SVD of a given mxn matrix (m>=n), returns U,S; where A=U*S*V
     * @details U and V are unitary, and S is a diagonal matrix.
     * Where s_i >= 0, and s_i >= s_i+1 (which means that the biggest number is the first one and the smallest the last one)
     * @todo This version is quite innefficient, look for a real and mathematical implementation (not the algorithm found in Wikipedia!!)
     * @param rInputMatrix The matrix where perform the SVD
     * @param rUMatrix The unitary U matrix
     * @param rSMatrix The diagonal S matrix
     * @param rVMatrix The unitary V matrix
     * @param ThisParameters The configuration parameters
     * @return iter: The number of iterations
     */
    static inline std::size_t SingularValueDecomposition(
        const MatrixType& rInputMatrix,
        MatrixType& rUMatrix,
        MatrixType& rSMatrix,
        MatrixType& rVMatrix,
        Parameters ThisParameters)
    {
        // Validating defaults
        Parameters default_parameters = Parameters(R"(
        {
            "type_svd"             : "Jacobi",
            "tolerance"            : 0.0,
            "max_iter"             : 200
        })");
        default_parameters["tolerance"].SetDouble(std::numeric_limits<double>::epsilon());
        ThisParameters.RecursivelyValidateAndAssignDefaults(default_parameters);

        const std::string& r_type_svd = ThisParameters["type_svd"].GetString();
        const double tolerance = ThisParameters["tolerance"].GetDouble();
        const double max_iter = ThisParameters["max_iter"].GetInt();
        return SingularValueDecomposition(rInputMatrix, rUMatrix, rSMatrix, rVMatrix, r_type_svd, tolerance, max_iter);
    }

    /**
     * @brief This function gives the SVD of a given mxn matrix (m>=n), returns U,S; where A=U*S*V
     * @details U and V are unitary, and S is a diagonal matrix.
     * Where s_i >= 0, and s_i >= s_i+1 (which means that the biggest number is the first one and the smallest the last one)
     * @todo This version is quite innefficient, look for a real and mathematical implementation (not the algorithm found in Wikipedia!!)
     * @param rInputMatrix The matrix where perform the SVD
     * @param rUMatrix The unitary U matrix
     * @param rSMatrix The diagonal S matrix
     * @param rVMatrix The unitary V matrix
     * @param TypeSVD The type of SVD algorithm (Jacobi by default)
     * @param Tolerance The tolerance considered
     * @param MaxIter Maximum number of iterations
     * @return iter: The number of iterations
     */
    static inline std::size_t SingularValueDecomposition(
        const MatrixType& rInputMatrix,
        MatrixType& rUMatrix,
        MatrixType& rSMatrix,
        MatrixType& rVMatrix,
        const std::string& TypeSVD = "Jacobi",
        const TDataType Tolerance = std::numeric_limits<double>::epsilon(),
        const IndexType MaxIter = 200)
    {
        if (TypeSVD == "Jacobi") {
            return JacobiSingularValueDecomposition(rInputMatrix, rUMatrix, rSMatrix, rVMatrix, Tolerance, MaxIter);
        } else {
            KRATOS_ERROR << "SVD Type not implemented" << std::endl;
        }
    }

    /**
     * @brief This function gives the Jacobi SVD of a given mxn matrix (m>=n), returns U,S; where A=U*S*V
     * @details U and V are unitary, and S is a diagonal matrix.
     * Where s_i >= 0, and s_i >= s_i+1 (which means that the biggest number is the first one and the smallest the last one)
     * @todo This version is quite innefficient, look for a real and mathematical implementation (not the algorithm found in Wikipedia!!)
     * @param rInputMatrix The matrix where perform the SVD
     * @param rUMatrix The unitary U matrix
     * @param rSMatrix The diagonal S matrix
     * @param rVMatrix The unitary V matrix
     * @param Tolerance The tolerance considered
     * @param MaxIter Maximum number of iterations
     * @return iter: The number of iterations
     */

    static inline std::size_t JacobiSingularValueDecomposition(
        const MatrixType& rInputMatrix,
        MatrixType& rUMatrix,
        MatrixType& rSMatrix,
        MatrixType& rVMatrix,
        const TDataType Tolerance = std::numeric_limits<double>::epsilon(),
        const IndexType MaxIter = 200)
    {
        const SizeType m = rInputMatrix.size1();
        const SizeType n = rInputMatrix.size2();

        if(rSMatrix.size1() != m || rSMatrix.size2() != n) {
            rSMatrix.resize(m, n, false);
        }
        noalias(rSMatrix) = rInputMatrix;

        if(rUMatrix.size1() != m || rUMatrix.size2() != m) {
            rUMatrix.resize(m, m, false);
        }
        noalias(rUMatrix) = IdentityMatrix(m);

        if(rVMatrix.size1() != n || rVMatrix.size2() != n) {
            rVMatrix.resize(n, n, false);
        }
        noalias(rVMatrix) = IdentityMatrix(n);

        const TDataType relative_tolerance = Tolerance * LocalSpaceType::TwoNorm(rInputMatrix);

        IndexType iter = 0;

        // We create the auxiliar matrices (for aliased operations)
        MatrixType auxiliar_matrix_mn(m, n);
        MatrixType auxiliar_matrix_m(m, m);
        MatrixType auxiliar_matrix_n(n, n);

        // More auxiliar operators
        MatrixType j1(m, m);
        MatrixType j2(n, n);

        // We compute Jacobi
        while (LocalSpaceType::JacobiNorm(rSMatrix) > relative_tolerance) {
            for (IndexType i = 0; i < n; i++) {
                for (IndexType j = i+1; j < n; j++) {
                    Jacobi(j1, j2, rSMatrix, m, n, i, j);

                    const MatrixType aux_matrix = prod(rSMatrix, j2);
                    noalias(rSMatrix) = prod(j1, aux_matrix);
                    noalias(auxiliar_matrix_m) = prod(rUMatrix, trans(j1));
                    noalias(rUMatrix) = auxiliar_matrix_m;
                    noalias(auxiliar_matrix_n) = prod(trans(j2), rVMatrix);
                    noalias(rVMatrix) = auxiliar_matrix_n;
                }

                for (IndexType j = n; j < m; j++) {
                    MatrixType j1(m, m);

                    Jacobi(j1, rSMatrix, m, i, j);

                    noalias(auxiliar_matrix_mn) = prod(j1, rSMatrix);
                    noalias(rSMatrix) = auxiliar_matrix_mn;
                    noalias(auxiliar_matrix_m) = prod(rUMatrix, trans(j1));
                    noalias(rUMatrix) = auxiliar_matrix_m;
                }
            }

            ++iter;
            if (iter > MaxIter) {
                KRATOS_WARNING("JacobiSingularValueDecomposition") << "Maximum number of iterations " << MaxIter << " reached." << std::endl;
                break;
            }
        }

        return iter;
    }

    /**
     * @brief This function gives the Jacobi SVD of a given 2x2 matrix, returns U,S; where A=U*S*V
     * @details U and V are unitary, and S is a diagonal matrix.
     * Where s_i >= 0, and s_i >= s_i+1
     * @param rInputMatrix The matrix where perform the SVD
     * @param rUMatrix The unitary U matrix
     * @param rSMatrix The diagonal S matrix
     * @param rVMatrix The unitary V matrix
     */
    static inline void SingularValueDecomposition2x2(
        const MatrixType& rInputMatrix,
        MatrixType& rUMatrix,
        MatrixType& rSMatrix,
        MatrixType& rVMatrix
        )
    {
        const TDataType t = (rInputMatrix(0, 1) - rInputMatrix(1, 0))/(rInputMatrix(0, 0) + rInputMatrix(1, 1));
        const TDataType c = 1.0/std::sqrt(1.0 + t*t);
        const TDataType s = t*c;
        MatrixType r_matrix(2, 2);
        r_matrix(0, 0) =  c;
        r_matrix(0, 1) = -s;
        r_matrix(1, 0) =  s;
        r_matrix(1, 1) =  c;

        MatrixType m_matrix = prod(r_matrix, rInputMatrix);

        SingularValueDecomposition2x2Symmetric(m_matrix, rUMatrix, rSMatrix, rVMatrix);

        MatrixType auxiliar_matrix_m(rUMatrix.size1(), rUMatrix.size2());
        noalias(auxiliar_matrix_m) = prod(trans(r_matrix), rUMatrix);
        noalias(rUMatrix) = auxiliar_matrix_m;
    }

	/**
     * This function gives the Jacobi SVD of a given 2x2 matrix, returns U,S; where A=U*S*V
     * U and V are unitary, and S is a diagonal matrix.
     * Where s_i >= 0, and s_i >= s_i+1
     * @param rInputMatrix The matrix where perform the SVD
     * @param rUMatrix The unitary U matrix
     * @param rSMatrix The diagonal S matrix
     * @param rVMatrix The unitary V matrix
     */
    static inline void SingularValueDecomposition2x2Symmetric(
        const MatrixType& rInputMatrix,
        MatrixType& rUMatrix,
        MatrixType& rSMatrix,
        MatrixType& rVMatrix
        )
    {
        if(rSMatrix.size1() != 2 || rSMatrix.size2() != 2) {
            rSMatrix.resize(2 ,2, false);
        }
        if(rUMatrix.size1() != 2 || rUMatrix.size2() != 2) {
            rUMatrix.resize(2, 2, false);
        }
        if(rVMatrix.size1() != 2 || rVMatrix.size2() != 2) {
            rVMatrix.resize(2, 2, false);
        }

        if (std::abs(rInputMatrix(1, 0)) < ZeroTolerance) { // Already symmetric
            noalias(rSMatrix) = rInputMatrix;
            noalias(rUMatrix) = IdentityMatrix(2);
            noalias(rVMatrix) = rUMatrix;
        } else {
            const TDataType w = rInputMatrix(0, 0);
            const TDataType y = rInputMatrix(1, 0);
            const TDataType z = rInputMatrix(1, 1);
            const TDataType ro = (z - w)/(2.0 * y);
            const TDataType t = MathUtils<TDataType>::Sign(ro)/(std::abs(ro) + std::sqrt(1 + ro * ro));
            const TDataType c = 1.0/(std::sqrt(1.0 + t*t));
            const TDataType s = t*c;

            rUMatrix(0, 0) =  c;
            rUMatrix(0, 1) =  s;
            rUMatrix(1, 0) = -s;
            rUMatrix(1, 1) =  c;
            noalias(rVMatrix) = trans(rUMatrix);

            noalias(rSMatrix) = prod(trans(rUMatrix), MatrixType(prod(rInputMatrix, trans(rVMatrix))));
        }

        MatrixType z_matrix(2, 2);
        z_matrix(0, 0) = MathUtils<TDataType>::Sign(rSMatrix(0, 0));
        z_matrix(0, 1) = 0.0;
        z_matrix(1, 0) = 0.0;
        z_matrix(1, 1) = MathUtils<TDataType>::Sign(rSMatrix(1, 1));

        // Auxiliar matrix for alias operations
        MatrixType aux_2_2_matrix(2, 2);
        noalias(aux_2_2_matrix) = prod(rUMatrix, z_matrix);
        noalias(rUMatrix) = aux_2_2_matrix;
        noalias(aux_2_2_matrix) = prod(z_matrix, rSMatrix);
        noalias(rSMatrix) = aux_2_2_matrix;

        if (rSMatrix(0, 0) < rSMatrix(1, 1)) {
            MatrixType p_matrix(2, 2);
            p_matrix(0, 0) = 0.0;
            p_matrix(0, 1) = 1.0;
            p_matrix(1, 0) = 1.0;
            p_matrix(1, 1) = 0.0;

            noalias(aux_2_2_matrix) = prod(rUMatrix, p_matrix);
            noalias(rUMatrix) = aux_2_2_matrix;
            const MatrixType aux_matrix = prod(rSMatrix, p_matrix);
            noalias(rSMatrix) = prod(p_matrix, aux_matrix);
            noalias(aux_2_2_matrix) = prod(p_matrix, rVMatrix);
            noalias(rVMatrix) = aux_2_2_matrix;
        }
    }

    /**
     * @brief This method computes the Jacobi rotation operation
     * @param rJ1 First Jacobi matrix
     * @param rJ2 Second Jacobi matrix
     * @param rInputMatrix The matrix to compute the Jacobi tolerance
     * @param Size1 The size of the matrix (number of rows)
     * @param Size2 The size of the matrix (number of columns)
     * @param Index1 The index to compute (row)
     * @param Index2 The index to compute (column)
     */
    static inline void Jacobi(
        MatrixType& rJ1,
        MatrixType& rJ2,
        const MatrixType& rInputMatrix,
        const SizeType Size1,
        const SizeType Size2,
        const SizeType Index1,
        const SizeType Index2
        )
    {
        MatrixType b_matrix(2,2);
        b_matrix(0, 0) = rInputMatrix(Index1, Index1);
        b_matrix(0, 1) = rInputMatrix(Index1, Index2);
        b_matrix(1, 0) = rInputMatrix(Index2, Index1);
        b_matrix(1, 1) = rInputMatrix(Index2, Index2);

        MatrixType u_matrix, s_matrix, v_matrix;

        SingularValueDecomposition2x2(b_matrix, u_matrix, s_matrix, v_matrix);

        rJ1 = IdentityMatrix(Size1);
        rJ1(Index1, Index1) = u_matrix(0, 0);
        rJ1(Index1, Index2) = u_matrix(1, 0);
        rJ1(Index2, Index1) = u_matrix(0, 1);
        rJ1(Index2, Index2) = u_matrix(1, 1);

        rJ2 = IdentityMatrix(Size2);
        rJ2(Index1, Index1) = v_matrix(0, 0);
        rJ2(Index1, Index2) = v_matrix(1, 0);
        rJ2(Index2, Index1) = v_matrix(0, 1);
        rJ2(Index2, Index2) = v_matrix(1, 1);
    }

    /**
     * @brief This method computes the Jacobi rotation operation
     * @param rJ1 First Jacobi matrix
     * @param rInputMatrix The matrix to compute the Jacobi tolerance
     * @param Size1 The size of the matrix (number of rows)
     * @param Size2 The size of the matrix (number of columns)
     * @param Index1 The index to compute (row)
     * @param Index2 The index to compute (column)
     */
    static inline void Jacobi(
        MatrixType& rJ1,
        const MatrixType& rInputMatrix,
        const SizeType Size1,
        const SizeType Index1,
        const SizeType Index2
        )
    {
        MatrixType b_matrix(2,2);
        b_matrix(0, 0) = rInputMatrix(Index1, Index1);
        b_matrix(0, 1) = 0.0;
        b_matrix(1, 0) = rInputMatrix(Index2, Index1);
        b_matrix(1, 1) = 0.0;

        MatrixType u_matrix, s_matrix, v_matrix;

        SingularValueDecomposition2x2(b_matrix, u_matrix, s_matrix, v_matrix);

        noalias(rJ1) = IdentityMatrix(Size1);
        rJ1(Index1, Index1) = u_matrix(0, 0);
        rJ1(Index1, Index2) = u_matrix(1, 0);
        rJ1(Index2, Index1) = u_matrix(0, 1);
        rJ1(Index2, Index2) = u_matrix(1, 1);
    }

    /**
     * @brief This method computes the condition number using the SVD
     * @details The condition number can be estimated as the ratio between the largest singular value and the smallest singular value
     * @param rInputMatrix The matrix to be evaluated
     * @param Tolerance The tolerance considered
     * @return condition_number: The ratio between the largest SV and the smallest SV
     */
    static inline TDataType SVDConditionNumber(
        const MatrixType& rInputMatrix,
        const std::string TypeSVD = "Jacobi",
        const TDataType Tolerance = std::numeric_limits<double>::epsilon(),
        const IndexType MaxIter = 200)
    {
        MatrixType u_matrix, s_matrix, v_matrix;
        SingularValueDecomposition(rInputMatrix, u_matrix, s_matrix, v_matrix, TypeSVD, Tolerance, MaxIter);

        const SizeType size_s = s_matrix.size1();
        const TDataType condition_number = s_matrix(0, 0)/s_matrix(size_s - 1, size_s - 1);

        return condition_number;
    }

    ///@}
    ///@name Access
    ///@{


    ///@}
    ///@name Inquiry
    ///@{


    ///@}
    ///@name Input and output
    ///@{

    ///@}
    ///@name Friends
    ///@{

private:

    ///@name Private static Member Variables
    ///@{

    ///@}
    ///@name Private member Variables
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
    ///@name Private LifeCycle
    ///@{

    ///@}
    ///@name Unaccessible methods
    ///@{

    SVDUtils(void);

    SVDUtils(SVDUtils& rSource);

}; /* Class SVDUtils */

///@name Type Definitions
///@{


///@}
///@name Input and output
///@{

}  /* namespace Kratos.*/

#endif /* KRATOS_SVD_UTILS  defined */
