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

#if !defined(KRATOS_TRUSS_FIC_ELEMENT_LINEAR_3D2N_H_INCLUDED )
#define  KRATOS_TRUSS_FIC_ELEMENT_LINEAR_3D2N_H_INCLUDED

// System includes

// External includes

// Project includes
#include "custom_elements/truss_element_linear_3D2N.hpp"
#include "includes/define.h"
#include "includes/variables.h"

namespace Kratos {
/**
 * @class TrussFICElementLinear3D2N
 *
 * @author Ignasi de Pouplana
 */

class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) TrussFICElementLinear3D2N : public TrussElementLinear3D2N
{
public:
    KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(TrussFICElementLinear3D2N);

    TrussFICElementLinear3D2N() {};
    TrussFICElementLinear3D2N(IndexType NewId,
                           GeometryType::Pointer pGeometry);
    TrussFICElementLinear3D2N(IndexType NewId,
                           GeometryType::Pointer pGeometry,
                           PropertiesType::Pointer pProperties);


    ~TrussFICElementLinear3D2N() override;

    /**
     * @brief Creates a new element
     * @param NewId The Id of the new created element
     * @param pGeom The pointer to the geometry of the element
     * @param pProperties The pointer to property
     * @return The pointer to the created element
     */
    Element::Pointer Create(
        IndexType NewId,
        GeometryType::Pointer pGeom,
        PropertiesType::Pointer pProperties
    ) const override;

    /**
     * @brief Creates a new element
     * @param NewId The Id of the new created element
     * @param ThisNodes The array containing nodes
     * @param pProperties The pointer to property
     * @return The pointer to the created element
     */
    Element::Pointer Create(
        IndexType NewId,
        NodesArrayType const& ThisNodes,
        PropertiesType::Pointer pProperties
    ) const override;

    /**
     * @brief This function is designed to make the element to assemble an rRHS vector identified by a variable rRHSVariable by assembling it to the nodes on the variable rDestinationVariable (double version)
     * @details The "AddEXplicit" FUNCTIONS THE ONLY FUNCTIONS IN WHICH AN ELEMENT IS ALLOWED TO WRITE ON ITS NODES.
     * The caller is expected to ensure thread safety hence SET/UNSETLOCK MUST BE PERFORMED IN THE STRATEGY BEFORE CALLING THIS FUNCTION
     * @param rRHSVector input variable containing the RHS vector to be assembled
     * @param rRHSVariable variable describing the type of the RHS vector to be assembled
     * @param rDestinationVariable variable in the database to which the rRHSVector will be assembled
     * @param rCurrentProcessInfo the current process info instance
     */
    void AddExplicitContribution(
        const VectorType& rRHSVector,
        const Variable<VectorType>& rRHSVariable,
        const Variable<double >& rDestinationVariable,
        const ProcessInfo& rCurrentProcessInfo
        ) override;

    /**
     * @brief This function is designed to make the element to assemble an rRHS vector identified by a variable rRHSVariable by assembling it to the nodes on the variable (array_1d<double, 3>) version rDestinationVariable.
     * @details The "AddEXplicit" FUNCTIONS THE ONLY FUNCTIONS IN WHICH AN ELEMENT IS ALLOWED TO WRITE ON ITS NODES.
     * The caller is expected to ensure thread safety hence SET/UNSETLOCK MUST BE PERFORMED IN THE STRATEGY BEFORE CALLING THIS FUNCTION
     * @param rRHSVector input variable containing the RHS vector to be assembled
     * @param rRHSVariable variable describing the type of the RHS vector to be assembled
     * @param rDestinationVariable variable in the database to which the rRHSVector will be assembled
     * @param rCurrentProcessInfo the current process info instance
     */
    void AddExplicitContribution(const VectorType& rRHSVector,
        const Variable<VectorType>& rRHSVariable,
        const Variable<array_1d<double, 3> >& rDestinationVariable,
        const ProcessInfo& rCurrentProcessInfo
        ) override;

    int Check(const ProcessInfo& rCurrentProcessInfo) const override;

protected:

    /**
     * @brief This method computes directly the lumped mass vector
     * @param rMassVector The lumped mass vector
     */
    void CalculateLumpedMassVector(VectorType& rMassVector);

    /**
     * @brief This method computes directly the lumped stiffness vector
     * @param rStiffnessVector The lumped stiffness vector
     */
    void CalculateLumpedStiffnessVector(VectorType& rStiffnessVector,const ProcessInfo& rCurrentProcessInfo);

    /**
     * @brief This method computes directly the lumped damping vector
     * @param rDampingVector The lumped damping vector
     */
    void CalculateLumpedDampingVector(VectorType& rDampingVector, const ProcessInfo& rCurrentProcessInfo);

    void CalculateFrequencyMatrix(MatrixType& rMatrix, const VectorType& rMassVector, const VectorType& rNumNeighElemsVector, const ProcessInfo& rCurrentProcessInfo);

    // void GetAuxiliaryVelocityVector(Vector& rValues, int Step) const;


private:

    friend class Serializer;
    void save(Serializer& rSerializer) const override;
    void load(Serializer& rSerializer) override;
};


}


#endif
