#if !defined(CONSISTENT_L2_ELEMENT_TO_NODE_PROJECTION_UTILITY_H)
#define  CONSISTENT_L2_ELEMENT_TO_NODE_PROJECTION_UTILITY_H

#include "includes/model_part.h"
#include "includes/ublas_interface.h"
#include "utilities/variable_utils.h"
#include "utilities/parallel_utilities.h"

namespace Kratos{


template<unsigned int TDim, unsigned int TNumNodes>
class ConsistentL2ElementToNodeProjection
{
public:

	static_assert(TDim == 2 || TDim == 3,
		"ConsistentL2ElementToNodeProjection is only implemented for 2D and 3D elements");

	ConsistentL2ElementToNodeProjection(
		const Variable<double>& rVariable,
		ModelPart& rModelPart)
	: mrVariable(rVariable), mrModelPart(rModelPart)
	{}

	void Project(const std::size_t NumberOfIterations)
	{
		VariableUtils().SetNonHistoricalVariableToZero(mrVariable, mrModelPart.Nodes());
		SystemOfEquations system_of_equations(mrModelPart.NumberOfNodes());
		
		InitialAssembly(system_of_equations);

		for(unsigned int i=0; i<NumberOfIterations; ++i)
		{
			IterationAssembly(system_of_equations);
			Solve(system_of_equations);
		}
	}

protected:
	const Variable<double>& mrVariable;
	ModelPart& mrModelPart;

	struct SystemOfEquations
	{
		SystemOfEquations(const std::size_t size = 0)
			: mSize(size), Mu(Vector(size)), Q(Vector(size))
		{
			mNodeToDof.reserve(size);
		}
		
		// System size
		const std::size_t mSize;

		// Computed every step:
		Vector Mu;

		// Cached data
		Vector Q;

		// Node to dof mapping
		std::unordered_map<std::size_t, std::size_t> mNodeToDof;

	private:
		SystemOfEquations() : mSize(0) {};
	};

	/* @brief Reduction to assemble a global vector from elemental ones
	 *
	 */
	struct AssemblyReduction
	{
	    typedef Vector value_type;
	    typedef value_type return_type;

	    typedef std::array<unsigned int, TNumNodes> DofType;
	    typedef array_1d<double, TNumNodes> ArrayType;
	    typedef std::size_t SizeType;
	    typedef std::tuple<const DofType, const ArrayType, const SizeType> ReturnType; 

	    value_type mGlobalVector;
	    bool mVectorIsInitialized = false;

	    return_type GetValue() const
	    {
	        return mGlobalVector;
	    }

	    /// NON-THREADSAFE (fast) value of reduction, to be used within a single thread
	    void LocalReduce(const ReturnType& rLambdaReturn)
	    {
	    	const auto& dofs 		 = std::get<0>(rLambdaReturn);
	    	const auto& local_vector = std::get<1>(rLambdaReturn);

	    	if(!mVectorIsInitialized)
    		{
	    		const auto system_size  = std::get<2>(rLambdaReturn);
    			mGlobalVector = Vector(system_size, 0.0);
    			mVectorIsInitialized = true;
    		}

	        for(unsigned int i=0; i<TNumNodes; ++i)
	        {
	        	mGlobalVector[dofs[i]] += local_vector(i);
	        }

	    }

	    /// THREADSAFE (needs some sort of lock guard) reduction, to be used to sync threads
	    void ThreadSafeReduce(AssemblyReduction& rOther)
	    {
	        if(!rOther.mVectorIsInitialized) return;

	        #pragma omp critical
	        {
		        if(!mVectorIsInitialized)
	        	{
	        		mGlobalVector.swap(rOther.mGlobalVector);
	        		mVectorIsInitialized = true;
	        		rOther.mVectorIsInitialized = false;
	        	} else {
	        		noalias(mGlobalVector) += rOther.mGlobalVector;
	        	}
	    	}
	    }

	    /** @brief Boilerplate to ensure const-ness is well deduced.
		 * Automatic template deduction fails to set most types to const, causing the compilation to fail
		 * or sometimes causing undefined behaviour. This function bypasses it by explicitly stating the types.
		 */
	    static const ReturnType AssemblyReturn(
			const DofType& rDofs,
			const ArrayType& rVector,
			const SizeType& rSystemSize)
	    {
	    	return std::tie<const DofType, const ArrayType, const SizeType>
	    	(
	    		rDofs,
	    		rVector,
	    		rSystemSize
	    	);
	    }
	};

	/**
	* Constructs the cached vector:
	* - Q = \int elemental_value* N_i dV
	* These vectors do not change each iteration and therefore can be cached
	*/
	void InitialAssembly(SystemOfEquations& rSystemOfEquations) const
	{
		KRATOS_TRY

		// Mapping form node to dof
		unsigned int i=0;
		for(const auto& r_node : mrModelPart.Nodes())
		{
			rSystemOfEquations.mNodeToDof[r_node.Id()] = i;
			++i;
		}

		// Asembling Q
		rSystemOfEquations.Q = 
		block_for_each<AssemblyReduction>(mrModelPart.Elements(), [&](const Element& rElement)
		{
			const auto& r_geometry = rElement.GetGeometry();
	        const double volume = ComputeVolume(r_geometry);
	        const double nodal_volume = volume / static_cast<double>(TNumNodes);

			// Obtaining local vectors
			const array_1d<double, TNumNodes> q(TNumNodes, rElement.GetValue(mrVariable) * nodal_volume);

			// Obtaining dofs
			std::array<unsigned int, TNumNodes> dofs;
			for(unsigned int i=0; i<TNumNodes; ++i)
			{
				dofs[i] = rSystemOfEquations.mNodeToDof[r_geometry[i].Id()];
			}

			return AssemblyReduction::AssemblyReturn(dofs, q, rSystemOfEquations.mSize);
		});

		KRATOS_CATCH("")
	}


	/**
	 *	Assembles the iteration-dependent vector:
	 *  - Mu = M_consistent * prev_iteration_nodal_values
	 */
	void IterationAssembly(SystemOfEquations& rSystemOfEquations) const
	{
		KRATOS_TRY

		// Asembly

		rSystemOfEquations.Mu = 
		block_for_each<AssemblyReduction>(mrModelPart.Elements(), 
			[&](const Element& rElement)
		{
			const auto& r_geometry = rElement.GetGeometry();

	        constexpr auto integration_method = GeometryData::GI_GAUSS_2;
			const auto& r_integration_points = r_geometry.IntegrationPoints(integration_method);
		    const auto& r_N = r_geometry.ShapeFunctionsValues(integration_method);

			// Obtaining previous iteration data and DoFs
			array_1d<double, TNumNodes> nodal_values;
			std::array<unsigned int, TNumNodes> dofs;
			for(unsigned int i=0; i<TNumNodes; ++i)
			{
				nodal_values[i] = r_geometry[i].GetValue(mrVariable);
				dofs[i] = rSystemOfEquations.mNodeToDof[r_geometry[i].Id()];
			}

			// Obtaining local mass matrix
			BoundedMatrix<double, TNumNodes, TNumNodes> M_consistent = ZeroMatrix(TNumNodes, TNumNodes);
			for(unsigned int g=0; g<r_integration_points.size(); ++g)
			{
				const auto N = row(r_N, g);
				const auto w = r_integration_points[g].Weight();
		    	const auto J = r_geometry.DeterminantOfJacobian(g, integration_method);
				M_consistent += w * J * outer_prod(N, N);
			}

			// Obtaining local matrices
			const array_1d<double, TNumNodes> elemental_Mu = prod(M_consistent, nodal_values);

			return AssemblyReduction::AssemblyReturn(dofs, elemental_Mu, rSystemOfEquations.mSize);
		});

		KRATOS_CATCH("")
	}

	void Solve(SystemOfEquations& rSystemOfEquations)
	{
		KRATOS_TRY

		// Solving
		Vector RHS = rSystemOfEquations.Q - rSystemOfEquations.Mu;

		// Storing result
		block_for_each(mrModelPart.Nodes(), [&](Node<3>& r_node)
		{
			const auto dof = rSystemOfEquations.mNodeToDof[r_node.Id()];
			const double lhs = r_node.GetValue(NODAL_AREA);
			AtomicAdd(r_node.GetValue(mrVariable), RHS[dof] / lhs);
		});

		KRATOS_CATCH("")
	}

	/**
	 * @brief Computes the volume for 2D elements
	 */
	double ComputeVolume(const Geometry<Node<3>>& rGeometry) const
	{
		switch(TDim){
			case 2: return rGeometry.Area();
			case 3: return rGeometry.Volume();
		}
		KRATOS_ERROR << "ConsistentL2ElementToNodeProjection is only implemented for 2D and 3D" << std::endl;
	}
};

}

#endif