# importing the Kratos Library
from . import structural_responses
from . import partitioning_responses

def CreateResponseFunction(response_name,response_type,response_settings,model):

    if response_type == "mass": 
        return structural_responses.MassResponseFunction(response_name,response_settings,model)
    elif response_type == "interface":
        return partitioning_responses.InterfaceResponseFunction(response_name,response_settings,model)      
    elif response_type == "partition_mass":
        return partitioning_responses.PartitionMassResponseFunction(response_name,response_settings,model)                     