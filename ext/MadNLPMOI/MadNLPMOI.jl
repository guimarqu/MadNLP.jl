module MadNLPMOI

import MadNLP
import NLPModels
import MathOptInterface as MOI
import MathOptInterface.Utilities as MOIU
using Printf: @printf

function __init__()
    setglobal!(MadNLP, :Optimizer, Optimizer)
    return
end

include("MOI_utils.jl")
include("MOI_wrapper.jl")
include("diagnostics.jl")

end # module MadNLPMOI
