using POMDPs

# POMDPs.add("GenerativeModels")
# POMDPs.add("POMDPToolbox")
# POMDPs.add("MCTS")
# POMDPs.add("POMCP")

try
    Pkg.clone("https://github.com/tawheeler/Vec.jl.git")
catch
    println("already installed.")
end

try
    Pkg.clone("https://github.com/tawheeler/AutomotiveDrivingModels.jl.git")
catch
    println("already installed.")
end

try
    Pkg.clone("https://github.com/tawheeler/AutoViz.jl.git")
catch
    println("already installed.")
end


try
    Pkg.clone("https://github.com/slundberg/PmapProgressMeter.jl.git")
catch
    println("already installed.")
end

try
    Pkg.clone("https://github.com/JuliaPlots/StatPlots.jl.git")
catch
    println("already installed.")
end
