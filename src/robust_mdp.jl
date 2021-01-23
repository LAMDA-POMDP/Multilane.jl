#=
type RobustNoCrashMDP <: RobustMDP{MLPhysicalState, MLAction}
    base::MLMDP
end
RobustMCTS.representative_mdp(rmdp::RobustNoCrashMDP) = StochasticBehaviorNoCrashMDP(rmdp.base)

function RobustMCTS.next_model(gen::RandomModelGenerator, rmdp::RobustNoCrashMDP, s::MLPhysicalState, a::MLAction)
    behaviors = Dict{Int, Union{Nothing,BehaviorModel}}(1=>nothing)
    for c in s.cars
        # TODO instead of just using sample, try to pick models that will be bad
        behaviors[c.id] = rand(gen.rng, rmdp.base.dmodel.behaviors)
    end
    return FixedBehaviorNoCrashMDP(behaviors, rmdp.base)
end
=#

abstract type EmbeddedBehaviorMDP <: MDP{MLPhysicalState, MLAction} end

POMDPs.actions(p::EmbeddedBehaviorMDP) = actions(p.base)
POMDPs.actions(p::EmbeddedBehaviorMDP, s::MLPhysicalState, as::NoCrashActionSpace) = actions(p.base, s, as)
# create_action(p::EmbeddedBehaviorMDP) = create_action(p.base)
create_state(p::EmbeddedBehaviorMDP) = MLPhysicalState(false, 0.0, 0.0, [])
POMDPs.discount(p::EmbeddedBehaviorMDP) = discount(p.base)
# reward(p::EmbeddedBehaviorMDP, s::MLPhysicalState, a::MLAction, sp::MLPhysicalState)

mutable struct FixedBehaviorNoCrashMDP <: EmbeddedBehaviorMDP
    behaviors::Dict{Int,Union{Nothing,BehaviorModel}}
    base::MLMDP
end

function POMDPs.gen(mdp::FixedBehaviorNoCrashMDP, s::MLPhysicalState, a::MLAction, rng::AbstractRNG)
    full_s = MLState(s, Vector{CarState}(undef, length(s.cars)))
    for (i,c) in enumerate(s.cars) 
        full_s.cars[i] = CarState(c, mdp.behaviors[c.id])
    end
    full_sp = @gen(:sp)(mdp.base, full_s, a, rng)
    sp = create_state(mdp)
    for c in sp.cars
        if !haskey(mdp.behaviors, c.id)
            mdp.behaviors[c.id] = c.behavior
        end
    end
    return (sp=MLPhysicalState(full_sp), r=reward(mdp.base, full_s, a, full_sp), info=nothing)
end

mutable struct StochasticBehaviorNoCrashMDP <: EmbeddedBehaviorMDP
    base::MLMDP
end

function POMDPs.gen(mdp::StochasticBehaviorNoCrashMDP, s::MLPhysicalState, a::MLAction, rng::AbstractRNG)
    full_s = MLState(s, Vector{CarState}(undef, length(s.cars)))
    for i in 1:length(s.cars) 
        full_s.cars[i] = CarState(s.cars[i], rand(rng, mdp.base.dmodel.behaviors))
    end
    full_sp = @gen(:sp)(mdp.base, full_s, a, rng)
    return (sp=MLPhysicalState(full_sp), r=reward(mdp.base, full_s, a, full_sp), info=nothing)
end

function POMDPs.initialstate(mdp::Union{FixedBehaviorNoCrashMDP,StochasticBehaviorNoCrashMDP})
    ImplicitDistribution() do rng
        full_state = rand(rng, initialstate(mdp.base))
        return MLPhysicalState(full_state)
    end
end

function initial_state(mdp::Union{FixedBehaviorNoCrashMDP,StochasticBehaviorNoCrashMDP}, rng::AbstractRNG)
    full_state = initial_state(mdp.base, rng)
    return MLPhysicalState(full_state)
end

#=
type RobustMLSolver <: Solver
    rsolver
end

type RobustMLPolicy <: Policy{MLState}
    rpolicy
end

function action(p::RobustMLPolicy, s::MLState, a::MLAction=MLAction(0,0))
    mdp = representative_mdp(p.rpolicy.rmdp)
    as = actions(mdp, MLPhysicalState(s), actions(mdp))
    if length(as) == 1
        return collect(as)[1]
    end
    action(p.rpolicy, MLPhysicalState(s))
end

function solve(s::RobustMLSolver, mdp::MLMDP)
    RobustMLPolicy(solve(s.rsolver, RobustNoCrashMDP(mdp::MLMDP)))
end
=#
