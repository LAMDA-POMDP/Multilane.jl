# XXX this is all kind of hacky

mutable struct MLPOMDPSolver <: Solver
    solver
    updater::Union{Nothing,Any}
end
function set_rng!(s::MLPOMDPSolver, rng::AbstractRNG)
    set_rng!(s.solver, rng)
    set_rng!(s.updater, rng)
end

mutable struct MLPOMDPAgent <: Policy
    updater::Updater
    previous_belief::Union{Nothing,Any}
    policy::Policy
    previous_action::Union{Nothing,MLAction}
end

function solve(solver::MLPOMDPSolver, problem::MLMDP, up=get(solver.updater, nothing))
    internal_problem = MLPOMDP{MLState, MLAction, MLPhysicalState, typeof(problem.dmodel), typeof(problem.rmodel)}(problem.dmodel, problem.rmodel, problem.discount)
    policy = solve(solver.solver, internal_problem)
    if up === nothing
        up = POMDPs.updater(policy)
    else
        set_problem!(up, internal_problem)
    end
    return MLPOMDPAgent(up, nothing, policy, nothing)
end

function action(agent::MLPOMDPAgent, state::MLState)
    o = MLPhysicalState(state)
    if nothing === (agent.previous_belief)
        belief = initialize_belief(agent.updater, ParticleGenerator(agent.policy.problem, state))
    else
        belief = update(agent.updater, agent.previous_belief, agent.previous_action, o)
    end
    a = action(agent.policy, belief)
    agent.previous_action = a
    agent.previous_belief = belief
    return a
end

mutable struct ParticleGenerator
    physical::MLPhysicalState
    behaviors::BehaviorGenerator
end
function ParticleGenerator(problem::NoCrashProblem, state::Union{MLState, MLPhysicalState})
    return ParticleGenerator(MLPhysicalState(state),
                problem.dmodel.behaviors)
end

function rand(rng::AbstractRNG, gen::ParticleGenerator,
              full_s::MLState=MLState(gen.physical,
                                      Vector{CarState}(
                                            length(gen.physical.cars))))
    s = gen.physical
    resize!(full_s.cars, length(s.cars))
    for i in 1:length(s.cars)
        behavior = rand(rng, gen.behaviors)
        full_s.cars[i] = CarState(s.cars[i], behavior)
    end
    return full_s
end


#=
type BasicMLReinvigorator <: POMCP.ParticleReinvigorator
    N::Int # target number of particles
    p_change::Float64 # (independent) probability of the behavior of each car changing
    behaviors::AbstractVector
    weights::Weights
end

# ideas for future reinvigorators
# 1) change behaviors of newer cars more often 
# 2) change behaviors of cars at the edges of the lanes more often
# 3) change behaviors of the cars that are furthest from their observations most often

function POMCP.reinvigorate!(pc::ParticleCollection, r::BasicMLReinvigorator, old_node::POMCP.BeliefNode, a, o)
    while length(pc.particles) < r.N
        new_s = 
    end
    return pc
end

function POMCP.handle_unseen_observation(r::BasicMLReinvigorator, old_node::POMCP.BeliefNode, a, o)
    pc = ParticleCollection{MLState}()

end
=#
