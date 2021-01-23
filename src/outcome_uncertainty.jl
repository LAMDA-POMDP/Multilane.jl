"""
MDP with all state uncertainty modeled by outcome uncertainty
"""
struct OutcomeMDP{M} <: MDP{MLPhysicalState, MLAction}
    mdp::M
end

function POMDPs.gen(m::OutcomeMDP, s::MLPhysicalState, a::MLAction, rng::AbstractRNG)
    cars = CarState[]
    for c in s.cars
        cs = CarState(c, rand(rng, m.mdp.dmodel.behaviors))
        push!(cars, cs)
    end
    mls = MLState(s, cars)
    mlsp, r = @gen(:sp,:r)(m.mdp, mls, a, rng)
    return (sp=MLPhysicalState(mlsp), r=r, info=nothing)
end

POMDPs.actions(m::OutcomeMDP, s::MLPhysicalState) = actions(m.mdp, s)
POMDPs.discount(m::OutcomeMDP) = discount(m.mdp)

struct OutcomeSolver <: Solver
    solver::Solver
end

struct OutcomePlanner{P} <: Policy
    planner::P
end

function POMDPs.solve(sol::OutcomeSolver, mdp)
    omdp = OutcomeMDP(mdp)
    return OutcomePlanner(solve(sol.solver, omdp))
end

function POMDPModelTools.action_info(p::OutcomePlanner, s::MLState)
    ps = MLPhysicalState(s)
    return action_info(p.planner, ps)
end
POMDPs.action(p::OutcomePlanner, s::MLState) = first(action_info(p, s))

srand(p::OutcomePlanner, s) = srand(p.planner, s)
