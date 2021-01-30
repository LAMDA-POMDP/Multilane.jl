# redundant
struct QMDPState{B,S}
    isstate::Bool
    b::Union{Nothing,B}
    s::Union{Nothing,S}

    function QMDPState{B,S}(isstate::Bool, x) where {B,S}
        if isstate
            return new(isstate, nothing, x)
        else
            return new(isstate, x, nothing)
        end
    end
end

struct QMDPWrapper{M,B,S,A} <: MDP{QMDPState{B,S}, A}
    mdp::M
end

function QMDPWrapper(mdp::MDP, B::Type)
    S = statetype(mdp)
    A = actiontype(mdp)
    return QMDPWrapper{typeof(mdp), B, S, A}(mdp)
end

state(m::QMDPWrapper{<:Any,B,S,<:Any}, s::S) where {B,S} = QMDPState{B,S}(true, s)
state(m::QMDPWrapper{<:Any,B,S,<:Any}, b::B) where {B,S} = QMDPState{B,S}(false, b)

function POMDPs.isterminal(m::QMDPWrapper, s::QMDPState)
    if s.isstate
        return isterminal(m, s.s)
    else
        return isterminal(m, s.b)
    end
end

function POMDPs.gen(m::QMDPWrapper, s::QMDPState, a, rng::AbstractRNG)
    if s.isstate
        sp, r = @gen(:sp,:r)(m.mdp, s.s, a, rng)
    else
        sp, r = @gen(:sp,:r)(m.mdp, rand(rng, s.b), a, rng)
    end
    return (sp=statetype(m)(true, sp), r=r)
end

POMDPs.discount(m::QMDPWrapper) = discount(m.mdp)
function POMDPs.actions(m::QMDPWrapper, s::QMDPState)
    if s.isstate
        return actions(m.mdp, s.s)
    else
        # Gallium.@enter actions(m.mdp, s.b)
        return actions(m.mdp, s.b)
    end
end

struct GenQMDPSolver <: Solver
    solver
end

struct GenQMDPPolicy{P<:Policy, Q<:QMDPWrapper} <: Policy
    policy::P
    qmdp::Q
end

POMDPs.solve(sol::GenQMDPSolver, qmdp::QMDPWrapper) = GenQMDPPolicy(solve(sol.solver, qmdp), qmdp)

function POMDPModelTools.action_info(p::GenQMDPPolicy, b)
    # XXX if this ever makes it into the toolbox, need to get the mdp some other way
    s = QMDPState{typeof(b), statetype(p.qmdp.mdp)}(false, b)
    return action_info(p.policy, s)
end

POMDPs.action(p::GenQMDPPolicy, b) = first(action_info(p, b))
