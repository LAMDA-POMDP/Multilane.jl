##Crashing
include(joinpath("..","src","crash.jl"))

function test_cross2d()
	println("\t\tTesting cross2d")
end

function test_line_segment_intersect()
	println("\t\tTesting line_segment_intersect")
	X = Array{Float64,1}[[0.;0.],[2.;0.],[2.;2.],[0;2.]]
	Y = Array{Float64,1}[[1.;1.],[3.;1.],[3.;3.],[1.;3.]]
	@assert(!line_segment_intersect(X[1],X[2],Y[1],Y[2]))
	@assert(!line_segment_intersect(X[1],X[2],Y[2],Y[3]))
	@assert(!line_segment_intersect(Y[1],Y[2],X[1],X[2]))
	@assert(!line_segment_intersect(Y[2],Y[3],X[1],X[2]))
	@assert(line_segment_intersect(X[2],X[3],Y[1],Y[2]))
	@assert(line_segment_intersect(Y[1],Y[2],X[2],X[3]))
	@assert(line_segment_intersect(X[2],X[3],X[2],X[3]))
	@assert(line_segment_intersect(X[2],X[3],X[3],X[2]))
	@assert(!line_segment_intersect(X[1],X[2],Y[2],Y[1]))
	@assert(!line_segment_intersect(X[1],X[2],Y[3],Y[2]))
	@assert(!line_segment_intersect(Y[1],Y[2],X[2],X[1]))
	@assert(!line_segment_intersect(Y[2],Y[3],X[2],X[1]))
	@assert(line_segment_intersect(X[2],X[3],Y[2],Y[1]))
	@assert(line_segment_intersect(Y[1],Y[2],X[3],X[2]))
	@assert(line_segment_intersect(X[2],X[3],X[3],X[2]))
end

function poly_to_line_segments(X::Array{Array{Float64,1}})
	X_ = [hcat(X[i],X[i+1]) for i=1:(length(X)-1)]
	push!(X_,hcat(X[end],X[1]))
	return X_
end

function test_poly_intersect()
	println("\t\tTesting poly_intersect")
	#CASE: no intersection
	X = Array{Float64,1}[[0.;0.],[1.;0.],[1.;1.],[0;1.]]
	Y = Array{Float64,1}[[3.;3.],[4.;3.],[4.;4.],[3.;4.]]
	@assert(!poly_intersect(poly_to_line_segments(X), poly_to_line_segments(Y)))
	#CASE: intersect at segments --not working
	X = Array{Float64,1}[[0.;0.],[2.;0.],[2.;2.],[0;2.]]
	Y = Array{Float64,1}[[1.;1.],[3.;1.],[3.;3.],[1.;3.]]
	@assert(poly_intersect(poly_to_line_segments(X), poly_to_line_segments(Y)))
	#CASE: one inside other --not supported yet; will fail test!
	X = Array{Float64,1}[[0.;0.],[3.;0.],[3.;3.],[0;3.]]
	Y = Array{Float64,1}[[1.;1.],[2.;1.],[2.;2.],[1.;2.]]
	@assert(poly_intersect(poly_to_line_segments(X), poly_to_line_segments(Y)))
	#CASE: same poly
	X = Array{Float64,1}[[0.;0.],[2.;0.],[2.;2.],[0;2.]]
	Y = Array{Float64,1}[[0.;0.],[2.;0.],[2.;2.],[0;2.]]
	@assert(poly_intersect(poly_to_line_segments(X), poly_to_line_segments(Y)))
	#CASE: same with 1d offset
	X = Array{Float64,1}[[0.;0.],[2.;0.],[2.;2.],[0;2.]]
	Y = Array{Float64,1}[[1.;0.],[3.;0.],[3.;2.],[1;2.]]
	@assert(poly_intersect(poly_to_line_segments(X), poly_to_line_segments(Y)))
	#And permutations of the above
	##TODO
end

function is_crash(mdp::NoCrashMDP,s::MLState,a::MLAction)
	sp, r = @gen(:sp,:r)(mdp,s,a,MersenneTwister(1))
	return is_crash(mdp,s,sp)
end

function test_is_crash()
	println("\t\tTesting is_crash")
	nb_lanes = 2
	pp = PhysicalParam(nb_lanes,lane_length=12.) #2.=>col_length=8
	nb_cars=2
	rmodel = NoCrashRewardModel()
	dmodel = NoCrashIDMMOBILModel(nb_cars, pp)
    p = NoCrashMDP{NoCrashRewardModel, typeof(dmodel.behaviors)}(dmodel, rmodel, 1., true);

	bs = IDMMOBILBehavior[IDMMOBILBehavior(x[1],x[2],x[3],idx) for (idx,x) in enumerate(Iterators.product(["cautious","normal","aggressive"],[pp.v_slow;pp.v_med;pp.v_fast],[pp.l_car]))][:]

	#Env Car out of bounds
	cs_oob = CarState(-1.,1,27.,0,bs[1],1)
	#env car going slow in right lane: (41,1) corresponds to right in front of agent car area + 0.25m or 0.5m
	cs_r_slow = CarState(10.,1,27.,0,IDMMOBILBehavior("aggressive",27.,4.,1),2)
	#env car going fast in right lane: (7,1) corresponds to right behind the agent car area + 0.25m or 0.5m
	cs_r_fast = CarState(1.75,1,35.,0,IDMMOBILBehavior("aggressive",35.,4.,1),3)
	#env car going at a medium speed in the right lane heading left
	cs_lchange = CarState(6.,1,30.,1,IDMMOBILBehavior("normal",30.,4.,1),4)
	#env car going at a medium speed in the left lane heading right
	cs_rchange = CarState(6.,2,30.,-1,IDMMOBILBehavior("normal",30.,4.,1),5)
	#env car is just chilling in the right/btwn/lhigh lane
	cs_rchill = CarState(6.,1,30.,1,IDMMOBILBehavior("normal",30.,4.,1),6)
	cs_mchill = CarState(6.,1.5,30.,1,IDMMOBILBehavior("normal",30.,4.,1),7)
	cs_hchill = CarState(7.,1,30.,1,IDMMOBILBehavior("normal",30.,4.,1),8)

	##TODO: change these tests borrowed from test_reward to something more meaningful
	#env car going slow in right lane: (41,1) corresponds to right in front of agent car area + 0.25m or 0.5m
	_cs_l_slow = CarState(0.75,1,27.,0,IDMMOBILBehavior("aggressive",27.,4.,1),9)
	#env car going fast in right lane: (7,1) corresponds to right behind the agent car area + 0.25m or 0.5m
	_cs_r_fast = CarState(12.,1,35.,0,IDMMOBILBehavior("aggressive",35.,4.,1),10)
	s = MLState(1,27.,CarState[_cs_r_fast],6.)
	a = MLAction(0,0)
	#going out of top boundary
	@assert(!is_crash(p,s,a))
	#going out of bottom boundary
	s = MLState(1,35.,CarState[_cs_l_slow],6.)
	a = MLAction(0,1)
	@assert(!is_crash(p,s,a))
	#CASE: do nothing = no costs
	@assert(!is_crash(p,MLState(1,27.,CarState[cs_oob],6.),MLAction(0,0)))
	#CASE: moving = cost
	#=
	@assert(!is_crash(p,MLState(1,27.,CarState[cs_oob]),MLAction(1,0)))
	@assert(!is_crash(p,MLState(1,27.,CarState[cs_oob]),MLAction(-1,0)))
	@assert(!is_crash(p,MLState(1,27.,CarState[cs_oob]),MLAction(0,1)))
	@assert(!is_crash(p,MLState(1,27.,CarState[cs_oob]),MLAction(1,1)))
	@assert(!is_crash(p,MLState(1,27.,CarState[cs_oob]),MLAction(-1,1)))
	@assert(!is_crash(p,MLState(1,27.,CarState[cs_oob]),MLAction(1,-1)))
	@assert(!is_crash(p,MLState(1,27.,CarState[cs_oob]),MLAction(-1,-1)))
	@assert(!is_crash(p,MLState(1,27.,CarState[cs_oob]),MLAction(0,-1)))
	=#
	#Case: cars occupy same space
	@assert(is_crash(p,MLState(1,30.,CarState[cs_rchill],6.),MLState(1,30.,CarState[cs_rchill],6.)))
	@assert(is_crash(p,MLState(1,30.,CarState[cs_mchill],6.),MLState(1,30.,CarState[cs_mchill],6.)))
	@assert(is_crash(p,MLState(1,30.,CarState[cs_hchill],6.),MLState(1,30.,CarState[cs_hchill],6.)))
	#CASE: cars intersect; vertically (gets railroaded from behind)
	# removed because of noise crash prevention
    # @assert(is_crash(p,MLState(1,27.,CarState[cs_r_fast],6.),MLAction(-1,0)))
	# @assert(is_crash(p,MLState(1,35.,CarState[cs_r_slow],6.),MLAction(1,0)))
	#CASE: cars intersect; horizontally (across lanes)
    # removed because of noise crash prevention
	# @assert(is_crash(p,MLState(1,30.,CarState[cs_rchange],6.),MLAction(0,1)))
	# @assert(is_crash(p,MLState(2,30.,CarState[cs_lchange],6.),MLAction(0,-1)))
	#NOTE edge cases:
	"""
	If a car is going so fast (or slow) that it just completely blows out of the frame
	"""
end


function test_crash()
	println("\tTesting Crashing")
	test_cross2d()
	test_line_segment_intersect()
	test_poly_intersect()
	test_is_crash()
end
