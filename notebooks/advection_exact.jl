### Exact Solution for Laplacian System from abaas

using CombinatorialSpaces
using CombinatorialSpaces.DiscreteExteriorCalculus: inv_hodge_star
using DifferentialEquations
using CairoMakie
function dual(s::EmbeddedDeltaSet2D{O, P}) where {O, P}
  sd = EmbeddedDeltaDualComplex2D{O, eltype(P), P}(s)
  subdivide_duals!(sd, Circumcenter())
  sd
end

# Define mesh
s = EmbeddedDeltaSet2D("meshes/pipe_fine.stl")
sd = dual(s)

# Define boundary
boundary_e = findall(x -> x != 0, boundary(Val{2},s) * fill(1,ntriangles(s)))
boundary_v = unique(vcat(s[boundary_e,:src],s[boundary_e,:tgt]))

# Define constants
a = 100.0
b = 30.0
C_k = 0.5

# Initial conditions
using Distributions
# Square Wave
x1 = -30.0
x2 = -20.0
y1 = -5.0
y2 = 5.0
c_sq = VForm([(x1 <= p[1] <= x2) && (y1 <= p[2] <= y2) ? 1 : 0 for p in s[:point]])
#Gaussian curve
c_dist = MvNormal([(x1+x2)/2, 0.0], [3.0, 3.0])
c_gauss = VForm([pdf(c_dist, [p[1], p[2]]) for p in s[:point]])

# Initialize straight 1-form
velocity(x) = begin
  amp = 0.01 * 100
  amp * Point{3,Float64}(1,0,0)
end
v = ♭(sd, DualVectorField(velocity.(sd[triangle_center(sd),:dual_point])))

# Initialize the Lie derivative operator
println("Setting up problem")

# Initialize physics sim
vf(du, u, p, t) = begin
  du .= -(1 ./ ⋆(Val{0}, sd).diag) .* ℒ(sd, v, ⋆(sd, VForm(u))).data
  du[boundary_v] .= 0.0
end

# Run the simulation
println("Computing sim")
tspan = (0.0,50.0)
prob = ODEProblem(vf, c_sq.data, tspan, [C_k])
sol_sq = solve(prob)
println("Finished computing")

# Run the simulation
println("Computing sim")
tspan = (0.0,50.0)
prob = ODEProblem(vf, c_gauss.data, tspan, [C_k])
sol_gauss = solve(prob)
println("Finished computing")

# Plot simulation result
times = range(tspan[1], tspan[2], length=100)
colors = [sol_sq(t) for t in times]
println("Plotting Result")
fig, ax, ob = mesh(s, color=colors[1],
                   colorrange= (minimum(vcat(colors...)),
                                maximum(vcat(colors...))))
ax.aspect = AxisAspect(100.0/30.0)
framerate = 30
record(fig, "gifs/advection_square.gif", collect(1:length(collect(times))); framerate = framerate) do i
  ob.color = colors[i]
end

# Plot simulation result
times = range(tspan[1], tspan[2], length=100)
colors = [sol_gauss(t) for t in times]
println("Plotting Result")
fig, ax, ob = mesh(s, color=colors[1],
                   colorrange= (minimum(vcat(colors...)),
                                maximum(vcat(colors...))))
ax.aspect = AxisAspect(100.0/30.0)
framerate = 30
record(fig, "gifs/advection_gauss.gif", collect(1:length(collect(times))); framerate = framerate) do i
  ob.color = colors[i]
end

# Plot the exact solution for Square Wave
# Square Wave
x1 = -30.0
x2 = -20.0
y1 = -5.0
y2 = 5.0
u_exact = 0.01 * 100
s_exact_sol(p, t) = ((x1+u_exact*t) <= p[1] <= (x2+u_exact*t)) && (y1 <= p[2] <= y2) ? 1 : 0
errors = [[s_exact_sol(p, t) for p in s[:point]] for t in times]
println("Plotting Exact")
fig, ax, ob = mesh(s, color=errors[1],
                   colorrange=(minimum(vcat(errors...)),
                              maximum(vcat(errors...))))
ax.aspect = AxisAspect(100.0/30.0)
framerate = 30
record(fig, "gifs/actual_square_adv.gif", collect(1:length(collect(times))); framerate = framerate) do i
  ob.color = errors[i]
end

# Plot the exact solution for Gaussian Wave
# Gaussian
x1 = -30.0
x2 = -20.0
y1 = -5.0
y2 = 5.0
u_exact = 0.01 * 100
c_dist_adv(t) = MvNormal([(x1+x2)/2+u_exact*t, 0.0], [3.0, 3.0])
c_exact_sol(p, t) = pdf(c_dist_adv(t), [p[1], p[2]])
errors = [[c_exact_sol(p, t) for p in s[:point]] for t in times]
println("Plotting Exact")
fig, ax, ob = mesh(s, color=errors[1],
                   colorrange=(minimum(vcat(errors...)),
                              maximum(vcat(errors...))))
ax.aspect = AxisAspect(100.0/30.0)
framerate = 30
record(fig, "gifs/actual_gauss_adv.gif", collect(1:length(collect(times))); framerate = framerate) do i
  ob.color = errors[i]
end

# Plot the pointwise difference between the exact and simulated results
errors = [(sol_sq(t) .- [c_exact_sol(p, t) for p in s[:point]]) #=./ [exact_sol(p, t) for p in s[:point]]=# for t in times]
@show minimum(vcat(errors...))
@show maximum(vcat(errors...))
r_max = maximum(abs.(errors[end]))
println("Plotting Error")
fig, ax, ob = mesh(s, color=errors[1],
                   colorrange=(-r_max, r_max),
                   colormap=:bluesreds)
ax.aspect = AxisAspect(100.0/30.0)
framerate = 30
record(fig, "gifs/error_square_adv.gif", collect(1:length(collect(times))); framerate = framerate) do i
  ob.color = errors[i]
end

# Plot the pointwise difference between the exact and simulated results
errors = [(sol_gauss(t) .- [c_exact_sol(p, t) for p in s[:point]]) #=./ [exact_sol(p, t) for p in s[:point]]=# for t in times]
@show minimum(vcat(errors...))
@show maximum(vcat(errors...))
r_max = maximum(abs.(errors[end]))
println("Plotting Error")
fig, ax, ob = mesh(s, color=errors[1],
                   colorrange=(-r_max, r_max),
                   colormap=:bluesreds)
ax.aspect = AxisAspect(100.0/30.0)
framerate = 30
record(fig, "gifs/error_gauss_adv.gif", collect(1:length(collect(times))); framerate = framerate) do i
  ob.color = errors[i]
end
