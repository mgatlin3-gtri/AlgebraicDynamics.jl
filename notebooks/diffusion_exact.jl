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
c = VForm([sin(π/a*(p[1] + a/2))*sin(π/b*(p[2] + b/2)) + 1 for p in s[:point]])

# Plot the initial state of the laplacian
@show minimum(∇²(sd, c).data)
@show maximum(∇²(sd, c).data)
r_max = maximum(∇²(sd, c).data)
fig, ax, ob = mesh(s, color=∇²(sd, c).data, colormap=:bluesreds, colorrange=(-r_max, r_max))
ax.aspect = AxisAspect(100.0/30.0)
save("lap.svg", fig)
wireframe!(sd, linewidth=0.3)
wireframe!(s, linewidth=0.3, color=:black)
save("init_cond.svg", fig)

# Initialize the laplacian operator
println("Setting up problem")
lapl = ∇²(Val{0}, sd)

# Initialize physics sim
vf(du, u, p, t) = begin
  du .= (-1 * p[1]) * (lapl * u)
  du[boundary_v] .= 0.0
end

# Run the simulation
println("Computing sim")
tspan = (0.0,100.0)
prob = ODEProblem(vf, c.data, tspan, [C_k])
sol = solve(prob)
println("Finished computing")

# Plot simulation result
times = range(tspan[1], tspan[2], length=300)
colors = [sol(t) for t in times]
println("Plotting Result")
fig, ax, ob = mesh(s, color=colors[1],
                   colorrange= (minimum(vcat(colors...)),
                                maximum(vcat(colors...))))
ax.aspect = AxisAspect(100.0/30.0)
framerate = 30
record(fig, "diffusion.gif", collect(1:length(collect(times))); framerate = framerate) do i
  ob.color = colors[i]
end

# Plot the exact solution
exact_sol(p, t) = sin(π/a*(p[1] + a/2))*sin(π/b*(p[2] + b/2))*exp(-π^2*(1/a^2 + 1/b^2)*C_k*t) + 1
errors = [[exact_sol(p, t) for p in s[:point]] for t in times]
println("Plotting Exact")
fig, ax, ob = mesh(s, color=errors[1],
                   colorrange=(minimum(vcat(errors...)),
                              maximum(vcat(errors...))))
ax.aspect = AxisAspect(100.0/30.0)
framerate = 30
record(fig, "actual.gif", collect(1:length(collect(times))); framerate = framerate) do i
  ob.color = errors[i]
end

# Plot the pointwise difference between the exact and simulated results
errors = [(sol(t) .- [exact_sol(p, t) for p in s[:point]]) #=./ [exact_sol(p, t) for p in s[:point]]=# for t in times]
@show minimum(vcat(errors...))
@show maximum(vcat(errors...))
r_max = maximum(abs.(errors[end]))
println("Plotting Error")
fig, ax, ob = mesh(s, color=errors[1],
                   colorrange=(-r_max, r_max),
                   colormap=:bluesreds)
ax.aspect = AxisAspect(100.0/30.0)
framerate = 30
record(fig, "error.gif", collect(1:length(collect(times))); framerate = framerate) do i
  ob.color = errors[i]
end