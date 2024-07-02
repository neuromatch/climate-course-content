### A Pluto.jl notebook ###
# v0.19.24

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 1c8d2d00-b7d9-11eb-35c4-47f2a2aa1593
begin
    import Pkg
	ENV["JULIA_MARGO_LOAD_PYPLOT"] = "no thank you"
    Pkg.activate(mktempdir())
    Pkg.add([
        Pkg.PackageSpec(name="Plots", version="1"),
        Pkg.PackageSpec(name="ClimateMARGO", version=v"0.3.3"),
        Pkg.PackageSpec(name="PlutoUI", version="0.7"),
        Pkg.PackageSpec(name="HypertextLiteral", version="0.9"),
		Pkg.PackageSpec(name="Underscores", version="2"),
    ])
	
	using Plots
	using Plots.Colors
	using ClimateMARGO
	using ClimateMARGO.Models
	using ClimateMARGO.Optimization
	using ClimateMARGO.Diagnostics
	using PlutoUI
    using HypertextLiteral
	using Underscores
	
	Plots.default(linewidth=5)
end;

# ╔═╡ 9a48a08e-7281-473c-8afc-7ad3e0771269
TableOfContents()

# ╔═╡ 94415ff2-32a2-4b0f-9911-3b93e202f548
const initial_1 = Dict("M" => [2090, 6]);

# ╔═╡ 50d24c91-61ae-4544-98fa-5749bafe3d41
# md"""
# ## Overview of the climate problem: from greenhouse gas emissions to climate suffering

# Human emissions of greenhouse gases, especially Carbon Dioxide (CO₂), increase the stock of greenhouse gases in the atmosphere. For every molecule of CO₂ emitted, about 50% are taken up by plants, soils, or the ocean within a few years, while the rest remains in the atmosphere. (The effects of other greenhouse gases, such as Methane and CFCs, and other forcing agents, can approximately be converted into the "CO₂-equivalent"– or CO₂ₑ– concentrations that would lead to the same climate forcing).

# Greenhouse gases get their name because they trap invisible heat radiation emitted by Earth's surface and atmosphere from escaping to space, much like greenhouses trap hot air from rising when it is warmed by the Sun. This "greenhouse effect" causes the temperature to rise globally, although some places warm *more* and *faster* than others. Warmer temperatures exacerbate both the frequency and intensity of "natural" disasters, such as heat waves, coastal flooding from major hurricanes, and inland flooding from torrential rain. These climate impacts lead to enhanced climate suffering, which economics typically attempt to quantify suffering in terms of lost money or welfare.

# In the interactive article below, we invite you to explore the benefits of emissions mitigation and carbon dioxide removal in reducing climate suffering, and the trade-offs with their costs.
# """

# ╔═╡ e810a90f-f964-4d7d-acdb-fc3a159dc12e
const initial_2 = Dict("M" => [2080, .7]);

# ╔═╡ a3422533-2b78-4bc2-92bd-737da3c8982d
const initial_3 = Dict("M" => [2080, .7]);

# ╔═╡ 51037451-0fea-4021-8824-56911970b97b
const initial_x = Dict("G" => [2030, 1]);

# ╔═╡ 3094a9eb-074d-46c3-9c1e-0a9c94c6ad43
blob(el, color = "red") = @htl("""<div style="
background: $(color);
padding: 1em 1em 1em 1em;
border-radius: 2em;
">$(el)</div>""")

# ╔═╡ 0b31eac2-8efd-47cd-9571-a2053846343b
function infeasablewarning(x)
	@htl("""
<margo-infeasible>
	$(x)
	<margo-infeasible-label>No solution found</margo-infeasible-label>
</margo-infeasible>

<style>
margo-infeasible > * {
	opacity: .1;
}
margo-infeasible > margo-infeasible-label {
	opacity: .7;
	display: block;
	position: absolute;
	transform: translate(-50%, -50%);
	font-family: "Vollkorn", system-ui;
	font-style: italic;
	font-size: 40px;
	top: 50%;
	left: 50%;
	
    white-space: nowrap;
}


</style>
	""")
end

# ╔═╡ ca104939-a6ca-4e70-a47a-1eb3c32db18f
status_ok(x) = x ∈ [
    "OPTIMAL",
    "LOCALLY_SOLVED",
    "ALMOST_OPTIMAL",
    "ALMOST_LOCALLY_SOLVED"
  ]

# ╔═╡ cf90139c-13d8-42a7-aba3-8c431e7854b8
feasibility_overlay(x) = status_ok(x.status) ? as_html : infeasablewarning

# ╔═╡ bd2bfa3c-a42e-4975-a543-84541f66b1c1
begin
	hidecloack(name) = HTML("""
	<style>
	plj-cloack.$(name) {
		opacity: 0;
		display: block;
	}
	</style>
	""")
	
	"A trick to hide a cell without creating a variable dependency, to make it simpler for PlutoSliderServer.jl."
	cloak(name) = x -> @htl("<plj-cloack class=$(name)>$(x)</plj-cloak>")
end

# ╔═╡ b81de514-2506-4243-8235-0b54dd4a7ec9
colors = (
	baseline=colorant"#dddddd",
	baseline_emissions=colorant"#dddddd",
	baseline_concentrations=colorant"#dddddd",
	baseline_temperature=colorant"#dddddd",
	baseline_damages=colorant"#dddddd",
	temperature=colorant"#edc949",
	above_paris=colorant"#e1575910",
	M=colorant"#4E79A7",
	R=colorant"#F28E2C",
	A=colorant"#59A14F",
	G=colorant"#E15759",
	T_max=colorant"#00000080",
	controls=colorant"#af7aa1",
	damages=colorant"#e15759",
	avoided_damages=colorant"#e49734",
	benefits=colorant"#7abf5e",
	emissions=colorant"brown",
	emissions_1=colorant"#4E79A7",
	concentrations=colorant"brown",
)

# ╔═╡ 73e01bd8-f56b-4bb5-a9a2-85ad223c9e9b
names = (
	baseline="Baseline",
	baseline_emissions="Baseline",
	baseline_concentrations="Baseline",
	baseline_temperature="Baseline",
	baseline_damages="Baseline",
	temperature="Temperature",
	above_paris="Above Paris",
	M="Mitigation",
	R="Removal",
	A="Adaptation",
	G="Geo-engineering",
	T_max="Goal temperature",
	controls="Controls",
	damages="Damages",
	avoided_damages="Avoided Damages",
	benefits="Benefits",
	emissions="Emissions",
	emissions_1="Emissions",
	concentrations="Concentrations",
)

# ╔═╡ ae92ba1f-5175-4704-8240-2de8432df752
@assert keys(colors) == keys(names)

# ╔═╡ 8ac04d55-9034-4c29-879b-3b10887a616d
begin
	struct BondDefault
		x
		default
	end
	
	Base.get(bd::BondDefault) = bd.default
	Base.show(io::IO, m::MIME"text/html", bd::BondDefault) = Base.show(io, m, bd.x)
	
	BondDefault
end

# ╔═╡ e846c6e2-aa63-40db-8592-c9563bbbdd40
@bind which_graph_2 Select([
		"Emissions"
		"Concentrations"
		"Temperature"
		])

# ╔═╡ 70173466-c9b5-4227-8fba-6256fc1ecace
Tmax_9_slider = @bind Tmax_9 Slider(0:0.1:5; default=2);

# ╔═╡ 6bcb9b9e-e0ab-45d3-b9b9-3d7282f89df6
allow_overshoot_9_cb = @bind allow_overshoot_9 CheckBox();

# ╔═╡ b428e2d3-e1a9-4e4e-a64f-61048572102f
function multiplier(unit::Real, factor::Real=2, suffix::String="%")
	h = @htl("""
	
	
	<script>
		const unit = $(unit)
		const factor = $(factor)
		const suffix = $(suffix)
  const input = html`<input type=range min=-1 max=1 step=.01 value=0>`;
  const output = html`<input disabled style="width: 1.8em; display: inline-block;overflow-x: hidden;"></input>`;
  // const output = html``;

  const left = Math.round(100 / factor) + "%";
  const right = Math.round(100 * factor) + "%";

  const reset = html`<a href="#" title="Reset" style='padding-left: .5em'><img width="14" src="https://cdn.jsdelivr.net/gh/ionic-team/ionicons@5.0.0/src/svg/arrow-undo-sharp.svg"></img></a>`;
  const span = html`<div style="margin-left: 2em;">\${left}\${input}\${right}\${reset}</div>`;

  const on_slider = () => {
    output.value = Math.round(100 * Math.pow(factor, input.value));
    input.title = Math.round(100 * Math.pow(factor, input.value)) + "%";

    reset.style.opacity = input.valueAsNumber == 0 ? "0" : "1";
  };
  input.oninput = on_slider;
  on_slider();

  //   const on_box = () => {
  //     input.value = output.value;

  //     reset.style.opacity = input.valueAsNumber == 100 ? "0" : "1";
  //   };
  //   output.oninput = on_box;

  reset.onclick = (e) => {
    input.value = 0;
    on_slider();
		e.preventDefault()
    span.dispatchEvent(new CustomEvent("input", {}));
  };

  Object.defineProperty(span, "value", {
    get: () => unit * Math.pow(factor, input.value),
    set: val => {
      input.value = Math.log2(val / unit) / Math.log2(factor);
      on_slider();
    }
  });

  return span;
	</script>
	""")
	
	BondDefault(h, unit)
end

# ╔═╡ 8cab3d28-a457-4ccc-b053-38cd003bf4d1
function Carousel(
		elementsList;
		wraparound::Bool=false,
		peek::Bool=true,
	)
	
	@assert peek
	
    carouselHTML = map(elementsList) do element
        @htl("""<div class="carousel-slide">
            $(element)
        </div>""")
    end
	
    h = @htl("""
<div>
    <style>
    .carousel-box{
        width: 100%;
        overflow: hidden;
    }
    .carousel-container{
        top: 0;
        left: 0;
        display: flex;
        width: 100%;
        flex-flow: row nowrap;
        transform: translate(10%, 0px);
        transition: transform 200ms ease-in-out;
    }
    .carousel-controls{
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .carousel-controls button{
        margin: 8px;
        width: 6em;
    }
    .carousel-slide {
        min-width: 80%;
    }
    </style>
		
    <script>
        const div = currentScript.parentElement
        const buttons = div.querySelectorAll("button")
		
		const max = $(length(elementsList))

		let count = 0
		
		const mod = (n, m) => ((n % m) + m) % m
		const clamp = (x, a, b) => Math.max(Math.min(x, b), a)
		
		const update_ui = (count) => {
			buttons[0].disabled = !$(wraparound) && count === 0
			buttons[1].disabled = !$(wraparound) && count === max - 1
		
			div.querySelector(".carousel-container").style = `transform: translate(\${10-count*80}%, 0px)`;
		}
		
		const onclick = (e) => {
			const new_count = count + parseInt(e.target.dataset.value)
			if($(wraparound)){
				count = mod(new_count, max)
			} else {
				count = clamp(new_count, 0, max - 1)
			}
			
            
			div.value = count + 1
			div.dispatchEvent(new CustomEvent("input"))
			update_ui(div.value - 1)
            e.preventDefault()
        }
        buttons.forEach(button => button.addEventListener("click", onclick))
        div.value = count + 1
		update_ui(div.value - 1)
    </script>
		
    <div class="carousel-box">
        <div class="carousel-container">
            $(carouselHTML)
        </div>
    </div>
		
    <div class="carousel-controls">
        <button data-value="-1">Previous</button>
        <button data-value="1">Next</button>
    </div>
</div>
    """)
	
	BondDefault(h,1)
end

# ╔═╡ c7cbc172-daed-406f-b24b-5da2cc234c29
preindustrial_concentrations = 280

# ╔═╡ b440cd13-36a9-4c54-9d80-ac3fa7c2900e
end_of_oil = 2150 # cannot mitigate when fossil fuels are depleted

# ╔═╡ ec760706-15ac-4a50-a67e-c338d70f3b0a
pp = (;
	((k, (:color => c, :label => n))
	for (k, c, n) in zip(keys(names), colors, names))...
);

# ╔═╡ bb4b25e4-0db5-414b-a384-0a27fe7efb66
gauss_stdev = 30

# ╔═╡ 013807a0-bddb-448b-9300-f7f559e48a45
begin
	default_usage_error = :(error("Example usage:\n\n@intially [1,2] @bind x f(x)\n"))
	
	macro initially(::Any)
		default_usage_error
	end
	
	macro initially(default, bind_expr::Expr)
		if bind_expr.head != :macrocall || bind_expr.args[1] != Symbol("@bind")
			return default_usage_error
		end
		
		# warn if the first argument is a @bind
		if default isa Expr && default.head == :macrocall && default.args[1] == Symbol("@bind")
			return default_usage_error
		end
			
		esc(intially_function(default, bind_expr))
	end
	
	
	function intially_function(default, bind_expr)
		sym = bind_expr.args[3]
		@gensym setval bond

		quote
			if !@isdefined($sym)
				$sym = $default
			end

			$setval = $sym


			$bond = @bind $sym $(bind_expr.args[4])
			PlutoRunner.Bond

			if $sym isa Missing
				$sym = $setval
			end

			$bond
		end
	end
end

# ╔═╡ 4e91fb48-fc5e-409e-9a7e-bf846f1d211d
html"""
<style>

margo-knob {
	display: block;
	cursor: pointer;
	width: 32px;
	height: 32px;
	transform: translate(-8px, -16px);
	background: red;
	border-radius: 100%;
	border-width: 5px;
	border-style: solid;
	border-color: rgb(255 255 255 / 43%);
	border-opacity: .2;

    position: absolute;
	top: 0px;
	left: 0px;
}

margo-knob-label {
	transform: translate(32px, -8px);
    display: block;
    position: absolute;
    left: 0;
    top: 0;
	white-space: nowrap;
    background: #d6eccb;
    font-family: system-ui;
    padding: .4em;
    border-radius: 11px;
    font-weight: 600;
	pointer-events: none;
	opacity: 0;
}

.wiggle margo-knob {
	animation: wiggle-margo-knob 5s ease-in-out;
	animation-delay: 600ms;
}
.wiggle margo-knob-label {
	animation: fadeout 1s ease-in-out;
	animation-delay: 3s;
	animation-fill-mode: both;
}

@keyframes fadeout {
	from {
		opacity: 1;
	}
	to {
		opactiy: 0;
	}
}

@keyframes wiggle-margo-knob {
	0% {
		transform: translate(-8px, -16px);
	}
	2% {
		transform: translate(8px, -16px);
	}
	5% {
		transform: translate(-24px, -16px);
	}
	10% {
		transform: translate(-8px, -16px);
	}
	/* 15% {
		transform: translate(-8px, -16px);
	}
	17% {
		transform: translate(-8px, 0px);
	}
	20% {
		transform: translate(-8px, -32px);
	}
	25% {
		transform: translate(-8px, -16px);
	}*/
}

</style>
"""

# ╔═╡ 3c7271ab-ece5-4ae2-a8dd-dc3670f300f7
# initial_mrga_1 = Dict(
# 	"M" => [2070, 0.7],
# 	"R" => [2100, 0.4],
# 	"G" => [2170, 0.3],
# 	"A" => [2110, 0.1],
# )

# ╔═╡ dcf265c1-f09b-483e-a361-d54c6c7500c1
# @initially initial_mrga_1 @bind input_8 begin
	
# 	controls_8 = MRGA(
# 		gaussish(input_8["M"]...),
# 		gaussish(input_8["R"]...),
# 		gaussish(input_8["G"]...),
# 		gaussish(input_8["A"]...),
# 	)
	
	
# 	plotclicktracker2(
# 		plot_controls(controls_8),
# 		initial_mrga_1,
# 	)
# end

# ╔═╡ 10c015ec-780c-4453-83cb-12dd0f09f358
function plotclicktracker(p::Plots.Plot; draggable::Bool=false)

	# we need to render the plot before its dimensions are available:
	# plot_render = repr(MIME"image/svg+xml"(),  p)
	plot_render = repr(MIME"image/svg+xml"(),  p)

	# these are the _bounding boxes_ of our plot
	big = bbox(p.layout)
	small = plotarea(p[1])

	# the axis limits
	xl = xlims(p)
	yl = ylims(p)

	# with this information, we can form the linear transformation from 
	# screen coordinate -> plot coordinate

	# this is done on the JS side, to avoid one step in the Julia side
	# we send the linear coefficients:
	r = (
		x_offset = xl[1] - (xl[2] - xl[1]) * small.x0[1] / small.a[1],
		x_scale = (big.a[1] / small.a[1]) * (xl[2] - xl[1]),
		y_offset = (yl[2] - yl[1]) + (small.x0[2] / small.a[2]) * (yl[2] - yl[1]) + yl[1],
		y_scale = -(big.a[2]/ small.a[2]) * (yl[2] - yl[1]),
		x_min = xl[1], # TODO: add margin
		x_max = xl[2],
		y_min = yl[1],
		y_max = yl[2],
	)

	HTML("""<script id="hello">

		const body = $(PlutoRunner.publish_to_js(plot_render))
		const mime = "image/svg+xml"


		const img = this ?? document.createElement("img")


		let url = URL.createObjectURL(new Blob([body], { type: mime }))

		img.type = mime
		img.src = url
		img.draggable = false
		img.style.cursor = "pointer"
		
		const clamp = (x,a,b) => Math.min(Math.max(x, a), b)
		img.transform = f => [
			clamp(f[0] * $(r.x_scale) + $(r.x_offset), $(r.x_min), $(r.x_max)),
			clamp(f[1] * $(r.y_scale) + $(r.y_offset), $(r.y_min), $(r.y_max)),
		]
		img.fired = false
		
		const val = {current: undefined }
		


		if(this == null) {

		Object.defineProperty(img, "value", {
			get: () => val.current,
			set: () => {},
		})
		
		const handle_mouse = (e) => {
			const svgrect = img.getBoundingClientRect()
			const f = [
				(e.clientX - svgrect.left) / svgrect.width, 
				(e.clientY - svgrect.top) / svgrect.height
			]
			if(img.fired === false){
				img.fired = true
				val.current = img.transform(f)
				img.dispatchEvent(new CustomEvent("input"), {})
			}
		}



		img.addEventListener("click", onclick)

		img.addEventListener("pointerdown", e => {
			if($(draggable)){
				img.addEventListener("pointermove", handle_mouse);
			}
			handle_mouse(e);
		});
		const mouseup = e => {
			img.removeEventListener("pointermove", handle_mouse);
		};
		document.addEventListener("pointerup", mouseup);
		document.addEventListener("pointerleave", mouseup);
		}



		return img
		</script>""")
end

# ╔═╡ 7e540eaf-8700-4176-a96c-77ee2e4c384b
years = 2020:12.0:2200

# ╔═╡ 646591c4-cb60-41cd-beb9-506807ce17d2
function gaussish(mean, magnitude)
	my_stdev = gauss_stdev * (1 + magnitude);
	map(years) do t
	clamp(
		(1.5 *
		magnitude *
		(-0.4 +
		  exp(
			(-1 * ((t - mean) * (t - mean))) / (2 * my_stdev * my_stdev)
		  ))) /
		(1.0 - 0.4),
		0.0,
		1.0
	)
	end
end

# ╔═╡ 6fb77b13-7a54-4d1d-9985-4735318680e1
function expcontrol(mean, magnitude)
	sulfur_fraction = 0.33
	initial_value = sulfur_fraction
	map(years) do t
	sulfur_fraction + (magnitude-sulfur_fraction)*(1-exp(-(t-2020)/((mean-2020))))
	end
end

# ╔═╡ 8fa94ec9-1fab-41b9-a7e6-1917e975e4ff
function default_parameters()::ClimateModelParameters
	result = deepcopy(ClimateMARGO.IO.included_configurations["default"])
	result.domain = years isa Domain ? years : Domain(step(years), first(years), last(years))
	result.economics.baseline_emissions = ramp_emissions(result.domain)
    result.economics.extra_CO₂ = zeros(size(result.economics.baseline_emissions))
	return result
end

# ╔═╡ 785c428d-d4f7-431e-94d7-039b0708a78a
function opt_controls_temp(model_parameters = default_parameters(); opt_parameters...)
    
    model = ClimateModel(model_parameters)

    model_optimizer = optimize_controls!(model; opt_parameters..., print_raw_status=false)
	
	(
		result=model, 
		status=ClimateMARGO.Optimization.JuMP.termination_status(model_optimizer) |> string,
	)
    # return Dict(
    #     :model_parameters => model_parameters,
    #     model_results(model)...
    # )
end

# ╔═╡ 2dcd5669-c725-40b9-84c4-f8399f6e924b
bigbreak = html"""
<div style="height: 10em;"></div>
""";

# ╔═╡ b8f9efec-63ac-4e58-93cf-9f7199b78451
function setfieldconvert!(value, name::Symbol, x)
    setfield!(value, name, convert(typeof(getfield(value, name)), x))
end

# ╔═╡ 371991c7-13dd-46f6-a730-ad89f43c6f0e
function enforce_maxslope!(controls;
		dt=step(years),
		max_slope=Dict("mitigate"=>1. /40., "remove"=>1. /40., "geoeng"=>1. /80., "adapt"=> 0.)
    )
    controls.mitigate[1] = 0.0
    controls.remove[1] = 0.0
    controls.geoeng[1] = 0.0
    # controls.adapt[1] = 0.0


    for i in 2:length(controls.mitigate)
        controls.mitigate[i] = clamp(
            controls.mitigate[i], 
            controls.mitigate[i-1] - max_slope["mitigate"]*dt, 
            controls.mitigate[i-1] + max_slope["mitigate"]*dt
        )
        controls.remove[i] = clamp(
            controls.remove[i], 
            controls.remove[i-1] - max_slope["remove"]*dt, 
            controls.remove[i-1] + max_slope["remove"]*dt
        )
        controls.geoeng[i] = clamp(
            controls.geoeng[i], 
            controls.geoeng[i-1] - max_slope["geoeng"]*dt, 
            controls.geoeng[i-1] + max_slope["geoeng"]*dt
        )
        controls.adapt[i] = clamp(
            controls.adapt[i], 
            controls.adapt[i-1] - max_slope["adapt"]*dt, 
            controls.adapt[i-1] + max_slope["adapt"]*dt
        )
    end
end

# ╔═╡ e815d175-1c47-4aef-bd89-e7fdc3e4912e
function enforce_maxslope!2(controls;
		dt=step(years),
		max_slope=Dict("mitigate"=>1. /40., "remove"=>1. /40., "geoeng"=>1. /80., "adapt"=> 0.)
    )
    controls.mitigate[1] = 0.0
    controls.remove[1] = 0.0
    controls.geoeng[1] = 0.0
    # controls.adapt[1] = 0.0


    for i in 2:length(controls.mitigate)
        controls.mitigate[i] = clamp(
            controls.mitigate[i], 
            controls.mitigate[i-1] - max_slope["mitigate"]*dt, 
            controls.mitigate[i-1] + max_slope["mitigate"]*dt
        )
        controls.remove[i] = clamp(
            controls.remove[i], 
            controls.remove[i-1] - max_slope["remove"]*dt, 
            controls.remove[i-1] + max_slope["remove"]*dt
        )
        # controls.geoeng[i] = clamp(
        #     controls.geoeng[i], 
        #     controls.geoeng[i-1] - max_slope["geoeng"]*dt, 
        #     controls.geoeng[i-1] + max_slope["geoeng"]*dt
        # )
        controls.adapt[i] = clamp(
            controls.adapt[i], 
            controls.adapt[i-1] - max_slope["adapt"]*dt, 
            controls.adapt[i-1] + max_slope["adapt"]*dt
        )
    end
end

# ╔═╡ b7ca316b-6fa6-4c2e-b43b-cddb08aaabbb
function costs_dict(costs, model)
    Dict(
        :discounted => costs,
        :total_discounted => sum(costs .* model.domain.dt),
    )
end

# ╔═╡ 0a3be2ea-6af6-43c0-b8fb-e453bc2b703b

model_results(model::ClimateModel) = Dict(
    :controls => model.controls,
    :computed => Dict(
        :temperatures => Dict(
            :baseline => T_adapt(model),
            :M => T_adapt(model; M=true),
            :MR => T_adapt(model; M=true, R=true),
            :MRG => T_adapt(model; M=true, R=true, G=true),
            :MRGA => T_adapt(model; M=true, R=true, G=true, A=true),
        ),
        :emissions => Dict(
            :baseline => effective_emissions(model),
            :M => effective_emissions(model; M=true),
            :MRGA => effective_emissions(model; M=true, R=true),
        ),
        :concentrations => Dict(
            :baseline => c(model),
            :M => c(model; M=true),
            :MRGA => c(model; M=true, R=true),
        ),
        :damages => Dict(
            :baseline => costs_dict(damage(model; discounting=true), model),
            :MRGA => costs_dict(damage(model; M=true, R=true, G=true, A=true, discounting=true), model),
        ),
        :costs => Dict(
            :M => costs_dict(cost(model; M=true, discounting=true), model),
            :R => costs_dict(cost(model; R=true, discounting=true), model),
            :G => costs_dict(cost(model; G=true, discounting=true), model),
            :A => costs_dict(cost(model; A=true, discounting=true), model),
            :MRGA => costs_dict(cost(model; M=true, R=true, G=true, A=true, discounting=true), model),
        ),
    ),
)


# ╔═╡ eb0c961d-42cf-4219-a36e-cd492fa31f6b
const cost_bars_scale = 70

# ╔═╡ ec5d87a6-354b-4f1d-bb73-b3db08589d9b
total_discounted(costs, model) = sum(costs .* model.domain.dt)

# ╔═╡ 70f01a4d-0aa3-4cd5-ad71-452c490c61ac
colors_js = Dict((k,string("#", hex(v))) for (k,v) in pairs(colors));

# ╔═╡ ac779b93-e19e-41de-94cb-6a2a919bcd2e
names_js = Dict(pairs(names));

# ╔═╡ 5c484595-4646-484f-9e75-a4a3b4c2af9b
function plotclicktracker2(p::Plots.Plot, initial::Dict; draggable::Bool=true)

	# we need to render the plot before its dimensions are available:
	# plot_render = repr(MIME"image/svg+xml"(),  p)
	plot_render = repr(MIME"image/svg+xml"(),  p)

	# these are the _bounding boxes_ of our plot
	big = bbox(p.layout)
	small = plotarea(p[1])

	# the axis limits
	xl = xlims(p)
	yl = ylims(p)

	# with this information, we can form the linear transformation from 
	# screen coordinate -> plot coordinate

	# this is done on the JS side, to avoid one step in the Julia side
	# we send the linear coefficients:
	r = (
		x_offset = xl[1] - (xl[2] - xl[1]) * small.x0[1] / small.a[1],
		x_scale = (big.a[1] / small.a[1]) * (xl[2] - xl[1]),
		y_offset = (yl[2] - yl[1]) + (small.x0[2] / small.a[2]) * (yl[2] - yl[1]) + yl[1],
		y_scale = -(big.a[2]/ small.a[2]) * (yl[2] - yl[1]),
		x_min = xl[1], # TODO: add margin
		x_max = xl[2],
		y_min = yl[1],
		y_max = yl[2],
		aspect_ratio = big.a[1] / big.a[2],
	)

	@htl("""<script id="hello">
		
		const initial = $(initial)
		
		
		const colors = $(colors_js)
		const names = $(names_js)
		

		const body = $(PlutoRunner.publish_to_js(plot_render))
		const mime = "image/svg+xml"

		
		const knob = (name) => {
			const k = html`<margo-knob id=\${name}><margo-knob-label>👈 Move me!</margo-knob-label></margo-knob>`
			k.style.backgroundColor = colors[name]
			return k
		}
		
		const aspect_ratio = $(r.aspect_ratio)
		
		const wrapper = this ?? html`
			<div style='touch-action: none;'>
				<img style='width: 100%; aspect-ratio: \${aspect_ratio}; background: white;'>
				\${Object.keys(initial).map(knob)}
			</div>
		`
		const img = wrapper.firstElementChild

		let url = URL.createObjectURL(new Blob([body], { type: mime }))
		invalidation.then(() => {
			URL.revokeObjectURL(url)
		})
		
		// Call `fetch` on the URL to trigger the browser to make it ready. 
		let fetch_promise = fetch(url)
		Promise.race([
			fetch_promise, 
   			invalidation.then(x => null)
		]).then((r) => {
			if(r != null) {
				img.type = mime
				img.src = url
				img.draggable = false
			}
		})
		
		const clamp = (x,a,b) => Math.min(Math.max(x, a), b)
		wrapper.transform = f => [
			clamp(f[0] * $(r.x_scale) + $(r.x_offset), $(r.x_min), $(r.x_max)),
			clamp(f[1] * $(r.y_scale) + $(r.y_offset), $(r.y_min), $(r.y_max)),
		]
		wrapper.inversetransform = f => [
			(f[0] - $(r.x_offset)) / $(r.x_scale),
			(f[1] - $(r.y_offset)) / $(r.y_scale),
		]
		
		const set_knob_coord = (k, coord) => {
			const svgrect = img.getBoundingClientRect()
			const r = wrapper.inversetransform(coord)
			k.style.left = `\${r[0] * svgrect.width}px`
			k.style.top = `\${r[1] * svgrect.height}px`
		}
		
		wrapper.fired_already = false
		
		
		wrapper.last_render_time = Date.now()
		

		// If running for the first time
		if(this == null) {
			console.log("Creating new plotclicktracker...")
		
			// will contain the currently dragging HTMLElement
			const dragging = { current: undefined }
		
			const value = {...initial}
		
			Object.defineProperty(wrapper, "value", {
				get: () => value,
				set: (x) => {
					/* console.log("old", value, "new", x)
					Object.assign(value, x)
					Object.entries(value).forEach(([k,v]) => {
						set_knob_coord(
							wrapper.querySelector(`margo-knob#\${k}`),
							v
						)
					}) */
				},
			})
		
		
			
			
			////
			// Event listener for pointer move
		
			const allow_only_one_event_per_render = false
			const on_pointer_move = (e) => {
				if(Object.keys(initial).includes(dragging.current.id)){

					const svgrect = img.getBoundingClientRect()
					const f = [
						(e.clientX - svgrect.left) / svgrect.width, 
						(e.clientY - svgrect.top) / svgrect.height
					]
					if(!allow_only_one_event_per_render || wrapper.fired_already === false){
						const new_coord = wrapper.transform(f)
						value[dragging.current.id] = new_coord
						set_knob_coord(dragging.current, new_coord)
		
						wrapper.classList.toggle("wiggle", false)
						wrapper.fired_already = true
						wrapper.dispatchEvent(new CustomEvent("input"), {})
					}
				}
			}

			
			////
			// Add the listeners

			wrapper.addEventListener("pointerdown", e => {
				window.getSelection().empty()
		
				
				dragging.current = e.target
				if($(draggable)){
					wrapper.addEventListener("pointermove", on_pointer_move);
				}
				on_pointer_move(e);
			});
			const mouseup = e => {
				wrapper.removeEventListener("pointermove", on_pointer_move);
			};
			document.addEventListener("pointerup", mouseup);
			document.addEventListener("pointerleave", mouseup);
			wrapper.onselectstart = () => false
		
		
		
			////
			// Set knobs to initial positions, using the inverse transformation
		
			new Promise(r => {
				img.onload = r
			}).then(() => {
				Array.from(wrapper.querySelectorAll("margo-knob")).forEach(k => {
					set_knob_coord(k, initial[k.id])
				})
			})
		
			////
			// Intersection observer to trigger to wiggle animation
			const observer = new IntersectionObserver((es) => {
				es.forEach((e) => {
					if(Date.now() - wrapper.last_render_time > 500){
						wrapper.classList.toggle("wiggle", e.isIntersecting)
					}
				})
			}, {
				rootMargin: `20px`,
				threshold: 1,
			})
		
			observer.observe(wrapper)
			wrapper.classList.toggle("wiggle", true)
		}



		return wrapper
		</script>""")
end

# ╔═╡ 060cbeab-5503-4eda-95d8-3f554765b2ee
begin
	mutable struct MRGA{T}
		M::T
		R::T
		G::T
		A::T
	end
	# function MRGA(M::TM,R::TR=nothing,G::TG=nothing) where {TM,TR,TG}
	# 	MRGA{Union{TM,TR,TG,Nothing}}(M,R,G,nothing)
	# end
	function MRGA(;M::TM=nothing,R::TR=nothing,G::TG=nothing,A::TA=nothing) where {TM,TR,TG,TA}
		MRGA{Union{TM,TR,TG,TA}}(M,R,G,A)
	end
	
	MRGA(x) = MRGA(x,x,x,x)
	
	splat(mrga::MRGA) = (:M => mrga.M, :R => mrga.R, :G => mrga.G, :A => mrga.A)
	
	Base.collect(mrga::MRGA) = [mrga.M, mrga.R, mrga.G, mrga.A]
	
	Base.getindex(x::MRGA, pos::MRGA{Bool}) = collect(x)[collect(pos)]

	Base.getindex(x::MRGA, tech::Symbol) = getfield(x, tech)

	Base.eachindex(m::MRGA) = (:M,:R,:G,:A)

	Base.enumerate(mrga::MRGA) = ((:M, mrga.M), (:R, mrga.R), (:G, mrga.G), (:A, mrga.A))

	Base.any(m::MRGA{Bool}) = m.M || m.R || m.G || m.A

	Base.all(m::MRGA{Bool}) = m.M && m.R && m.G && m.A

	MRGA
end

# ╔═╡ 1bc00b68-15c5-48fe-9994-dc2f543542df
md"""
# Tutorial 3 - IAMs and SSPs
Week 2, Day 3: IPCC Socio-economic Basis

By Neuromatch Academy

Content creators: Maximilian Puelma Touzel

Content reviewers:  Peter Ohue, Derick Temfack, Zahra Khodakaramimaghsoud, Peizhen Yang, Younkap Nina DuplexLaura Paccini, Sloane Garelick, Abigail Bodner, Manisha Sinha, Agustina Pesce, Dionessa Biton, Cheng Zhang, Jenna Pearson, Chi Zhang, Ohad Zivan

Content editors: Jenna Pearson, Chi Zhang, Ohad Zivan

Production editors: Wesley Banfield, Jenna Pearson, Chi Zhang, Ohad Zivan

Our 2023 Sponsors: NASA TOPS
"""

# ╔═╡ 2dacd4cd-68bd-4c31-857e-d8cab04b5180
md"""
# Tutorial Objectives
In this tutorial you'll explore how climate policy best adapts to potential obstacles that can arise in the upcoming years.

You'll run [MARGO](https://github.com/ClimateMARGO/ClimateMARGO.jl), a simple climate model, to explore what it takes to avoid the catastrophic impacts of global warming.

This is a `Pluto` notebook written in the `Julia` programming language. It's similar to the `Python`-written `Jupyter` notebooks you've been working with, with the added benefit of easy interactivity. We rely completely on that interactivity here and so you won't have to code anything or need to know anything about `Julia` or `Pluto`. Just read on and follow the instructions.  

*This notebook has been modified extensively for use in ClimateMatch from the [original MARGO notebook](https://margo.plutojl.org/introduction.html) by [Henri F. Drake](https://hdrake.github.io/) and [Fons van der Plas](https://github.com/fonsp/)*
"""

# ╔═╡ 5e5fc687-32be-4308-b3e5-499e25013afe
md"""
# Setup
"""

# ╔═╡ eda48867-7998-400b-98c9-fcd7ef762dc7
md"""Helper functions at end of notebook"""

# ╔═╡ 331c45b7-b5f2-4a78-b180-5b918d1806ee
md"""
# Section 1: Emissions mitigation and carbon dioxide removal to minimize climate suffering
"""

#The code in this webpage is *reactive*, meaning that the graphs and numbers automatically update whenever you change the climate model's inputs.

# ╔═╡ f3c9ee51-f43a-45ca-b604-7e09f28bdbd2
md"""
Recall the causal chain of climate change from slide 3 from Tutorial 2's video: 

Emissions → GHG → Forcing → Temperature → Damages

Each arrow is an potential opportunity to reduce damages: mitigation, Carbon Dioxide Removal (CDR), Solar Radiation Management (SRM), and adaptation, respectively. MARGO is a simple integrated assessment model focussed on simply demonstrating the effects of these 4 controls.
"""

# ╔═╡ 14f314b1-663d-4415-adc1-bc46869b5ee9
md"First let's look at the baseline scenario where emissions continue to grow linearly until 2100 and then fall to 0. In this hypothetical no-policy world, the fall to 0 could be because of public health concerns regarding other forms of air pollution, the development of new zero-carbon technologies, running out of extractable fossil fuels resources, and/or collapse of industrial capacity due to climate damages."

# ╔═╡ 0c00a7da-fa98-4253-9770-9a88bb02b849
md"""
## Section 1.1: _Can you limit human-caused global warming to "well-below 2ºC"?_

What would our emission-reducing efforts over time have to look like to limit warming to less than 2 degrees? Try it by lowering emissions using the blue dot in the plot below while watching the yellow curve showing the corresponding time course of temperature in the plot below it:
"""

# ╔═╡ 8f7f21bf-ec5d-4fa8-934a-14231e781f70
<<<<<<< HEAD
md"""The length of the bars above this text show the mitigation costs (i.e. the control costs) and the benefit (i.e. the avoided damages) and their difference as the net benefit.
=======
md"""The length of the bars above this text show 
- the mitigation cost (top purple bar),
- the benefit (the avoided damages; the middle organe bar) and 
- their difference as the net benefit (the bottom green bar).
>>>>>>> main
"""

# ╔═╡ f4b380c9-0ef4-4951-9954-3ddf92ddddb8
md"""
Here is a schematic from the MARGO paper that shows the components of this simple, idealized model. 
"""

# ╔═╡ 6533c123-34fe-4c0d-9ecc-7fef11379253
md"""
![image](https://raw.githubusercontent.com/ClimateMARGO/MARGO-gifs/0824fda21d17a846c0f8c594c6cac9630d9ebcdf/Margo_schematic.png)

_Figure from: [ClimateMARGO.jl](https://github.com/ClimateMARGO/ClimateMARGO.jl)_
"""

# ╔═╡ ec325089-8418-4fed-ac0e-e8ae21b433ab
md"""
## Section 1.2: Mitigating emissions
Human greenhouse gas emissions are the result of fossil fuel burning (e.g. for transportation, electricity generation, heating, industry), unsustainable agriculture, and land use changes. We refer to any actions or policies that reduce these emissions as *mitigation*.

The MARGO model lumps all potential mitigation into a single number: the fraction of *baseline* emissions that are mitigated in a given year. Baseline emissions are the emissions that would arise in a hypothetical future world absent of climate policy.

In the plot below (y-axis is the fraction of baseline emissions that are mitigated in a given year), *drag the blue dot around* to vary the amount and timing of mitigation, and observe how these changes affect key climate variables, using the drop-box menu: CO₂ₑ emissions, CO₂ₑ concentrations, and global temperature.
"""

# ╔═╡ 76411fcd-a8bb-422e-b063-8a460c522fe4
md"""Pick "Emissions", "Concentrations", or "Temperature" from this dropdown menu to show the respective plot below."""

# ╔═╡ 99d950f9-e3bf-4c18-9f82-0fec6dcb89d6
md"""
For a mitigation scheme that keeps temperature to below 3 degrees, by what year do our emissions have to have dropped by 50%? What about for 2 degrees?
"""

# ╔═╡ 30218715-6469-4a0f-bf90-f3243219e7b5
md"""
## Section 1.3: Cost & damages
So far in this tutorial we have looked at the effects that changes in CO2 emissions and mitigation have on climate (e.g. atmospheric CO2 concentrations and global temperature). It’s also essential to assess the socioeconomic impact of mitigation efforts.

Use the `Next` button to move through the 3 points below and watch the relevant plot appear below.
"""

# ╔═╡ 8433cb38-915a-46c1-b3db-8e7905351c1b
@bind cost_benefits_narrative_slide Carousel([
		  md"""
		### 1. The costs of climate suffering

		In the absence of climate action, temperatures would rise over 4.5ºC above preindustrial levels (1800 to 1850 average), causing catastrophic climate impacts. MARGO attempts to quantify this suffering by translating the degree of warming into economic damages (in \$ / year). The curve below shows how climate damages rise over time, as a percentage of the World Gross Domestic Product (WGDP) in that year, due to uncontrolled temperature increases.

		""",

		md"""### 2. Avoiding climate damages
		Emissions mitigation limits future warming and climate suffering (_Damages_ curve). The economic benefits of mitigation are given by the difference in damages relative to the no-policy scenario (_Baseline_ curve minus _Damages_ curve).

		In the figure below, drag around the blue dot to change the future mitigation strategy, and observe how the _Avoided damages_ (the grey area) change!

		""",

		md"""### 3. Cost-benefit analysis

		Unfortunately, mitigating CO₂ₑ emissions also carries a cost. In MARGO, the cost of mitigation control is proportional to the fraction of CO₂ₑ emissions that have been mitigated in a given year, increasing up to a maximum of $70 per metric ton of CO₂ₑ at 100% mitigation.

		This naturally leads to a **cost-benefit analysis**. We search for the most beneficial, or *optimal*, scenario: the one with the *maximum net present benefits*. In the figure below, try finding a mitigation strategy that optimizes these _Net benefits_.
		"""
]; wraparound=false)

# ╔═╡ 11d62228-476c-4616-9e7d-de6c05a6a53d
if cost_benefits_narrative_slide == 1
	hidecloack("cost_benefits_narrative_input")
end

<<<<<<< HEAD
# ╔═╡ 4c7fccc5-450c-4903-96a6-ce36ff60d280
md"""
## Section 1.4: Picking up the slack: carbon dioxide removal

While substantial emissions mitigations are necessary to reduce future climate suffering, they can not make up for the hundreds of billions of tons of CO₂ that humans have already emitted. However, both natural and technological methods for removing CO₂ from the atmosphere exist. Although they are presently miniscule compared to the tens-of-gigatons scale of global emissions, some experts expect that CO₂ removal methods will play a key role in the future once major mitigation goal have been accomplished. In MARGO, we do not distinguish between different carbon dioxide removal methods, and further assume that the carbon is stored permanently.

*Drag the yellow dot in the figure below to modify the amount and timing of carbon dioxide removal*.
"""

=======
>>>>>>> main
# ╔═╡ b2d65726-df99-4710-9d03-9f6838036c87
md"""
# Section 2: MARGO's automated optimization

<<<<<<< HEAD
In the above example, *you* manually adjusted the timing and amount of mitigation and carbon dioxide removal to achieve some desired temperature, but did not have much control on the shape of curves. Using a computer algorithm, we can do this optimization step *automatically* and *faster*, without having to assume anything about the shape of the mitigation and carbon dioxide removal curves. This optimization (see the bottom right corner of the model schematic shown above) maximizes the difference of benefits (i.e. avoided damages) and costs (i.e. control costs) under the constraint of a maximum temperature.

Adjust the parameters below to assess their effect on global temperature. First, you can adjust the maximum temperature using the slider, which will set the goal temperature in the bottom plot. Selecting the *allow temperature overshoot* option will allow the projected temperature to temporily exceed the maximum temperature you selected. Next, you can enable various controls (mitigation, removal, geo-engineering, and adaptation) and set the cost multiplier for each which will adjust the costs associated with each approach for reducing damages. Observe how changes in these various settings affect the global temperature projections in the bottom plot.
=======
In the above example, *you* manually adjusted the timing and amount of mitigation and carbon dioxide removal, but did not have much control on the shape of curves. Using a computer algorithm, we can do this optimization step *automatically* and *faster*, without having to assume anything about the shape of the mitigation and carbon dioxide removal curves.

The horizontal black line in the bottom plot shows the maximum temperature you select using the slider.
>>>>>>> main
"""

# ╔═╡ a0a1bb20-ec9b-446d-a36a-272840b8d35c
blob(
	md"""
	#### Maximum temperature

	`0.0 °C` $(Tmax_9_slider) `5.0 °C`
	
	_Allow **temperature overshoot**:_ $(allow_overshoot_9_cb)

	""",
	"#c5710014"
)

# ╔═╡ c1971e19-8ce1-4f80-a00f-2815330e5753
md"""
We can see
- how which of the 4 classes of controls we use (checkbox), and
- how expensive each is (the cost multiplier sliders) 
affects the optimal solution.
"""

# ╔═╡ 5f9090d4-1a40-4217-b537-c497cf92976a
md"""
Expore the space of optimal solutions within different scenarios you can think of. 

For example, mitigation and adaptation are considered two important responses to climate change now. Are they? If we restrict ourselves to only mitigation and adaptation measures, do we use both in equal amounts? When do we start doing them? The result you'll find suggests misplaced emphasis on adaptation relative to mitigation now.

Much of the solution space here uses carbon dioxide removal (orange) at substantial (>10% levels) starting in the next decade or two. Although natural and technological methods for removing CO₂ from the atmosphere exist now, they are presently miniscule compared to the tens-of-gigatons scale of global emissions. Some experts believe that the challenges in scaling these technologies will be overcome and they will play a key role in the future. Other experts argue there are inherent limitations to such technologies and useful only as fine adjustments later in the century and that any emphasis on them now takes away from more effective mitigation efforts. Toggle carbon removal on and off to see the effect. Do we still use these removal controls if they are more expensive?
"""

# ╔═╡ 38ff6efc-ccf1-4ca3-8437-152784a98a9e
md"""
# Section 3: From Unaware to Purposeful Solar Radiation Management

Various regulations arising from well-established public health concerns (mostly regarding air quality) have been successful at significantly reducing over the last decades the amount of aerosols we emit when burning fossil fuels (e.g. China, *circa* 2010 and sulfur emissions from freight shipping, *circa* 2020). 

<<<<<<< HEAD
Over the year or so that it takes the typical aerosol particle to fall out of the atmosphere, they reflect sunlight to space and have an effective negative contribution to radiative forcing. The IPCC estimates at least 0.5 degrees of warming has been masked by yearly sulfur emissions alone (i.e. if we stopped emitting sulfur the global average temperature would go up by 0.5 degrees as existing sulfur aerosols fall out of the atmosphere). The effective warming effects of reducing aerosol emissions are typically absent from assessments of how aerosols affect public health. 
=======
Over the year or so that it takes the typical aerosol particle to fall out of the atmosphere after being emitted, they reflect sunlight to space and have an effective negative contribution to radiative forcing. The IPCC estimates at least 0.5 degs of warming has been masked by sulfur alone. 
>>>>>>> main

The effective warming effects of reducing aerosol emissions are typically absent from assessments of how aerosols affect public health. At the expense of accelerated global warming, our society is likely to continue reducing aerosols for the public health benefit in the near future. 

<<<<<<< HEAD
Let's use MARGO to simulate how we manage (purposefully or not) our aerosol emissions using a baseline (uncontrolled) scenario in which we are ignorant of the associated warming of their reduction and do not apply any of the other controls (mitigation, removal, or adaptation). Our control is a reduction knob of the total aerosol forcing budget of 1.5 $$W/m^2$$ normalized from 0 (no reduction) to 1 (full reduction). Consistent with the large sulfur reductions in recent years, let's assume we've eliminated sulfur so our initial control is a pulse up to 1/3 (a socalled _termination shock_).

What are the effects on global temperature of continuing to reduce our aerosol emissions?
"""

# ╔═╡ edcb60ca-6dc8-4d6b-b722-d3b2a621a305
md"""
# Summary
In this tutorial you explored components of MARGO, a simple climate model with dynamic mitigation controls that allow us to assess the measures needed to avoid the worst impacts of global warming. First, varied the amount and timing of CO2 emissions and mitigation efforts to investigate how these changes affect global warming. You then explored the socioeconomic impact of mitigation efforts on factors such as control costs, climate damages and net benefits and also assessed the combined impact of carbon dioxide removal and mitigation efforts on global CO2.  Finally, you further analyzed the socioeconomic impact of global warming and mitigation efforts using MARGO’s automated optimization computer algorithm.
"""

# ╔═╡ 3f44b23a-f772-4988-9f35-e477b3439219
md"""
# Resources
The model code from the MARGO simulations used in this tutorial can be accessed [here](https://github.com/ClimateMARGO/ClimateMARGO.jl).
=======
Estimation uncertainty of aerosol forcing is relatively large but the current best estimates are -1.5 to -1 $$W/m^2$$, with sulfate alone contributing about -0.5 $$W/m^2$$. 
More recent analysis suggests aerosol reductions over the last decades are in the process of significantly increasing the warming rate over the next decade ([Hansen et al. 2022](https://arxiv.org/ftp/arxiv/papers/2212/2212.04474.pdf)).

Let's use MARGO to simulate how we manage (purposefully or not) our aerosol emissions. We'll use a baseline (i.e. uncontrolled) scenario in which we are ignorant of the associated warming of thier reduction. We also assume we do not apply any of the other controls (mitiation, removal, or adaptation). Our control is a reduction knob of a total aerosol forcing budget of 1.5 $$W/m^2$$ normalized from 0 (no reduction) to 1 (full reduction). Consistent with the large sulfur reductions in recent years, let's assume we've eliminated sulfur so our initial control is a pulse up to 1/3 (a socalled _termination shock_).

To assess the effects on global temperature of different scenarios of aerosol reductions, move around the red dot below.
"""

# ╔═╡ 1bb0a463-4998-44fc-b5f2-998bddf6aee4
md"""
- By how much does temperature go up relative to the baseline if we remove all aerosols? If we remove 50%?
- What is the temperature difference between removing them immediately versus over the rest of the century?
- What if over the next two decades we deployed non-controversial (available and safe) SRM technology (e.g. surface reflectors) at scale that were able to replicate the cooling effects of the sulfur we have just removed? Do we get back to the baseline?
>>>>>>> main
"""

# ╔═╡ 4e1524e9-27d2-45a7-9ebd-21e6e369c4a7
md"""
# Appendix
"""

# ╔═╡ d5ab2398-2904-4b44-a855-e4729996bc0d
md"""
Example of running the model:
"""

# ╔═╡ 14623e1f-7719-47b1-8854-8070d5ef8e17
md"""
## Plot functions
"""

# ╔═╡ a9b1e7fa-0318-41d8-b720-b8615c047bcd
plot_controls(c::ClimateMARGO.Models.Controls) = plot_controls(MRGA(
		c.mitigate, 
		c.remove, 
		c.geoeng,
		c.adapt
))

# ╔═╡ d9d20714-0689-449f-8e52-603dc804c93f
yearticks = collect(2020:20:2200)

# ╔═╡ cabc3214-1036-433b-aae1-6964bb780be8
function finish!(p)
	plot!(p;
		xlim=(2020,2201),
		xticks=yearticks,
		size=(680,200),
		grid=false,
	)
end

# ╔═╡ 2fec1e12-0218-4e93-a6b5-3711e6910d79
function plot_costs(result::ClimateModel; 
		show_baseline::Bool=true,
		show_controls::Bool=true,
		show_damages::Bool=true,
		title="Control costs & climate damages"
	)
	
	p = plot(; 
		ylim=(0,25.1), left_margin=2.25Plots.Measures.mm,
		ylabel="trillion USD / year",
	)
	title === nothing || plot!(p; title=title)
	

	# baseline
	show_baseline && plot!(p,
		years, damage(result; discounting=true);
		pp.baseline_damages...,
		fillrange=zero(years),
		fillopacity=.2,
		linestyle=:dash,
	)
	
	# control costs
	controlled_damages = damage(result; M=true, R=true, G=true, A=true, discounting=true)
	
	show_controls && plot!(p,
		years, controlled_damages .+ cost(result; M=true, R=true, G=true, A=true, discounting=true);
		fillrange=controlled_damages,
		fillopacity=.2,
		pp.controls...
	)
	

	# controlled damages
	show_damages && plot!(p,
		years, controlled_damages;
		fillrange=zero(years),
		fillopacity=.2,
		pp.damages...
	)
	
	finish!(p)
	
end

# ╔═╡ cff9f952-4850-4d55-bb8d-c0a759d1b7d8
function plot_concentrations(result::ClimateModel; 
		relative_to_preindustrial::Bool=true)
	Tmax = 5
	p = relative_to_preindustrial ? plot(; 
		ylim=(0,4.5),
		yticks=[1,2,3,4],
		yformatter=x -> string(Int(x), "×"),
		title="Atmospheric CO₂ₑ concentration, relative to 1800-1850",
	) : plot(;
		ylim=(0,1400),
		ylabel="ppm",
		title="Atmospheric CO₂ₑ concentration",
	)
	
	factor = relative_to_preindustrial ? preindustrial_concentrations : 1

	# baseline
	plot!(p,
		years, c(result) ./ factor;
		pp.baseline_concentrations...,
		linestyle=:dash,
	)
	# controlled temperature
	plot!(p,
		years, c(result; M=true, R=true) ./ factor;
		pp.concentrations...
	)
	

	finish!(p)
end

# ╔═╡ c73c89a7-f652-4554-95e9-20f47a818996
function plot_controls(controls::MRGA; title=nothing, reverse=false)
	
	p = plot(; 
		ylim=(0,1),
	)
	title === nothing || plot!(p; title=title)
	
	for (tech, c) in enumerate(controls)
		if c !== nothing
			which = tech === :M ? (years .< end_of_oil) : eachindex(years)
			if ~reverse
				plot!(p,
					years[which], c[which];
					pp[tech]...
				)
			else
				plot!(p,
					years[which], -c[which];
					pp[tech]...
				)
			end
		end
	end

	finish!(p)
	
end

# ╔═╡ 373cba19-511e-483e-8bfe-14bbb92d5a21
@initially initial_x @bind input_x begin
	
	local t = input_x["G"][1]
	local y = input_x["G"][2]
	
	controls_x = MRGA(
		G=-expcontrol(t, y)
	)
	# controls_x = MRGA(
	# 	G=gaussish(input_x["G"]...),
	# )
	# controls_x = MRGA(
	# 	G=gaussish(t, y*clamp(.07 * (10 - y), 0, 1))
	# )

		
	plotclicktracker2(
<<<<<<< HEAD
		plot_controls(controls_x; title="Controlled reduction of aerosols (1=full reduction of 1.5 W/m2)", reverse=true),
=======
		plot_controls(controls_x; title="Controlled reduction of aerosols (1=full reduction of 1.5 W/m2)",reverse=true),
>>>>>>> main
		initial_x
	)
end

# ╔═╡ 6634bcf1-8af6-4000-9b00-a5b4c02596c6
function plot_emissions(result::ClimateModel)
	
	p = plot(; 
		ylim=(-3,11), left_margin=1.75Plots.Measures.mm, 
		ylabel="ppm / year",
		title="Global CO₂ₑ emissions",
	)

	
	

	# baseline
	plot!(p,
		years, effective_emissions(result);
		pp.baseline_emissions...,
		linestyle=:dash,
	)
	# controlled
	plot!(p,
		years, effective_emissions(result; M=true, R=true);
		fillrange=zero(years),
		fillopacity=.2,
		pp.emissions...
	)
	

	finish!(p)
	
end

# ╔═╡ 424940e1-06ef-453a-8ffb-deb24dadb334
function plot_emissions_pretty(result::ClimateModel)
	# offset the x values so that framestyle=:origin will make the y-axis pass through 2020 instead of 0. yuck
	R = x -> x + 2020
	L = x -> x - 2020
	
	Tmax = 5
	p = plot(; 
		ylim=(-3,11), 
		ylabel="ppm / year",
		framestyle = :origin,
		xformatter=string ∘ Int ∘ R,
	)

	
	

	# baseline
	plot!(p,
		L.(years), effective_emissions(result);
		pp.baseline_emissions...,
		linestyle=:dash,
	)
	# controlled temperature
	plot!(p,
		L.(years), effective_emissions(result; M=true, R=true);
		fillrange=zero(L.(years)),
		fillopacity=.2,
		pp.emissions...
	)
	

	finish!(p)
	
	plot!(p;
		xlim=L.(extrema(years)),
		xticks = L.(yearticks),
		)
end

# ╔═╡ 700f982d-85da-4dc1-9319-f3b2527d0308
function plot_temp(result::ClimateModel)
	Tmax = 6
	
	# setup
	p = plot(; 
		ylim=(0,Tmax), 
		yticks=[0,1,2,3], 
		yformatter=x -> string("+", Int(x), " °C"),
		title="Global warming relative to 1800-1850",
	)
	
	# shade dangerously high temperatures
	for a in [2,3]
		plot!(p,
			collect(extrema(years)),
			[a,a],
			linewidth=0,
			label=nothing,
			fillrange=[Tmax,Tmax],
			fillcolor=colors.above_paris
		)
	end

	# baseline
	plot!(p,
		years, T_adapt(result; splat(MRGA(false))...);
		pp.baseline_temperature...,
		linestyle=:dash,
	)
	# controlled temperature
	plot!(p,
		years, T_adapt(result; splat(MRGA(true))...);
		pp.temperature...
	)
	

	finish!(p)
end

# ╔═╡ ab557633-e0b5-4439-bc81-d274770f2e65
md"""
## Plot dots input magic
"""

# ╔═╡ 2758b185-cd54-484e-bb7d-d4cfcd2d39f4
md"""
## Running the model
"""

# ╔═╡ 611c25ab-a454-4d52-b8fb-a58b0d1f5ca6
function forward_controls_temp(controls::MRGA=MRGA(), model_parameters=default_parameters())
    
    
    model = ClimateModel(model_parameters)

    translations = Dict(
        :M => :mitigate,
        :R => :remove,
        :G => :geoeng,
        :A => :adapt,
    )
    for (k, v) in enumerate(controls)
		if v !== nothing
	        setfieldconvert!(model.controls, translations[Symbol(k)], copy(v))
		end
    end

    enforce_maxslope!(model.controls)

	model
    # return Dict(
    #     :model_parameters => model_parameters,
    #     model_results(model)...
    # )
end

# ╔═╡ 9aa73ce0-cec6-4d53-bbbc-f5c85de7b521
@initially initial_1 @bind input_1 begin
	
	local t = input_1["M"][1]
	local y = input_1["M"][2]
	
	controls_1 = MRGA(
		M=gaussish(t, clamp(.07 * (10 - y), 0, 1)),
		R=gaussish(t, clamp(.07 * (10 - y) * 0.25, 0, 1)),
	)
	
	result_1 = forward_controls_temp(controls_1)
	
	plotclicktracker2(
		plot_emissions(result_1),
		initial_1
	)
end

# ╔═╡ 65d31fbf-322d-459a-a2dd-2894edbecc4d
plot_temp(result_1)

# ╔═╡ ff2b1c0a-e419-4f41-aa3b-d017642ffc13
@initially initial_2 @bind input_2 begin
	
	
	controls_2 = MRGA(
		M=gaussish(input_2["M"]...),
	)
	
	result_2 = forward_controls_temp(controls_2)
	
	plotclicktracker2(
		plot_controls(controls_2; title="Deployment of mitigation"),
		initial_2
	)
end

# ╔═╡ 02851ee9-8050-4821-b3c9-1f65c9b8135b
if which_graph_2 == "Emissions"
	plot_emissions(result_2)
elseif which_graph_2 == "Concentrations"
	plot_concentrations(result_2; relative_to_preindustrial=true)
else
	plot_temp(result_2)
end

# ╔═╡ f4203dcf-b251-4e2b-be07-922bc7c4496d
(@initially initial_3 @bind input_3 begin
	
	
	controls_3 = MRGA(
		M=gaussish(input_3["M"]...),
	)
	
	result_3 = forward_controls_temp(controls_3)
	
	plotclicktracker2(
		plot_controls(controls_3; title="Deployment of mitigation"),
		initial_3
	)
end) |> cloak("cost_benefits_narrative_input")

# ╔═╡ 3e26d311-6abc-4b2c-ada4-f8a3171d9f75
if cost_benefits_narrative_slide == 1
	local uncontrolled = ClimateModel(default_parameters())
	plot_costs(uncontrolled; show_controls=false, show_baseline=false)
elseif cost_benefits_narrative_slide == 2
	plot_costs(result_3; show_controls=false)
else
	plot_costs(result_3)
end

# ╔═╡ 182b58bf-3733-400e-9d91-425615d36f1b

function forward_controls_temp2(controls::MRGA=MRGA(), model_parameters=default_parameters())
    model_parameters.economics.Finf =1.5
    
    model = ClimateModel(model_parameters)

    translations = Dict(
        :M => :mitigate,
        :R => :remove,
        :G => :geoeng,
        :A => :adapt,
    )
    for (k, v) in enumerate(controls)
		if v !== nothing
	        setfieldconvert!(model.controls, translations[Symbol(k)], copy(v))
		end
    end

    enforce_maxslope!2(model.controls)

	model
    # return Dict(
    #     :model_parameters => model_parameters,
    #     model_results(model)...
    # )
end

# ╔═╡ 37c056fc-a3e9-47ad-89d5-72e6a36a243d
result_x = forward_controls_temp2(controls_x);

# ╔═╡ 5a80d968-2664-4b4f-89d1-845279e419ee
plot_temp(result_x)

# ╔═╡ 89752d91-9c8e-4203-b6f1-bdad41386b31
shortname = MRGA("M","R","G","A")

# ╔═╡ ff2709a4-516f-4066-b5b2-617ac0e5f20c
mediumname = MRGA("mitigate", "remove", "geoeng", "adapt")

# ╔═╡ 2821b722-75c2-4072-b142-d13553a84b7b
longname = MRGA("Mitigation", "Removal", "Geo-engineering", "Adaptation")

# ╔═╡ 8e89f521-c19d-4f87-9497-f9b61c19c176
let
	default_cost = let
		e = default_parameters().economics
		MRGA(e.mitigate_cost, e.remove_cost, e.geoeng_cost, e.adapt_cost)
	end
	blob(
		@htl("""
	<h4>Which controls?</h4>

	<style>

	.controltable thead th,
	.controltable tbody td {
	  text-align: center;
	}

	.controltable input[type=range] {
	  width: 10em;
	}

	</style>

	<table class="controltable">
	<thead>
	<th></th><th>Enabled?</th><th style="text-align: center;">Cost multiplier</th>
	</thead>
	<tbody>

		<tr>
		<th>$(longname.M)</th>
		<td>$(@bind enable_M_9 CheckBox(;default=true))</td>
		<td>$(@bind cost_M_9 multiplier(default_cost.M, 4))</td>
		</tr>
		
		<tr>
		<th>$(longname.R)</th>
		<td>$(@bind enable_R_9 CheckBox(;default=true))</td>
		<td>$(@bind cost_R_9 multiplier(default_cost.R, 4))</td>
		</tr>
		
		<tr>
		<th>$(longname.G)</th>
		<td>$(@bind enable_G_9 CheckBox(;default=false))</td>
		<td>$(@bind cost_G_9 multiplier(default_cost.G, 4))</td>
		</tr>
		
		<tr>
		<th>$(longname.A)</th>
		<td>$(@bind enable_A_9 CheckBox(;default=false))</td>
		<td>$(@bind cost_A_9 multiplier(default_cost.A, 4))</td>
		</tr>
		
	</tbody>
	</table>
		"""),
		"#c500b40a"
	)
end

# ╔═╡ a83e47fa-4b48-4bbc-b210-382d1cf19f55
control_enabled_9 = MRGA(
	enable_M_9,
	enable_R_9,
	enable_G_9,
	enable_A_9,
);

# ╔═╡ 242f3109-244b-4884-a0e9-6ea8950ca47e
control_cost_9 = MRGA(
	Float64(cost_M_9),
	Float64(cost_R_9),
	Float64(cost_G_9),
	Float64(cost_A_9),
);

# ╔═╡ f861935a-8b03-426e-aebe-6963e034ad49
output_9 = let
	parameters = default_parameters()
	
	parameters.economics.mitigate_cost = control_cost_9.M
	parameters.economics.remove_cost = control_cost_9.R
	parameters.economics.geoeng_cost = control_cost_9.G
	parameters.economics.adapt_cost = control_cost_9.A
	# modify the parameters here!
	
	opt_controls_temp(parameters;
		temp_overshoot=allow_overshoot_9 ? 999.0 : Tmax_9,
		temp_goal=Tmax_9,
		max_deployment=let
			e = control_enabled_9
			Dict(
				"mitigate" => e.M ? 1.0 : 0.0, 
				"remove" => e.R ? 1.0 : 0.0, 
				"geoeng" => e.G ? 1.0 : 0.0, 
				"adapt" => e.A ? 0.4 : 0.0, 
			)
		end,
	)
end

# ╔═╡ 6978acad-9cac-4490-85fb-7e43d9558aca
plot_controls(output_9.result.controls) |> feasibility_overlay(output_9)

# ╔═╡ 7a435e46-4f36-4037-a9a6-d296b20bf6ac
plot!(plot_temp(output_9.result),
	years, zero(years) .+ Tmax_9;
	lw=2,
	pp.T_max...
	)

# ╔═╡ 7ffad0f8-082b-4ca1-84f7-37c08d5f7266
md"""
## Cost bars
"""

# ╔═╡ 608b50e7-4419-4dfb-8d9e-5144d4034c05
function avoided_damages_bars(result)
	td(x) = total_discounted(x, result)
	
	baseline_damages = td(damage(result; discounting=true))
	controlled_damages = td(damage(result; splat(MRGA(true))..., discounting=true))
	
	avoided_damages = baseline_damages - controlled_damages
	
	costs = td(cost(result; splat(MRGA(true))..., discounting=true))
	
	@htl("""
		
		<script>
		
		const colors = $(colors_js)
		const names = $(names_js)
		
  const baseline_damages = $(baseline_damages);
  const controlled_damages = $(controlled_damages);
  const avoided_damages = $(avoided_damages);

  const costs = $(costs);

  const scale = $(cost_bars_scale);


  const bar = (offset, width, color) =>
    html`<span style="margin-left: \${offset}%; width: \${width}%; opacity: .7; display: inline-block; background: \${color}; height: 1.2em; margin-bottom: -.2em;"></span>`;

  return html`
   <div>
\${bar(0, controlled_damages / scale, colors.damages)}
<span style="opacity: .6;">Controlled damages: <b>\${Math.ceil(
    controlled_damages
  )} trillion USD</b>.
  </div>

<div style="border-bottom: 2px solid #eee; margin-bottom: 4px;">
  \${bar(0, baseline_damages / scale, colors.baseline)}
  <span style="opacity: .6;">Baseline damages: <b>\${Math.ceil(
    baseline_damages
  )} trillion USD</b>.</span>
</div>
<div style="font-style: italic;">
\${bar(
  controlled_damages / scale,
  avoided_damages / scale,
  colors.avoided_damages
)}
<span>Avoided damages: <b>\${Math.ceil(avoided_damages)} trillion USD</b>.</span>
</div>
`;

	</script>
		""")
end

# ╔═╡ 2c1416cf-9b6b-40a0-b714-16853c7e1f1d
if cost_benefits_narrative_slide >= 2
	avoided_damages_bars(result_3)
end

# ╔═╡ 31a30755-1d8b-451b-8c9a-2c32a3a1d0b4
function cost_bars(result; offset_damages=false)
	td(x) = total_discounted(x, result)
	
	baseline_damages = td(damage(result; discounting=true))
	controlled_damages = td(damage(result; splat(MRGA(true))..., discounting=true))
	
	avoided_damages = baseline_damages - controlled_damages
	
	costs = td(cost(result; splat(MRGA(true))..., discounting=true))
	
	@htl("""
		
		<script>
		
		const colors = $(colors_js)
		const names = $(names_js)
		
  const baseline_damages = $(baseline_damages);
  const controlled_damages = $(controlled_damages);
  const avoided_damages = $(avoided_damages);

  const costs = $(costs);

  const scale = $(cost_bars_scale);

  const bar = (offset, width, color) =>
    html`<span style="margin-left: \${offset}%; width: \${width}%; opacity: .7; display: inline-block; background: \${color}; height: 1.2em; margin-bottom: -.2em;"></span>`;

  //   <div>
  // \${bar(0, baseline_damages / scale, colors.baseline)}
  // Baseline damages: <b>\${Math.ceil(baseline_damages)} trillion USD</b>.
  // </div>

  const extra_offset = $(offset_damages) ? controlled_damages / scale : 0;

  return html`

<div>
\${bar(extra_offset, costs / scale, colors.controls)}
<span  style="opacity: .6;">Control costs: <b>\${Math.ceil(
    costs
  )} trillion USD</b>.</span>
</div>
<div style="border-bottom: 2px solid #eee; margin-bottom: 4px;">
\${bar(extra_offset, avoided_damages / scale, colors.avoided_damages)}
<span style="opacity: .6;">Avoided damages: <b>\${Math.ceil(
    avoided_damages
  )} trillion USD</b>.</span>
</div>
<div style="font-style: italic;" title="Net benefits: Avoided damages minus the cost of getting there.">
\${bar(
  extra_offset + costs / scale,
  (avoided_damages - costs) / scale,
  colors.benefits
)}
<span>Net benefits: <b>\${Math.ceil(
    avoided_damages - costs
  )} trillion USD</b>.</span>
</div>`;

	</script>
		""")
end

# ╔═╡ 470d2f6f-fe97-4edd-8aaa-142bc8046fe8
cost_bars(result_1)

# ╔═╡ 5154aac7-812d-447f-8435-b8209d45fe04
if cost_benefits_narrative_slide >= 3
	cost_bars(result_3; offset_damages=true)
else
	bigbreak
end

# ╔═╡ 7f9df132-61de-4fec-a674-176c4a43335c
md"""
## MRGA struct
"""

# ╔═╡ 354b9d8a-7c3f-456b-9da9-4396ac975743
function MR(x::T,y::T) where T
	MRGA{T}(x, y, zero(x), zero(x))
end

# ╔═╡ Cell order:
# ╟─9a48a08e-7281-473c-8afc-7ad3e0771269
# ╟─1bc00b68-15c5-48fe-9994-dc2f543542df
# ╟─2dacd4cd-68bd-4c31-857e-d8cab04b5180
# ╟─5e5fc687-32be-4308-b3e5-499e25013afe
# ╟─eda48867-7998-400b-98c9-fcd7ef762dc7
# ╟─331c45b7-b5f2-4a78-b180-5b918d1806ee
# ╟─f3c9ee51-f43a-45ca-b604-7e09f28bdbd2
# ╟─14f314b1-663d-4415-adc1-bc46869b5ee9
# ╟─0c00a7da-fa98-4253-9770-9a88bb02b849
# ╟─9aa73ce0-cec6-4d53-bbbc-f5c85de7b521
# ╟─65d31fbf-322d-459a-a2dd-2894edbecc4d
# ╟─470d2f6f-fe97-4edd-8aaa-142bc8046fe8
# ╟─8f7f21bf-ec5d-4fa8-934a-14231e781f70
# ╟─94415ff2-32a2-4b0f-9911-3b93e202f548
# ╟─f4b380c9-0ef4-4951-9954-3ddf92ddddb8
# ╟─6533c123-34fe-4c0d-9ecc-7fef11379253
# ╟─50d24c91-61ae-4544-98fa-5749bafe3d41
# ╟─ec325089-8418-4fed-ac0e-e8ae21b433ab
# ╟─ff2b1c0a-e419-4f41-aa3b-d017642ffc13
# ╟─e846c6e2-aa63-40db-8592-c9563bbbdd40
# ╟─76411fcd-a8bb-422e-b063-8a460c522fe4
# ╟─02851ee9-8050-4821-b3c9-1f65c9b8135b
# ╟─99d950f9-e3bf-4c18-9f82-0fec6dcb89d6
# ╟─e810a90f-f964-4d7d-acdb-fc3a159dc12e
# ╟─30218715-6469-4a0f-bf90-f3243219e7b5
# ╟─8433cb38-915a-46c1-b3db-8e7905351c1b
# ╟─3e26d311-6abc-4b2c-ada4-f8a3171d9f75
# ╟─f4203dcf-b251-4e2b-be07-922bc7c4496d
# ╟─2c1416cf-9b6b-40a0-b714-16853c7e1f1d
# ╟─5154aac7-812d-447f-8435-b8209d45fe04
# ╟─11d62228-476c-4616-9e7d-de6c05a6a53d
# ╟─a3422533-2b78-4bc2-92bd-737da3c8982d
# ╟─b2d65726-df99-4710-9d03-9f6838036c87
# ╟─70173466-c9b5-4227-8fba-6256fc1ecace
# ╟─6bcb9b9e-e0ab-45d3-b9b9-3d7282f89df6
# ╟─a0a1bb20-ec9b-446d-a36a-272840b8d35c
# ╟─c1971e19-8ce1-4f80-a00f-2815330e5753
# ╟─8e89f521-c19d-4f87-9497-f9b61c19c176
# ╟─6978acad-9cac-4490-85fb-7e43d9558aca
# ╟─7a435e46-4f36-4037-a9a6-d296b20bf6ac
# ╟─5f9090d4-1a40-4217-b537-c497cf92976a
# ╟─a83e47fa-4b48-4bbc-b210-382d1cf19f55
# ╟─242f3109-244b-4884-a0e9-6ea8950ca47e
# ╟─38ff6efc-ccf1-4ca3-8437-152784a98a9e
# ╟─51037451-0fea-4021-8824-56911970b97b
# ╟─373cba19-511e-483e-8bfe-14bbb92d5a21
# ╟─37c056fc-a3e9-47ad-89d5-72e6a36a243d
# ╟─5a80d968-2664-4b4f-89d1-845279e419ee
<<<<<<< HEAD
# ╟─edcb60ca-6dc8-4d6b-b722-d3b2a621a305
# ╟─3f44b23a-f772-4988-9f35-e477b3439219
=======
# ╟─1bb0a463-4998-44fc-b5f2-998bddf6aee4
>>>>>>> main
# ╟─4e1524e9-27d2-45a7-9ebd-21e6e369c4a7
# ╟─1c8d2d00-b7d9-11eb-35c4-47f2a2aa1593
# ╟─d5ab2398-2904-4b44-a855-e4729996bc0d
# ╟─f861935a-8b03-426e-aebe-6963e034ad49
# ╟─3094a9eb-074d-46c3-9c1e-0a9c94c6ad43
# ╟─b428e2d3-e1a9-4e4e-a64f-61048572102f
# ╟─0b31eac2-8efd-47cd-9571-a2053846343b
# ╟─ca104939-a6ca-4e70-a47a-1eb3c32db18f
# ╟─cf90139c-13d8-42a7-aba3-8c431e7854b8
# ╟─bd2bfa3c-a42e-4975-a543-84541f66b1c1
# ╟─8cab3d28-a457-4ccc-b053-38cd003bf4d1
# ╟─b81de514-2506-4243-8235-0b54dd4a7ec9
# ╟─73e01bd8-f56b-4bb5-a9a2-85ad223c9e9b
# ╟─ae92ba1f-5175-4704-8240-2de8432df752
# ╟─8ac04d55-9034-4c29-879b-3b10887a616d
# ╟─14623e1f-7719-47b1-8854-8070d5ef8e17
# ╟─2fec1e12-0218-4e93-a6b5-3711e6910d79
# ╟─cff9f952-4850-4d55-bb8d-c0a759d1b7d8
# ╟─c73c89a7-f652-4554-95e9-20f47a818996
# ╟─a9b1e7fa-0318-41d8-b720-b8615c047bcd
# ╟─6634bcf1-8af6-4000-9b00-a5b4c02596c6
# ╟─424940e1-06ef-453a-8ffb-deb24dadb334
# ╟─700f982d-85da-4dc1-9319-f3b2527d0308
# ╟─cabc3214-1036-433b-aae1-6964bb780be8
# ╟─d9d20714-0689-449f-8e52-603dc804c93f
# ╟─c7cbc172-daed-406f-b24b-5da2cc234c29
# ╟─b440cd13-36a9-4c54-9d80-ac3fa7c2900e
# ╟─ec760706-15ac-4a50-a67e-c338d70f3b0a
# ╟─ab557633-e0b5-4439-bc81-d274770f2e65
# ╟─bb4b25e4-0db5-414b-a384-0a27fe7efb66
# ╟─646591c4-cb60-41cd-beb9-506807ce17d2
# ╟─6fb77b13-7a54-4d1d-9985-4735318680e1
# ╟─5c484595-4646-484f-9e75-a4a3b4c2af9b
# ╟─013807a0-bddb-448b-9300-f7f559e48a45
# ╟─4e91fb48-fc5e-409e-9a7e-bf846f1d211d
# ╟─3c7271ab-ece5-4ae2-a8dd-dc3670f300f7
# ╟─dcf265c1-f09b-483e-a361-d54c6c7500c1
# ╟─10c015ec-780c-4453-83cb-12dd0f09f358
# ╟─2758b185-cd54-484e-bb7d-d4cfcd2d39f4
# ╟─8fa94ec9-1fab-41b9-a7e6-1917e975e4ff
# ╟─611c25ab-a454-4d52-b8fb-a58b0d1f5ca6
# ╟─182b58bf-3733-400e-9d91-425615d36f1b
# ╟─785c428d-d4f7-431e-94d7-039b0708a78a
# ╟─7e540eaf-8700-4176-a96c-77ee2e4c384b
# ╟─89752d91-9c8e-4203-b6f1-bdad41386b31
# ╟─ff2709a4-516f-4066-b5b2-617ac0e5f20c
# ╟─2821b722-75c2-4072-b142-d13553a84b7b
# ╟─2dcd5669-c725-40b9-84c4-f8399f6e924b
# ╟─b8f9efec-63ac-4e58-93cf-9f7199b78451
# ╟─371991c7-13dd-46f6-a730-ad89f43c6f0e
# ╟─e815d175-1c47-4aef-bd89-e7fdc3e4912e
# ╟─0a3be2ea-6af6-43c0-b8fb-e453bc2b703b
# ╟─b7ca316b-6fa6-4c2e-b43b-cddb08aaabbb
# ╟─7ffad0f8-082b-4ca1-84f7-37c08d5f7266
# ╟─eb0c961d-42cf-4219-a36e-cd492fa31f6b
# ╟─608b50e7-4419-4dfb-8d9e-5144d4034c05
# ╟─31a30755-1d8b-451b-8c9a-2c32a3a1d0b4
# ╟─ec5d87a6-354b-4f1d-bb73-b3db08589d9b
# ╟─70f01a4d-0aa3-4cd5-ad71-452c490c61ac
# ╟─ac779b93-e19e-41de-94cb-6a2a919bcd2e
# ╟─7f9df132-61de-4fec-a674-176c4a43335c
# ╟─060cbeab-5503-4eda-95d8-3f554765b2ee
# ╟─354b9d8a-7c3f-456b-9da9-4396ac975743
