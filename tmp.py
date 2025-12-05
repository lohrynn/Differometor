from optimization import VoyagerProblem
import jax
import jax.numpy as jnp

vp = VoyagerProblem()

n_samples = 100

# n_samples random arrays as parameters (shape 48, values between -10 and 10)
params = [jnp.array(jax.random.uniform(jax.random.PRNGKey(i), (48,), minval=-10, maxval=10)) for i in range(n_samples)]

losses = jnp.array([vp.objective_function(params) for params in params])

best_index = jnp.argmin(losses)

vp.output_to_files(
    best_params=params[best_index],
    losses=losses,
    algorithm_str="random_sampling",
    hyper_param_str=f"n_samples_{n_samples}",
    hyper_param_str_in_filename=True,
)