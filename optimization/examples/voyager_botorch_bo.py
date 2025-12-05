from optimization import VoyagerProblem, BotorchBO
import jax.numpy as jnp
# Whole workflow of opimization with adam

vp = VoyagerProblem()

optimizer = BotorchBO(vp)

_, _, losses, wti =optimizer.optimize(save_to_file=True,
                   wall_times=[30, 60, 180],)

print("Best loss:", jnp.min(losses))
print("Wall time indices:", wti)