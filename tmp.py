import torch
import jax.numpy as jnp
import jax
import numpy as np
from jaxtyping import Array
import time
from optimization.voyager.voyager_problem import VoyagerProblem

def t2j(a):
    """Convert torch array to jax array."""
    return jax.dlpack.from_dlpack(a)


def j2t(a):
    """Convert jax array to torch array."""
    return torch.utils.dlpack.from_dlpack(a)


t = torch.rand(48)
t2 = torch.rand(48)
j = jax.random.uniform(jax.random.key(0), (50, 48,))
j2 = jax.random.uniform(jax.random.key(1), (50, 48))

vp = VoyagerProblem()

v_obj = jax.vmap(vp.objective_function, in_axes=0)


# for pop in range(1,10):
#     for _ in range(5):
#         j_pop = jax.random.uniform(jax.random.key(pop), (pop,48))
#         start = time.time()
#         _ = v_obj(j_pop)
#         end = time.time()
#         print(f"Voyager objective function for pop {pop} (x1):", (end - start) * 1000, "ms")


# for _ in range(5):
#     start = time.time()
#     v_obj(j_pop)
#     end = time.time()
#     print("Voyager vectorized objective function (x1):", (end - start) * 1000, "ms")


# for _ in range(5):
#     start = time.time()
#     vp.objective_function(j)
#     end = time.time()
#     print("Voyager objective function (x1):", (end - start) * 1000, "ms")
    
# start = time.time()
# for _ in range(100):
#     vp.objective_function(j)
# end = time.time()
# print("Voyager objective function (x100):", (end - start) / 100 * 1000, "ms")



for _ in range(5):
    start = time.time()
    _ = j2t(j)
    end = time.time()
    print("JAX to Torch (x1):", (end - start) * 1000, "ms")

for _ in range(5):
    start = time.time()
    _ = j2t(j2)
    end = time.time()
    print("JAX2 to Torch (x1):", (end - start) * 1000, "ms")

start = time.time()
for _ in range(10000):
    _ = j2t(j)
end = time.time()
print("JAX to Torch (x10000):", (end - start) / 10000 * 1000, "ms")


# _ = t2j(t)

# print("1st Torch to JAX:", timeit.timeit(lambda: t2j(t), number=1) * 1000, "ms")
# print("2nd Torch to JAX:", timeit.timeit(lambda: t2j(t), number=1) * 1000, "ms")
# print("1st Torch2 to JAX:", timeit.timeit(lambda: t2j(t2), number=1) * 1000, "ms")
# print("2nd Torch2 to JAX:", timeit.timeit(lambda: t2j(t2), number=1) * 1000, "ms")

# print("1st JAX to Torch:", timeit.timeit(lambda: j2t(j), number=1) * 1000, "ms")
# print("2nd JAX to Torch:", timeit.timeit(lambda: j2t(j), number=1) * 1000, "ms")



# print()

# print("Torch to Torch:", timeit.timeit(lambda: j2t(t2j(t)), number=1) * 1000, "ms")
# print("Torch to Torch (x1000):", timeit.timeit(lambda: j2t(t2j(t)), number=1000) * 1000, "ms")

# print()

# print("Torch to JAX (x1000):", timeit.timeit(lambda: t2j(t), number=1000) * 1000, "ms")

