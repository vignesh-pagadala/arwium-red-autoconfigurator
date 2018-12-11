import random

print("Here is a single sample from a uniform random variable")
print(random.random())
print("Here is a list of three samples:")
uniSamples = [random.random(), random.random(), random.random()]
print(uniSamples)
print("Here is a list of three exponential samples:")
expSamples = [random.expovariate(1.0), random.expovariate(1.0), random.expovariate(1.0)]
print(expSamples)