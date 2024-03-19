# Hypothesis-SBI
## Hypothesis-SBI works in the following way:

They key of the algorithm are metrics that are each constraint for a specific range or single value. In analogy to fSBI these metrics will be applied one after the other. 

__In general:__
- A metric can be declared as broad, then all specified parameters in the model will be varied to find feasible combinations of parameters that satify this metric
- A metric can be declared as specific, then it is bounded to specific parameters in the model and the search for feasible combinations of parameters is limited to these parameters


__In the estimation of the posterior p(θ|x) we will use the prinicple of fSBI:__
- We will first search for feasible combinations of parameters that satisfy broad metric (in fSBI) manner
- We will then use the trained posterior p(θ|x) as the prior for the ongoing search
- A second posterior p2(θ*|x) will be t rained onthe specific metric while other parameters are fixed (fixed in the sense
  that they will be sampled according to the prior p(θ|x))

## Example 

We have metrics that specify stable network dynamics such as
- The firing rate of the excitatory population should be in the range of 5-10 Hz
- The firing rate of the inhibitory population should be in the range of 5-10 Hz
etc. 

Such metrics can be specified as broad metrics, because each parameter in our network should 
play a role in satisfying these metrics. 

We then have task specific metrics e.g. a delayed memory task, in heres comes the hypothesis into play -
we hypothesise that a specific neuronal population or connection type e.g. I-to-I is responsible for the
functionality of that task in the circuit. 

By defining this metric as specific we are limiting the search for feasible combinations of parameters 
that we define as relevant for that parameters, while at the same time only searching in the overall subspace
of parameters that result in stable network dynamics (broad metrics).

Each parameter in the model should be specified by an identifier.

When we fit the posterior p(θ|x) we initally fit



