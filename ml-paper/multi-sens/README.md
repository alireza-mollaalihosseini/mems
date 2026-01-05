In this folder, the goal is to simulate multiple sensors and get the timeseries and extract some features from it and then run ridge regression. SO the state matrix will no longer be magnitudes of the frequencies. The number of sensors to be simulated are also high (100, 200, or even 1000).

The idea behind this simulation is to investigate the effect of MEMS sensors on the Aucostice Scene Classification with much more sensors. In previous simulation, it has been shown that the effect of single sensor with magnitude of frequencies were more or less similar to the audio samples withouth any sensor at all.

right now, a grid search for 101 various sensors with quality factor 500 is simulating and the lambda optimization plots with highest accuracy will save. later can these high accuracies be used for grid search plot.

but on the side, another simulation (for single a and u_dc and quality factor of 50) is processing to compare which quality factor will results in better performance.

