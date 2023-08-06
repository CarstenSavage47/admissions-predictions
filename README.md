# Admissions Predictions

## Overview

The purpose of this neural network is to predict whether a college is selective or not based on attributes.

The current model is using:
- 75th percentile ACT scores
- Admittance rate percentages
- Total enrollment
- Total out-of-state price
- Percent of enrollment that is non-white
- Historically black university status (dummy variable)
- Percent of total enrollment that is women

These are useful predictors for whether a college is selective. I define a 'selective' university as one that has an acceptance rate of less than 50%.

## Technical Notes

Evaluating kurtosis:
- Mesokurtic: Data follows a normal distribution
- Leptokurtic: Heavy tails on either side, indicating large outliers. Looks like Top-Thrill Dragster.
- Playtkurtic: Flat tails indicate that there aren't many outliers.

- A kurtosis value greater than +1 indicates the graph is very peaked. Leptokurtic.
- A kurtosis value less than -1 indicates the graph is relatively flat. Playtkurtic.
- A kurtosis value of 0 indicates that the graph follows a normal distribution. Mesokurtic.

Evaluating skewness:
- A negative value indicates the tail is on the left side of the distribution.
- A positive value indicates the tail is on the right side of the distribution.
- A value of zero indicates that there is no skewness in the distribution; it's perfectly symmetrical.


Thank you Venelin (https://curiousily.com/posts/build-your-first-neural-network-with-pytorch/) and StatQuest (https://www.youtube.com/watch?v=FHdlXe1bSe4) for creating fantastic guides to PyTorch.
