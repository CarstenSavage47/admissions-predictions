# Admissions Predictions

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


Thank you to Venelin (https://curiousily.com/posts/build-your-first-neural-network-with-pytorch/) and StatQuest (https://www.youtube.com/watch?v=FHdlXe1bSe4) for creating fantastic guides to PyTorch.
