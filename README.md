# Feature-Permutations
When it come to classic machine learning implementations, the order in which the features appear do not seem to matter that much. However, when working with deep neural networks everything matters. Passing the same dataset but with a different order of the features (columns) to the exact same neural network architecture will produce completely different results during training even if the seed is the same.

For this experiment, I am attempting to make feature permutations part of the learning process.
