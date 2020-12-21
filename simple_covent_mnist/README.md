<h1>Simple Covnet Model trained on MNIST</h1>

This model is the most basic example of a covnet model. The sizes of out inputs decrease as you go deeper into the nerwork because a covnet model learns local patterns instead of global (the model learns patterns in small 2d windows accross the image). To handle this I implemented MaxPooling instead of convolution. 

The accuracy difference between this covnet model and a classification model trained on the same data has a 68% (relative) increase.

The accuracy for this model is 99%
