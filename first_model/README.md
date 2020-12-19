<h1>Classification of Movie Reviews</h1>
This model classifies movie reviews as either positive or negative.

validation accuracy = 88%

loss accuracy = 33%

uncomment line 47 to see array of prediction confidence

terminal prints accuracy results (first value is loss, second is validation) at the very end

matplotlib saves accuracy graphs in plots directory

<h3>Model Details:</h3>
<ul>
  <li>sequential model with 3 layers of vector shape</li>
  <li>loss function = bionary_crossentropy</li>
  <li>activation = relu</li>
  <li>handles crossfitting simply; by lower the number epochs</li>
</ul>

<h2>matplotlib Graphs and Overfitting</h2>
<br />
Running 20 epochs causes the network to start predicting values that are specific to the training data. As seen in these graphs, although the training accuracy was high, the validation accuracy dives at around 4 epochs:

<p align="center"><img src="https://github.com/about14sheep/ml_keras/blob/master/first_model/plots/tv_graph_acc_20e.png"><img src="https://github.com/about14sheep/ml_keras/blob/master/first_model/plots/tv_graph_loss_20e.png"></p>
<br />

Running only four epochs is the best number for this dataset, however better methods will yield a validation accuracy higher than %88:

<p align="center"><img src="https://github.com/about14sheep/ml_keras/blob/master/first_model/plots/tv_graph_acc.png"><img src="https://github.com/about14sheep/ml_keras/blob/master/first_model/plots/tv_graph_loss.png"></p>

<h3>Regularization Methods</h3>

In order to handle the overfititng issue present in all machine learning models, there have been methods applied to regularize training inputs. A common method is to simply lower the layer size. Less size to store predictions causes the model to only store essential ones. These graphs show a slight improvent on validation acc.

<p align="center"><img src="https://github.com/about14sheep/ml_keras/blob/master/first_model/plots/tv_graph_acc_fitted.png"><img src="https://github.com/about14sheep/ml_keras/blob/master/first_model/plots/tv_graph_loss_fitted.png"></p>
<br />

Another method is using L1-L2 regularization. Which adds a cost for the model associated with larger weights. This will help keep the weights more regular and less varying. These graphs show an improvment:

<p align="center"><img src="https://github.com/about14sheep/ml_keras/blob/master/first_model/plots/tv_graph_acc_l2_regularized.png"><img src="https://github.com/about14sheep/ml_keras/blob/master/first_model/plots/tv_graph_loss_l2_regularized.png"></p>
<br />

Finally, we have the dropout method which is the most common regularization technigue. By adding 'dropout' layers to the model, you random set input values to zero. This will give you the dropout rate. When you test the model, instead of dropping out units, you scale them by a factor equal to the dropout rate.

<p align="center"><img src="https://github.com/about14sheep/ml_keras/blob/master/first_model/plots/tv_graph_acc_dropout_regularized.png"><img src="https://github.com/about14sheep/ml_keras/blob/master/first_model/plots/tv_graph_loss_dropout_regularized.png"></p>
<br />

