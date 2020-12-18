<h1>Classification of Newswires</h1>
This model classifies newswires from Reuters

validation accuracy = 78%

loss accuracy = 99%

terminal prints accuracy results (first value is loss, second is validation) at the very end

matplotlib saves accuracy graphs in plots directory

Because this model has 46 different categories to output, we had to change the layer-size in order to allow for 46 classes. So stepping up the layer size from 16 to 64 and them using a 'softmax', 46 unit, layer on the bottom allows for 46 classes

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

<p align="center"><img src="https://github.com/about14sheep/ml_keras/blob/master/classifying_newswires/plots/tv_graph_acc.png"><img src="https://github.com/about14sheep/ml_keras/blob/master/classifying_newswires/plots/tv_graph_loss.png"></p>
<br />
Running only 9 epochs is the best number for this dataset, however better methods will yield a validation accuracy higher than %88:

<p align="center"><img src="https://github.com/about14sheep/ml_keras/blob/master/classifying_newswires/plots/tv_graph_acc_9e.png"><img src="https://github.com/about14sheep/ml_keras/blob/master/classifying_newswires/plots/tv_graph_loss_9e.png"></p>
