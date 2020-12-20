<h1>Predicting House Prices</h1>
This model predicts the price of houses off the boston housing price training data

In stead of a classification style model, we use scalar regression to predict a single continuous value

Because the concept of accuracy doesnt apply to regression style models: I implement the k-fold validation process to validate the network while adjusting its parameters. Essentialy this splits the data into 3-4 parts, training a model off them seperatly, then takes the average validation between them. This method is used when you have a low amount of training data

matplotlib saves accuracy graphs in plots directory

<h3>Model Details:</h3>
<ul>
  <li>sequential model with 3 layers of vector shape</li>
  <li>loss function = mse</li>
  <li>activation = relu</li>
  <li>handles crossfitting simply; by lower the number epochs</li>
</ul>

<h2>K-Fold Graph</h2>
<br />
This graph represents validation scores across the 500 epochs. 

<p align="center"><img src="https://github.com/about14sheep/ml_keras/blob/master/predicting_house_price/plots/mae_validation.png"></p>
<br />
From the graph you can see that cutting the production model down to 80 epochs would result in less overfitting.
