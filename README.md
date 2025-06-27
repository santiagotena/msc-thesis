# Thesis
 
<p align="center">
<img src="https://github.com/santiagotena/assets/blob/master/thesis/title.png?raw=true" alt="Title" width=75% height=75%>
</p>

## Motivation

Graph Neural Networks are a subtype of deep learning algorithms.
They seem to leverage relationships between data points. Effective in molecular chemistry, social networks and traffic modeling.
They have been effectively used in the context of tabular data learning.

<p align="center">
<img src="https://github.com/santiagotena/assets/blob/master/thesis/gnn.png?raw=true" alt="gnn" width=20% height=20%>
</p>

## Research Questions

1. Can tabular data be used to define a graph structure that can be effectively used in a GNN for classification purposes?
2. Are categorical features enough to create meaningful graph representations?
3. How do graphs derived from categorical features differ from those derived from continuous features?
4. How do these GNN models compare to the algorithms commonly used for this kind of task?

## Datasets

<p align="center">
<img src="https://github.com/santiagotena/assets/blob/master/thesis/datasets.png?raw=true" alt="datasets" width=50% height=50%>
</p>

## Experimental Overview

<p align="center">
<img src="https://github.com/santiagotena/assets/blob/master/thesis/experiments.png?raw=true" alt="experiments" width=50% height=50%>
</p>

XGBoost: eXtreme Gradient Boosting
<br/> MLP: Multilayer Perceptron
<br/> SFV: Same Feature Value
<br/> KNN: K-Nearest Neighbors
<br/> GCN: Graph Convolutional Network

## Results

<p align="center">
<img src="https://github.com/santiagotena/assets/blob/master/thesis/results.png?raw=true" alt="results" width=50% height=50%>
</p>

## Conclusions

1. Can tabular data be used to define a graph structure that can be effectively used in a GNN for classification purposes?
<br/> Yes, graphs were built using the KNN and SFV approaches. A GCN received them as input.
2. Are categorical features enough to create meaningful graph representations?
<br/> The SFV-based approach produced a viable but not effective model relative to the other models present in this study.
3. How do graphs derived from categorical features differ from those derived from continuous features?
<br/> The KNN-based approach using categorical features had better performance than the SFV-based approach derived from categorical features.
4. How do these GNN models compare to the algorithms commonly used for this kind of task?
<br/> The SFV-based model was not able to compare favorably with any other model.
The KNN-based matched the performance of the XGBoost and MLP-based models with some datasets. 
The regularized MLP-based models were the best overall performers within the confines of this study.


## Usage

The files in this repository are meant to be run using Google Colab.

https://colab.research.google.com/

The runtime type used when developing these models was the T4 GPU between 10.2024 and 03.2025.
