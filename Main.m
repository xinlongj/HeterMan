clear all;
close all;
clc;

RealPositions               = load('RealPostions.txt');
FeatureVectors              = load('FeatureVectors.txt');
IndicationOfLabeledPosition = load('IndicationOfLabeledPosition.txt');
Orientation                 = load('Orientation.txt');

LaplacianOptions.NN = 15;
LaplacianOptions.GraphDistanceFunction = 'euclidean';
LaplacianOptions.GraphWeights = 'binary';
LaplacianOptions.GraphNormalize = 1;
LaplacianOptions.GraphWeightParam = 1;

weightOfOrientation         = 1;
weightOfManifold            = 1;

HeterMan(RealPositions, FeatureVectors, IndicationOfLabeledPosition, Orientation, LaplacianOptions, weightOfOrientation, weightOfManifold);
