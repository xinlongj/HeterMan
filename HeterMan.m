function [EstimatedPosition, TimeConsumption, Accuracy] = HeterMan(RealPositions, FeatureVectors, IndicationOfLabeledPosition, Orientation, LaplacianOptions, weightOfOrientation, weightOfManifold)

% Usage: HeterMan(RealPositions, FeatureVectors, IndicationOfLabeledPosition, Orientation, LaplacianOptions, weightOfOrientation, weightOfManifold)
% OR: [EstimatedPosition, TimeConsumption, Accuracy] = HeterMan(RealPositions, FeatureVectors, IndicationOfLabeledPosition, Orientation, LaplacianOptions, weightOfOrientation, weightOfManifold)
%
% Input:
% RealPositions                 - The real postions labeled by people
%                                 e.g. [x1 y1; x2 y2; ..., xm ym]
% FeatureVectors                - The Wi-Fi RSSI vectors received in the 'RealPositions'
% IndicationOfLabeledPosition   - Indication matrix to indicate labeled postions
% Orientation                   - The tangent of azimuth angle
% LaplacianOptions              - The parameters of Laplacian
%  .NN                          - The number of nearest neighbors
%  .GraphDistanceFunction       - The draph distance function
%                                 * 'euclidean'
%  .GraphWeights                - Weight type;
%                                 * 'distance' for distance weight
%                                 * 'binary' for binary weight
%                                 * 'heat' for heat kernel sigma                                   
%  .GraphNormalize              - 0 for normalized laplacian, 1 for not normalized laplacian
%  .GraphWeightParam            - The standard deviation when use 'heat' as GraphWeight
% weightOfOrientation           - The weight value of orientation constraint term
% weightOfManifold              - The weight value of manifold constraint term
%
%
%
% Output: 
% EstimatedPostion              - The estimated postion by HeterMan
% TimeConsumption               - Time (seconds) spent on predicting all testing data
% Accuracy                      - RMSE of EstimatedPositions and RealPositions

% Note: 
% RealPositions, FeatureVectors, IndicationOfLabeledPosition and Orientation should have equal number of rows
% RealPositions:                m * 2
% FeatureVectors:               m * N (N is the number of features)
% IndicationOfLabeledPosition:  m * 1 or 1 * m
% Orientation:                  m * 1 or 1 * m
%
%
% ------------------------------------------------------------------------
% Samples:
%
% LaplacianOptions.NN = 15;
% LaplacianOptions.GraphDistanceFunction = 'euclidean';
% LaplacianOptions.GraphWeights = 'binary';
% LaplacianOptions.GraphNormalize = 1;
% LaplacianOptions.GraphWeightParam = 1;
%
% HeterMan(RealPositions, FeatureVectors, IndicationOfLabeledPosition, Orientation, LaplacianOptions, 1, 1);
% ------------------------------------------------------------------------


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    Authors:   Xinlon Jiang    
%    Affiliate: Institute of Computing Technology, CAS
%    EMAIL:     jiangxinlong@ict.ac.cn
%    Paper:     《Heterogeneous Data Driven Manifold Regularization Model for Fingerprint Calibration Reduction》
%    Website：  http://ieeexplore.ieee.org/abstract/document/7816829/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

start_time = cputime; % Begining of HeterMan
% M indicate the tangent of azimuth angle
[m n] = size(Orientation);
if m < n
    m = n;
end
M = zeros(m, m);
for i = 1:1:m
    M(i, i) = Orientation(i);
end

% A is the Jordan form matrix
A = eye(m);
A(1,1) = 0;
for i = 2:1:m
    A(i,i-1) = -1;
end

% Calculate laplacian graph Lp
Lp = laplacian(FeatureVectors, 'nn', LaplacianOptions);

% Jp = diag(sigma_1, sigma_2, ..., sigma_m)
[m n] = size(IndicationOfLabeledPosition);
if m < n
    m = n;
end
Jp = zeros(m, m);
for i = 1:1:m
    if(1 == IndicationOfLabeledPosition(i))
          Jp(i, i) = 1;  
    end
end

% 构造Yp，只保留那些有label的，其余位置留 '0'
Yp = RealPositions;
for i=1:1:m
    if(0 == IndicationOfLabeledPosition(i))
        Yp(i,:) = 0;
    end
end


delta = weightOfOrientation;
gamma = weightOfManifold;

J0 = zeros(size(Jp));
Jp_hat = [Jp J0; J0 Jp];
L0 = zeros(size(Lp));
Lp_hat = [Lp L0; L0 Lp];
Yp_hat = [Yp(:,1); Yp(:,2)];

MM = [M * A -A];
K_hat = Jp_hat + delta * MM' * MM + gamma * Lp_hat;
P_hat = pinv(K_hat) * Jp_hat * Yp_hat;

[m n] = size(P_hat);
P = [P_hat(1:m/2,:) P_hat(m/2+1:m,:)];
end_time = cputime; % End of HeterMan

% Output parameters
EstimatedPosition = P;
TimeConsumption = end_time - start_time
[m n] = size(P);
dis = zeros(m,1);
for i=1:1:m
    dis(i,1) = norm(P(i,:) - RealPositions(i,:));
end
Accuracy = sqrt(mse(dis))