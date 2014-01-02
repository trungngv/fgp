function [rx,ClusterIndex,ClusterCenter,NumPoints,ClusterRadii]=KCenterClustering(d,N,X,K)
%
%     Gonzalez's farthest-point clustering algorithm.
%
%     O(N log K) version.
%
%     C++ Implementation.
%
%     loads KCenterClustering.dll
%
%% Input
%
%     * d                 --> data dimensionality.
%     * N                 --> number of source points.
%     * X                 --> d x N matrix of N source points in d dimensions.
%     * K			        --> number of clusters.
%
%% Ouput
%
%     * rx                     --> maximum radius of the clusters (rx).
%     * ClusterIndex     --> vector of length N where the i th element is   the cluster number to which the i th point  belongs. 
%                                     ClusterIndex[i] varies between 0 to K-1. 
%      * ClusterCenters  --> d x K matrix of K  cluster centers.
%      * NumPoints        --> number of points in each cluster.
%      * ClusterRadii      --> radius of each cluster.
%
%% Signature
%
% Author: Vikas Chandrakant Raykar
% E-Mail: vikas@cs.umd.edu
% Date:  6 April 2005, June 10 2005, August 23 2005
%
[rx,ClusterIndex,ClusterCenter,NumPoints,ClusterRadii]=mexmain(d,N,X,K);

