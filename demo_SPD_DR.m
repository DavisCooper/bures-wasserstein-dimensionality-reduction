clear;
clc;

addpath('local_manopt');
addpath('funcs');
load('toy_data.mat');
spdDR_Obj = spdDR;
spdDR_Obj.newDim = 10;
spdDR_Obj.trn_X = covD_Struct.trn_X;
spdDR_Obj.trn_y = covD_Struct.trn_y;

% AIRM
spdDR_Obj.metric = 1;
W_airm = spdDR_Obj.perform_graph_DA();

% Stein
spdDR_Obj.metric = 2;
W_stein = spdDR_Obj.perform_graph_DA();

% Jeffrey
spdDR_Obj.metric = 3;
W_jeff = spdDR_Obj.perform_graph_DA();

% log-Euclidean
spdDR_Obj.metric = 4;
W_le = spdDR_Obj.perform_graph_DA();

% Euclidean
spdDR_Obj.metric = 5;
W_e = spdDR_Obj.perform_graph_DA();


%Example of using the learned mappings to perform nearest neighbor
%classification

%AIRM
crr_AIRM = SPD_NN_Classifier(covD_Struct.trn_X,covD_Struct.trn_y,...
                             covD_Struct.tst_X,covD_Struct.tst_y,...
                             W_airm,1);
fprintf('Accuracy after dimensionality reduction using AIRM -->%.3f\n',crr_AIRM);

%Stein
crr_Stein = SPD_NN_Classifier(covD_Struct.trn_X,covD_Struct.trn_y,...
                             covD_Struct.tst_X,covD_Struct.tst_y,...
                             W_stein,2);
fprintf('Accuracy after dimensionality reduction using Stein -->%.3f\n',crr_Stein);

%Jeffrey
crr_Jef = SPD_NN_Classifier(covD_Struct.trn_X,covD_Struct.trn_y,...
                             covD_Struct.tst_X,covD_Struct.tst_y,...
                             W_jeff,3);
fprintf('Accuracy after dimensionality reduction using Jeffreys div. -->%.3f\n',crr_Jef);

%log-Euclidean
crr_leuc = SPD_NN_Classifier(covD_Struct.trn_X,covD_Struct.trn_y,...
                             covD_Struct.tst_X,covD_Struct.tst_y,...
                             W_le,4);
fprintf('Accuracy after dimensionality reduction using Log-Euclidean metric -->%.3f\n',crr_leuc);

%Euclidean
crr_euc = SPD_NN_Classifier(covD_Struct.trn_X,covD_Struct.trn_y,...
                             covD_Struct.tst_X,covD_Struct.tst_y,...
                             W_e,5);
fprintf('Accuracy after dimensionality reduction using Euclidean metric -->%.3f\n',crr_euc);
