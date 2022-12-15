%This demo shows how to call the weighted fuzzy C-means algorithm described in the paper:
%M.Hashemzadeh, A.Golzari oskouei and N.farajzadeh, "New fuzzy C-means clustering method
%based on feature-weight and cluster-weight learning", Applied Soft Computing, 2019.
%For the demonstration, the iris dataset of the above paper is used.
%Courtesy of A.Golzari Oskouei

clc
clear all
close all

%Load the dataset. The last column of dataset is true labels.
X=load('iris.mat');
X=X.iris;

class=X(:,end);
X(:,end)=[];    %delete last column (true labels) in clustering process

[N,d]=size(X);
X=(X(:,:)-min(X(:)))./(max(X(:)-min(X(:)))); %Normalize data between 0 and 1 (optinal)

%Algorithm parameters.
%---------------------
k=size(unique(class),1);  %number of clusters.
q=2;                      %the value for the feature weight updates.
p_init=0;                 %initial p.
p_max=0.5;                %maximum p.
p_step=0.01;              %p step.
t_max=100;                %maximum number of iterations.
beta_memory=0;            %amount of memory for the cluster weight updates.
Restarts=10;              %number of algorithm restarts.
fuzzy_degree=2;           %fuzzy membership degree
I=1;                      %The value of this parameter is in the range of (0 and 1]
landa=I./var(X);
landa(landa==inf)=1;
%---------------------

%Cluster the instances using the propsed procedure.
%---------------------------------------------------------
for repeat=1:Restarts
    fprintf('========================================================\n')
    fprintf('proposed clustering algorithm: Restart %d\n',repeat);
    
    %Randomly initialize the cluster centers.
    rand('state',repeat)
    tmp=randperm(N);
    M=X(tmp(1:k),:);
    
    %Execute proposed clustering algorithm.
    %Get the cluster assignments, the cluster centers and the cluster weight and feature weight.
    [Cluster_elem,M,W,Z]=FWCW_FCM(X,M,k,p_init,p_max,p_step,t_max,beta_memory,N,fuzzy_degree,d,q,landa);
    
    [~,Cluster]=max(Cluster_elem,[],1); %Hard clusters. Select the largest value for each sample among the clusters, and assign that sample to that cluster.
    
    % Evaluation metrics
    % Accurcy
    Accurcy(repeat) = calculateAccuracy(Cluster,class);
    
    % NMI
    NMI(repeat) = calculateNMI(Cluster,class);
    
    fprintf('End of Restart %d\n',repeat);
    fprintf('========================================================\n\n')
end

fprintf('Average accurcy over %d restarts: %f.\n',Restarts,mean(Accurcy));
fprintf('Average NMI over %d restarts: %f.\n',Restarts,mean(NMI));
