%% Hard margin with linear kernel
clc
clear all
close all
load train.mat
load test.mat

% standardization
TrData = train_data;
TrLabel = train_label;
TeData = test_data;
TeLabel = test_label;

mean_feature = mean(TrData,2);
SD_feature = std(TrData,1,2);

stndzd_Tr = (TrData-mean_feature)./SD_feature;
stndzd_Te = (TeData-mean_feature)./SD_feature;

C = 10^6;
didj = TrLabel*TrLabel';
K = stndzd_Tr'*stndzd_Tr;

%Mercer condition check
EG_values = eig(K);
if EG_values>=-1e-04
    check = true;
end
if check == true
    "Admissible kernel"
else
    "Non admissible kernel"
end
% calculate alpha
H = K.*didj;
f = -ones(2000,1);
A = [];
B = [];
Aeq = TrLabel';
beq = 0;
lb = zeros(2000,1);
ub = ones(2000,1)*C;
x0 = [];
options = optimset('LargeScale','off','MaxIter',1000);
x = quadprog(H,f,A,B,Aeq,beq,lb,ub,x0,options);

% select support vector
idx = find(x>=1e-04);
% discriminant function
wo = sum(x'.*TrLabel'.*stndzd_Tr,2);
bo = mean(1./TrLabel(idx)-stndzd_Tr(:,idx)'*wo);

%% Hard margin with polynomial kernel
clc
clear all
close all
load train.mat
load test.mat

% standardization
TrData = train_data;
TrLabel = train_label;
TeData = test_data;
TeLabel = test_label;

mean_feature = mean(TrData,2);
SD_feature = std(TrData,1,2);

stndzd_Tr = (TrData-mean_feature)./SD_feature;
stndzd_Te = (TeData-mean_feature)./SD_feature;

p = 2; C = 10^6;
didj = TrLabel*TrLabel';
K = (stndzd_Tr'*stndzd_Tr+1).^p;

%Mercer condition check
EG_values = eig(K);
if EG_values>=-1e-04
    check = true;
else
    check = false;
end
if check == true
    "Admissible kernel"
else
    "Non admissible kernel"
end
% calculate alpha
H = K.*didj;
f = -ones(2000,1);
A = [];
B = [];
Aeq = TrLabel';
beq = 0;
lb = zeros(2000,1);
ub = ones(2000,1)*C;
x0 = [];
options = optimset('LargeScale','off','MaxIter',1000);
x = quadprog(H,f,A,B,Aeq,beq,lb,ub,x0,options);

% select support vector
idx = find(x>=1e-04);    
% discriminant function
wo = sum(x'.*TrLabel'.*(stndzd_Tr'*stndzd_Tr+1).^p,1)';
bo_Tr = mean(TrLabel(idx)-wo(idx));
   

%% Soft margin with polynomial kernel
clc
clear all
close all
load train.mat
load test.mat

% standardization
TrData = train_data;
TrLabel = train_label;
TeData = test_data;
TeLabel = test_label;

mean_feature = mean(TrData,2);
SD_feature = std(TrData,1,2);

stndzd_Tr = (TrData-mean_feature)./SD_feature;
stndzd_Te = (TeData-mean_feature)./SD_feature;

p = 1; C = 0.1;
didj = TrLabel*TrLabel';
K = (stndzd_Tr'*stndzd_Tr+1).^p;

%Mercer condition check
EG_values = eig(K);
if EG_values>=-1e-04
    check = true;
else
    check = false;
end
if check == true
    "Admissible kernel"
else
    "Non admissible kernel"
end
% calculate alpha
H = didj.*K;
f = -ones(2000,1);
A = [];
B = [];
Aeq = TrLabel';
beq = 0;
lb = zeros(2000,1);
ub = ones(2000,1)*C;
x0 = [];
options = optimset('LargeScale','off','MaxIter',1000);
x = quadprog(H,f,A,B,Aeq,beq,lb,ub,x0,options);

% select support vector
idx = find (0<=x & x<=C);

% discriminant function
wo = sum(x'.*TrLabel'.*(stndzd_Tr'*stndzd_Tr+1).^p,1)';
bo_Tr = mean(TrLabel(idx)-wo(idx));
