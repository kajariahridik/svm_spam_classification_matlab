clc 
clear all
close all

load train.mat
load test.mat
% load eval.mat

% EvData = eval_data;
% EvLabel = eval_label;
id = randperm(1536,600);
EvData = test_data(:,id);
EvLabel = test_label(id);

% standardization
TrData = train_data;
TrLabel = train_label;
TeData = test_data;
TeLabel = test_label;

mean_feature = mean(TrData,2);
SD_feature = std(TrData,1,2);

stndzd_Tr = (TrData-mean_feature)./SD_feature;
stndzd_Te = (TeData-mean_feature)./SD_feature;
stndzd_Ev = (EvData-mean_feature)./SD_feature;

p = 1; C = 0.08;
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
wo_Tr = sum(x.*TrLabel.*(stndzd_Tr'*stndzd_Tr+1).^p,1)';
bo = mean(TrLabel(idx)-wo_Tr(idx));
g_Tr = wo_Tr + bo;

wo_Te = sum(x.*TrLabel.*(stndzd_Tr'*stndzd_Te+1).^p,1)';
g_Te = wo_Te + bo;

wo_Ev = sum(x.*TrLabel.*(stndzd_Tr'*stndzd_Ev+1).^p,1)';
g_Ev = wo_Ev + bo;

% Testing and training accuracies
Acc_Tr = 0;
pred_Tr = sign(g_Tr);
for i = 1:2000   
    
    if pred_Tr(i) == TrLabel(i,1)
        Acc_Tr = Acc_Tr+1;
    end
end
Training_Accuracy = Acc_Tr*100/2000

Acc_Te = 0;
pred_Te = sign(g_Te);
for i = 1:1536    
    
    if pred_Te(i) == TeLabel(i,1)
        Acc_Te = Acc_Te+1;
    end
end
Testing_Accuracy = Acc_Te*100/1536

Acc_Ev = 0;
pred_Ev = sign(g_Ev);
eval_predicted = pred_Ev';
for i = 1:600   
    
    if pred_Ev(i) == EvLabel(i,1)
        Acc_Ev = Acc_Ev+1;
    end
end
Evaluation_Accuracy = Acc_Ev*100/600
