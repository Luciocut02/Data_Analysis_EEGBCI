load('feature2_1.mat')
load('feature2_1rdef.mat')
load('metrics_ws_60_40r.mat')
load('metrics2_ws_60_40r.mat')

for s=1:25
    for se=1:5
         if s<10
    fileName=sprintf('sub-00%d_ses-0%d_task_motorimagery_eeg.mat', s, se);
       else
    fileName=sprintf('sub-0%d_ses-0%d_task_motorimagery_eeg.mat', s, se);
         end
labels_ld=load(fileName,'labels');
labels2=labels_ld.labels;
n_trial=size(labels2,2);
n_train=round(0.6*n_trial);
n_test=n_trial-n_train;
labels_tr=labels2(1,1:n_train);
labels_ts=labels2(1,n_train+1:end);
labels_tr=transpose(labels_tr);
labels_ts=transpose(labels_ts);
 x=sprintf('sub_%02d_ses_%02d_tr', s, se);
 y=sprintf('sub_%02d_ses_%02d_te1', s, se);
 z=sprintf('sub_%02d_ses_%02d_te2', s, se);
 p=sprintf('sub_%02d_ses_%02d_te3', s, se);
 x_x=eval(x);
 y_y=eval(y);
 z_z=eval(z);
 p_p=eval(p);
 x_x2=normalize(x_x,'zscore');
 y_y2=normalize(y_y,'zscore');
 z_z2=normalize(z_z,'zscore');
 p_p2=normalize(p_p,'zscore');
 NaN_columns1=any(isnan(x_x2), 1);
 x_x2=x_x2(:, ~NaN_columns1);
 y_y2=y_y2(:, ~NaN_columns1);
 z_z2=z_z2(:, ~NaN_columns1);
 p_p2=p_p2(:, ~NaN_columns1);


 tic;
 mod=fitcsvm(x_x2,labels_tr,'KernelFunction','polynomial','OptimizeHyperparameters','auto');
 close all
 time_train2(s,se,1)=toc;
 tic;
 predizioni=predict(mod,y_y2);
 time_predict1_1(s,se,1)=toc;
 tic;
 predizioni2=predict(mod,z_z2);
 time_predict2_1(s,se,1)=toc;
 tic;
 predizioni3=predict(mod,p_p2);
 time_predict3_1(s,se,1)=toc;
 accuracy_60_40_1(s,se,1)=sum(predizioni==labels_ts)/length(labels_ts)*100;
 accuracy_60_40_2(s,se,1)=sum(predizioni2==labels_ts)/length(labels_ts)*100;
 accuracy_60_40_3(s,se,1)=sum(predizioni3==labels_ts)/length(labels_ts)*100;
 positive_class = 1;
TP=sum((predizioni==positive_class) & (labels_ts==positive_class));
FP=sum((predizioni==positive_class) & (labels_ts~=positive_class));
FN=sum((predizioni~=positive_class) & (labels_ts==positive_class));
    precision1_60_40pc_1(s,se,1)=(TP)/(TP+FP);
    recall1_60_40pc_1(s,se,1)=(TP)/(TP+FN);
   positive_class = 2;
TP2=sum((predizioni==positive_class) & (labels_ts==positive_class));
FP2=sum((predizioni==positive_class) & (labels_ts~=positive_class));
FN2=sum((predizioni~=positive_class) & (labels_ts==positive_class));
    precision2_60_40pc_1(s,se,1)=(TP2)/(TP2+FP2);
    recall2_60_40pc_1(s,se,1)=(TP2)/(TP2+FN2);
 positive_class = 1;
TP=sum((predizioni2==positive_class) & (labels_ts==positive_class));
FP=sum((predizioni2==positive_class) & (labels_ts~=positive_class));
FN=sum((predizioni2~=positive_class) & (labels_ts==positive_class));
    precision1_60_40pc_2(s,se,1)=(TP)/(TP+FP);
    recall1_60_40pc_2(s,se,1)=(TP)/(TP+FN);
   positive_class = 2;
TP2=sum((predizioni2==positive_class) & (labels_ts==positive_class));
FP2=sum((predizioni2==positive_class) & (labels_ts~=positive_class));
FN2=sum((predizioni2~=positive_class) & (labels_ts==positive_class));
    precision2_60_40pc_2(s,se,1)=(TP2)/(TP2+FP2);
    recall2_60_40pc_2(s,se,1)=(TP2)/(TP2+FN2);
    positive_class = 1;
TP=sum((predizioni3==positive_class) & (labels_ts==positive_class));
FP=sum((predizioni3==positive_class) & (labels_ts~=positive_class));
FN=sum((predizioni3~=positive_class) & (labels_ts==positive_class));
    precision1_60_40pc_3(s,se,1)=(TP)/(TP+FP);
    recall1_60_40pc_3(s,se,1)=(TP)/(TP+FN);
   positive_class = 2;
TP2=sum((predizioni3==positive_class) & (labels_ts==positive_class));
FP2=sum((predizioni3==positive_class) & (labels_ts~=positive_class));
FN2=sum((predizioni3~=positive_class) & (labels_ts==positive_class));
    precision2_60_40pc_3(s,se,1)=(TP2)/(TP2+FP2);
    recall2_60_40pc_3(s,se,1)=(TP2)/(TP2+FN2);

 tic;
 mod2=fitcnet(x_x2,labels_tr,'OptimizeHyperparameters','auto');
 close all
 time_train2(s,se,2)=toc;
 tic;
 predizioni4=predict(mod2,y_y2);
 time_predict1_1(s,se,2)=toc;
 tic;
 predizioni5=predict(mod2,z_z2);
 time_predict2_1(s,se,2)=toc;
 tic;
 predizioni6=predict(mod2,p_p2);
 time_predict3_1(s,se,2)=toc;

 accuracy_60_40_1(s,se,2)=sum(predizioni4==labels_ts)/length(labels_ts)*100;
 accuracy_60_40_2(s,se,2)=sum(predizioni5==labels_ts)/length(labels_ts)*100;
 accuracy_60_40_3(s,se,2)=sum(predizioni6==labels_ts)/length(labels_ts)*100;
 positive_class = 1;
TP=sum((predizioni4==positive_class) & (labels_ts==positive_class));
FP=sum((predizioni4==positive_class) & (labels_ts~=positive_class));
FN=sum((predizioni4~=positive_class) & (labels_ts==positive_class));
    precision1_60_40pc_1(s,se,2)=(TP)/(TP+FP);
    recall1_60_40pc_1(s,se,2)=(TP)/(TP+FN);
   positive_class = 2;
TP2=sum((predizioni4==positive_class) & (labels_ts==positive_class));
FP2=sum((predizioni4==positive_class) & (labels_ts~=positive_class));
FN2=sum((predizioni4~=positive_class) & (labels_ts==positive_class));
    precision2_60_40pc_1(s,se,2)=(TP2)/(TP2+FP2);
    recall2_60_40pc_1(s,se,2)=(TP2)/(TP2+FN2);
    positive_class = 1;
TP=sum((predizioni5==positive_class) & (labels_ts==positive_class));
FP=sum((predizioni5==positive_class) & (labels_ts~=positive_class));
FN=sum((predizioni5~=positive_class) & (labels_ts==positive_class));
    precision1_60_40pc_2(s,se,2)=(TP)/(TP+FP);
    recall1_60_40pc_2(s,se,2)=(TP)/(TP+FN);
   positive_class = 2;
TP2=sum((predizioni5==positive_class) & (labels_ts==positive_class));
FP2=sum((predizioni5==positive_class) & (labels_ts~=positive_class));
FN2=sum((predizioni5~=positive_class) & (labels_ts==positive_class));
    precision2_60_40pc_2(s,se,2)=(TP2)/(TP2+FP2);
    recall2_60_40pc_2(s,se,2)=(TP2)/(TP2+FN2);
    positive_class = 1;
TP=sum((predizioni6==positive_class) & (labels_ts==positive_class));
FP=sum((predizioni6==positive_class) & (labels_ts~=positive_class));
FN=sum((predizioni6~=positive_class) & (labels_ts==positive_class));
    precision1_60_40pc_3(s,se,2)=(TP)/(TP+FP);
    recall1_60_40pc_3(s,se,2)=(TP)/(TP+FN);
   positive_class = 2;
TP2=sum((predizioni6==positive_class) & (labels_ts==positive_class));
FP2=sum((predizioni6==positive_class) & (labels_ts~=positive_class));
FN2=sum((predizioni6~=positive_class) & (labels_ts==positive_class));
    precision2_60_40pc_3(s,se,2)=(TP2)/(TP2+FP2);
    recall2_60_40pc_3(s,se,2)=(TP2)/(TP2+FN2);

 tic;
 mod3=fitcensemble(x_x,labels_tr,'OptimizeHyperparameters','auto');
 close all
 time_train2(s,se,3)=toc;
 tic;
 predizioni7=predict(mod3,y_y);
 time_predict1_1(s,se,3)=toc;
 tic;
 predizioni8=predict(mod3,z_z);
 time_predict2_1(s,se,3)=toc;
 tic;
 predizioni9=predict(mod3,p_p);
 time_predict3_1(s,se,3)=toc;
 accuracy_60_40_1(s,se,3)=sum(predizioni7==labels_ts)/length(labels_ts)*100;
 accuracy_60_40_2(s,se,3)=sum(predizioni8==labels_ts)/length(labels_ts)*100;
 accuracy_60_40_3(s,se,3)=sum(predizioni9==labels_ts)/length(labels_ts)*100;

 positive_class = 1;
TP=sum((predizioni7==positive_class) & (labels_ts==positive_class));
FP=sum((predizioni7==positive_class) & (labels_ts~=positive_class));
FN=sum((predizioni7~=positive_class) & (labels_ts==positive_class));
    precision1_60_40pc_1(s,se,3)=(TP)/(TP+FP);
    recall1_60_40pc_1(s,se,3)=(TP)/(TP+FN);
   positive_class = 2;
TP2=sum((predizioni7==positive_class) & (labels_ts==positive_class));
FP2=sum((predizioni7==positive_class) & (labels_ts~=positive_class));
FN2=sum((predizioni7~=positive_class) & (labels_ts==positive_class));
    precision2_60_40pc_1(s,se,3)=(TP2)/(TP2+FP2);
    recall2_60_40pc_1(s,se,3)=(TP2)/(TP2+FN2);

     positive_class = 1;
TP=sum((predizioni8==positive_class) & (labels_ts==positive_class));
FP=sum((predizioni8==positive_class) & (labels_ts~=positive_class));
FN=sum((predizioni8~=positive_class) & (labels_ts==positive_class));
    precision1_60_40pc_2(s,se,3)=(TP)/(TP+FP);
    recall1_60_40pc_2(s,se,3)=(TP)/(TP+FN);
   positive_class = 2;
TP2=sum((predizioni8==positive_class) & (labels_ts==positive_class));
FP2=sum((predizioni8==positive_class) & (labels_ts~=positive_class));
FN2=sum((predizioni8~=positive_class) & (labels_ts==positive_class));
    precision2_60_40pc_2(s,se,3)=(TP2)/(TP2+FP2);
    recall2_60_40pc_2(s,se,3)=(TP2)/(TP2+FN2);

     positive_class = 1;
TP=sum((predizioni9==positive_class) & (labels_ts==positive_class));
FP=sum((predizioni9==positive_class) & (labels_ts~=positive_class));
FN=sum((predizioni9~=positive_class) & (labels_ts==positive_class));
    precision1_60_40pc_3(s,se,3)=(TP)/(TP+FP);
    recall1_60_40pc_3(s,se,3)=(TP)/(TP+FN);
   positive_class = 2;
TP2=sum((predizioni9==positive_class) & (labels_ts==positive_class));
FP2=sum((predizioni9==positive_class) & (labels_ts~=positive_class));
FN2=sum((predizioni9~=positive_class) & (labels_ts==positive_class));
    precision2_60_40pc_3(s,se,3)=(TP2)/(TP2+FP2);
    recall2_60_40pc_3(s,se,3)=(TP2)/(TP2+FN2);
    end
end
save('metrics_ws_60_40r.mat','accuracy_60_40_1','accuracy_60_40_2','accuracy_60_40_3')
save('metrics2_ws_60_40r.mat','precision1_60_40pc_1','precision1_60_40pc_2','precision1_60_40pc_3','precision2_60_40pc_1','precision2_60_40pc_2','precision2_60_40pc_3','recall1_60_40pc_1','recall2_60_40pc_1','recall1_60_40pc_2','recall2_60_40pc_2','recall1_60_40pc_3','recall2_60_40pc_3')
save('time_60_40r.mat','time_train2','time_predict1_1','time_predict2_1','time_predict3_1')