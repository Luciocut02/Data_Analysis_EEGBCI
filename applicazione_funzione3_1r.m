featureStruct3_1r= struct();
load('noise_data.mat')
load('noise_data2.mat')
load('noise_data3.mat')
for s = 1:25
    for se =1:5
        if s < 10
            fileName=sprintf('sub-00%d_ses-0%d_task_motorimagery_eeg.mat', s, se);
        else
            fileName=sprintf('sub-0%d_ses-0%d_task_motorimagery_eeg.mat', s, se);
        end
      
        p=sprintf('sub_%d_%d_noise',s,se);
        t=sprintf('sub_%d_%d_noise2',s,se);
        v=sprintf('sub_%d_%d_noise3',s,se);
        p_p=eval(p);
        t_t=eval(t);
        v_v=eval(v);
        dataStruct=load(fileName,'labels');
        labels=dataStruct.labels;
        [feature_sogg_40pc,feature_sogg_60pc,time_extraction_feature3_1]=feature_freq_time_domain3_1r(p_p, labels);
        [feature_sogg_40pc2,feature_sogg_60pc2,time_extraction_feature3_2]=feature_freq_time_domain3_1r(t_t, labels);
        [feature_sogg_40pc3,feature_sogg_60pc3,time_extraction_feature3_3]=feature_freq_time_domain3_1r(v_v, labels);
        matrix_name = sprintf('sub_%02d_ses_%02d_tr1', s, se);
        matrix_name2= sprintf('sub_%02d_ses_%02d_te1', s, se);
        matrix_name3=sprintf('time_feature_%d_%d1',s,se);
        matrix_name4 = sprintf('sub_%02d_ses_%02d_tr2', s, se);
        matrix_name5= sprintf('sub_%02d_ses_%02d_te2', s, se);
        matrix_name6=sprintf('time_feature_%d_%d2',s,se);
        matrix_name7= sprintf('sub_%02d_ses_%02d_tr3', s, se);
        matrix_name8= sprintf('sub_%02d_ses_%02d_te3', s, se);
        matrix_name9=sprintf('time_feature_%d_%d3',s,se);
        featureStruct3_1r.(matrix_name)=feature_sogg_40pc;
        featureStruct3_1r.(matrix_name2)=feature_sogg_60pc;
        featureStruct3_1r.(matrix_name3)=time_extraction_feature3_1;
        featureStruct3_1r.(matrix_name4)=feature_sogg_40pc2;
        featureStruct3_1r.(matrix_name5)=feature_sogg_60pc2;
        featureStruct3_1r.(matrix_name6)=time_extraction_feature3_2;
        featureStruct3_1r.(matrix_name7)=feature_sogg_40pc3;
        featureStruct3_1r.(matrix_name8)=feature_sogg_60pc3;
        featureStruct3_1r.(matrix_name9)=time_extraction_feature3_3;
   end
end
save('feature3_1r.mat', '-struct', 'featureStruct3_1r');
load('feature3_1r.mat')
for sub=1:25
    for ses=1:5
        if sub<10
    fileName=sprintf('sub-00%d_ses-0%d_task_motorimagery_eeg.mat', sub, ses);
       else
    fileName=sprintf('sub-0%d_ses-0%d_task_motorimagery_eeg.mat', sub, ses);
        end
labels_ld=load(fileName,'labels');
labels2=labels_ld.labels;
x=sprintf('sub_%02d_ses_%02d_tr1',sub,ses);
y=sprintf('sub_%02d_ses_%02d_te1',sub,ses);
z=sprintf('sub_%02d_ses_%02d_tr2',sub,ses);
p=sprintf('sub_%02d_ses_%02d_te2',sub,ses);
q=sprintf('sub_%02d_ses_%02d_tr3',sub,ses);
r=sprintf('sub_%02d_ses_%02d_te3',sub,ses);
x_x=eval(x);
y_y=eval(y);
z_z=eval(z);
p_p=eval(p);
q_q=eval(q);
r_r=eval(r);
n_trial=size(labels2,2);
n_train=round(0.4*n_trial);
n_test=n_trial-n_train;
labels_tr=labels2(1,1:n_train);
labels_ts=labels2(1,n_train+1:end);
labels_tr=transpose(labels_tr);
labels_ts=transpose(labels_ts);

x_x=normalize(x_x,'zscore');
y_y=normalize(y_y,'zscore');
NaN_columns1=any(isnan(x_x), 1);
x_x2=x_x(:, ~NaN_columns1);
y_y2=y_y(:, ~NaN_columns1);
z_z=normalize(z_z,'zscore');
p_p=normalize(p_p,'zscore');
NaN_columns1=any(isnan(z_z), 1);
z_z2=z_z(:, ~NaN_columns1);
p_p2=p_p(:, ~NaN_columns1);
q_q=normalize(q_q,'zscore');
r_r=normalize(r_r,'zscore');
NaN_columns1=any(isnan(q_q), 1);
q_q2=q_q(:, ~NaN_columns1);
r_r2=r_r(:, ~NaN_columns1);


tic;
modello=fitcsvm(x_x2,labels_tr,'KernelFunction','polynomial','OptimizeHyperparameters','auto');
time_matrix_train_40_60_3_1(sub,ses,1)=toc;
close all
tic;
predizioni=predict(modello,y_y2);
time_matrix_predict_40_60_3_1(sub,ses,1)=toc;
accuracy_40_60_3_1(sub,ses,1)=sum(predizioni==labels_ts)/length(labels_ts)*100;
positive_class = 1;
TP=sum((predizioni==positive_class) & (labels_ts==positive_class));
FP=sum((predizioni==positive_class) & (labels_ts~=positive_class));
FN=sum((predizioni~=positive_class) & (labels_ts==positive_class));
    precision1_40_60pc_3_1(sub,ses,1)=(TP)/(TP+FP);
    recall1_40_60pc_3_1(sub,ses,1)=(TP)/(TP+FN);
   positive_class = 2;
TP2=sum((predizioni==positive_class) & (labels_ts==positive_class));
FP2=sum((predizioni==positive_class) & (labels_ts~=positive_class));
FN2=sum((predizioni~=positive_class) & (labels_ts==positive_class));
    precision2_40_60pc_3_1(sub,ses,1)=(TP2)/(TP2+FP2);
    recall2_40_60pc_3_1(sub,ses,1)=(TP2)/(TP2+FN2);

    tic;
modello2=fitcsvm(z_z2,labels_tr,'KernelFunction','polynomial','OptimizeHyperparameters','auto');
close all
time_matrix_train_40_60_3_2(sub,ses,1)=toc;
  tic;
predizioni2=predict(modello2,p_p2);
time_matrix_predict_40_60_3_2(sub,ses,1)=toc;
accuracy_40_60_3_2(sub,ses,1)=sum(predizioni2==labels_ts)/length(labels_ts)*100;
positive_class = 1;
TP=sum((predizioni==positive_class) & (labels_ts==positive_class));
FP=sum((predizioni==positive_class) & (labels_ts~=positive_class));
FN=sum((predizioni~=positive_class) & (labels_ts==positive_class));
    precision1_40_60pc_3_2(sub,ses,1)=(TP)/(TP+FP);
    recall1_40_60pc_3_2(sub,ses,1)=(TP)/(TP+FN);
    positive_class = 2;
TP2=sum((predizioni==positive_class) & (labels_ts==positive_class));
FP2=sum((predizioni==positive_class) & (labels_ts~=positive_class));
FN2=sum((predizioni~=positive_class) & (labels_ts==positive_class));
    precision2_40_60pc_3_2(sub,ses,1)=(TP2)/(TP2+FP2);
    recall2_40_60pc_3_2(sub,ses,1)=(TP2)/(TP2+FN2);

     tic;
modello3=fitcsvm(q_q2,labels_tr,'KernelFunction','polynomial','OptimizeHyperparameters','auto');
close all
time_matrix_train_40_60_3_3(sub,ses,1)=toc;
  tic;
predizioni3=predict(modello3,r_r2);
time_matrix_predict_40_60_3_3(sub,ses,1)=toc;
accuracy_40_60_3_3(sub,ses,1)=sum(predizioni3==labels_ts)/length(labels_ts)*100;
positive_class = 1;
TP=sum((predizioni==positive_class) & (labels_ts==positive_class));
FP=sum((predizioni==positive_class) & (labels_ts~=positive_class));
FN=sum((predizioni~=positive_class) & (labels_ts==positive_class));
    precision1_40_60pc_3_3(sub,ses,1)=(TP)/(TP+FP);
    recall1_40_60pc_3_3(sub,ses,1)=(TP)/(TP+FN);
    positive_class = 2;
TP2=sum((predizioni==positive_class) & (labels_ts==positive_class));
FP2=sum((predizioni==positive_class) & (labels_ts~=positive_class));
FN2=sum((predizioni~=positive_class) & (labels_ts==positive_class));
    precision2_40_60pc_3_3(sub,ses,1)=(TP2)/(TP2+FP2);
    recall2_40_60pc_3_3(sub,ses,1)=(TP2)/(TP2+FN2);
    
    end
end

for sub=1:25 
    for ses=1:5
        if sub<10
    fileName=sprintf('sub-00%d_ses-0%d_task_motorimagery_eeg.mat', sub, ses);
       else
    fileName=sprintf('sub-0%d_ses-0%d_task_motorimagery_eeg.mat', sub, ses);
       end

labels_ld=load(fileName,'labels');
labels2=labels_ld.labels;
x=sprintf('sub_%02d_ses_%02d_tr1',sub,ses);
y=sprintf('sub_%02d_ses_%02d_te1',sub,ses);
x_x=eval(x);
y_y=eval(y);
z=sprintf('sub_%02d_ses_%02d_tr2',sub,ses);
p=sprintf('sub_%02d_ses_%02d_te2',sub,ses);
q=sprintf('sub_%02d_ses_%02d_tr3',sub,ses);
r=sprintf('sub_%02d_ses_%02d_te3',sub,ses);
z_z=eval(z);
p_p=eval(p);
q_q=eval(q);
r_r=eval(r);

n_trial=size(labels2,2);
n_train=round(0.4*n_trial);
n_test=n_trial-n_train;
labels_tr=labels2(1,1:n_train);
labels_ts=labels2(1,n_train+1:end);
labels_tr=transpose(labels_tr);
labels_ts=transpose(labels_ts);
x_x=normalize(x_x,'zscore');
y_y=normalize(y_y,'zscore');
NaN_columns1=any(isnan(x_x), 1);
x_x2=x_x(:, ~NaN_columns1);
y_y2=y_y(:, ~NaN_columns1);
z_z=normalize(z_z,'zscore');
p_p=normalize(p_p,'zscore');
NaN_columns1=any(isnan(z_z), 1);
z_z2=z_z(:, ~NaN_columns1);
p_p2=p_p(:, ~NaN_columns1);
q_q=normalize(q_q,'zscore');
r_r=normalize(r_r,'zscore');
NaN_columns1=any(isnan(q_q), 1);
q_q2=q_q(:, ~NaN_columns1);
r_r2=r_r(:, ~NaN_columns1);


tic;
modello=fitcnet(x_x2,labels_tr,'OptimizeHyperparameters','auto');
close all
time_matrix_train_40_60_3_1(sub,ses,2)=toc;
tic;
predizioni=predict(modello,y_y2);
time_matrix_predict_40_60_3_1(sub,ses,2)=toc;
  
accuracy_40_60_3_1(sub,ses,2)=sum(predizioni==labels_ts)/length(labels_ts)*100;
positive_class = 1;

TP=sum((predizioni==positive_class) & (labels_ts==positive_class));
FP=sum((predizioni==positive_class) & (labels_ts~=positive_class));
FN=sum((predizioni~=positive_class) & (labels_ts==positive_class));
    precision1_40_60pc_3_1(sub,ses,2)=(TP)/(TP+FP);
    recall1_40_60pc_3_1(sub,ses,2)=(TP)/(TP+FN);

    positive_class = 2;

TP2=sum((predizioni==positive_class) & (labels_ts==positive_class));
FP2=sum((predizioni==positive_class) & (labels_ts~=positive_class));
FN2=sum((predizioni~=positive_class) & (labels_ts==positive_class));
    precision2_40_60pc_3_1(sub,ses,2)=(TP2)/(TP2+FP2);
    recall2_40_60pc_3_1(sub,ses,2)=(TP2)/(TP2+FN2);

    tic;
modello2=fitcnet(z_z2,labels_tr,'OptimizeHyperparameters','auto');
close all
time_matrix_train_40_60_3_2(sub,ses,2)=toc;
  tic;
predizioni2=predict(modello2,p_p2);
time_matrix_predict_40_60_3_2(sub,ses,2)=toc;
accuracy_40_60_3_2(sub,ses,2)=sum(predizioni2==labels_ts)/length(labels_ts)*100;
positive_class = 1;
TP=sum((predizioni==positive_class) & (labels_ts==positive_class));
FP=sum((predizioni==positive_class) & (labels_ts~=positive_class));
FN=sum((predizioni~=positive_class) & (labels_ts==positive_class));
    precision1_40_60pc_3_2(sub,ses,2)=(TP)/(TP+FP);
    recall1_40_60pc_3_2(sub,ses,2)=(TP)/(TP+FN);
    positive_class = 2;
TP2=sum((predizioni==positive_class) & (labels_ts==positive_class));
FP2=sum((predizioni==positive_class) & (labels_ts~=positive_class));
FN2=sum((predizioni~=positive_class) & (labels_ts==positive_class));
    precision2_40_60pc_3_2(sub,ses,2)=(TP2)/(TP2+FP2);
    recall2_40_60pc_3_2(sub,ses,2)=(TP2)/(TP2+FN2);

     tic;
modello3=fitcnet(q_q2,labels_tr,'OptimizeHyperparameters','auto');
close all
time_matrix_train_40_60_3_3(sub,ses,2)=toc;
  tic;
predizioni3=predict(modello3,r_r2);
time_matrix_predict_40_60_3_3(sub,ses,2)=toc;
accuracy_40_60_3_3(sub,ses,2)=sum(predizioni3==labels_ts)/length(labels_ts)*100;
positive_class = 1;
TP=sum((predizioni==positive_class) & (labels_ts==positive_class));
FP=sum((predizioni==positive_class) & (labels_ts~=positive_class));
FN=sum((predizioni~=positive_class) & (labels_ts==positive_class));
    precision1_40_60pc_3_3(sub,ses,2)=(TP)/(TP+FP);
    recall1_40_60pc_3_3(sub,ses,2)=(TP)/(TP+FN);
    positive_class = 2;
TP2=sum((predizioni==positive_class) & (labels_ts==positive_class));
FP2=sum((predizioni==positive_class) & (labels_ts~=positive_class));
FN2=sum((predizioni~=positive_class) & (labels_ts==positive_class));
    precision2_40_60pc_3_3(sub,ses,2)=(TP2)/(TP2+FP2);
    recall2_40_60pc_3_3(sub,ses,2)=(TP2)/(TP2+FN2);
    end
end

for sub=1:25
    for ses=1:5
x=sprintf('sub_%02d_ses_%02d_tr1',sub,ses);
y=sprintf('sub_%02d_ses_%02d_te1',sub,ses);
x_x=eval(x);
y_y=eval(y);
z=sprintf('sub_%02d_ses_%02d_tr2',sub,ses);
p=sprintf('sub_%02d_ses_%02d_te2',sub,ses);
q=sprintf('sub_%02d_ses_%02d_tr3',sub,ses);
r=sprintf('sub_%02d_ses_%02d_te3',sub,ses);
z_z=eval(z);
p_p=eval(p);
q_q=eval(q);
r_r=eval(r);
if sub<10
    fileName=sprintf('sub-00%d_ses-0%d_task_motorimagery_eeg.mat', sub, ses);
else
    fileName=sprintf('sub-0%d_ses-0%d_task_motorimagery_eeg.mat', sub, ses);
end
labels_ld=load(fileName,'labels');
labels2=labels_ld.labels;
n_trial=size(labels2,2);
n_train=round(0.4*n_trial);
n_test=n_trial-n_train;
labels_tr=labels2(1,1:n_train);
labels_ts=labels2(1,n_train+1:end);
labels_tr=transpose(labels_tr);
labels_ts=transpose(labels_ts);
tic;
modello=fitcensemble(x_x,labels_tr,'OptimizeHyperparameters','auto');
close all
time_matrix_train_40_60_3_1(sub,ses,3)=toc;
tic;
predizioni=predict(modello,y_y);
time_matrix_predict_40_60_3_1(sub,ses,3)=toc;
accuracy_40_60_3_1(sub,ses,3)=sum(predizioni==labels_ts)/length(labels_ts)*100;
positive_class = 1;

TP=sum((predizioni==positive_class) & (labels_ts==positive_class));
FP=sum((predizioni==positive_class) & (labels_ts~=positive_class));
FN=sum((predizioni~=positive_class) & (labels_ts==positive_class));
    precision1_40_60pc_3_1(sub,ses,3)=(TP)/(TP+FP);
    recall1_40_60pc_3_1(sub,ses,3)=(TP)/(TP+FN);

    positive_class = 2;

TP2=sum((predizioni==positive_class) & (labels_ts==positive_class));
FP2=sum((predizioni==positive_class) & (labels_ts~=positive_class));
FN2=sum((predizioni~=positive_class) & (labels_ts==positive_class));
    precision2_40_60pc_3_1(sub,ses,3)=(TP2)/(TP2+FP2);
    recall2_40_60pc_3_1(sub,ses,3)=(TP2)/(TP2+FN2);

    tic;
modello2=fitcensemble(z_z,labels_tr,'OptimizeHyperparameters','auto');
close all
time_matrix_train_40_60_3_2(sub,ses,3)=toc;
  tic;
predizioni2=predict(modello2,p_p);
time_matrix_predict_40_60_3_2(sub,ses,3)=toc;
accuracy_40_60_3_2(sub,ses,3)=sum(predizioni2==labels_ts)/length(labels_ts)*100;
positive_class = 1;
TP=sum((predizioni==positive_class) & (labels_ts==positive_class));
FP=sum((predizioni==positive_class) & (labels_ts~=positive_class));
FN=sum((predizioni~=positive_class) & (labels_ts==positive_class));
    precision1_40_60pc_3_2(sub,ses,3)=(TP)/(TP+FP);
    recall1_40_60pc_3_2(sub,ses,3)=(TP)/(TP+FN);
    positive_class = 2;
TP2=sum((predizioni==positive_class) & (labels_ts==positive_class));
FP2=sum((predizioni==positive_class) & (labels_ts~=positive_class));
FN2=sum((predizioni~=positive_class) & (labels_ts==positive_class));
    precision2_40_60pc_3_2(sub,ses,3)=(TP2)/(TP2+FP2);
    recall2_40_60pc_3_2(sub,ses,3)=(TP2)/(TP2+FN2);

     tic;
modello3=fitcensemble(q_q,labels_tr,'OptimizeHyperparameters','auto');
close all
time_matrix_train_40_60_3_3(sub,ses,3)=toc;
  tic;
predizioni3=predict(modello3,r_r);
time_matrix_predict_40_60_3_3(sub,ses,3)=toc;
accuracy_40_60_3_3(sub,ses,3)=sum(predizioni3==labels_ts)/length(labels_ts)*100;
positive_class = 1;
TP=sum((predizioni==positive_class) & (labels_ts==positive_class));
FP=sum((predizioni==positive_class) & (labels_ts~=positive_class));
FN=sum((predizioni~=positive_class) & (labels_ts==positive_class));
    precision1_40_60pc_3_3(sub,ses,3)=(TP)/(TP+FP);
    recall1_40_60pc_3_3(sub,ses,3)=(TP)/(TP+FN);
    positive_class = 2;
TP2=sum((predizioni==positive_class) & (labels_ts==positive_class));
FP2=sum((predizioni==positive_class) & (labels_ts~=positive_class));
FN2=sum((predizioni~=positive_class) & (labels_ts==positive_class));
    precision2_40_60pc_3_3(sub,ses,3)=(TP2)/(TP2+FP2);
    recall2_40_60pc_3_3(sub,ses,3)=(TP2)/(TP2+FN2);
    end
end
save('metrics_ws_40_60r.mat','accuracy_40_60_3_1','accuracy_40_60_3_2','accuracy_40_60_3_3')
save('metrics2_ws_40_60r.mat','precision1_40_60pc_3_1','precision1_40_60pc_3_2','precision1_40_60pc_3_3','precision2_40_60pc_3_1','precision2_40_60pc_3_2','precision2_40_60pc_3_3','recall1_40_60pc_3_1','recall2_40_60pc_3_1','recall1_40_60pc_3_2','recall2_40_60pc_3_2','recall1_40_60pc_3_3','recall2_40_60pc_3_3')
save('time_40_60r.mat','time_matrix_train_40_60_3_1','time_matrix_train_40_60_3_2','time_matrix_train_40_60_3_3','time_matrix_predict_40_60_3_1','time_matrix_predict_40_60_3_2','time_matrix_predict_40_60_3_3')
