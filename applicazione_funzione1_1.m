featureStruct1_1= struct();
for s = 1:25
    for se =1:5
        if s < 10
            fileName=sprintf('sub-00%d_ses-0%d_task_motorimagery_eeg.mat', s, se);
        else
            fileName=sprintf('sub-0%d_ses-0%d_task_motorimagery_eeg.mat', s, se);
        end
        cd  C:\Users\lucio\OneDrive\Desktop\Tesi3\
        dataStruct = load(fileName,'data','labels');
        data=dataStruct.data;
        labels=dataStruct.labels;
        [feature_sogg_80pc,feature_sogg_20pc,time_extraction_feature1_1]=feature_freq_time_domain1_1(data, labels);
        matrix_name=sprintf('sub_%02d_ses_%02d_tr', s, se);
        matrix_name2= sprintf('sub_%02d_ses_%02d_te', s, se);
        matrix_name3=sprintf('time_feature_%d_%d',s,se);
        featureStruct1_1.(matrix_name)=feature_sogg_80pc;
        featureStruct1_1.(matrix_name2)=feature_sogg_20pc;
        featureStruct1_1.(matrix_name3)=time_extraction_feature1_1;
   end
end
save('feature1_1.mat', '-struct', 'featureStruct1_1');
load('feature1_1.mat')
for sub=1:25
    for ses=1:5
        if sub<10
    fileName=sprintf('sub-00%d_ses-0%d_task_motorimagery_eeg.mat', sub, ses);
       else
    fileName=sprintf('sub-0%d_ses-0%d_task_motorimagery_eeg.mat', sub, ses);
        end
cd  C:\Users\lucio\OneDrive\Desktop\Tesi3\
labels_ld=load(fileName,'labels');
labels2=labels_ld.labels;
x=sprintf('sub_%02d_ses_%02d_tr',sub,ses);
y=sprintf('sub_%02d_ses_%02d_te',sub,ses);
x_x=eval(x);
y_y=eval(y);
n_trial=size(labels2,2);
n_train=round(0.8*n_trial);
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

tic;
modello=fitcsvm(x_x2,labels_tr,'KernelFunction','polynomial','OptimizeHyperparameters','auto');
time_matrix_train_80_20_1_1(sub,ses,1)=toc;
  
tic;
predizioni=predict(modello,y_y2);
time_matrix_predict_80_20_1_1(sub,ses,1)=toc;
accuracy_80_20_1_1(sub,ses,1)=sum(predizioni==labels_ts)/length(labels_ts)*100;

positive_class = 1;

TP=sum((predizioni==positive_class) & (labels_ts==positive_class));
FP=sum((predizioni==positive_class) & (labels_ts~=positive_class));
FN=sum((predizioni~=positive_class) & (labels_ts==positive_class));
    precision1_80_20pc_1_1(sub,ses,1)=(TP)/(TP+FP);
    recall1_80_20pc_1_1(sub,ses,1)=(TP)/(TP+FN);

    positive_class = 2;

TP2=sum((predizioni==positive_class) & (labels_ts==positive_class));
FP2=sum((predizioni==positive_class) & (labels_ts~=positive_class));
FN2=sum((predizioni~=positive_class) & (labels_ts==positive_class));
    precision2_80_20pc_1_1(sub,ses,1)=(TP2)/(TP2+FP2);
    recall2_80_20pc_1_1(sub,ses,1)=(TP2)/(TP2+FN2);
    
    end
end

for sub=15:25 
    for ses=1:5
        if sub<10
    fileName=sprintf('sub-00%d_ses-0%d_task_motorimagery_eeg.mat', sub, ses);
       else
    fileName=sprintf('sub-0%d_ses-0%d_task_motorimagery_eeg.mat', sub, ses);
       end
cd  C:\Users\lucio\OneDrive\Desktop\Tesi3\
labels_ld=load(fileName,'labels');
labels2=labels_ld.labels;
x=sprintf('sub_%02d_ses_%02d_tr',sub,ses);
y=sprintf('sub_%02d_ses_%02d_te',sub,ses);
x_x=eval(x);
y_y=eval(y);

n_trial=size(labels2,2);
n_train=round(0.8*n_trial);
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
tic;
modello=fitcnet(x_x2,labels_tr,'OptimizeHyperparameters','auto');
time_matrix_train_80_20_1_1(sub,ses,2)=toc;
tic;
predizioni=predict(modello,y_y2);
time_matrix_predict_80_20_1_1(sub,ses,2)=toc;
  
accuracy_80_20_1_1(sub,ses,2)=sum(predizioni==labels_ts)/length(labels_ts)*100;
positive_class = 1;

TP=sum((predizioni==positive_class) & (labels_ts==positive_class));
FP=sum((predizioni==positive_class) & (labels_ts~=positive_class));
FN=sum((predizioni~=positive_class) & (labels_ts==positive_class));
    precision1_80_20pc_1_1(sub,ses,2)=(TP)/(TP+FP);
    recall1_80_20pc_1_1(sub,ses,2)=(TP)/(TP+FN);

    positive_class = 2;

TP2=sum((predizioni==positive_class) & (labels_ts==positive_class));
FP2=sum((predizioni==positive_class) & (labels_ts~=positive_class));
FN2=sum((predizioni~=positive_class) & (labels_ts==positive_class));
    precision2_80_20pc_1_1(sub,ses,2)=(TP2)/(TP2+FP2);
    recall2_80_20pc_1_1(sub,ses,2)=(TP2)/(TP2+FN2);
    end
end

for sub=1:25
    for ses=1:5
x=sprintf('sub_%02d_ses_%02d_tr',sub,ses);
y=sprintf('sub_%02d_ses_%02d_te',sub,ses);
x_x=eval(x);
y_y=eval(y);
if sub<10
    fileName=sprintf('sub-00%d_ses-0%d_task_motorimagery_eeg.mat', sub, ses);
else
    fileName=sprintf('sub-0%d_ses-0%d_task_motorimagery_eeg.mat', sub, ses);
end
cd  C:\Users\lucio\OneDrive\Desktop\Tesi3\
labels_ld=load(fileName,'labels');
labels2=labels_ld.labels;
n_trial=size(labels2,2);
n_train=round(0.8*n_trial);
n_test=n_trial-n_train;
labels_tr=labels2(1,1:n_train);
labels_ts=labels2(1,n_train+1:end);
labels_tr=transpose(labels_tr);
labels_ts=transpose(labels_ts);
tic;
modello=fitcensemble(x_x,labels_tr,'OptimizeHyperparameters','auto');
time_matrix_train_80_20_1_1(sub,ses,3)=toc;
tic;
predizioni=predict(modello,y_y);
time_matrix_predict_80_20_1_1(sub,ses,3)=toc;
accuracy_80_20_1_1(sub,ses,3)=sum(predizioni==labels_ts)/length(labels_ts)*100;
positive_class = 1;

TP=sum((predizioni==positive_class) & (labels_ts==positive_class));
FP=sum((predizioni==positive_class) & (labels_ts~=positive_class));
FN=sum((predizioni~=positive_class) & (labels_ts==positive_class));
    precision1_80_20pc_1_1(sub,ses,3)=(TP)/(TP+FP);
    recall1_80_20pc_1_1(sub,ses,3)=(TP)/(TP+FN);

    positive_class = 2;

TP2=sum((predizioni==positive_class) & (labels_ts==positive_class));
FP2=sum((predizioni==positive_class) & (labels_ts~=positive_class));
FN2=sum((predizioni~=positive_class) & (labels_ts==positive_class));
    precision2_80_20pc_1_1(sub,ses,3)=(TP2)/(TP2+FP2);
    recall2_80_20pc_1_1(sub,ses,3)=(TP2)/(TP2+FN2);
    end
end