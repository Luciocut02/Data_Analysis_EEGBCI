load('features1_1cs.mat')
mean_accuracy_class_80_20_cs=mean(accuracy_80_20cs1_1,1);
mean_accuracy_sub_80_20_cs=mean(accuracy_80_20cs1_1,2);
%sogg1,2,4,6,7,13,20,21 scelti perchè i + performanti 
accuracy_cs1_def(1:2,:)=accuracy_80_20cs1_1(1:2,:);
accuracy_cs1_def(3,:)=accuracy_80_20cs1_1(4,:);
accuracy_cs1_def(4:5,:)=accuracy_80_20cs1_1(6:7,:);
accuracy_cs1_def(6,:)=accuracy_80_20cs1_1(13,:);
accuracy_cs1_def(7:8,:)=accuracy_80_20cs1_1(20:21,:);
mean_cs1_def_cl=mean(accuracy_cs1_def,1);
mean_cs1_def_sub=mean(accuracy_cs1_def,2);
for sub=1:25
    x=sprintf('time_extraction_featurecs1_%d',sub);
    y=sprintf('feature_sogg_80cs_%d',sub);
    z=sprintf('feature_sogg_20cs_%d',sub);
    x_x=eval(x);
    y_y=eval(y);
    z_z=eval(z);
    time_train_featurecs1(sub,1)=(x_x*size(y_y,1))/(size(y_y,1)+size(x_x,1));
    time_test_featurecs1(sub,1)=(x_x*size(z_z,1))/(size(y_y,1)+size(x_x,1));
end
for sub=1:25
mean_extract_features(sub,1)=(time_train_featurecs1(sub,1)+time_test_featurecs1(sub,1));
end
mean_extract_features=mean(mean_extract_features);

precision1_cs1_def(1:2,:)=precision1_80_20cs1_1(1:2,:);
precision1_cs1_def(3,:)=precision1_80_20cs1_1(4,:);
precision1_cs1_def(4:5,:)=precision1_80_20cs1_1(6:7,:);
precision1_cs1_def(6,:)=precision1_80_20cs1_1(13,:);
precision1_cs1_def(7:8,:)=precision1_80_20cs1_1(20:21,:);
mean_cs1_pc1_cl=mean(precision1_cs1_def,1);
mean_cs1_pc1_sub=mean(precision1_cs1_def,2);

precision2_cs1_def(1:2,:)=precision2_80_20cs1_1(1:2,:);
precision2_cs1_def(3,:)=precision2_80_20cs1_1(4,:);
precision2_cs1_def(4:5,:)=precision2_80_20cs1_1(6:7,:);
precision2_cs1_def(6,:)=precision2_80_20cs1_1(13,:);
precision2_cs1_def(7:8,:)=precision2_80_20cs1_1(20:21,:);
mean_cs1_pc2_cl=mean(precision2_cs1_def,1);
mean_cs1_pc2_sub=mean(precision2_cs1_def,2);

recall1_cs1_def(1:2,:)=precision1_80_20cs1_1(1:2,:);
recall1_cs1_def(3,:)=precision1_80_20cs1_1(4,:);
recall1_cs1_def(4:5,:)=precision1_80_20cs1_1(6:7,:);
recall1_cs1_def(6,:)=precision1_80_20cs1_1(13,:);
recall1_cs1_def(7:8,:)=precision1_80_20cs1_1(20:21,:);
mean_cs1_rc1_cl=mean(recall1_cs1_def,1);
mean_cs1_rc1_sub=mean(recall1_cs1_def,2);

recall2_cs1_def(1:2,:)=precision1_80_20cs1_1(1:2,:);
recall2_cs1_def(3,:)=precision1_80_20cs1_1(4,:);
recall2_cs1_def(4:5,:)=precision1_80_20cs1_1(6:7,:);
recall2_cs1_def(6,:)=precision1_80_20cs1_1(13,:);
recall2_cs1_def(7:8,:)=precision1_80_20cs1_1(20:21,:);
mean_cs1_rc2_cl=mean(recall2_cs1_def,1);
mean_cs1_rc2_sub=mean(recall2_cs1_def,2);


time_train_cs1_featdef(1:2,1)=time_train_featurecs1(1:2);
time_train_cs1_featdef(3,1)=time_train_featurecs1(4,1);
time_train_cs1_featdef(4:5,1)=time_train_featurecs1(6:7,1);
time_train_cs1_featdef(6,1)=time_train_featurecs1(13,1);
time_train_cs1_featdef(7:8,1)=time_train_featurecs1(20:21,1);
cs1_def_cl_time_train_feat=mean(time_train_cs1_featdef,1);
cs1_def_sub_time_train_feat=mean(time_train_cs1_featdef,2);
time_test_cs1_featdef(1:2,1)=time_test_featurecs1(1:2,1);
time_test_cs1_featdef(3,1)=time_test_featurecs1(4,1);
time_test_cs1_featdef(4:5,1)=time_test_featurecs1(6:7,1);
time_test_cs1_featdef(6,1)=time_test_featurecs1(13,1);
time_test_cs1_featdef(7:8,1)=time_test_featurecs1(20:21,1);
cs1_def_cl_time_train_feat=mean(time_test_cs1_featdef,1);
cs1_def_sub_time_train_feat=mean(time_test_cs1_featdef,2);

time_train_cs1_def(1:2,:)=time_train_80_20cs1_1(1:2,:);
time_train_cs1_def(3,:)=time_train_80_20cs1_1(4,:);
time_train_cs1_def(4:5,:)=time_train_80_20cs1_1(6:7,:);
time_train_cs1_def(6,:)=time_train_80_20cs1_1(13,:);
time_train_cs1_def(7:8,:)=time_train_80_20cs1_1(20:21,:);
cs1_def_cl_time_train=mean(time_train_cs1_def,1);
cs1_def_sub_time_train=mean(time_train_cs1_def,2);
time_predict_cs1_def(1:2,:)=time_predict_80_20cs1_1(1:2,:);
time_predict_cs1_def(3,:)=time_predict_80_20cs1_1(4,:);
time_predict_cs1_def(4:5,:)=time_predict_80_20cs1_1(6:7,:);
time_predict_cs1_def(6,:)=time_predict_80_20cs1_1(13,:);
time_predict_cs1_def(7:8,:)=time_predict_80_20cs1_1(20:21,:);
cs1_def_cl_time_test=mean(time_predict_cs1_def,1);
cs1_def_sub_time_test=mean(time_predict_cs1_def,2);


    figure;
    subplot(2,4,1)
    plot(time_train_cs1_featdef(:,:),'-o')
    ylabel('mean time for extraction features [s]')
    xlabel('subjects')
    subplot(2,4,2)
    plot(time_test_cs1_featdef(:,:),'-o')
    ylabel('mean time for extraction features [s]')
    xlabel('subjects')
    subplot(2,4,3)
    plot(time_train_cs1_def(:,:),'-o')
    ylabel('time for train [s]')
    xlabel('subjects')
    legend('SVM','NET','RF')
    % subplot(4,2,4)
    % plot(time_predict_cs2_def(:,i),'-o')
    % ylabel('time for test [s]')
    % xlabel('subjects')
    subplot(2,4,4)
    plot(accuracy_cs1_def(:,:),'-o')
    ylabel('accuracy [%]')
    xlabel('subjects')
    legend('SVM','NET','RF')
    subplot(2,4,5)
    plot(precision1_cs1_def(:,:),'-o')
    ylabel('Precision 1')
    xlabel('subjects')
    legend('SVM','NET','RF')
    subplot(2,4,6)
    plot(precision2_cs1_def(:,:),'-o')
    ylabel('Precision 1')
    xlabel('subjects')
    legend('SVM','NET','RF')
    subplot(2,4,7)
    plot(recall1_cs1_def(:,:),'-o')
    ylabel('Recall 1')
    xlabel('subjects')
    legend('SVM','NET','RF')
    subplot(2,4,8)
    plot(recall2_cs1_def(:,:),'-o')
    ylabel('Recall 2')
    xlabel('subjects')
    legend('SVM','NET','RF')

figure;
c=accuracy_cs1_def';
boxplot(c)
xlabel('Accuracy distribution')
title('classification cross-session 80-20%')

%60-40%--->sogg 1,2,4,6,13,15,20,23
accuracy_cs2_def(1:2,:)=accuracy_60_40cs2_1(1:2,:);
accuracy_cs2_def(3,:)=accuracy_60_40cs2_1(4,:);
accuracy_cs2_def(4,:)=accuracy_60_40cs2_1(6,:);
accuracy_cs2_def(5,:)=accuracy_60_40cs2_1(13,:);
accuracy_cs2_def(6,:)=accuracy_60_40cs2_1(15,:);
accuracy_cs2_def(7,:)=accuracy_60_40cs2_1(20,:);
accuracy_cs2_def(8,:)=accuracy_60_40cs2_1(23,:);
mean_cs2_def_cl=mean(accuracy_cs2_def,1);
mean_cs2_def_sub=mean(accuracy_cs2_def,2);
for sub=1:25
    x=sprintf('time_extraction_featurecs2_%d',sub);
    y=sprintf('feature_sogg_60cs_%d',sub);
    z=sprintf('feature_sogg_40cs_%d',sub);
    x_x=eval(x);
    y_y=eval(y);
    z_z=eval(z);
    time_train_featurecs2(sub,1)=(x_x*size(y_y,1))/(size(y_y,1)+size(x_x,1));
    time_test_featurecs2(sub,1)=(x_x*size(z_z,1))/(size(y_y,1)+size(x_x,1));
end

precision1_cs2_def(1:2,:)=precision1_60_40cs2_1(1:2,:);
precision1_cs2_def(3,:)=precision1_60_40cs2_1(4,:);
precision1_cs2_def(4,:)=precision1_60_40cs2_1(6,:);
precision1_cs2_def(5,:)=precision1_60_40cs2_1(13,:);
precision1_cs2_def(6,:)=precision1_60_40cs2_1(15,:);
precision1_cs2_def(7,:)=precision1_60_40cs2_1(20,:);
precision1_cs2_def(8,:)=precision1_60_40cs2_1(23,:);
mean_cs2_pc1_cl=mean(precision1_cs2_def,1);
mean_cs2_pc1_sub=mean(precision1_cs2_def,2);

precision2_cs2_def(1:2,:)=precision2_60_40cs2_1(1:2,:);
precision2_cs2_def(3,:)=precision2_60_40cs2_1(4,:);
precision2_cs2_def(4,:)=precision2_60_40cs2_1(6,:);
precision2_cs2_def(5,:)=precision2_60_40cs2_1(13,:);
precision2_cs2_def(6,:)=precision2_60_40cs2_1(15,:);
precision2_cs2_def(7,:)=precision2_60_40cs2_1(20,:);
precision2_cs2_def(8,:)=precision2_60_40cs2_1(23,:);
mean_cs2_pc2_cl=mean(precision2_cs2_def,1);
mean_cs2_pc2_sub=mean(precision2_cs2_def,2);

recall1_cs2_def(1:2,:)=recall1_60_40cs2_1(1:2,:);
recall1_cs2_def(3,:)=recall1_60_40cs2_1(4,:);
recall1_cs2_def(4,:)=recall1_60_40cs2_1(6,:);
recall1_cs2_def(5,:)=recall1_60_40cs2_1(13,:);
recall1_cs2_def(6,:)=recall1_60_40cs2_1(15,:);
recall1_cs2_def(7,:)=recall1_60_40cs2_1(20,:);
recall1_cs2_def(8,:)=recall1_60_40cs2_1(23,:);
mean_cs2_rc1_cl=mean(recall1_cs2_def,1);
mean_cs2_rc1_sub=mean(recall1_cs2_def,2);

recall2_cs2_def(1:2,:)=recall2_60_40cs2_1(1:2,:);
recall2_cs2_def(3,:)=recall2_60_40cs2_1(4,:);
recall2_cs2_def(4,:)=recall2_60_40cs2_1(6,:);
recall2_cs2_def(5,:)=recall2_60_40cs2_1(13,:);
recall2_cs2_def(6,:)=recall2_60_40cs2_1(15,:);
recall2_cs2_def(7,:)=recall2_60_40cs2_1(20,:);
recall2_cs2_def(8,:)=recall2_60_40cs2_1(23,:);
mean_cs2_rc2_cl=mean(recall2_cs2_def,1);
mean_cs2_rc2_sub=mean(recall2_cs2_def,2);

time_train_cs2_featdef(1:2,1)=time_train_featurecs2(1:2,1);
time_train_cs2_featdef(3,1)=time_train_featurecs2(4,1);
time_train_cs2_featdef(4,1)=time_train_featurecs2(6,1);
time_train_cs2_featdef(5,1)=time_train_featurecs2(13,1);
time_train_cs2_featdef(6,1)=time_train_featurecs2(15,1);
time_train_cs2_featdef(7,1)=time_train_featurecs2(20,1);
time_train_cs2_featdef(8,1)=time_train_featurecs2(23,1);
cs2_def_cl_time_train_feat=mean(time_train_cs2_featdef,1);
cs2_def_sub_time_train_feat=mean(time_train_cs2_featdef,2);
time_test_cs2_featdef(1:2,1)=time_test_featurecs2(1:2,1);
time_test_cs2_featdef(3,1)=time_test_featurecs2(4,1);
time_test_cs2_featdef(4,1)=time_test_featurecs2(6,1);
time_test_cs2_featdef(5,1)=time_test_featurecs2(13,1);
time_test_cs2_featdef(6,1)=time_test_featurecs2(15,1);
time_test_cs2_featdef(7,1)=time_test_featurecs2(20,1);
time_test_cs2_featdef(8,1)=time_test_featurecs2(23,1);
cs2_def_cl_time_train_feat=mean(time_test_cs2_featdef,1);
cs2_def_sub_time_train_feat=mean(time_test_cs2_featdef,2);

time_train_cs2_def(1:2,:)=time_train_60_40cs2_1(1:2,:);
time_train_cs2_def(3,:)=time_train_60_40cs2_1(4,:);
time_train_cs2_def(4,:)=time_train_60_40cs2_1(6,:);
time_train_cs2_def(5,:)=time_train_60_40cs2_1(13,:);
time_train_cs2_def(6,:)=time_train_60_40cs2_1(15,:);
time_train_cs2_def(7,:)=time_train_60_40cs2_1(20,:);
time_train_cs2_def(8,:)=time_train_60_40cs2_1(23,:);
cs2_def_cl_time_train=mean(time_train_cs2_def,1);
cs2_def_sub_time_train=mean(time_train_cs2_def,2);
time_predict_cs2_def(1:2,:)=time_predict_60_40cs2_1(1:2,:);
time_predict_cs2_def(3,:)=time_predict_60_40cs2_1(4,:);
time_predict_cs2_def(4,:)=time_predict_60_40cs2_1(6,:);
time_predict_cs2_def(5,:)=time_predict_60_40cs2_1(13,:);
time_predict_cs2_def(6,:)=time_predict_60_40cs2_1(15,:);
time_predict_cs2_def(7,:)=time_predict_60_40cs2_1(20,:);
time_predict_cs2_def(8,:)=time_predict_60_40cs2_1(23,:);
cs2_def_cl_time_test=mean(time_predict_cs2_def,1);
cs2_def_sub_time_test=mean(time_predict_cs2_def,2);


    figure;
    subplot(2,4,1)
    plot(time_train_cs2_featdef(:,1),'-o')
    ylabel('mean time for extraction features [s]')
    xlabel('subjects')
    subplot(2,4,2)
    plot(time_test_cs2_featdef(:,1),'-o')
    ylabel('mean time for extraction features [s]')
    xlabel('subjects')
    subplot(2,4,3)
    plot(time_train_cs2_def(:,:),'-o')
    ylabel('time for train [s]')
    xlabel('subjects')
    legend('SVM','NET','RF')
    % subplot(4,2,4)
    % % plot(time_predict_cs2_def(:,i),'-o')
    % % ylabel('time for test [s]')
    % % xlabel('subjects')
    subplot(2,4,4)
    plot(accuracy_cs2_def(:,:),'-o')
    ylabel('accuracy [%]')
    xlabel('subjects')
    legend('SVM','NET','RF')
    subplot(2,4,5)
    plot(precision1_cs2_def(:,:),'-o')
    ylabel('precision for class 1')
    xlabel('subjects')
    legend('SVM','NET','RF')
    subplot(2,4,6)
    plot(precision2_cs2_def(:,:),'-o')
    ylabel('precision for class 2')
    xlabel('subjects')
    legend('SVM','NET','RF')
    subplot(2,4,7)
    plot(recall1_cs2_def(:,:),'-o')
    ylabel('recall for class 1')
    xlabel('subjects')
    legend('SVM','NET','RF')
    subplot(2,4,8)
    plot(recall2_cs2_def(:,:),'-o')
    ylabel('recall for class 2')
    xlabel('subjects')
    legend('SVM','NET','RF')

figure;
c=accuracy_cs2_def';
boxplot(c)
xlabel('Accuracy distribution')
title('classification cross-session 60-40%')

%40-60%--->scegliere i soggetti quando finisce di runnare
%ricavo accuracy
accuracy_cs3_def(1:2,1:2)=accuracy_40_60cs3_1(1:2,2:3);
accuracy_cs3_def(3,1:2)=accuracy_40_60cs3_1(4,2:3);
accuracy_cs3_def(4:5,1:2)=accuracy_40_60cs3_1(6:7,2:3);
accuracy_cs3_def(6,1:2)=accuracy_40_60cs3_1(13,2:3);
accuracy_cs3_def(7:9,1:2)=accuracy_40_60cs3_1(19:21,2:3);
mean_cs3_def_cl=mean(accuracy_cs3_def,1);
mean_cs3_def_sub=mean(accuracy_cs3_def,2);
%ricavo tempi di estrazione features per tutti i soggetti
for sub=1:25
    x=sprintf('time_extraction_featurecs3_%d',sub);
    y=sprintf('feature_sogg_40cs_%d',sub);
    z=sprintf('feature_sogg_60cs_%d',sub);
    x_x=eval(x);
    y_y=eval(y);
    z_z=eval(z);
    time_train_featurecs3(sub,1)=(x_x*size(y_y,1))/(size(y_y,1)+size(x_x,1));
    time_test_featurecs3(sub,1)=(x_x*size(z_z,1))/(size(y_y,1)+size(x_x,1));
end
%ricavo precision e recall
precision1_cs3_def(1:2,1:2)=precision1_40_60cs3_1(1:2,2:3);
precision1_cs3_def(3,1:2)=precision1_40_60cs3_1(4,2:3);
precision1_cs3_def(4:5,1:2)=precision1_40_60cs3_1(6:7,2:3);
precision1_cs3_def(6,1:2)=precision1_40_60cs3_1(13,2:3);
precision1_cs3_def(7:9,1:2)=precision1_40_60cs3_1(19:21,2:3);
mean_cs3_pc1_cl=mean(precision1_cs3_def,1);
mean_cs3_pc1_sub=mean(precision1_cs3_def,2);

precision2_cs3_def(1:2,1:2)=precision2_40_60cs3_1(1:2,2:3);
precision2_cs3_def(3,1:2)=precision2_40_60cs3_1(4,2:3);
precision2_cs3_def(4:5,1:2)=precision2_40_60cs3_1(6:7,2:3);
precision2_cs3_def(6,1:2)=precision2_40_60cs3_1(13,2:3);
precision2_cs3_def(7:9,1:2)=precision2_40_60cs3_1(19:21,2:3);
mean_cs3_pc2_cl=mean(precision2_cs3_def,1);
mean_cs3_pc2_sub=mean(precision2_cs3_def,2);

recall1_cs3_def(1:2,1:2)=recall1_40_60cs3_1(1:2,2:3);
recall1_cs3_def(3,1:2)=recall1_40_60cs3_1(4,2:3);
recall1_cs3_def(4:5,1:2)=recall1_40_60cs3_1(6:7,2:3);
recaall1_cs3_def(6,1:2)=recall1_40_60cs3_1(13,2:3);
recall1_cs3_def(7:9,1:2)=recall1_40_60cs3_1(19:21,2:3);
mean_cs3_rc1_cl=mean(recall1_cs3_def,1);
mean_cs3_rc1_sub=mean(recall1_cs3_def,2);

recall2_cs3_def(1:2,1:2)=recall2_40_60cs3_1(1:2,2:3);
recall2_cs3_def(3,1:2)=recall2_40_60cs3_1(4,2:3);
recall2_cs3_def(4:5,1:2)=recall2_40_60cs3_1(6:7,2:3);
recaall2_cs3_def(6,1:2)=recall2_40_60cs3_1(13,2:3);
recall2_cs3_def(7:9,1:2)=recall2_40_60cs3_1(19:21,2:3);
mean_cs3_rc2_cl=mean(recall2_cs3_def,1);
mean_cs3_rc2_sub=mean(recall2_cs3_def,2);
%ricavo tempi di estrazione features per i soggetti scelti
time_train_cs3_featdef(1:2,1)=time_train_featurecs3_1(1:2);
time_train_cs3_featdef(3,1)=time_train_featurecs3_1(4,1);
time_train_cs3_featdef(4:5,1)=time_train_featurecs3_1(6:7,1);
time_train_cs3_featdef(6,1)=time_train_featurecs3_1(13,1);
time_train_cs3_featdef(7:9,1)=time_train_featurecs3_1(19:21,1);
cs3_def_cl_time_train_feat=mean(time_train_cs3_featdef,1);
cs3_def_sub_time_train_feat=mean(time_train_cs3_featdef,2);
time_test_cs3_featdef(1:2,1)=time_test_featurecs3_1(1:2,1);
time_test_cs3_featdef(3,1)=time_test_featurecs3_1(4,1);
time_test_cs3_featdef(4:5,1)=time_test_featurecs3_1(6:7,1);
time_test_cs3_featdef(6,1)=time_test_featurecs3_1(13,1);
time_test_cs3_featdef(7:9,1)=time_est_featurecs3_1(19:21,1);
cs3_def_cl_time_train_feat=mean(time_test_cs3_featdef,1);
cs3_def_sub_time_train_feat=mean(time_test_cs3_featdef,2);
%ricavo tempi di addestramento e test dei classificatori scelti e per i soggetti
%scelti
time_train_cs3_def(1:2,1:2)=time_train_40_60cs3_1(1:2,2:3);
time_train_cs3_def(3,1:2)=time_train_40_60cs3_1(4,2:3);
time_train_cs3_def(4:5,1:2)=time_train_40_60cs3_1(6:7,2:3);
time_train_cs3_def(6,1:2)=time_train_40_60cs3_1(13,2:3);
time_train_cs3_def(7:9,1:2)=time_train_40_60cs3_1(19:21,2:3);
cs2_def_cl_time_train=mean(time_train_cs3_def,1);
cs2_def_sub_time_train=mean(time_train_cs3_def,2);
time_predict_cs3_def(1:2,1:2)=time_predict_40_60cs3_1(1:2,2:3);
time_predict_cs3_def(3,1:2)=time_predict_40_60cs3_1(4,2:3);
time_predict_cs3_def(4:5,1:2)=time_predict_60_20cs3_1(6:7,2:3);
time_predict_cs3_def(6,1:2)=time_predict_40_60cs3_1(13,2:3);
time_predict_cs3_def(7:9,1:2)=time_predict_40_60cs3_1(19:21,2:3);
cs3_def_cl_time_test=mean(time_predict_cs3_def,1);
cs3_def_sub_time_test=mean(time_predict_cs3_def,2);

for i=1:2
    figure;
    subplot(4,2,1)
    plot(time_train_cs3_featdef(:,1),'-o')
    ylabel('mean time for extraction features [s]')
    xlabel('subjects')
    subplot(4,2,2)
    plot(time_test_cs3_featdef(:,1),'-o')
    ylabel('mean time for extraction features [s]')
    xlabel('subjects')
    subplot(4,2,3)
    plot(time_train_cs3_def(:,i),'-o')
    ylabel('time for train [s]')
    xlabel('subjects')
    % subplot(4,2,4)
    % plot(time_predict_cs3_def(:,i),'-o')
    % ylabel('time for test [s]')
    % xlabel('subjects')
    subplot(4,2,5)
    plot(accuracy_cs3_def(:,i),'-o')
    ylabel('accuracy [%]')
    xlabel('subjects')
end

figure;
c=accuracy_cs3_def';
boxplot(c)
xlabel('Accuracy distribution')
title('classification cross-session 40-60%')


%20-80%--->scegliere i soggetti quando finisce di runnare
accuracy_cs4_def(1:2,1:2)=accuracy_20_80cs4_1(1:2,2:3);
accuracy_cs4_def(3,1:2)=accuracy_20_80cs4_1(4,2:3);
accuracy_cs4_def(4:5,1:2)=accuracy_20_80cs4_1(6:7,2:3);
accuracy_cs4_def(6,1:2)=accuracy_20_80cs4_1(13,2:3);
accuracy_cs4_def(7:9,1:2)=accuracy_20_80cs4_1(19:21,2:3);
mean_cs4_def_cl=mean(accuracy_cs4_def,1);
mean_cs4_def_sub=mean(accuracy_cs4_def,2);
for sub=1:25
    x=sprintf('time_extraction_featurecs4_%d',sub);
    y=sprintf('feature_sogg_20cs_%d',sub);
    z=sprintf('feature_sogg_80cs_%d',sub);
    x_x=eval(x);
    y_y=eval(y);
    z_z=eval(z);
    time_train_featurecs4(sub,1)=(x_x*size(y_y,1))/(size(y_y,1)+size(x_x,1));
    time_test_featurecs4(sub,1)=(x_x*size(z_z,1))/(size(y_y,1)+size(x_x,1));
end

precision1_cs4_def(1:2,1:2)=precision1_20_80cs4_1(1:2,2:3);
precision1_cs4_def(3,1:2)=precision1_20_80cs4_1(4,2:3);
precision1_cs4_def(4:5,1:2)=precision1_20_80cs4_1(6:7,2:3);
precision1_cs4_def(6,1:2)=precision1_20_80cs4_1(13,2:3);
precision1_cs4_def(7:9,1:2)=precision1_20_80cs4_1(19:21,2:3);
mean_cs4_pc1_cl=mean(precision1_cs4_def,1);
mean_cs4_pc1_sub=mean(precision1_cs4_def,2);

precision2_cs4_def(1:2,1:2)=precision2_20_80cs4_1(1:2,2:3);
precision2_cs4_def(3,1:2)=precision2_20_80cs4_1(4,2:3);
precision2_cs4_def(4:5,1:2)=precision2_20_80cs4_1(6:7,2:3);
precision2_cs4_def(6,1:2)=precision2_20_80cs4_1(13,2:3);
precision2_cs4_def(7:9,1:2)=precision2_20_80cs4_1(19:21,2:3);
mean_cs4_pc2_cl=mean(precision2_cs4_def,1);
mean_cs4_pc2_sub=mean(precision2_cs4_def,2);

recall1_cs4_def(1:2,1:2)=recall1_20_80cs4_1(1:2,2:3);
recall1_cs4_def(3,1:2)=recall1_20_80cs4_1(4,2:3);
recall1_cs4_def(4:5,1:2)=recall1_20_80cs4_1(6:7,2:3);
recaall1_cs4_def(6,1:2)=recall1_20_80cs4_1(13,2:3);
recall1_cs4_def(7:9,1:2)=recall1_20_80cs4_1(19:21,2:3);
mean_cs4_rc1_cl=mean(recall1_cs4_def,1);
mean_cs4_rc1_sub=mean(recall1_cs4_def,2);

recall2_cs4_def(1:2,1:2)=recall2_20_80cs4_1(1:2,2:3);
recall2_cs4_def(3,1:2)=recall2_20_80cs4_1(4,2:3);
recall2_cs4_def(4:5,1:2)=recall2_20_80cs4_1(6:7,2:3);
recaall2_cs4_def(6,1:2)=recall2_20_80cs4_1(13,2:3);
recall2_cs4_def(7:9,1:2)=recall2_20_80cs4_1(19:21,2:3);
mean_cs4_rc2_cl=mean(recall2_cs4_def,1);
mean_cs4_rc2_sub=mean(recall2_cs4_def,2);



time_train_cs4_featdef(1:2,1)=time_train_featurecs4_1(1:2);
time_train_cs4_featdef(3,1)=time_train_featurecs4_1(4,1);
time_train_cs4_featdef(4:5,1)=time_train_featurecs4_1(6:7,1);
time_train_cs4_featdef(6,1)=time_train_featurecs4_1(13,1);
time_train_cs4_featdef(7:9,1)=time_train_featurecs4_1(19:21,1);
cs4_def_cl_time_train_feat=mean(time_train_cs4_featdef,1);
cs4_def_sub_time_train_feat=mean(time_train_cs4_featdef,2);
time_test_cs4_featdef(1:2,1)=time_test_featurecs4_1(1:2,1);
time_test_cs4_featdef(3,1)=time_test_featurecs4_1(4,1);
time_test_cs4_featdef(4:5,1)=time_test_featurecs4_1(6:7,1);
time_test_cs4_featdef(6,1)=time_test_featurecs4_1(13,1);
time_test_cs4_featdef(7:9,1)=time_est_featurecs4_1(19:21,1);
cs4_def_cl_time_train_feat=mean(time_test_cs4_featdef,1);
cs4_def_sub_time_train_feat=mean(time_test_cs4_featdef,2);

time_train_cs4_def(1:2,1:2)=time_train_20_80cs4_1(1:2,2:3);
time_train_cs4_def(3,1:2)=time_train_20_80cs4_1(4,2:3);
time_train_cs4_def(4:5,1:2)=time_train_20_80cs4_1(6:7,2:3);
time_train_cs4_def(6,1:2)=time_train_20_80cs4_1(13,2:3);
time_train_cs4_def(7:9,1:2)=time_train_20_80cs4_1(19:21,2:3);
cs4_def_cl_time_train=mean(time_train_cs4_def,1);
cs4_def_sub_time_train=mean(time_train_cs4_def,2);
time_predict_cs4_def(1:2,1:2)=time_predict_20_80cs4_1(1:2,2:3);
time_predict_cs4_def(3,1:2)=time_predict_20_80cs4_1(4,2:3);
time_predict_cs4_def(4:5,1:2)=time_predict_20_80cs4_1(6:7,2:3);
time_predict_cs4_def(6,1:2)=time_predict_20_80cs4_1(13,2:3);
time_predict_cs4_def(7:9,1:2)=time_predict_420_80cs4_1(19:21,2:3);
cs4_def_cl_time_test=mean(time_predict_cs4_def,1);
cs4_def_sub_time_test=mean(time_predict_cs4_def,2);

for i=1:2
    figure;
    subplot(4,2,1)
    plot(time_train_cs4_featdef(:,1),'-o')
    ylabel('mean time for extraction features [s]')
    xlabel('subjects')
    subplot(4,2,2)
    plot(time_test_cs4_featdef(:,1),'-o')
    ylabel('mean time for extraction features [s]')
    xlabel('subjects')
    subplot(4,2,3)
    plot(time_train_cs4_def(:,i),'-o')
    ylabel('time for train [s]')
    xlabel('subjects')
    subplot(4,2,4)
    % plot(time_predict_cs4_def(:,i),'-o')
    % ylabel('time for test [s]')
    % xlabel('subjects')
    subplot(4,2,5)
    plot(accuracy_cs4_def(:,i),'-o')
    ylabel('accuracy [%]')
    xlabel('subjects')
end

figure;
c=accuracy_cs4_def';
boxplot(c)
xlabel('Accuracy distribution')
title('classification cross-session 20-80%')
    