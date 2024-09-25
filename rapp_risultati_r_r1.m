save('results1wsr1.mat','accuracy_80_20def2_1','time_matrix_train_80_20def2_1','time_matrix_predict_80_20def2_1');
save('results2wsr1.mat','accuracy_60_40def2_1','time_matrix_train_60_40def2_1','time_matrix_predict_60_40def2_1');
save('results3wsr1.mat','accuracy_40_60def2_1','time_matrix_train_40_60def2_1','time_matrix_predict_40_60def2_1');
save('results4wsr1.mat','accuracy_20_80def2_1','time_matrix_train_20_80def2_1','time_matrix_predict_20_80def2_1');
save('results1wsr1_1.mat',' precision1_80_20def2_1',' precision2_80_20def2_1','recall1_80_20def2_1',' recall2_80_20def2_1');
save('results2wsr1_1.mat',' precision1_60_40def2_1',' precision2_60_40def2_1','recall1_60_40def2_1',' recall2_60_40def2_1');
save('results3wsr1_1.mat',' precision1_40_60def2_1',' precision2_40_60def2_1','recall1_40_60def2_1',' recall2_40_60def2_1');
save('results4wsr1_1.mat',' precision1_20_80def2_1',' precision2_20_80def2_1','recall1_20_80def2_1',' recall2_20_80def2_1');
for sub=1:25
        for cl=1:3
        [val,idx]=sort(accuracy_80_20_1_1(sub,1:5,cl),'descend');
        accuracy_80_20def(sub,1:5,cl)=val;
        accuracy_80_20def(sub,6:10,cl)=idx;
        [val2,idx2]=sort(accuracy_60_40_2_1(sub,1:5,cl),'descend');
        accuracy_60_40def(sub,1:5,cl)=val2;
        accuracy_60_40def(sub,6:10,cl)=idx2;
        [val3,idx3]=sort(accuracy_40_60_3_1(sub,1:5,cl),'descend');
        accuracy_40_60def(sub,1:5,cl)=val3;
        accuracy_40_60def(sub,6:10,cl)=idx3;
        [val4,idx4]=sort(accuracy_20_80_4_1(sub,1:5,cl),'descend');
        accuracy_20_80def(sub,1:5,cl)=val4;
        accuracy_20_80def(sub,6:10,cl)=idx4;
        end
end
for sub=1:25
        for cl=1:3
        accuracy_80_20def2_1(sub,1:2,cl)=accuracy_80_20def(sub,1:2,cl);
        accuracy_80_20def2_1(sub,3:4,cl)=accuracy_80_20def(sub,6:7,cl);
        accuracy_60_40def2_1(sub,1:2,cl)=accuracy_60_40def(sub,1:2,cl);
        accuracy_60_40def2_1(sub,3:4,cl)=accuracy_60_40def(sub,6:7,cl);
        accuracy_40_60def2_1(sub,1:2,cl)=accuracy_40_60def(sub,1:2,cl);
        accuracy_40_60def2_1(sub,3:4,cl)=accuracy_40_60def(sub,6:7,cl);
        accuracy_20_80def2_1(sub,1:2,cl)=accuracy_20_80def(sub,1:2,cl);
        accuracy_20_80def2_1(sub,3:4,cl)=accuracy_20_80def(sub,6:7,cl);
        end
end
for sub=1:25
    for cl=1:3
        time_matrix_train_80_20def2_1(sub,1:2,cl)=time_matrix_train_80_20_1_1(sub,accuracy_80_20def2_1(sub,3:4,cl),cl);
         time_matrix_train_60_40def2_1(sub,1:2,cl)=time_matrix_train_60_40_2_1(sub,accuracy_60_40def2_1(sub,3:4,cl),cl);
          time_matrix_train_40_60def2_1(sub,1:2,cl)=time_matrix_train_40_60_3_1(sub,accuracy_40_60def2_1(sub,3:4,cl),cl);
           time_matrix_train_20_80def2_1(sub,1:2,cl)=time_matrix_train_20_80_4_1(sub,accuracy_20_80def2_1(sub,3:4,cl),cl);
           time_matrix_predict_80_20def2_1(sub,1:2,cl)=time_matrix_predict_80_20_1_1(sub,accuracy_80_20def2_1(sub,3:4,cl),cl);
         time_matrix_predict_60_40def2_1(sub,1:2,cl)=time_matrix_predict_60_40_2_1(sub,accuracy_60_40def2_1(sub,3:4,cl),cl);
          time_matrix_predict_40_60def2_1(sub,1:2,cl)=time_matrix_predict_40_60_3_1(sub,accuracy_40_60def2_1(sub,3:4,cl),cl);
           time_matrix_predict_20_80def2_1(sub,1:2,cl)=time_matrix_predict_20_80_4_1(sub,accuracy_20_80def2_1(sub,3:4,cl),cl);
    end
end

for sub=1:25
    for cl=1:3
        precision1_80_20def2_1(sub,1:2,cl)=precision1_80_20pc_1_1(sub,accuracy_80_20def2_1(sub,3:4,cl),cl);
        precision2_80_20def2_1(sub,1:2,cl)=precision2_80_20pc_1_1(sub,accuracy_80_20def2_1(sub,3:4,cl),cl);
        recall1_80_20def2_1(sub,1:2,cl)=recall1_80_20pc_1_1(sub,accuracy_80_20def2_1(sub,3:4,cl),cl);
        recall2_80_20def2_1(sub,1:2,cl)=recall2_80_20pc_1_1(sub,accuracy_80_20def2_1(sub,3:4,cl),cl);

        precision1_60_40def2_1(sub,1:2,cl)=precision1_60_40pc_2_1(sub,accuracy_60_40def2_1(sub,3:4,cl),cl);
        precision2_60_40def2_1(sub,1:2,cl)=precision2_60_40pc_2_1(sub,accuracy_60_40def2_1(sub,3:4,cl),cl);
        recall1_60_40def2_1(sub,1:2,cl)=recall1_60_40pc_2_1(sub,accuracy_60_40def2_1(sub,3:4,cl),cl);
        recall2_60_40def2_1(sub,1:2,cl)=recall2_60_40pc_2_1(sub,accuracy_60_40def2_1(sub,3:4,cl),cl);

        precision1_40_60def2_1(sub,1:2,cl)=precision1_40_60pc_3_1(sub,accuracy_40_60def2_1(sub,3:4,cl),cl);
        precision2_40_60def2_1(sub,1:2,cl)=precision2_40_60pc_3_1(sub,accuracy_40_60def2_1(sub,3:4,cl),cl);
        recall1_40_60def2_1(sub,1:2,cl)=recall1_40_60pc_3_1(sub,accuracy_40_60def2_1(sub,3:4,cl),cl);
        recall2_40_60def2_1(sub,1:2,cl)=recall2_40_60pc_3_1(sub,accuracy_40_60def2_1(sub,3:4,cl),cl);

        precision1_20_80def2_1(sub,1:2,cl)=precision1_20_80pc_4_1(sub,accuracy_20_80def2_1(sub,3:4,cl),cl);
        precision2_20_80def2_1(sub,1:2,cl)=precision2_20_80pc_4_1(sub,accuracy_20_80def2_1(sub,3:4,cl),cl);
        recall1_20_80def2_1(sub,1:2,cl)=recall1_20_80pc_4_1(sub,accuracy_20_80def2_1(sub,3:4,cl),cl);
        recall2_20_80def2_1(sub,1:2,cl)=recall2_20_80pc_4_1(sub,accuracy_20_80def2_1(sub,3:4,cl),cl);
    end
end

%Da qui inizia rappresentazione definitiva


for d=1:3
    figure;
    c=bar(squeeze(accuracy_80_20def2_1(:,:,d)));
    if d==1
        xlabel('subjects and sessions')
        ylabel('Accuracy %')
        legend('session 1','session 2','session 3','session 4','session 5')
        title('SVM classification 80-20%')
    elseif d==2
        xlabel('subjects and sessions')
        ylabel('Accuracy %')
        legend('session 1','session 2','session 3','session 4','session 5')
        title('NET classification 80-20%')
    else
        xlabel('subjects and sessions')
        ylabel('Accuracy %')
        legend('session 1','session 2','session 3','session 4','session 5')
        title('RF classification 80-20%')
    end
end

for d=1:3
    figure;
    c=bar(squeeze(accuracy_60_40def2_1(:,:,d)));
    if d==1
        xlabel('subjects and sessions')
        ylabel('Accuracy %')
        legend('session 1','session 2','session 3','session 4','session 5')
        title('SVM classification 60-40%')
    elseif d==2
        xlabel('subjects and sessions')
        ylabel('Accuracy %')
        legend('session 1','session 2','session 3','session 4','session 5')
        title('NET classification 60-40%')
    else
        xlabel('subjects and sessions')
        ylabel('Accuracy %')
        legend('session 1','session 2','session 3','session 4','session 5')
        title('RF classification 60-40%')
    end
end

for d=1:3
    figure;
    c=bar(squeeze(accuracy_40_60def2_1(:,:,d)));
    if d==1
        xlabel('subjects and sessions')
        ylabel('Accuracy %')
        legend('session 1','session 2','session 3','session 4','session 5')
        title('SVM classification 40-60%')
    elseif d==2
        xlabel('subjects and sessions')
        ylabel('Accuracy %')
        legend('session 1','session 2','session 3','session 4','session 5')
        title('NET classification 40-60%')
    else
        xlabel('subjects and sessions')
        ylabel('Accuracy %')
        legend('session 1','session 2','session 3','session 4','session 5')
        title('RF classification 40-60%')
    end
end

for d=1:3
    figure;
    c=bar(squeeze(accuracy_20_80def2_1(:,:,d)));
    if d==1
        xlabel('subjects and sessions')
        ylabel('Accuracy %')
        legend('session 1','session 2','session 3','session 4','session 5')
        title('SVM classification 20-80%')
    elseif d==2
        xlabel('subjects and sessions')
        ylabel('Accuracy %')
        legend('session 1','session 2','session 3','session 4','session 5')
        title('NET classification 20-80%')
    else
        xlabel('subjects and sessions')
        ylabel('Accuracy %')
        legend('session 1','session 2','session 3','session 4','session 5')
        title('RF classification 20-80%')
    end
end

mean_sub_80_20=mean(accuracy_80_20def2_1(:,1:2,:),2);
mean_class_80_20=mean(mean_sub_80_20,1);
mean_sub_60_40=mean(accuracy_60_40def2_1(:,1:2,:),2);
mean_class_60_40=mean(mean_sub_60_40,1);
mean_sub_40_60=mean(accuracy_40_60def2_1(:,1:2,:),2);
mean_class_40_60=mean(mean_sub_40_60,1);
mean_sub_20_80=mean(accuracy_20_80def2_1(:,1:2,:),2);
mean_class_20_80=mean(mean_sub_20_80);

time_train_80_20_sub=mean(time_matrix_train_80_20def2_1,2);
time_train_80_20_class=mean(time_train_80_20_sub,1);
time_predict_80_20_sub=mean(time_matrix_predict_80_20def2_1,2);
time_predict_80_20_class=mean(time_predict_80_20_sub,1);
time_train_60_40_sub=mean(time_matrix_train_60_40def2_1,2);
time_train_60_40_class=mean(time_train_60_40_sub,1);
time_predict_60_40_sub=mean(time_matrix_predict_60_40def2_1,2);
time_predict_60_40_class=mean(time_predict_60_40_sub,1);
time_train_40_60_sub=mean(time_matrix_train_40_60def2_1,2);
time_train_40_60_class=mean(time_train_40_60_sub,1);
time_predict_40_60_sub=mean(time_matrix_predict_40_60def2_1,2);
time_predict_40_60_class=mean(time_predict_40_60_sub,1);
time_train_20_80_sub=mean(time_matrix_train_20_80def2_1,2);
time_train_20_80_class=mean(time_train_20_80_sub,1);
time_predict_20_80_sub=mean(time_matrix_predict_20_80def2_1,2);
time_predict_20_80_class=mean(time_predict_20_80_sub,1);


g=squeeze(mean_sub_80_20)';
figure;
boxplot(g);
xlabel('distribution of accuracy')
title('claassification 80-20%')
figure;
g=squeeze(mean_sub_60_40)';
fig1=boxplot(g);
xlabel('distribution of accuracy')
title('claassification 60-40%')
figure;
g=squeeze(mean_sub_40_60)';
fig2=boxplot(g);
xlabel('distribution of accuracy')
title('claassification 40-60%')
figure;
g=squeeze(mean_sub_20_80)';
fig3=boxplot(g);
xlabel('distribution of accuracy')
title('claassification 20-80%')

figure;
subplot(4,3,1)
plot(squeeze(mean_class_80_20),'-o')
xlabel('classificator')
ylabel('accuracy mean (%)')
title('classification 80-20%')
subplot(4,3,2)
plot(squeeze(time_train_80_20_class),'-o')
xlabel('classificator')
ylabel(' mean time [s]')
title('classification 80-20% time for training')
subplot(4,3,3)
plot(squeeze(time_predict_80_20_class),'-o')
xlabel('classificator')
ylabel(' mean time [s]')
title('classification 80-20% time for prediction')
subplot(4,3,4)
plot(squeeze(mean_class_60_40),'-o')
xlabel('classificator')
ylabel('accuracy mean (%)')
title('classification 60-40%')
subplot(4,3,5)
plot(squeeze(time_train_60_40_class),'-o')
xlabel('classificator')
ylabel(' mean time [s]')
title('classification 60-40% time for training')
subplot(4,3,6)
plot(squeeze(time_predict_60_40_class),'-o')
xlabel('classificator')
ylabel(' mean time [s]')
title('classification 60-40% time for prediction')
subplot(4,3,7)
plot(squeeze(mean_class_40_60),'-o')
xlabel('classificator')
ylabel('accuracy mean (%)')
title('classification 40-60%')
subplot(4,3,8)
plot(squeeze(time_train_40_60_class),'-o')
xlabel('classificator')
ylabel(' mean time [s]')
title('classification 40-60% time for training')
subplot(4,3,9)
plot(squeeze(time_predict_40_60_class),'-o')
xlabel('classificator')
ylabel(' mean time [s]')
title('classification 40-60% time for prediction')
subplot(4,3,10)
plot(squeeze(mean_class_20_80),'-o')
xlabel('classificator')
ylabel('accuracy mean (%)')
title('classification 20-80%')
subplot(4,3,11)
plot(squeeze(time_train_20_80_class),'-o')
xlabel('classificator')
ylabel(' mean time [s]')
title('classification 20-80% time for training')
subplot(4,3,12)
plot(squeeze(time_predict_20_80_class),'-o')
xlabel('classificator')
ylabel(' mean time [s]')
title('classification 20-80% time for prediction')


figure;
subplot(4,3,1)
plot(squeeze(mean_sub_80_20),'-o')
xlabel('subjects')
ylabel('accuracy mean (%)')
title('classification 80-20%')
legend('svm','net','rf')
subplot(4,3,2)
plot(squeeze(time_train_80_20_sub),'-o')
xlabel('subjects')
ylabel(' mean time [s]')
title('classification 80-20% time for training')
legend('svm','net','rf')
subplot(4,3,3)
plot(squeeze(time_predict_80_20_sub),'-o')
xlabel('subjects')
ylabel(' mean time [s]')
title('classification 80-20% time for prediction')
legend('svm','net','rf')
subplot(4,3,4)
plot(squeeze(mean_sub_60_40),'-o')
xlabel('subjects')
ylabel('accuracy mean (%)')
title('classification 80-20%')
legend('svm','net','rf')
subplot(4,3,5)
plot(squeeze(time_train_60_40_sub),'-o')
xlabel('subjects')
ylabel(' mean time [s]')
title('classification 80-20% time for training')
legend('svm','net','rf')
subplot(4,3,6)
plot(squeeze(time_predict_60_40_sub),'-o')
xlabel('subjects')
ylabel(' mean time [s]')
title('classification 60-40% time for prediction')
legend('svm','net','rf')
subplot(4,3,7)
plot(squeeze(mean_sub_40_60),'-o')
xlabel('subjects')
ylabel('accuracy mean (%)')
title('classification 40-60%')
legend('svm','net','rf')
subplot(4,3,8)
plot(squeeze(time_train_40_60_sub),'-o')
xlabel('subjects')
ylabel(' mean time [s]')
title('classification 40-60% time for training')
legend('svm','net','rf')
subplot(4,3,9)
plot(squeeze(time_predict_40_60_sub),'-o')
xlabel('subjects')
ylabel(' mean time [s]')
title('classification 40-60% time for prediction')
legend('svm','net','rf')
subplot(4,3,10)
plot(squeeze(mean_sub_20_80),'-o')
xlabel('subjects')
ylabel('accuracy mean (%)')
title('classification 20-80%')
legend('svm','net','rf')
subplot(4,3,11)
plot(squeeze(time_train_20_80_sub),'-o')
xlabel('subjects')
ylabel(' mean time [s]')
title('classification 20-80% time for training')
legend('svm','net','rf')
subplot(4,3,12)
plot(squeeze(time_predict_20_80_sub),'-o')
xlabel('subjects')
ylabel(' mean time [s]')
title('classification 20-80% time for prediction')
legend('svm','net','rf')

load('feature1_1r2.mat')
for sub=1:25
    for ses=1:5
        t=sprintf('time_feature_%d_%d1',sub,ses);
        d=sprintf('sub_%02d_ses_%02d_tr1',sub,ses);
        w=sprintf('sub_%02d_ses_%02d_te1',sub,ses);
        t_t=eval(t);
        d_d=eval(d);
        w_w=eval(w);
        time_extraction_features_train(sub,ses)=t_t*size(d_d,1)/(size(d_d,1)+size(w_w,1));
        time_extraction_features_test(sub,ses)=t_t*size(w_w,1)/(size(d_d,1)+size(w_w,1));
    end
end
for sub=1:25
mean_extract_features(sub,1)=(time_extraction_features_train(sub,1)+time_extraction_features_test(sub,1));
end
mean_extract_features1=mean(mean_extract_features);


load('feature2_1rdef.mat')
for sub=1:25
    for ses=1:5
        t=sprintf('time_feature_%d_%d1',sub,ses);
        d=sprintf('sub_%02d_ses_%02d_tr1',sub,ses);
        w=sprintf('sub_%02d_ses_%02d_te1',sub,ses);
        t_t=eval(t);
        d_d=eval(d);
        w_w=eval(w);
        time_extraction_features_train_2(sub,ses)=t_t*size(d_d,1)/(size(d_d,1)+size(w_w,1));
        time_extraction_features_test_2(sub,ses)=t_t*size(w_w,1)/(size(d_d,1)+size(w_w,1));
    end
end

load('feature3_1r.mat')
for sub=1:25
    for ses=1:5
        t=sprintf('time_feature_%d_%d1',sub,ses);
        d=sprintf('sub_%02d_ses_%02d_tr1',sub,ses);
        w=sprintf('sub_%02d_ses_%02d_te1',sub,ses);
        t_t=eval(t);
        d_d=eval(d);
        w_w=eval(w);
        time_extraction_features_train_3(sub,ses)=t_t*size(d_d,1)/(size(d_d,1)+size(w_w,1));
        time_extraction_features_test_3(sub,ses)=t_t*size(w_w,1)/(size(d_d,1)+size(w_w,1));
    end
end

load('feature4_1r.mat')
load('feature4_1r2.mat')
for sub=1:25
    for ses=1:5
        t=sprintf('time_feature_%d_%d1',sub,ses);
        d=sprintf('sub_%02d_ses_%02d_tr1',sub,ses);
        w=sprintf('sub_%02d_ses_%02d_te1',sub,ses);
        t_t=eval(t);
        d_d=eval(d);
        w_w=eval(w);
        time_extraction_features_train_4(sub,ses)=t_t*size(d_d,1)/(size(d_d,1)+size(w_w,1));
        time_extraction_features_test_4(sub,ses)=t_t*size(w_w,1)/(size(d_d,1)+size(w_w,1));
    end
end

subplot(4,2,1)
plot(time_extraction_features_train,'-o')
xlabel('subjects and sessions')
legend('session 1','session 2','session 3','session 4','session 5')
ylabel('time [s]')
title('Train 80% of data')
subplot(4,2,2)
plot(time_extraction_features_test,'-o')
xlabel('subjects and sessions')
legend('session 1','session 2','session 3','session 4','session 5')
ylabel('time [s]')
title('Test 20% of data')
subplot(4,2,3)
plot(time_extraction_features_train_2,'-o')
xlabel('subjects and sessions')
legend('session 1','session 2','session 3','session 4','session 5')
ylabel('time [s]')
title('Train 60% of data')
subplot(4,2,4)
plot(time_extraction_features_test_2,'-o')
xlabel('subjects and sessions')
legend('session 1','session 2','session 3','session 4','session 5')
ylabel('time [s]')
title('Test 40% of data')
subplot(4,2,5)
plot(time_extraction_features_train_3,'-o')
xlabel('subjects and sessions')
legend('session 1','session 2','session 3','session 4','session 5')
ylabel('time [s]')
title('Train 40% of data')
subplot(4,2,6)
plot(time_extraction_features_test_3,'-o')
xlabel('subjects and sessions')
legend('session 1','session 2','session 3','session 4','session 5')
ylabel('time [s]')
title(' test 60% of data')
subplot(4,2,7)
plot(time_extraction_features_train_4,'-o')
xlabel('subjects and sessions')
legend('session 1','session 2','session 3','session 4','session 5')
ylabel('time [s]')
title('Train 20% of data')
subplot(4,2,8)
plot(time_extraction_features_test_4,'-o')
xlabel('subjects and sessions')
legend('session 1','session 2','session 3','session 4','session 5')
ylabel('time [s]')
title(' test 80% of data')

for sub=1:25
    for cl=1:3
    
    time_traindef(sub,1:2,cl)=time_extraction_features_train(sub,accuracy_80_20def2_1(sub,3:4,cl));
    time_testdef(sub,1:2,cl)=time_extraction_features_test(sub,accuracy_80_20def2_1(sub,3:4,cl));


    time_traindef2(sub,1:2,cl)=time_extraction_features_train_2(sub,accuracy_60_40def2_1(sub,3:4,cl));
    time_testdef2(sub,1:2,cl)=time_extraction_features_test_2(sub,accuracy_60_40def2_1(sub,3:4,cl));


    time_traindef3(sub,1:2,cl)=time_extraction_features_train_3(sub,accuracy_40_60def2_1(sub,3:4,cl));
    time_testdef3(sub,1:2,cl)=time_extraction_features_test_3(sub,accuracy_40_60def2_1(sub,3:4,cl));


    time_traindef4(sub,1:2,cl)=time_extraction_features_train_4(sub,accuracy_20_80def2_1(sub,3:4,cl));
    time_testdef4(sub,1:2,cl)=time_extraction_features_test_4(sub,accuracy_20_80def2_1(sub,3:4,cl));
    end
end
%INSIEME DI TUTTI I RISULTATI PRE-PROCESSING-LEARNING TIME-PREDICT
%TIME-ACCURACY-PRECISION1/2-RECALL1/2
%80-20
for i=1:2
figure;
subplot(2,4,1)
plot(squeeze(time_traindef(:,i,:)),'-o')
xlabel('subjects')
ylabel('mean time for extraction features-train [s]')
title(['session index' num2str(i)])
legend('SVM','NET','RF')
subplot(2,4,2)
plot(squeeze(time_testdef(:,i,:)),'-o')
xlabel('subjects')
ylabel('mean time for extraction features-test [s]')
title(['session index' num2str(i)])
legend('SVM','NET','RF')
subplot(2,4,3)
plot(squeeze(time_matrix_train_80_20def2_1(:,i,:)),'-o')
xlabel('subjects')
ylabel('mean time for train[s]')
title(['session index' num2str(i)])
legend('SVM','NET','RF')
% subplot(2,4,4)
% plot(squeeze(time_matrix_test_80_20def(:,i,:)),'-o')
% xlabel('subjects')
% ylabel('mean time for test[s]')
% title(['session index' num2str(i)])
% legend('SVM','NET','RF')
subplot(2,4,4)
plot(squeeze(accuracy_80_20def2_1(:,i,:)),'-o')
xlabel('subjects')
ylabel('accuracy (%)')
title(['session index' num2str(i)])
legend('SVM','NET','RF')
subplot(2,4,5)
plot(squeeze(precision1_80_20def2_1(:,i,:)),'-o')
xlabel('subjects')
ylabel('precision for class 1')
title(['session index' num2str(i)])
legend('SVM','NET','RF')
subplot(2,4,6)
plot(squeeze(precision2_80_20def2_1(:,i,:)),'-o')
xlabel('subjects')
ylabel('precision for class 2')
title(['session index' num2str(i)])
legend('SVM','NET','RF')
subplot(2,4,7)
plot(squeeze(recall1_80_20def2_1(:,i,:)),'-o')
xlabel('subjects')
ylabel('recall for class 1')
title(['session index' num2str(i)])
legend('SVM','NET','RF')
subplot(2,4,8)
plot(squeeze(recall2_80_20def2_1(:,i,:)),'-o')
xlabel('subjects')
ylabel('recall for class 2')
title(['session index' num2str(i)])
legend('SVM','NET','RF')
end
%60-40
for i=1:2
figure;
subplot(2,4,1)
plot(squeeze(time_traindef2(:,i,:)),'-o')
xlabel('subjects')
ylabel('mean time for extraction features-train [s]')
title(['session index' num2str(i)])
legend('SVM','NET','RF')
subplot(2,4,2)
plot(squeeze(time_testdef2(:,i,:)),'-o')
xlabel('subjects')
ylabel('mean time for extraction features-test [s]')
title(['session index' num2str(i)])
legend('SVM','NET','RF')
subplot(2,4,3)
plot(squeeze(time_matrix_train_60_40def2_1(:,i,:)),'-o')
xlabel('subjects')
ylabel(' time for train[s]')
title(['session index' num2str(i)])
legend('SVM','NET','RF')
% subplot(2,4,4)
% plot(squeeze(time_matrix_test_60_40def(:,i,:)),'-o')
% xlabel('subjects')
% ylabel('mean time for test[s]')
% title(['session index' num2str(i)])
% legend('SVM','NET','RF')
subplot(2,4,4)
plot(squeeze(accuracy_60_40def2_1(:,i,:)),'-o')
xlabel('subjects')
ylabel('accuracy (%)')
title(['session index' num2str(i)])
legend('SVM','NET','RF')
subplot(2,4,5)
plot(squeeze(precision1_60_40def2_1(:,i,:)),'-o')
xlabel('subjects')
ylabel('precision for class 1')
title(['session index' num2str(i)])
legend('SVM','NET','RF')
subplot(2,4,6)
plot(squeeze(precision2_60_40def2_1(:,i,:)),'-o')
xlabel('subjects')
ylabel('precision for class 2')
title(['session index' num2str(i)])
legend('SVM','NET','RF')
subplot(2,4,7)
plot(squeeze(recall1_60_40def2_1(:,i,:)),'-o')
xlabel('subjects')
ylabel('recall for class 1')
title(['session index' num2str(i)])
legend('SVM','NET','RF')
subplot(2,4,8)
plot(squeeze(recall2_60_40def2_1(:,i,:)),'-o')
xlabel('subjects')
ylabel('recall for class 2')
title(['session index' num2str(i)])
legend('SVM','NET','RF')
end
%40-60
for i=1:2
figure;
subplot(2,4,1)
plot(squeeze(time_traindef3(:,i,:)),'-o')
xlabel('subjects')
ylabel('mean time for extraction features-train [s]')
title(['session index' num2str(i)])
legend('SVM','NET','RF')
subplot(2,4,2)
plot(squeeze(time_testdef3(:,i,:)),'-o')
xlabel('subjects')
ylabel('mean time for extraction features-test [s]')
title(['session index' num2str(i)])
legend('SVM','NET','RF')
subplot(2,4,3)
plot(squeeze(time_matrix_train_40_60def2_1(:,i,:)),'-o')
xlabel('subjects')
ylabel(' time for train[s]')
title(['session index' num2str(i)])
legend('SVM','NET','RF')
% subplot(2,4,4)
% plot(squeeze(time_matrix_test_40_60def(:,i,:)),'-o')
% xlabel('subjects')
% ylabel('mean time for test[s]')
% title(['session index' num2str(i)])
% legend('SVM','NET','RF')
subplot(2,4,4)
plot(squeeze(accuracy_40_60def2_1(:,i,:)),'-o')
xlabel('subjects')
ylabel('accuracy (%)')
title(['session index' num2str(i)])
legend('SVM','NET','RF')
subplot(2,4,5)
plot(squeeze(precision1_40_60def2_1(:,i,:)),'-o')
xlabel('subjects')
ylabel('precision for class 1')
title(['session index' num2str(i)])
legend('SVM','NET','RF')
subplot(2,4,6)
plot(squeeze(precision2_40_60def2_1(:,i,:)),'-o')
xlabel('subjects')
ylabel('precision for class 2')
title(['session index' num2str(i)])
legend('SVM','NET','RF')
subplot(2,4,7)
plot(squeeze(recall1_40_60def2_1(:,i,:)),'-o')
xlabel('subjects')
ylabel('recall for class 1')
title(['session index' num2str(i)])
legend('SVM','NET','RF')
subplot(2,4,8)
plot(squeeze(recall2_40_60def2_1(:,i,:)),'-o')
xlabel('subjects')
ylabel('recall for class 2')
title(['session index' num2str(i)])
legend('SVM','NET','RF')
end
%20-80
for i=1:2
figure;
subplot(2,4,1)
plot(squeeze(time_traindef4(:,i,:)),'-o')
xlabel('subjects')
ylabel('mean time for extraction features-train [s]')
title(['session index' num2str(i)])
legend('SVM','NET','RF')
subplot(2,4,2)
plot(squeeze(time_testdef4(:,i,:)),'-o')
xlabel('subjects')
ylabel('mean time for extraction features-test [s]')
title(['session index' num2str(i)])
legend('SVM','NET','RF')
subplot(2,4,3)
plot(squeeze(time_matrix_train_20_80def2_1(:,i,:)),'-o')
xlabel('subjects')
ylabel(' time for train[s]')
title(['session index' num2str(i)])
legend('SVM','NET','RF')
% subplot(2,4,4)
% plot(squeeze(time_matrix_test_20_80def(:,i,:)),'-o')
% xlabel('subjects')
% ylabel('mean time for test[s]')
% title(['session index' num2str(i)])
% legend('SVM','NET','RF')
subplot(2,4,4)
plot(squeeze(accuracy_20_80def2_1(:,i,:)),'-o')
xlabel('subjects')
ylabel('accuracy (%)')
title(['session index' num2str(i)])
legend('SVM','NET','RF')
subplot(2,4,5)
plot(squeeze(precision1_20_80def2_1(:,i,:)),'-o')
xlabel('subjects')
ylabel('precision for class 1')
title(['session index' num2str(i)])
legend('SVM','NET','RF')
subplot(2,4,6)
plot(squeeze(precision2_20_80def2_1(:,i,:)),'-o')
xlabel('subjects')
ylabel('precision for class 2')
title(['session index' num2str(i)])
legend('SVM','NET','RF')
subplot(2,4,7)
plot(squeeze(recall1_20_80def2_1(:,i,:)),'-o')
xlabel('subjects')
ylabel('recall for class 1')
title(['session index' num2str(i)])
legend('SVM','NET','RF')
subplot(2,4,8)
plot(squeeze(recall2_20_80def2_1(:,i,:)),'-o')
xlabel('subjects')
ylabel('recall for class 2')
title(['session index' num2str(i)])
legend('SVM','NET','RF')
end