for sub=1:25
    for cl=1:3
    [val,idx]=sort(accuracy_80_20_cs2(sub,1,cl),1,'descend');
    accuracy_80_20_csr2_def(sub,1,cl)=val;
    accuracy_80_20_csr2_def(sub,2,cl)=idx;
    [val2,idx2]=sort(accuracy_60_40_cs2(sub,1,cl),1,'descend');
    accuracy_60_40_csr2_def(sub,1,cl)=val2;
    accuracy_60_40_csr2_def(sub,2,cl)=idx2;
    end
end
%choose the min threesold to select the subjects.


for sub=1:25
    for cl=1:3
        if accuracy_80_20_csr2_def(sub,1,cl)>0.65
           accuracy_80_20_csr2_def2(sub,1,cl)=accuracy_80_20_csr2_def(sub,1,cl);
           accuracy_80_20_csr2_def2(sub,2,cl)=accuracy_80_20_csr2_def(sub,2,cl);
        end
    end
end
for sub=1:25
    for cl=1:3
        if accuracy_60_40_csr2_def(sub,1,cl)>0.60
           accuracy_60_40_csr2_def2(sub,1,cl)=accuracy_60_40_csr2_def(sub,1,cl);
           accuracy_60_40_csr2_def2(sub,2,cl)=accuracy_60_40_csr2_def(sub,2,cl);
        end
    end
end
for sub=1:length(accuracy_80_20_csr2_def2)
    for cl=1:3
    time_train_80_20_csr2_def(sub,1,cl)=time_train_80_20_cs2(accuracy_80_20_csr2_def2(sub,2,cl),1,cl);
    time_predict_80_20_csr2_def(sub,1,cl)=time_predict_80_20_cs2(accuracy_80_20_csr2_def2(sub,2,cl),1,cl);
    end
end

for sub=1:length(accuracy_60_40_csr2_def2)
    for cl=1:3
    time_train_60_40_csr2_def(sub,1,cl)=time_train_60_40_cs2(accuracy_60_40_csr2_def2(sub,2,cl),1,cl);
    time_predict_60_40_csr2_def(sub,1,cl)=time_predict_60_40_cs2(accuracy_60_40_csr2_def2(sub,2,cl),1,cl);
    end
end

for sub=1:length(accuracy_80_20_csr2_def2)
    for cl=1:3
    precision1_80_20_csr2_def=precision1_80_20_cs2(accuracy_80_20_csr2_def2(sub,2,cl),1,cl);
    precision2_80_20_csr2_def=precision2_80_20_cs2(accuracy_80_20_csr2_def2(sub,2,cl),1,cl);
    recall1_80_20_csr2_def=recall1_80_20_cs2(accuracy_80_20_csr2_def2(sub,2,cl),1,cl);
    recall2_80_20_csr2_def=recall2_80_20_cs2(accuracy_80_20_csr2_def2(sub,2,cl),1,cl);
    end
end
for sub=1:length(accuracy_60_40_csr2_def2)
    for cl=1:3
       precision1_60_40_csr2_def=precision1_60_40_cs2(accuracy_60_40_csr2_def2(sub,2,cl),1,cl);
       precision2_60_40_csr2_def=precision2_60_40_cs2(accuracy_60_40_csr2_def2(sub,2,cl),1,cl);
       recall1_60_40_csr2_def=recall1_60_40_cs2(accuracy_60_40_csr2_def2(sub,2,cl),1,cl);
       recall2_60_40_csr2_def=recall2_60_40_cs2(accuracy_60_40_csr2_def2(sub,2,cl),1,cl);
    end
end
%plot
for d=1:3
    figure;
    c=bar(squeeze(accuracy_80_20_csr2_def2(:,:,d)));
    if d==1
        xlabel('subjects')
        ylabel('Accuracy %')
        title('SVM classification 80-20%')
    elseif d==2
        xlabel('subjects')
        ylabel('Accuracy %')
        title('NET classification 80-20%')
    else
        xlabel('subjects')
        ylabel('Accuracy %')
        title('RF classification 80-20%')
    end
end

for d=1:3
    figure;
    c=bar(squeeze(accuracy_60_40_csr2_def2(:,:,d)));
    if d==1
        xlabel('subjects')
        ylabel('Accuracy %')
        title('SVM classification 60-40%')
    elseif d==2
        xlabel('subjects')
        ylabel('Accuracy %')
        title('NET classification 60-40%')
    else
        xlabel('subjects')
        ylabel('Accuracy %')
        title('RF classification 60-40%')
    end
end
mean_sub_80_20cs2=mean(accuracy_80_20_csr2_def2(:,1,:),3);
mean_class_80_20cs2=mean(mean_sub_80_20cs2,1);
mean_sub_60_40cs2=mean(accuracy_60_40_csr2_def2(:,1,:),3);
mean_class_60_40cs2=mean(mean_sub_60_40cs2,1);

time_train_80_20_subcs2=mean(time_train_80_20_csr2_def,2);
time_train_80_20_classcs2=mean(time_train_80_20_subcs2,1);
time_predict_80_20_subcs2=mean(time_predict_80_20_csr2_def,2);
time_predict_80_20_classcs2=mean(time_predict_80_20_subcs2,1);
time_train_60_40_subcs2=mean(time_train_60_40_csr2_def,2);
time_train_60_40_classcs2=mean(time_train_60_40_subcs2,1);
time_predict_60_40_subcs2=mean(time_predict_60_40_csr2_def,2);
time_predict_60_40_classcs2=mean(time_predict_60_40_subcs2,1);

g=squeeze(mean_sub_80_20cs2)';
figure;
boxplot(g);
xlabel('distribution of accuracy')
title('claassification 80-20%')
figure;
g=squeeze(mean_sub_60_40cs2)';
fig1=boxplot(g);
xlabel('distribution of accuracy')
title('claassification 60-40%')



figure;
subplot(2,3,1)
plot(squeeze(time_train_80_20_csr2_def(:,1,:)),'-o')
xlabel('subjects')
ylabel('mean time for train[s]')
legend('SVM','NET','RF')
% subplot(2,4,4)
% plot(squeeze(time_matrix_test_80_20def(:,i,:)),'-o')
% xlabel('subjects')
% ylabel('mean time for test[s]')
% title(['session index' num2str(i)])
% legend('SVM','NET','RF')
subplot(2,3,1)
plot(squeeze(accuracy_80_20_csr2_def2(:,1,:)),'-o')
xlabel('subjects')
ylabel('accuracy (%)')
legend('SVM','NET','RF')
subplot(2,3,3)
plot(squeeze(precision1_80_20_csr2_def(:,1,:)),'-o')
xlabel('subjects')
ylabel('precision for class 1')
legend('SVM','NET','RF')
subplot(2,3,3)
plot(squeeze(precision2_80_20_csr2_def(:,1,:)),'-o')
xlabel('subjects')
ylabel('precision for class 2')
legend('SVM','NET','RF')
subplot(2,3,5)
plot(squeeze(recall1_80_20_csr2_def(:,1,:)),'-o')
xlabel('subjects')
ylabel('recall for class 1')
legend('SVM','NET','RF')
subplot(2,3,86)
plot(squeeze(recall2_80_20_csr2_def(:,1,:)),'-o')
xlabel('subjects')
ylabel('recall for class 2')
legend('SVM','NET','RF')

figure;
subplot(2,3,1)
plot(squeeze(time_train_60_40_csr2_def(:,1,:)),'-o')
xlabel('subjects')
ylabel('mean time for train[s]')
legend('SVM','NET','RF')
% subplot(2,4,4)
% plot(squeeze(time_matrix_test_80_20def(:,i,:)),'-o')
% xlabel('subjects')
% ylabel('mean time for test[s]')
% title(['session index' num2str(i)])
% legend('SVM','NET','RF')
subplot(2,3,1)
plot(squeeze(accuracy_60_40_csr2_def2(:,1,:)),'-o')
xlabel('subjects')
ylabel('accuracy (%)')
legend('SVM','NET','RF')
subplot(2,3,3)
plot(squeeze(precision1_60_40_csr2_def(:,1,:)),'-o')
xlabel('subjects')
ylabel('precision for class 1')
legend('SVM','NET','RF')
subplot(2,3,3)
plot(squeeze(precision2_60_40_csr2_def(:,1,:)),'-o')
xlabel('subjects')
ylabel('precision for class 2')
legend('SVM','NET','RF')
subplot(2,3,5)
plot(squeeze(recall1_60_40_csr2_def(:,1,:)),'-o')
xlabel('subjects')
ylabel('recall for class 1')
legend('SVM','NET','RF')
subplot(2,3,86)
plot(squeeze(recall2_60_40_csr2_def(:,1,:)),'-o')
xlabel('subjects')
ylabel('recall for class 2')
legend('SVM','NET','RF')
