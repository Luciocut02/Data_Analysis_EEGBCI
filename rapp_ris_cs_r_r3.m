accuracy_80_20_csr_3=squeeze(accuracy_80_20_csr_3);
for sub=1:25
    if accuracy_80_20_csr_3(sub,1:3)>65
        accuracy_80_20_csr3_def(sub,1:3)=accuracy_80_20_csr_3(sub,1:3);
    end
end


val1=find((accuracy_80_20_csr3_def(1:20)~=0));
accuracy_80_20_csr3_def2=zeros(size(val1,1),3);
for sub=1:size(val1,2)
       accuracy_80_20_csr3_def2(sub,1:3)=accuracy_80_20_csr3_def(val1(1,sub),1:3);
end

accuracy_60_40_csr_3=squeeze(accuracy_60_40_csr_3);
for sub=1:25
 
    if accuracy_60_40_csr_3(sub,:)>60
        accuracy_60_40_csr3_def(sub,:)=accuracy_60_40_csr_3(sub,:);
    end
   
end

val2=find((accuracy_60_40_csr3_def(1:23)~=0));
accuracy_60_40_csr3_def2=zeros(size(val2,1),3);
for sub=1:size(val1,2)
       accuracy_60_40_csr3_def2(sub,1:3)=accuracy_60_40_csr3_def(val2(1,sub),1:3);
end


for sub=1:size(accuracy_80_20_csr3_def2,1)
    for cl=1:3
    time_train_80_20_csr3_def3(sub,1,cl)=time_train_80_20_csr_3(val1(1,sub),1,cl);
    time_predict_80_20_csr3_def3(sub,1,cl)=time_predict_80_20_csr_3(val1(1,sub),1,cl);
    end
end

for sub=1:size(accuracy_60_40_csr3_def2,1)
    for cl=1:3
    time_train_60_40_csr3_def3(sub,1,cl)=time_train_60_40_csr_3(val2(1,sub),1,cl);
    time_predict_60_40_csr3_def3(sub,1,cl)=time_predict_60_40_csr_3(val2(1,sub),1,cl);
    end
end

for sub=1:size(accuracy_80_20_csr3_def2,1)
    for cl=1:3
    precision1_80_20_csr3_def3(sub,1,cl)=precision1_80_20_csr_3(val1(1,sub),1,cl);
    precision2_80_20_csr3_def3(sub,1,cl)=precision2_80_20_csr_3(val1(1,sub),1,cl);
    recall1_80_20_csr3_def3(sub,1,cl)=recall1_80_20_csr_3(val1(1,sub),1,cl);
    recall2_80_20_csr3_def3(sub,1,cl)=recall2_80_20_csr_3(val1(1,sub),1,cl);
    end
end
for sub=1:size(accuracy_60_40_csr3_def2,1)
    for cl=1:3
       precision1_60_40_csr3_def3(sub,1,cl)=precision1_60_40_csr_3(val2(1,sub),1,cl);
       precision2_60_40_csr3_def3(sub,1,cl)=precision2_60_40_csr_3(val2(1,sub),1,cl);
       recall1_60_40_csr3_def3(sub,1,cl)=recall1_60_40_csr_3(val2(1,sub),1,cl);
       recall2_60_40_csr3_def3(sub,1,cl)=recall2_60_40_csr_3(val2(1,sub),1,cl);
    end
end
%plot
for d=1:3
    figure;
    c=bar(squeeze(accuracy_80_20_csr3_def2(:,d)));
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
    c=bar(squeeze(accuracy_60_40_csr3_def2(:,d)));
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
mean_sub_80_20cs3=mean(accuracy_80_20_csr3_def2,2);
mean_class_80_20cs3=mean(accuracy_80_20_csr3_def2,1);
mean_sub_60_40cs3=mean(accuracy_60_40_csr3_def2,2);
mean_class_60_40cs3=mean(accuracy_60_40_csr3_def2,1);

time_train_80_20_subcs3=mean(time_train_80_20_csr3_def3,3);
time_train_80_20_classcs3=mean(time_train_80_20_csr3_def3,1);
time_predict_80_20_subcs3=mean(time_predict_80_20_csr3_def3,2);
time_predict_80_20_classcs3=mean(time_predict_80_20_subcs3,1);
time_train_60_40_subcs3=mean(time_train_60_40_csr3_def3,3);
time_train_60_40_classcs3=mean(time_train_60_40_csr3_def3,1);
time_predict_60_40_subcs3=mean(time_predict_60_40_csr3_def3,2);
time_predict_60_40_classcs3=mean(time_predict_60_40_subcs3,1);

g=squeeze(mean_sub_80_20cs3)';
figure;
boxplot(g);
xlabel('distribution of accuracy')
title('claassification 80-20%')
figure;
g=squeeze(mean_sub_60_40cs3)';
fig1=boxplot(g);
xlabel('distribution of accuracy')
title('claassification 60-40%')



figure;
subplot(2,3,1)
plot(squeeze(time_train_80_20_csr3_def3(:,1,:)),'-o')
xlabel('subjects')
ylabel('mean time for train[s]')
legend('SVM','NET','RF')
% subplot(2,4,4)
% plot(squeeze(time_matrix_test_80_20def(:,i,:)),'-o')
% xlabel('subjects')
% ylabel('mean time for test[s]')
% title(['session index' num2str(i)])
% legend('SVM','NET','RF')
subplot(2,3,2)
plot(squeeze(accuracy_80_20_csr3_def2(:,:)),'-o')
xlabel('subjects')
ylabel('accuracy (%)')
legend('SVM','NET','RF')
subplot(2,3,3)
plot(squeeze(precision1_80_20_csr3_def3(:,1,:)),'-o')
xlabel('subjects')
ylabel('precision for class 1')
legend('SVM','NET','RF')
subplot(2,3,4)
plot(squeeze(precision2_80_20_csr3_def3(:,1,:)),'-o')
xlabel('subjects')
ylabel('precision for class 2')
legend('SVM','NET','RF')
subplot(2,3,5)
plot(squeeze(recall1_80_20_csr3_def3(:,1,:)),'-o')
xlabel('subjects')
ylabel('recall for class 1')
legend('SVM','NET','RF')
subplot(2,3,6)
plot(squeeze(recall2_80_20_csr3_def3(:,1,:)),'-o')
xlabel('subjects')
ylabel('recall for class 2')
legend('SVM','NET','RF')

figure;
subplot(2,3,1)
plot(squeeze(time_train_60_40_csr3_def3(:,1,:)),'-o')
xlabel('subjects')
ylabel('mean time for train[s]')
legend('SVM','NET','RF')
% subplot(2,4,4)
% plot(squeeze(time_matrix_test_80_20def(:,i,:)),'-o')
% xlabel('subjects')
% ylabel('mean time for test[s]')
% title(['session index' num2str(i)])
% legend('SVM','NET','RF')
subplot(2,3,2)
plot(squeeze(accuracy_60_40_csr3_def2(:,:)),'-o')
xlabel('subjects')
ylabel('accuracy (%)')
legend('SVM','NET','RF')
subplot(2,3,3)
plot(squeeze(precision1_60_40_csr3_def3(:,1,:)),'-o')
xlabel('subjects')
ylabel('precision for class 1')
legend('SVM','NET','RF')
subplot(2,3,4)
plot(squeeze(precision2_60_40_csr3_def3(:,1,:)),'-o')
xlabel('subjects')
ylabel('precision for class 2')
legend('SVM','NET','RF')
subplot(2,3,5)
plot(squeeze(recall1_60_40_csr3_def3(:,1,:)),'-o')
xlabel('subjects')
ylabel('recall for class 1')
legend('SVM','NET','RF')
subplot(2,3,6)
plot(squeeze(recall2_60_40_csr3_def3(:,1,:)),'-o')
xlabel('subjects')
ylabel('recall for class 2')
legend('SVM','NET','RF')