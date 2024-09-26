load('features1_1cs.mat')
load('feature1_1rdef.mat')
load('cs1.mat')
for sub=1:25
    oo=sprintf('feature_sogg_80cs_%d',sub);
    matrix_name=sprintf('sub_%02d_ses_05_tr1',sub);
    matrix_name2=sprintf('sub_%02d_ses_05_te1',sub);
    matrix_name3=sprintf('sub_%02d_ses_05_tr2',sub);
    matrix_name4=sprintf('sub_%02d_ses_05_te2',sub);
    matrix_name5=sprintf('sub_%02d_ses_05_tr3',sub);
    matrix_name6=sprintf('sub_%02d_ses_05_te3',sub);

    l1=sprintf('labels_train_80_20cs_%d',sub);
    l2=sprintf('labels_test_80_20cs_%d',sub);
    l1_1=eval(l1);
    l2_2=eval(l2);
   
    a_a=eval(oo);
    x_x=eval(matrix_name);
    y_y=eval(matrix_name2);
    z_z=eval(matrix_name3);
    p_p=eval(matrix_name4);
    q_q=eval(matrix_name5);
    r_r=eval(matrix_name6);
    
    features_sogg_20cs1=zeros(size(x_x,1)+size(y_y,1),size(y_y,2));
    features_sogg_20cs2=zeros(size(z_z,1)+size(p_p,1),size(y_y,2));
    features_sogg_20cs3=zeros(size(q_q,1)+size(r_r,1),size(y_y,2));

     features_sogg_20cs1(1:size(x_x,1),:)=x_x(1:size(x_x,1),:);
     features_sogg_20cs1(size(x_x,1)+1:size(x_x,1)+size(y_y,1),:)=y_y(1:size(y_y,1),:);
     features_sogg_20cs2(1:size(z_z,1),:)=z_z(1:size(z_z,1),:);
     features_sogg_20cs2(size(z_z,1)+1:size(z_z,1)+size(p_p,1),:)=p_p(1:size(p_p,1),:);
     features_sogg_20cs3(1:size(q_q,1),:)=q_q(1:size(q_q,1),:);
     features_sogg_20cs3(size(q_q,1)+1:size(q_q,1)+size(r_r,1),:)=r_r(1:size(r_r,1),:);
   

    data_train=normalize(a_a,'zscore');
    data_test1=normalize(features_sogg_20cs1,'zscore');
    data_test2=normalize(features_sogg_20cs2,'zscore');
    data_test3=normalize(features_sogg_20cs3,'zscore');
    
    tic;
    modello=fitcsvm(data_train,l1_1,'KernelFunction','polynomial','OptimizeHyperparameters','auto');
    close all
    time_train1(sub,1)=toc;
    predizioni1=predict(modello,data_test1);
    predizioni2=predict(modello,data_test2);
    predizioni3=predict(modello,data_test3);
    tic;
    modello2=fitcnet(data_train,l1_1,'OptimizeHyperparameters','auto');
    close all
    time_train1(sub,2)=toc;
    predizioni4=predict(modello2,data_test1);
    predizioni5=predict(modello2,data_test2);
    predizioni6=predict(modello2,data_test3);
    tic;
    modello3=fitcensemble(a_a,l1_1,'OptimizeHyperparameters','auto');
    close all
    time_train(sub,3)=toc;
    predizioni7=predict(modello3,features_sogg_20cs1);
    predizioni8=predict(modello3,features_sogg_20cs2);
    predizioni9=predict(modello3,features_sogg_20cs3);

    accuracy_80_20cs1(sub,1)=sum(predizioni1==l2_2)/length(l2_2)*100;
    accuracy_80_20cs2(sub,1)=sum(predizioni2==l2_2)/length(l2_2)*100;
    accuracy_80_20cs3(sub,1)=sum(predizioni3==l2_2)/length(l2_2)*100;

     positive_class = 1;
TP=sum((predizioni1==positive_class) & (l_l2==positive_class));
FP=sum((predizioni1==positive_class) & (l_l2~=positive_class));
FN=sum((predizioni1~=positive_class) & (l_l2==positive_class));
    precision1_80_20cs_1(sub,1)=(TP)/(TP+FP);
    recall1_80_20cs_1(sub,1)=(TP)/(TP+FN);
   positive_class = 2;
TP2=sum((predizioni1==positive_class) & (l_l2==positive_class));
FP2=sum((predizioni1==positive_class) & (l_l2~=positive_class));
FN2=sum((predizioni1~=positive_class) & (l_l2==positive_class));
    precision2_80_20cs_1(sub,1)=(TP2)/(TP2+FP2);
    recall2_80_20cs_1(sub,1)=(TP2)/(TP2+FN2);

    positive_class = 1;
TP=sum((predizioni2==positive_class) & (l_l2==positive_class));
FP=sum((predizioni2==positive_class) & (l_l2~=positive_class));
FN=sum((predizioni2~=positive_class) & (l_l2==positive_class));
    precision1_80_20cs_2(sub,1)=(TP)/(TP+FP);
    recall1_80_20cs_2(sub,1)=(TP)/(TP+FN);
   positive_class = 2;
TP2=sum((predizioni2==positive_class) & (l_l2==positive_class));
FP2=sum((predizioni2==positive_class) & (l_l2~=positive_class));
FN2=sum((predizioni2~=positive_class) & (l_l2==positive_class));
    precision2_80_20cs_2(sub,1)=(TP2)/(TP2+FP2);
    recall2_80_20cs_2(sub,1)=(TP2)/(TP2+FN2);

    positive_class = 1;
TP=sum((predizioni3==positive_class) & (l_l2==positive_class));
FP=sum((predizioni3==positive_class) & (l_l2~=positive_class));
FN=sum((predizioni3~=positive_class) & (l_l2==positive_class));
    precision1_80_20cs_3(sub,1)=(TP)/(TP+FP);
    recall1_80_20cs_3(sub,1)=(TP)/(TP+FN);
   positive_class = 2;
TP2=sum((predizioni3==positive_class) & (l_l2==positive_class));
FP2=sum((predizioni3==positive_class) & (l_l2~=positive_class));
FN2=sum((predizioni3~=positive_class) & (l_l2==positive_class));
    precision2_80_20cs_3(sub,1)=(TP2)/(TP2+FP2);
    recall2_80_20cs_3(sub,1)=(TP2)/(TP2+FN2);


    accuracy_80_20cs1(sub,2)=sum(predizioni4==l2_2)/length(l2_2)*100;
    accuracy_80_20cs2(sub,2)=sum(predizioni5==l2_2)/length(l2_2)*100;
    accuracy_80_20cs3(sub,2)=sum(predizioni6==l2_2)/length(l2_2)*100;
         positive_class = 1;
TP=sum((predizioni4==positive_class) & (l_l2==positive_class));
FP=sum((predizioni4==positive_class) & (l_l2~=positive_class));
FN=sum((predizioni4~=positive_class) & (l_l2==positive_class));
    precision1_80_20cs_1(sub,2)=(TP)/(TP+FP);
    recall1_80_20cs_1(sub,2)=(TP)/(TP+FN);
   positive_class = 2;
TP2=sum((predizioni4==positive_class) & (l_l2==positive_class));
FP2=sum((predizioni4==positive_class) & (l_l2~=positive_class));
FN2=sum((predizioni4~=positive_class) & (l_l2==positive_class));
    precision2_80_20cs_1(sub,2)=(TP2)/(TP2+FP2);
    recall2_80_20cs_1(sub,2)=(TP2)/(TP2+FN2);

    positive_class = 1;
TP=sum((predizioni5==positive_class) & (l_l2==positive_class));
FP=sum((predizioni5==positive_class) & (l_l2~=positive_class));
FN=sum((predizioni5~=positive_class) & (l_l2==positive_class));
    precision1_80_20cs_2(sub,2)=(TP)/(TP+FP);
    recall1_80_20cs_2(sub,2)=(TP)/(TP+FN);
   positive_class = 2;
TP2=sum((predizioni5==positive_class) & (l_l2==positive_class));
FP2=sum((predizioni5==positive_class) & (l_l2~=positive_class));
FN2=sum((predizioni5~=positive_class) & (l_l2==positive_class));
    precision2_80_20cs_2(sub,2)=(TP2)/(TP2+FP2);
    recall2_80_20cs_2(sub,2)=(TP2)/(TP2+FN2);

    positive_class = 1;
TP=sum((predizioni6==positive_class) & (l_l2==positive_class));
FP=sum((predizioni6==positive_class) & (l_l2~=positive_class));
FN=sum((predizioni6~=positive_class) & (l_l2==positive_class));
    precision1_80_20cs_3(sub,2)=(TP)/(TP+FP);
    recall1_80_20cs_3(sub,2)=(TP)/(TP+FN);
   positive_class = 2;
TP2=sum((predizioni6==positive_class) & (l_l2==positive_class));
FP2=sum((predizioni6==positive_class) & (l_l2~=positive_class));
FN2=sum((predizioni6~=positive_class) & (l_l2==positive_class));
    precision2_80_20cs_3(sub,2)=(TP2)/(TP2+FP2);
    recall2_80_20cs_3(sub,2)=(TP2)/(TP2+FN2);

    accuracy_80_20cs1(sub,3)=sum(predizioni7==l2_2)/length(l2_2)*100;
    accuracy_80_20cs2(sub,3)=sum(predizioni8==l2_2)/length(l2_2)*100;
    accuracy_80_20cs3(sub,3)=sum(predizioni9==l2_2)/length(l2_2)*100;
             positive_class = 1;
TP=sum((predizioni7==positive_class) & (l_l2==positive_class));
FP=sum((predizioni7==positive_class) & (l_l2~=positive_class));
FN=sum((predizioni7~=positive_class) & (l_l2==positive_class));
    precision1_80_20cs_1(sub,3)=(TP)/(TP+FP);
    recall1_80_20cs_1(sub,3)=(TP)/(TP+FN);
   positive_class = 2;
TP2=sum((predizioni7==positive_class) & (l_l2==positive_class));
FP2=sum((predizioni7==positive_class) & (l_l2~=positive_class));
FN2=sum((predizioni7~=positive_class) & (l_l2==positive_class));
    precision2_80_20cs_1(sub,3)=(TP2)/(TP2+FP2);
    recall2_80_20cs_1(sub,3)=(TP2)/(TP2+FN2);

    positive_class = 1;
TP=sum((predizioni8==positive_class) & (l_l2==positive_class));
FP=sum((predizioni8==positive_class) & (l_l2~=positive_class));
FN=sum((predizioni8~=positive_class) & (l_l2==positive_class));
    precision1_80_20cs_2(sub,3)=(TP)/(TP+FP);
    recall1_80_20cs_2(sub,3)=(TP)/(TP+FN);
   positive_class = 2;
TP2=sum((predizioni8==positive_class) & (l_l2==positive_class));
FP2=sum((predizioni8==positive_class) & (l_l2~=positive_class));
FN2=sum((predizioni8~=positive_class) & (l_l2==positive_class));
    precision2_80_20cs_2(sub,3)=(TP2)/(TP2+FP2);
    recall2_80_20cs_2(sub,3)=(TP2)/(TP2+FN2);

    positive_class = 1;
TP=sum((predizioni9==positive_class) & (l_l2==positive_class));
FP=sum((predizioni9==positive_class) & (l_l2~=positive_class));
FN=sum((predizioni9~=positive_class) & (l_l2==positive_class));
    precision1_80_20cs_3(sub,3)=(TP)/(TP+FP);
    recall1_80_20cs_3(sub,3)=(TP)/(TP+FN);
   positive_class = 2;
TP2=sum((predizioni9==positive_class) & (l_l2==positive_class));
FP2=sum((predizioni9==positive_class) & (l_l2~=positive_class));
FN2=sum((predizioni9~=positive_class) & (l_l2==positive_class));
    precision2_80_20cs_3(sub,3)=(TP2)/(TP2+FP2);
    recall2_80_20cs_3(sub,3)=(TP2)/(TP2+FN2);
end
save('metrics_cs_80_20_r_r.mat','accuracy_80_20cs1','accuracy_80_20cs2','accuracy_80_20cs3')
save('metrics2_cs_80_20_r_r.mat','precision1_80_20cs_1','precision1_80_20cs_2','precision1_80_20cs_3','precision2_80_20cs_1','precision2_80_20cs_2','precision2_80_20cs_3','recall1_80_20cs_1','recall1_80_20cs_2','recall1_80_20cs_3','recall2_80_20cs_1','recall2_80_20cs_2','recall2_80_20cs_3')
save('time_cs','time_train1')

