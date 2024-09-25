load('features4_1cs.mat')
load('feature4_1r.mat')
load('cs4.mat')
for sub=1:25
    oo=sprintf('feature_sogg_20cs_%d',sub);

    matrix_name3=sprintf('sub_%02d_ses_02_tr1',sub);
    matrix_name2_3=sprintf('sub_%02d_ses_02_te1',sub);
    matrix_name3_3=sprintf('sub_%02d_ses_02_tr2',sub);
    matrix_name4_3=sprintf('sub_%02d_ses_02_te2',sub);
    matrix_name5_3=sprintf('sub_%02d_ses_02_tr3',sub);
    matrix_name6_3=sprintf('sub_%02d_ses_02_te3',sub);

    matrix_name2=sprintf('sub_%02d_ses_03_tr1',sub);
    matrix_name2_2=sprintf('sub_%02d_ses_03_te1',sub);
    matrix_name3_2=sprintf('sub_%02d_ses_03_tr2',sub);
    matrix_name4_2=sprintf('sub_%02d_ses_03_te2',sub);
    matrix_name5_2=sprintf('sub_%02d_ses_03_tr3',sub);
    matrix_name6_2=sprintf('sub_%02d_ses_03_te3',sub);

    matrix_name1=sprintf('sub_%02d_ses_04_tr1',sub);
    matrix_name2_1=sprintf('sub_%02d_ses_04_te1',sub);
    matrix_name3_1=sprintf('sub_%02d_ses_04_tr2',sub);
    matrix_name4_1=sprintf('sub_%02d_ses_04_te2',sub);
    matrix_name5_1=sprintf('sub_%02d_ses_04_tr3',sub);
    matrix_name6_1=sprintf('sub_%02d_ses_04_te3',sub);

    matrix_name=sprintf('sub_%02d_ses_05_tr1',sub);
    matrix_name2=sprintf('sub_%02d_ses_05_te1',sub);
    matrix_name3=sprintf('sub_%02d_ses_05_tr2',sub);
    matrix_name4=sprintf('sub_%02d_ses_05_te2',sub);
    matrix_name5=sprintf('sub_%02d_ses_05_tr3',sub);
    matrix_name6=sprintf('sub_%02d_ses_05_te3',sub);

    l1=sprintf('labels_train_20_80cs_%d',sub);
    l2=sprintf('labels_test_20_80cs_%d',sub);
    l1_1=eval(l1);
    l2_2=eval(l2);
   
    a_a=eval(oo);

    x_x=eval(matrix_name);
    y_y=eval(matrix_name2);
    z_z=eval(matrix_name3);
    p_p=eval(matrix_name4);
    q_q=eval(matrix_name5);
    r_r=eval(matrix_name6);

    x_x1=eval(matrix_name1);
    y_y1=eval(matrix_name2_1);
    z_z1=eval(matrix_name3_1);
    p_p1=eval(matrix_name4_1);
    q_q1=eval(matrix_name5_1);
    r_r1=eval(matrix_name6_1);

    x_x2=eval(matrix_name2);
    y_y2=eval(matrix_name2_2);
    z_z2=eval(matrix_name3_2);
    p_p2=eval(matrix_name4_2);
    q_q2=eval(matrix_name5_2);
    r_r2=eval(matrix_name6_2);

    x_x3=eval(matrix_name3);
    y_y3=eval(matrix_name2_3);
    z_z3=eval(matrix_name3_3);
    p_p3=eval(matrix_name4_3);
    q_q3=eval(matrix_name5_3);
    r_r3=eval(matrix_name6_3);
    

    features_sogg_80cs1(1:size(x_x3,1),:)=x_x3(1:size(x_x3,1),:);
    features_sogg_80cs1(size(x_x3,1)+1:size(y_y3,1)+size(x_x3,1),:)=y_y3(1:size(y_y3,1),:);
    features_sogg_80cs1(size(y_y3,1)+size(x_x3,1)+1:size(y_y3,1)+size(x_x3,1)+size(x_x2,1),:)=x_x2(1:size(x_x2,1),:);
    features_sogg_80cs1(size(y_y3,1)+size(x_x3,1)+size(x_x2,1)+1:size(y_y3,1)+size(x_x3,1)+size(x_x2,1)+size(y_y2,1),:)=y_y2(1:size(y_y2,1),:);
    features_sogg_80cs1(size(y_y3,1)+size(x_x3,1)+size(x_x2,1)+size(y_y2,1)+1:size(y_y3,1)+size(x_x3,1)+size(x_x2,1)+size(y_y2,1)+size(x_x1,1),:)=x_x1(1:size(x_x1,1),:);
    features_sogg_80cs1(size(y_y3,1)+size(x_x3,1)+size(x_x2,1)+size(y_y2,1)+size(x_x1,1)+1:size(y_y3,1)+size(x_x3,1)+size(x_x2,1)+size(y_y2,1)+size(x_x1,1)+size(y_y1,1),:)=y_y1(1:size(y_y1,1),:);
    features_sogg_80cs1(size(y_y3,1)+size(x_x3,1)+size(x_x2,1)+size(y_y2,1)+size(x_x1,1)+size(y_y1,1)+1:size(y_y3,1)+size(x_x3,1)+size(x_x2,1)+size(y_y2,1)+size(x_x1,1)+size(y_y1,1)+size(x_x,1),:)=x_x(1:size(x_x,1),:);
    features_sogg_80cs1(size(y_y3,1)+size(x_x3,1)+size(x_x2,1)+size(y_y2,1)+size(x_x1,1)+size(y_y1,1)+size(x_x,1)+1:size(y_y,1)+size(y_y3,1)+size(x_x3,1)+size(x_x2,1)+size(y_y2,1)+size(x_x1,1)+size(y_y1,1)+size(x_x,1),:)=y_y(1:size(y_y,1),:);

   features_sogg_80cs2(1:size(z_z3,1),:)=z_z3(1:size(z_z3,1),:);
    features_sogg_80cs2(size(z_z3,1)+1:size(p_p3,1)+size(z_z3,1),:)=p_p3(1:size(p_p3,1),:);
    features_sogg_80cs2(size(p_p3,1)+size(z_z3,1)+1:size(p_p3,1)+size(z_z3,1)+size(z_z2,1),:)=z_z2(1:size(z_z2,1),:);
    features_sogg_80cs2(size(p_p3,1)+size(z_z3,1)+size(z_z2,1)+1:size(p_p3,1)+size(z_z3,1)+size(z_z2,1)+size(p_p2,1),:)=p_p2(1:size(p_p2,1),:);
    features_sogg_80cs2(size(p_p3,1)+size(z_z3,1)+size(z_z2,1)+size(p_p2,1)+1:size(p_p3,1)+size(z_z3,1)+size(z_z2,1)+size(p_p2,1)+size(z_z1,1),:)=z_z1(1:size(z_z1,1),:);
    features_sogg_80cs2(size(p_p3,1)+size(z_z3,1)+size(z_z2,1)+size(p_p2,1)+size(z_z1,1)+1:size(p_p3,1)+size(z_z3,1)+size(z_z2,1)+size(p_p2,1)+size(z_z1,1)+size(p_p1,1),:)=p_p1(1:size(p_p1,1),:);
    features_sogg_80cs2(size(p_p3,1)+size(z_z3,1)+size(z_z2,1)+size(p_p2,1)+size(z_z1,1)+size(p_p1,1)+1:size(p_p3,1)+size(z_z3,1)+size(z_z2,1)+size(p_p2,1)+size(z_z1,1)+size(p_p1,1)+size(z_z,1),:)=z_z(1:size(z_z,1),:);
    features_sogg_80cs2(size(p_p3,1)+size(z_z3,1)+size(z_z2,1)+size(p_p2,1)+size(z_z1,1)+size(p_p1,1)+size(z_z,1)+1:size(p_p,1)+size(p_p3,1)+size(z_z3,1)+size(z_z2,1)+size(p_p2,1)+size(z_z1,1)+size(p_p1,1)+size(z_z,1),:)=p_p(1:size(p_p,1),:);

   
   features_sogg_80cs3(1:size(q_q3,1),:)=q_q3(1:size(q_q3,1),:);
    features_sogg_80cs3(size(q_q3,1)+1:size(r_r3,1)+size(q_q3,1),:)=r_r3(1:size(r_r3,1),:);
    features_sogg_80cs3(size(r_r3,1)+size(q_q3,1)+1:size(r_r3,1)+size(q_q3,1)+size(q_q2,1),:)=q_q2(1:size(q_q2,1),:);
    features_sogg_80cs3(size(r_r3,1)+size(q_q3,1)+size(q_q2,1)+1:size(p_p3,1)+size(q_q3,1)+size(q_q2,1)+size(r_r2,1),:)=r_r2(1:size(r_r2,1),:);
    features_sogg_80cs3(size(r_r3,1)+size(q_q3,1)+size(q_q2,1)+size(r_r2,1)+1:size(r_r3,1)+size(q_q3,1)+size(q_q2,1)+size(r_r2,1)+size(q_q1,1),:)=q_q1(1:size(q_q1,1),:);
    features_sogg_80cs3(size(r_r3,1)+size(q_q3,1)+size(q_q2,1)+size(r_r2,1)+size(q_q1,1)+1:size(r_r3,1)+size(q_q3,1)+size(q_q2,1)+size(r_r2,1)+size(q_q1,1)+size(r_r1,1),:)=r_r1(1:size(r_r1,1),:);
    features_sogg_80cs3(size(r_r3,1)+size(q_q3,1)+size(q_q2,1)+size(r_r2,1)+size(q_q1,1)+size(r_r1,1)+1:size(r_r3,1)+size(q_q3,1)+size(q_q2,1)+size(r_r2,1)+size(q_q1,1)+size(r_r1,1)+size(q_q,1),:)=q_q(1:size(q_q,1),:);
    features_sogg_80cs3(size(r_r3,1)+size(q_q3,1)+size(q_q2,1)+size(r_r2,1)+size(q_q1,1)+size(r_r1,1)+size(q_q,1)+1:size(r_r,1)+size(r_r3,1)+size(q_q3,1)+size(q_q2,1)+size(r_r2,1)+size(q_q1,1)+size(r_r1,1)+size(q_q,1),:)=r_r(1:size(r_r,1),:);

    data_train=normalize(a_a,'zscore');
    data_test1=normalize(features_sogg80cs1,'zscore');
    data_test2=normalize(features_sogg80cs2,'zscore');
    data_test3=normalize(features_sogg80cs3,'zscore');
    tic;
    modello=fitcsvm(data_train,l1_1,'KernelFunction','polynomial','OptimizeHyperparameters','auto');
    time_train4(sub,1)=toc;
    predizioni1=predict(modello,data_test1);
    predizioni2=predict(modello,data_test2);
    predizioni3=predict(modello,data_test3);
    tic;
    modello2=fitcnet(data_train,l1_1,'OptimizeHyperparameters','auto');
    time_train4(sub,2)=toc;
    predizioni4=predict(modello2,data_test1);
    predizioni5=predict(modello2,data_test2);
    predizioni6=predict(modello2,data_test3);
    tic;
    modello3=fitcensemble(a_a,l1_1,'OptimizeHyperparameters','auto');
    time_train4(sub,3)=toc;
    predizioni7=predict(modello3,feature_sogg80cs1);
    predizioni8=predict(modello3,feature_sogg80cs2);
    predizioni9=predict(modello3,feature_sogg80cs3);

    accuracy_20_80cs1(sub,1)=sum(predizioni1==l2_2)/length(l2_2)*100;
    accuracy_20_80cs2(sub,1)=sum(predizioni2==l2_2)/length(l2_2)*100;
    accuracy_20_80cs3(sub,1)=sum(predizioni3==l2_2)/length(l2_2)*100;

     positive_class = 1;
TP=sum((predizioni1==positive_class) & (l_l2==positive_class));
FP=sum((predizioni1==positive_class) & (l_l2~=positive_class));
FN=sum((predizioni1~=positive_class) & (l_l2==positive_class));
    precision1_20_80cs_1(sub,1)=(TP)/(TP+FP);
    recall1_20_80cs_1(sub,1)=(TP)/(TP+FN);
   positive_class = 2;
TP2=sum((predizioni1==positive_class) & (l_l2==positive_class));
FP2=sum((predizioni1==positive_class) & (l_l2~=positive_class));
FN2=sum((predizioni1~=positive_class) & (l_l2==positive_class));
    precision2_20_80cs_1(sub,1)=(TP2)/(TP2+FP2);
    recall2_20_80cs_1(sub,1)=(TP2)/(TP2+FN2);

    positive_class = 1;
TP=sum((predizioni2==positive_class) & (l_l2==positive_class));
FP=sum((predizioni2==positive_class) & (l_l2~=positive_class));
FN=sum((predizioni2~=positive_class) & (l_l2==positive_class));
    precision1_20_80cs_2(sub,1)=(TP)/(TP+FP);
    recall1_20_80cs_2(sub,1)=(TP)/(TP+FN);
   positive_class = 2;
TP2=sum((predizioni2==positive_class) & (l_l2==positive_class));
FP2=sum((predizioni2==positive_class) & (l_l2~=positive_class));
FN2=sum((predizioni2~=positive_class) & (l_l2==positive_class));
    precision2_20_80cs_2(sub,1)=(TP2)/(TP2+FP2);
    recall2_20_80cs_2(sub,1)=(TP2)/(TP2+FN2);

    positive_class = 1;
TP=sum((predizioni3==positive_class) & (l_l2==positive_class));
FP=sum((predizioni3==positive_class) & (l_l2~=positive_class));
FN=sum((predizioni3~=positive_class) & (l_l2==positive_class));
    precision1_20_80cs_3(sub,1)=(TP)/(TP+FP);
    recall1_20_80cs_3(sub,1)=(TP)/(TP+FN);
   positive_class = 2;
TP2=sum((predizioni3==positive_class) & (l_l2==positive_class));
FP2=sum((predizioni3==positive_class) & (l_l2~=positive_class));
FN2=sum((predizioni3~=positive_class) & (l_l2==positive_class));
    precision2_20_80cs_3(sub,1)=(TP2)/(TP2+FP2);
    recall2_20_80cs_3(sub,1)=(TP2)/(TP2+FN2);


    accuracy_20_80cs1(sub,2)=sum(predizioni4==l2_2)/length(l2_2)*100;
    accuracy_20_80cs2(sub,2)=sum(predizioni5==l2_2)/length(l2_2)*100;
    accuracy_20_80cs3(sub,2)=sum(predizioni6==l2_2)/length(l2_2)*100;
         positive_class = 1;
TP=sum((predizioni4==positive_class) & (l_l2==positive_class));
FP=sum((predizioni4==positive_class) & (l_l2~=positive_class));
FN=sum((predizioni4~=positive_class) & (l_l2==positive_class));
    precision1_20_80cs_1(sub,2)=(TP)/(TP+FP);
    recall1_20_80cs_1(sub,2)=(TP)/(TP+FN);
   positive_class = 2;
TP2=sum((predizioni4==positive_class) & (l_l2==positive_class));
FP2=sum((predizioni4==positive_class) & (l_l2~=positive_class));
FN2=sum((predizioni4~=positive_class) & (l_l2==positive_class));
    precision2_20_80cs_1(sub,2)=(TP2)/(TP2+FP2);
    recall2_20_80cs_1(sub,2)=(TP2)/(TP2+FN2);

    positive_class = 1;
TP=sum((predizioni5==positive_class) & (l_l2==positive_class));
FP=sum((predizioni5==positive_class) & (l_l2~=positive_class));
FN=sum((predizioni5~=positive_class) & (l_l2==positive_class));
    precision1_20_80cs_2(sub,2)=(TP)/(TP+FP);
    recall1_20_80cs_2(sub,2)=(TP)/(TP+FN);
   positive_class = 2;
TP2=sum((predizioni5==positive_class) & (l_l2==positive_class));
FP2=sum((predizioni5==positive_class) & (l_l2~=positive_class));
FN2=sum((predizioni5~=positive_class) & (l_l2==positive_class));
    precision2_20_80cs_2(sub,2)=(TP2)/(TP2+FP2);
    recall2_20_80cs_2(sub,2)=(TP2)/(TP2+FN2);

    positive_class = 1;
TP=sum((predizioni6==positive_class) & (l_l2==positive_class));
FP=sum((predizioni6==positive_class) & (l_l2~=positive_class));
FN=sum((predizioni6~=positive_class) & (l_l2==positive_class));
    precision1_20_80cs_3(sub,2)=(TP)/(TP+FP);
    recall1_20_80cs_3(sub,2)=(TP)/(TP+FN);
   positive_class = 2;
TP2=sum((predizioni6==positive_class) & (l_l2==positive_class));
FP2=sum((predizioni6==positive_class) & (l_l2~=positive_class));
FN2=sum((predizioni6~=positive_class) & (l_l2==positive_class));
    precision2_20_80cs_3(sub,2)=(TP2)/(TP2+FP2);
    recall2_20_80cs_3(sub,2)=(TP2)/(TP2+FN2);

    accuracy_20_80cs1(sub,3)=sum(predizioni7==l2_2)/length(l2_2)*100;
    accuracy_20_80cs2(sub,3)=sum(predizioni8==l2_2)/length(l2_2)*100;
    accuracy_20_80cs3(sub,3)=sum(predizioni9==l2_2)/length(l2_2)*100;
             positive_class = 1;
TP=sum((predizioni7==positive_class) & (l_l2==positive_class));
FP=sum((predizioni7==positive_class) & (l_l2~=positive_class));
FN=sum((predizioni7~=positive_class) & (l_l2==positive_class));
    precision1_20_80cs_1(sub,3)=(TP)/(TP+FP);
    recall1_20_80cs_1(sub,3)=(TP)/(TP+FN);
   positive_class = 2;
TP2=sum((predizioni7==positive_class) & (l_l2==positive_class));
FP2=sum((predizioni7==positive_class) & (l_l2~=positive_class));
FN2=sum((predizioni7~=positive_class) & (l_l2==positive_class));
    precision2_20_80cs_1(sub,3)=(TP2)/(TP2+FP2);
    recall2_20_80cs_1(sub,3)=(TP2)/(TP2+FN2);

    positive_class = 1;
TP=sum((predizioni8==positive_class) & (l_l2==positive_class));
FP=sum((predizioni8==positive_class) & (l_l2~=positive_class));
FN=sum((predizioni8~=positive_class) & (l_l2==positive_class));
    precision1_20_80cs_2(sub,3)=(TP)/(TP+FP);
    recall1_20_80cs_2(sub,3)=(TP)/(TP+FN);
   positive_class = 2;
TP2=sum((predizioni8==positive_class) & (l_l2==positive_class));
FP2=sum((predizioni8==positive_class) & (l_l2~=positive_class));
FN2=sum((predizioni8~=positive_class) & (l_l2==positive_class));
    precision2_20_80cs_2(sub,3)=(TP2)/(TP2+FP2);
    recall2_20_80cs_2(sub,3)=(TP2)/(TP2+FN2);

    positive_class = 1;
TP=sum((predizioni9==positive_class) & (l_l2==positive_class));
FP=sum((predizioni9==positive_class) & (l_l2~=positive_class));
FN=sum((predizioni9~=positive_class) & (l_l2==positive_class));
    precision1_20_80cs_3(sub,3)=(TP)/(TP+FP);
    recall1_20_80cs_3(sub,3)=(TP)/(TP+FN);
   positive_class = 2;
TP2=sum((predizioni9==positive_class) & (l_l2==positive_class));
FP2=sum((predizioni9==positive_class) & (l_l2~=positive_class));
FN2=sum((predizioni9~=positive_class) & (l_l2==positive_class));
    precision2_20_80cs_3(sub,3)=(TP2)/(TP2+FP2);
    recall2_20_80cs_3(sub,3)=(TP2)/(TP2+FN2);
end
save('metrics_cs_20_80_r_r.mat','accuracy_20_80cs1','accuracy_20_80cs2','accuracy_20_80cs3')
save('metrics2_cs_20_80_r_r.mat','precision1_20_80cs_1','precision1_20_80cs_2','precision1_20_80cs_3','precision2_20_80cs_1','precision2_20_80cs_2','precision2_20_80cs_3','recall1_20_80cs_1','recall1_20_80cs_2','recall1_20_80cs_3','recall2_20_80cs_1','recall2_20_80cs_2','recall2_20_80cs_3')
save('time_cs','time_train4')