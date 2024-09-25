load('feature1_1r.mat')
load('feature1_1_1r.mat')
matrix_csr1_1=struct();
for su=1:25
    x=sprintf('sub_%02d_ses_1_tr1', su);
    y=sprintf('sub_%02d_ses_1_te1', su);
    z=sprintf('sub_%02d_ses_2_tr1', su);
    p=sprintf('sub_%02d_ses_2_te1', su);
    q=sprintf('sub_%02d_ses_3_tr1', su);
    r=sprintf('sub_%02d_ses_3_te1', su);
    s=sprintf('sub_%02d_ses_4_tr1', su);
    t=sprintf('sub_%02d_ses_4_te1', su);
    v=sprintf('sub_%02d_ses_5_tr1', su);
    g=sprintf('sub_%02d_ses_5_te1', su);
    x_x=eval(x);
    y_y=eval(y);
    z_z=eval(z);
    p_p=eval(p);
    q_q=eval(q);
    r_r=eval(r);
    s_s=eval(s);
    t_t=eval(t);
    v_v=eval(v);
    g_g=eval(g);
    matrix_train_80_20csr=zeros(size(x_x,1)+size(y_y,1)+size(z_z,1)+size(p_p,1)+size(q_q,1)+size(r_r,1)+size(s_s,1)+size(t_t,1),size(x_x,2));
    matrix_test_80_20csr=zeros(size(v_v,1)+size(g_g,1),size(v_v,2));
    matrix_train_80_20csr(1:size(x_x,1),:)=x_x(:,:);
    matrix_train_80_20csr(size(x_x,1)+1:size(x_x,1)+size(y_y,1),:)=y_y(:,:);
    matrix_train_80_20csr(size(x_x,1)+size(y_y,1)+1:size(x_x,1)+size(y_y,1)+size(z_z,1),:)=z_z(:,:);
    matrix_train_80_20csr(size(x_x,1)+size(y_y,1)+size(z_z,1)+1:size(x_x,1)+size(y_y,1)+size(z_z,1)+size(p_p,1),:)=p_p(:,:);
    matrix_train_80_20csr(size(x_x,1)+size(y_y,1)+size(z_z,1)+size(p_p,1)+1:size(x_x,1)+size(y_y,1)+size(z_z,1)+size(p_p,1)+size(q_q,1),:)=q_q(:,:);
    matrix_train_80_20csr(size(x_x,1)+size(y_y,1)+size(z_z,1)+size(p_p,1)+size(q_q,1)+1:size(x_x,1)+size(y_y,1)+size(z_z,1)+size(p_p,1)+size(q_q,1)+size(r_r,1),:)=r_r(:,:);
    matrix_train_80_20csr(size(x_x,1)+size(y_y,1)+size(z_z,1)+size(p_p,1)+size(q_q,1)+size(r_r,1)+1:size(x_x,1)+size(y_y,1)+size(z_z,1)+size(p_p,1)+size(q_q,1)+size(r_r,1)+size(s_s,1),:)=s_s(:,:);
    matrix_train_80_20csr(size(x_x,1)+size(y_y,1)+size(z_z,1)+size(p_p,1)+size(q_q,1)+size(r_r,1)+size(s_s,1)+1:size(x_x,1)+size(y_y,1)+size(z_z,1)+size(p_p,1)+size(q_q,1)+size(r_r,1)+size(s_s,1)+size(t_t,1),:)=t_t(:,:);
    matrix_test_80_20csr(1:size(v_v,1),:)=v_v(:,:);
    matrix_test_80_20csr(size(v_v,1)+1:size(v_v,1)+size(g_g,1),:)=v_v(:,:);
    name1=sprintf('matrix_train_80_20csr_%d_1',su);
    name2=sprintf('matrix_test_80_20csr_%d_1',su);
    matrix_csr1_1.(name1)=matrix_train_80_20csr;
    matrix_csr1_1.(name2)=matrix_test_80_20csr;
end

matrix_csr2_1=struct();
for su=1:25
    x=sprintf('sub_%02d_ses_1_tr2', su);
    y=sprintf('sub_%02d_ses_1_te2', su);
    z=sprintf('sub_%02d_ses_2_tr2', su);
    p=sprintf('sub_%02d_ses_2_te2', su);
    q=sprintf('sub_%02d_ses_3_tr2', su);
    r=sprintf('sub_%02d_ses_3_te2', su);
    s=sprintf('sub_%02d_ses_4_tr2', su);
    t=sprintf('sub_%02d_ses_4_te2', su);
    v=sprintf('sub_%02d_ses_5_tr2', su);
    g=sprintf('sub_%02d_ses_5_te2', su);
    x_x=eval(x);
    y_y=eval(y);
    z_z=eval(z);
    p_p=eval(p);
    q_q=eval(q);
    r_r=eval(r);
    s_s=eval(s);
    t_t=eval(t);
    v_v=eval(v);
    g_g=eval(g);
    matrix_train_80_20csr=zeros(size(x_x,1)+size(y_y,1)+size(z_z,1)+size(p_p,1)+size(q_q,1)+size(r_r,1)+size(s_s,1)+size(t_t,1),size(x_x,2));
    matrix_test_80_20csr=zeros(size(v_v,1)+size(g_g,1),size(v_v,2));
    matrix_train_80_20csr(1:size(x_x,1),:)=x_x(:,:);
    matrix_train_80_20csr(size(x_x,1)+1:size(x_x,1)+size(y_y,1),:)=y_y(:,:);
    matrix_train_80_20csr(size(x_x,1)+size(y_y,1)+1:size(x_x,1)+size(y_y,1)+size(z_z,1),:)=z_z(:,:);
    matrix_train_80_20csr(size(x_x,1)+size(y_y,1)+size(z_z,1)+1:size(x_x,1)+size(y_y,1)+size(z_z,1)+size(p_p,1),:)=p_p(:,:);
    matrix_train_80_20csr(size(x_x,1)+size(y_y,1)+size(z_z,1)+size(p_p,1)+1:size(x_x,1)+size(y_y,1)+size(z_z,1)+size(p_p,1)+size(q_q,1),:)=q_q(:,:);
    matrix_train_80_20csr(size(x_x,1)+size(y_y,1)+size(z_z,1)+size(p_p,1)+size(q_q,1)+1:size(x_x,1)+size(y_y,1)+size(z_z,1)+size(p_p,1)+size(q_q,1)+size(r_r,1),:)=r_r(:,:);
    matrix_train_80_20csr(size(x_x,1)+size(y_y,1)+size(z_z,1)+size(p_p,1)+size(q_q,1)+size(r_r,1)+1:size(x_x,1)+size(y_y,1)+size(z_z,1)+size(p_p,1)+size(q_q,1)+size(r_r,1)+size(s_s,1),:)=s_s(:,:);
    matrix_train_80_20csr(size(x_x,1)+size(y_y,1)+size(z_z,1)+size(p_p,1)+size(q_q,1)+size(r_r,1)+size(s_s,1)+1:size(x_x,1)+size(y_y,1)+size(z_z,1)+size(p_p,1)+size(q_q,1)+size(r_r,1)+size(s_s,1)+size(t_t,1),:)=t_t(:,:);
    matrix_test_80_20csr(1:size(v_v,1),:)=v_v(:,:);
    matrix_test_80_20csr(size(v_v,1)+1:size(v_v,1)+size(g_g,1),:)=v_v(:,:);
    name1=sprintf('matrix_train_80_20csr_%d_2',su);
    name2=sprintf('matrix_test_80_20csr_%d_2',su);
    matrix_csr2_1.(name1)=matrix_train_80_20csr;
    matrix_csr2_1.(name2)=matrix_test_80_20csr;
end

matrix_csr3_1=struct();
for su=1:25
    x=sprintf('sub_%02d_ses_1_tr3', su);
    y=sprintf('sub_%02d_ses_1_te3', su);
    z=sprintf('sub_%02d_ses_2_tr3', su);
    p=sprintf('sub_%02d_ses_2_te3', su);
    q=sprintf('sub_%02d_ses_3_tr3', su);
    r=sprintf('sub_%02d_ses_3_te3', su);
    s=sprintf('sub_%02d_ses_4_tr3', su);
    t=sprintf('sub_%02d_ses_4_te3', su);
    v=sprintf('sub_%02d_ses_5_tr3', su);
    g=sprintf('sub_%02d_ses_5_te3', su);
    x_x=eval(x);
    y_y=eval(y);
    z_z=eval(z);
    p_p=eval(p);
    q_q=eval(q);
    r_r=eval(r);
    s_s=eval(s);
    t_t=eval(t);
    v_v=eval(v);
    g_g=eval(g);
    matrix_train_80_20csr=zeros(size(x_x,1)+size(y_y,1)+size(z_z,1)+size(p_p,1)+size(q_q,1)+size(r_r,1)+size(s_s,1)+size(t_t,1),size(x_x,2));
    matrix_test_80_20csr=zeros(size(v_v,1)+size(g_g,1),size(v_v,2));
    matrix_train_80_20csr(1:size(x_x,1),:)=x_x(:,:);
    matrix_train_80_20csr(size(x_x,1)+1:size(x_x,1)+size(y_y,1),:)=y_y(:,:);
    matrix_train_80_20csr(size(x_x,1)+size(y_y,1)+1:size(x_x,1)+size(y_y,1)+size(z_z,1),:)=z_z(:,:);
    matrix_train_80_20csr(size(x_x,1)+size(y_y,1)+size(z_z,1)+1:size(x_x,1)+size(y_y,1)+size(z_z,1)+size(p_p,1),:)=p_p(:,:);
    matrix_train_80_20csr(size(x_x,1)+size(y_y,1)+size(z_z,1)+size(p_p,1)+1:size(x_x,1)+size(y_y,1)+size(z_z,1)+size(p_p,1)+size(q_q,1),:)=q_q(:,:);
    matrix_train_80_20csr(size(x_x,1)+size(y_y,1)+size(z_z,1)+size(p_p,1)+size(q_q,1)+1:size(x_x,1)+size(y_y,1)+size(z_z,1)+size(p_p,1)+size(q_q,1)+size(r_r,1),:)=r_r(:,:);
    matrix_train_80_20csr(size(x_x,1)+size(y_y,1)+size(z_z,1)+size(p_p,1)+size(q_q,1)+size(r_r,1)+1:size(x_x,1)+size(y_y,1)+size(z_z,1)+size(p_p,1)+size(q_q,1)+size(r_r,1)+size(s_s,1),:)=s_s(:,:);
    matrix_train_80_20csr(size(x_x,1)+size(y_y,1)+size(z_z,1)+size(p_p,1)+size(q_q,1)+size(r_r,1)+size(s_s,1)+1:size(x_x,1)+size(y_y,1)+size(z_z,1)+size(p_p,1)+size(q_q,1)+size(r_r,1)+size(s_s,1)+size(t_t,1),:)=t_t(:,:);
    matrix_test_80_20csr(1:size(v_v,1),:)=v_v(:,:);
    matrix_test_80_20csr(size(v_v,1)+1:size(v_v,1)+size(g_g,1),:)=v_v(:,:);
    name1=sprintf('matrix_train_80_20csr_%d_3',su);
    name2=sprintf('matrix_test_80_20csr_%d_3',su);
    matrix_csr3_1.(name1)=matrix_train_80_20csr;
    matrix_csr3_1.(name2)=matrix_test_80_20csr;
end

struct_labels_80_20csr=struct();
for sub=1:25
       if sub < 10
            fileName1=sprintf('sub-00%d_ses-01_task_motorimagery_eeg.mat', sub);
            fileName2=sprintf('sub-00%d_ses-02_task_motorimagery_eeg.mat', sub);
            fileName3=sprintf('sub-00%d_ses-03_task_motorimagery_eeg.mat', sub);
            fileName4=sprintf('sub-00%d_ses-04_task_motorimagery_eeg.mat', sub);
            fileName5=sprintf('sub-00%d_ses-05_task_motorimagery_eeg.mat', sub);
       else
            fileName1=sprintf('sub-0%d_ses-01_task_motorimagery_eeg.mat', sub);
            fileName2=sprintf('sub-0%d_ses-02_task_motorimagery_eeg.mat', sub);
            fileName3=sprintf('sub-0%d_ses-03_task_motorimagery_eeg.mat', sub);
            fileName4=sprintf('sub-0%d_ses-04_task_motorimagery_eeg.mat', sub);
            fileName5=sprintf('sub-0%d_ses-05_task_motorimagery_eeg.mat', sub);
       end
       x=load(fileName1,'labels');
       y=load(fileName2,'labels');
       z=load(fileName3,'labels');
       p=load(fileName4,'labels');
       q=load(fileName5,'labels');
       l1=x.labels;
       l2=y.labels;
       l3=z.labels;
       l4=p.labels;
       l5=q.labels;
       labels_train_80_20cs=zeros(size(l1,2) + size(l2,2) + size(l3,2) + size(l4,2), 1);
       labels_test_80_20cs=zeros(size(l5,2), 1);
       labels_train_80_20cs(1:size(l1,2),1)=l1(1,:);
       labels_train_80_20cs(size(l1,2)+1:size(l1,2)+size(l2,2),1)=l2(1,:);
       labels_train_80_20cs(size(l1,2)+size(l2,2)+1:size(l1,2)+size(l2,2)+size(l3,2),1)=l3(1,:);
       labels_train_80_20cs(size(l1,2)+size(l2,2)+size(l3,2)+1:size(l1,2)+size(l2,2)+size(l3,2)+size(l4,2),1)=l4(1,:);
       labels_test_80_20cs(1:size(l5,2),1)=l5(1,:);
       o=sprintf('labels_train_80_20cs_%d_r',sub);
       e=sprintf('labels_test_80_20cs_%d_r',sub);
       struct_labels_80_20csr.(o)=labels_train_80_20cs;
       struct_labels_80_20csr.(e)=labels_test_80_20cs;
end
save('labels_80_20csr.mat','-struct','struct_labels_80_20csr')

for su=1:25
    name1=sprintf('matrix_train_80_20csr_%d_1',su);
    name2=sprintf('matrix_test_80_20csr_%d_1',su);
    name3=sprintf('labels_train_80_20cs_%d_r',su);
    name4=sprintf('labels_test_80_20cs_%d_r',su);
    x_x=eval(name1);
    y_y=eval(name2);
    z_z=eval(name3);
    p_p=eval(name4);

    name5=sprintf('matrix_train_80_20csr_%d_2',su);
    name6=sprintf('matrix_test_80_20csr_%d_2',su);
    g_g=eval(name5);
    h_h=eval(name6);

    name7=sprintf('matrix_train_80_20csr_%d_3',su);
    name8=sprintf('matrix_test_80_20csr_%d_3',su);
    v_v=eval(name7);
    c_c=eval(name8);


    x_x=normalize(x_x,'zscore');
    y_y=normalize(y_y,'zscore');
    g_g=normalize(g_g,'zscore');
    h_h=normalize(h_h,'zscore');
    v_v=normalize(v_v,'zscore');
    c_c=normalize(c_c,'zscore');

    tic;
    mod=fitcsvm(x_x,z_z,'KernelFunction','polynomial','OptimizeHyperparameters','auto');
    time_train_80_20_csr_1(su,1,1)=toc;
    tic;
    predizioni=predict(mod,y_y);
    time_predict_80_20_csr_1(su,1,1)=toc;
    accuracy_80_20_csr_1(su,1,1)=sum(predizioni==p_p)/length(p_p)*100;
    TP=sum((predizioni==positive_class) & (p_p==positive_class));
FP=sum((predizioni==positive_class) & (p_p~=positive_class));
FN=sum((predizioni~=positive_class) & (p_p==positive_class));
precision1_80_20_csr_1(su,1,1)=(TP)/(TP+FP);
recall1_80_20_csr_1(su,1,1)=(TP)/(TP+FN);
positive_class = 2;
TP2=sum((predizioni==positive_class) & (p_p==positive_class));
FP2=sum((predizioni==positive_class) & (p_p~=positive_class));
FN2=sum((predizioni~=positive_class) & (p_p==positive_class));
precision2_80_20_csr_1(su,1,1)=(TP2)/(TP2+FP2);
recall2_80_20_csr_1(su,1,1)=(TP2)/(TP2+FN2);


    tic;
    mod=fitcsvm(g_g,z_z,'KernelFunction','polynomial','OptimizeHyperparameters','auto');
    time_train_80_20_csr_2(su,1,1)=toc;
    tic;
    predizioni=predict(mod,h_h);
    time_predict_80_20_csr_2(su,1,1)=toc;
    accuracy_80_20_csr_2(su,1,1)=sum(predizioni==p_p)/length(p_p)*100;
    TP=sum((predizioni==positive_class) & (p_p==positive_class));
FP=sum((predizioni==positive_class) & (p_p~=positive_class));
FN=sum((predizioni~=positive_class) & (p_p==positive_class));
precision1_80_20_csr_2(su,1,1)=(TP)/(TP+FP);
recall1_80_20_csr_2(su,1,1)=(TP)/(TP+FN);
positive_class = 2;
TP2=sum((predizioni==positive_class) & (p_p==positive_class));
FP2=sum((predizioni==positive_class) & (p_p~=positive_class));
FN2=sum((predizioni~=positive_class) & (p_p==positive_class));
precision2_80_20_csr_2(su,1,1)=(TP2)/(TP2+FP2);
recall2_80_20_csr_2(su,1,1)=(TP2)/(TP2+FN2);

    tic;
    mod=fitcsvm(v_v,z_z,'KernelFunction','polynomial','OptimizeHyperparameters','auto');
    time_train_80_20_csr_3(su,1,1)=toc;
    tic;
    predizioni=predict(mod,c_c);
    time_predict_80_20_csr_3(su,1,1)=toc;
    accuracy_80_20_csr_3(su,1,1)=sum(predizioni==p_p)/length(p_p)*100;
    TP=sum((predizioni==positive_class) & (p_p==positive_class));
FP=sum((predizioni==positive_class) & (p_p~=positive_class));
FN=sum((predizioni~=positive_class) & (p_p==positive_class));
precision1_80_20_csr_3(su,1,1)=(TP)/(TP+FP);
recall1_80_20_csr_3(su,1,1)=(TP)/(TP+FN);
positive_class = 2;
TP2=sum((predizioni==positive_class) & (p_p==positive_class));
FP2=sum((predizioni==positive_class) & (p_p~=positive_class));
FN2=sum((predizioni~=positive_class) & (p_p==positive_class));
precision2_80_20_csr_3(su,1,1)=(TP2)/(TP2+FP2);
recall2_80_20_csr_3(su,1,1)=(TP2)/(TP2+FN2);
end

for su=1:25
    name1=sprintf('matrix_train_80_20csr_%d_1',su);
    name2=sprintf('matrix_test_80_20csr_%d_1',su);
   
    x_x=eval(name1);
    y_y=eval(name2);
 

    name5=sprintf('matrix_train_80_20csr_%d_2',su);
    name6=sprintf('matrix_test_80_20csr_%d_2',su);
    g_g=eval(name5);
    h_h=eval(name6);

    name7=sprintf('matrix_train_80_20csr_%d_3',su);
    name8=sprintf('matrix_test_80_20csr_%d_3',su);
    v_v=eval(name7);
    c_c=eval(name8);

    name3=sprintf('labels_train_80_20cs_%d_r',su);
    name4=sprintf('labels_test_80_20cs_%d_r',su);
    z_z=eval(name3);
    p_p=eval(name4);


    x_x=normalize(x_x,'zscore');
    y_y=normalize(y_y,'zscore');
    g_g=normalize(g_g,'zscore');
    h_h=normalize(h_h,'zscore');
    v_v=normalize(v_v,'zscore');
    c_c=normalize(c_c,'zscore');

    tic;
    mod=fitcnet(x_x,z_z,'OptimizeHyperparameters','auto');
    time_train_80_20_csr_1(su,1,2)=toc;
    tic;
    predizioni=predict(mod,y_y);
    time_predict_80_20_csr_1(su,1,2)=toc;
    accuracy_80_20_csr_1(su,1,2)=sum(predizioni==p_p)/length(p_p)*100;
TP=sum((predizioni==positive_class) & (p_p==positive_class));
FP=sum((predizioni==positive_class) & (p_p~=positive_class));
FN=sum((predizioni~=positive_class) & (p_p==positive_class));
precision1_80_20_csr_1(su,1,2)=(TP)/(TP+FP);
recall1_80_20_csr_1(su,1,2)=(TP)/(TP+FN);
positive_class = 2;
TP2=sum((predizioni==positive_class) & (p_p==positive_class));
FP2=sum((predizioni==positive_class) & (p_p~=positive_class));
FN2=sum((predizioni~=positive_class) & (p_p==positive_class));
precision2_80_20_csr_1(su,1,2)=(TP2)/(TP2+FP2);
recall2_80_20_csr_1(su,1,2)=(TP2)/(TP2+FN2);

tic;
mod=fitcnet(g_g,z_z,'OptimizeHyperparameters','auto');
time_train_80_20_csr_2(su,1,2)=toc;
tic;
predizioni=predict(mod,h_h);
time_predict_80_20_csr_2(su,1,2)=toc;
accuracy_80_20_csr_2(su,1,2)=sum(predizioni==p_p)/length(p_p)*100;
TP=sum((predizioni==positive_class) & (p_p==positive_class));
FP=sum((predizioni==positive_class) & (p_p~=positive_class));
FN=sum((predizioni~=positive_class) & (p_p==positive_class));
precision1_80_20_csr_2(su,1,2)=(TP)/(TP+FP);
recall1_80_20_csr_2(su,1,2)=(TP)/(TP+FN);
positive_class = 2;
TP2=sum((predizioni==positive_class) & (p_p==positive_class));
FP2=sum((predizioni==positive_class) & (p_p~=positive_class));
FN2=sum((predizioni~=positive_class) & (p_p==positive_class));
precision2_80_20_csr_2(su,1,2)=(TP2)/(TP2+FP2);
recall2_80_20_csr_2(su,1,2)=(TP2)/(TP2+FN2);

tic;
mod=fitcnet(v_v,z_z,'OptimizeHyperparameters','auto');
time_train_80_20_csr_3(su,1,2)=toc;
tic;
predizioni=predict(mod,c_c);
time_predict_80_20_csr_3(su,1,2)=toc;
accuracy_80_20_csr_3(su,1,2)=sum(predizioni==p_p)/length(p_p)*100;
TP=sum((predizioni==positive_class) & (p_p==positive_class));
FP=sum((predizioni==positive_class) & (p_p~=positive_class));
FN=sum((predizioni~=positive_class) & (p_p==positive_class));
precision1_80_20_csr_3(su,1,2)=(TP)/(TP+FP);
recall1_80_20_csr_3(su,1,2)=(TP)/(TP+FN);
positive_class = 2;
TP2=sum((predizioni==positive_class) & (p_p==positive_class));
FP2=sum((predizioni==positive_class) & (p_p~=positive_class));
FN2=sum((predizioni~=positive_class) & (p_p==positive_class));
precision2_80_20_csr_3(su,1,2)=(TP2)/(TP2+FP2);
recall2_80_20_csr_3(su,1,2)=(TP2)/(TP2+FN2);
end

for su=1:25
    name1=sprintf('matrix_train_80_20csr_%d_1',su);
    name2=sprintf('matrix_test_80_20csr_%d_1',su);
    x_x=eval(name1);
    y_y=eval(name2);
 

    name5=sprintf('matrix_train_80_20csr_%d_2',su);
    name6=sprintf('matrix_test_80_20csr_%d_2',su);
    g_g=eval(name5);
    h_h=eval(name6);

    name7=sprintf('matrix_train_80_20csr_%d_3',su);
    name8=sprintf('matrix_test_80_20csr_%d_3',su);
    v_v=eval(name7);
    c_c=eval(name8);

    name3=sprintf('labels_train_80_20cs_%d_r',su);
    name4=sprintf('labels_test_80_20cs_%d_r',su);
    z_z=eval(name3);
    p_p=eval(name4);


    tic;
    mod=fitcensemble(x_x,z_z,'OptimizeHyperparameters','auto');
    time_train_80_20_csr_1(su,1,3)=toc;
    tic;
    predizioni=predict(mod,y_y);
    time_predict_80_20_csr_1(su,1,3)=toc;
    accuracy_80_20_csr_1(su,1,3)=sum(predizioni==p_p)/length(p_p)*100;
    TP=sum((predizioni==positive_class) & (p_p==positive_class));
FP=sum((predizioni==positive_class) & (p_p~=positive_class));
FN=sum((predizioni~=positive_class) & (p_p==positive_class));
precision1_80_20_csr_1(su,1,3)=(TP)/(TP+FP);
recall1_80_20_csr_1(su,1,3)=(TP)/(TP+FN);
positive_class = 2;
TP2=sum((predizioni==positive_class) & (p_p==positive_class));
FP2=sum((predizioni==positive_class) & (p_p~=positive_class));
FN2=sum((predizioni~=positive_class) & (p_p==positive_class));
precision2_80_20_csr_1(su,1,3)=(TP2)/(TP2+FP2);
recall2_80_20_csr_1(su,1,3)=(TP2)/(TP2+FN2);

    tic;
    mod=fitcensemble(g_g,z_z,'OptimizeHyperparameters','auto');
    time_train_80_20_csr_2(su,1,3)=toc;
    tic;
    predizioni=predict(mod,h_h);
    time_predict_80_20_csr_2(su,1,3)=toc;
    accuracy_80_20_csr_2(su,1,3)=sum(predizioni==p_p)/length(p_p)*100;
    positive_class = 1;
TP=sum((predizioni==positive_class) & (p_p==positive_class));
FP=sum((predizioni==positive_class) & (p_p~=positive_class));
FN=sum((predizioni~=positive_class) & (p_p==positive_class));
precision1_80_20_csr_2(su,1,3)=(TP)/(TP+FP);
recall1_80_20_csr_2(su,1,3)=(TP)/(TP+FN);
positive_class = 2;
TP2=sum((predizioni==positive_class) & (p_p==positive_class));
FP2=sum((predizioni==positive_class) & (p_p~=positive_class));
FN2=sum((predizioni~=positive_class) & (p_p==positive_class));
precision2_80_20_csr_2(su,1,3)=(TP2)/(TP2+FP2);
recall2_80_20_csr_2(su,1,3)=(TP2)/(TP2+FN2);




    tic;
    mod=fitcensemble(v_v,z_z,'OptimizeHyperparameters','auto');
    time_train_80_20_csr_3(su,1,3)=toc;
    tic;
    predizioni=predict(mod,c_c);
    time_predict_80_20_csr_3(su,1,3)=toc;
    accuracy_80_20_csr_3(su,1,3)=sum(predizioni==p_p)/length(p_p)*100;
TP=sum((predizioni==positive_class) & (p_p==positive_class));
FP=sum((predizioni==positive_class) & (p_p~=positive_class));
FN=sum((predizioni~=positive_class) & (p_p==positive_class));
precision1_80_20_csr_3(su,1,3)=(TP)/(TP+FP);
recall1_80_20_csr_3(su,1,3)=(TP)/(TP+FN);
positive_class = 2;
TP2=sum((predizioni==positive_class) & (p_p==positive_class));
FP2=sum((predizioni==positive_class) & (p_p~=positive_class));
FN2=sum((predizioni~=positive_class) & (p_p==positive_class));
precision2_80_20_csr_3(su,1,3)=(TP2)/(TP2+FP2);
recall2_80_20_csr_3(su,1,3)=(TP2)/(TP2+FN2);
end
save('metricsr_80_20.mat','accuracy_80_20_csr_1','accuracy_80_20_csr_2','accuracy_80_20_csr_3')
save('metrics2csr_80_20.mat','precision1_80_20_csr_1','precision1_80_20_csr_2','precision1_80_20_csr_3','precision2_80_20_csr_1','precision2_80_20_csr_2','precision2_80_20_csr_3','recall1_80_20_csr_1','recall1_80_20_csr_2','recall1_80_20_csr_3','recall2_80_20_csr_1','recall2_80_20_csr_2','recall2_80_20_csr_3')
save('time_cs_80_20r.mat','time_train_80_20_csr_1','time_predict_80_20_csr_1','time_train_80_20_csr_2','time_predict_80_20_csr_2','time_train_80_20_csr_3','time_predict_80_20_csr_3')