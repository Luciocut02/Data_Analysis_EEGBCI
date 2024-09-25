load('cs1.mat')
struct_csfeatures1_1=struct();
for sub=1:25
    w=sprintf('matrix_train_80_20cs_%d',sub);
    m=sprintf('matrix_test_80_20cs_%d',sub);
    n=sprintf('labels_train_80_20cs_%d',sub);
    u=sprintf('labels_test_80_20cs_%d',sub);
    ww=eval(w);
    mm=eval(m);
    nn=eval(n);
    uu=eval(u);
    [feature_sogg_80cs1_1,feature_sogg_20cs1_1,time_extraction_featurecs1_1]=feature_freq_time_domaincs1_1(ww,mm,nn);
    pp=sprintf('feature_sogg_80cs_%d',sub);
    oo=sprintf('feature_sogg_20cs_%d',sub);
    dd=sprintf('time_extraction_featurecs1_%d',sub);
    struct_csfeatures1_1.(pp)=feature_sogg_80cs1_1;
    struct_csfeatures1_1.(oo)=feature_sogg_20cs1_1;
    struct_csfeatures1_1.(dd)=time_extraction_featurecs1_1;
end
save('features1_1cs.mat','-struct','struct_csfeatures1_1')
load('features1_1cs.mat')
for sub=1:25
        g=sprintf('feature_sogg_80cs_%d',sub);
        y=sprintf('feature_sogg_20cs_%d',sub);
        k=sprintf('labels_train_80_20cs_%d',sub);
        j=sprintf('labels_test_80_20cs_%d',sub);

        g_g=eval(g);
        y_y=eval(y);
        k_k=eval(k);
        j_j=eval(j);

        g_g=normalize(g_g,'zscore');
        y_y=normalize(y_y,'zscore');

        tic;
        mod=fitcsvm(g_g,k_k,'KernelFunction','polynomial','OptimizeHyperparameters','auto');
        close all
        time_train_80_20cs1_1(sub,1)=toc;
        tic;
        predizioni=predict(mod,y_y);
        time_predict_80_20cs1_1(sub,1)=toc;
        accuracy_80_20cs1_1(sub,1)=sum(predizioni==j_j)/length(j_j)*100;
        positive_class = 1;

TP=sum((predizioni==positive_class) & (j_j==positive_class));
FP=sum((predizioni==positive_class) & (j_j~=positive_class));
FN=sum((predizioni~=positive_class) & (j_j==positive_class));
    precision1_80_20cs1_1(sub,1)=(TP)/(TP+FP);
    recall1_80_20cs1_1(sub,1)=(TP)/(TP+FN);

    positive_class = 2;

TP2=sum((predizioni==positive_class) & (j_j==positive_class));
FP2=sum((predizioni==positive_class) & (j_j~=positive_class));
FN2=sum((predizioni~=positive_class) & (j_j==positive_class));
    precision2_80_20cs1_1(sub,1)=(TP2)/(TP2+FP2);
    recall2_80_20cs1_1(sub,1)=(TP2)/(TP2+FN2);
end

for sub=1:25
        g=sprintf('feature_sogg_80cs_%d',sub);
        y=sprintf('feature_sogg_20cs_%d',sub);
        k=sprintf('labels_train_80_20cs_%d',sub);
        j=sprintf('labels_test_80_20cs_%d',sub);

        g_g=eval(g);
        y_y=eval(y);
        k_k=eval(k);
        j_j=eval(j);

        g_g=normalize(g_g,'zscore');
        y_y=normalize(y_y,'zscore');

        tic;
        mod=fitcnet(g_g,k_k,'OptimizeHyperparameters','auto');
        close all
        time_train_80_20cs1_1(sub,2)=toc;
        tic;
        predizioni=predict(mod,y_y);
        time_predict_80_20cs1_1(sub,2)=toc;
        accuracy_80_20cs1_1(sub,2)=sum(predizioni==j_j)/length(j_j)*100;
        positive_class = 1;

TP=sum((predizioni==positive_class) & (j_j==positive_class));
FP=sum((predizioni==positive_class) & (j_j~=positive_class));
FN=sum((predizioni~=positive_class) & (j_j==positive_class));
    precision1_80_20cs1_1(sub,2)=(TP)/(TP+FP);
    recall1_80_20cs1_1(sub,2)=(TP)/(TP+FN);

    positive_class = 2;

TP2=sum((predizioni==positive_class) & (j_j==positive_class));
FP2=sum((predizioni==positive_class) & (j_j~=positive_class));
FN2=sum((predizioni~=positive_class) & (j_j==positive_class));
    precision2_80_20cs1_1(sub,2)=(TP2)/(TP2+FP2);
    recall2_80_20cs1_1(sub,2)=(TP2)/(TP2+FN2);
end

for sub=1:25
        g=sprintf('feature_sogg_80cs_%d',sub);
        y=sprintf('feature_sogg_20cs_%d',sub);
        k=sprintf('labels_train_80_20cs_%d',sub);
        j=sprintf('labels_test_80_20cs_%d',sub);

        g_g=eval(g);
        y_y=eval(y);
        k_k=eval(k);
        j_j=eval(j);

      
        tic;
        mod=fitcensemble(g_g,k_k,'OptimizeHyperparameters','auto');
        close all
        time_train_80_20cs1_1(sub,3)=toc;
        tic;
        predizioni=predict(mod,y_y);
        time_predict_80_20cs1_1(sub,3)=toc;
        accuracy_80_20cs1_1(sub,3)=sum(predizioni==j_j)/length(j_j)*100;
        positive_class = 1;

TP=sum((predizioni==positive_class) & (j_j==positive_class));
FP=sum((predizioni==positive_class) & (j_j~=positive_class));
FN=sum((predizioni~=positive_class) & (j_j==positive_class));
    precision1_80_20cs1_1(sub,3)=(TP)/(TP+FP);
    recall1_80_20cs1_1(sub,3)=(TP)/(TP+FN);

    positive_class = 2;

TP2=sum((predizioni==positive_class) & (j_j==positive_class));
FP2=sum((predizioni==positive_class) & (j_j~=positive_class));
FN2=sum((predizioni~=positive_class) & (j_j==positive_class));
    precision2_80_20cs1_1(sub,3)=(TP2)/(TP2+FP2);
    recall2_80_20cs1_1(sub,3)=(TP2)/(TP2+FN2);
end
save('results1cs.mat','accuracy_80_20cs1_1','precision1_80_20cs1_1','precision2_80_20cs1_1','recall1_80_20cs1_1','recall2_80_20cs1_1')
save('timecs1.mat','time_train_80_20cs1_1','time_predict_80_20cs1_1')