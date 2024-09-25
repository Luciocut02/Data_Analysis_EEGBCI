load('cs3.mat')
struct_csfeatures3_1=struct();
for sub=1:25
    w=sprintf('matrix_train_40_60cs_%d',sub);
    m=sprintf('matrix_test_40_60cs_%d',sub);
    n=sprintf('labels_train_40_60cs_%d',sub);
    u=sprintf('labels_test_40_60cs_%d',sub);
    ww=eval(w);
    mm=eval(m);
    nn=eval(n);
    uu=eval(u);
    [feature_sogg_40cs2_1,feature_sogg_60cs2_1,time_extraction_featurecs3_1]=feature_freq_time_domaincs3_1(ww,mm,nn);
    pp=sprintf('feature_sogg_40cs_%d',sub);
    oo=sprintf('feature_sogg_60cs_%d',sub);
    dd=sprintf('time_extraction_featurecs3_%d',sub);
    struct_csfeatures3_1.(pp)=feature_sogg_60cs2_1;
    struct_csfeatures3_1.(oo)=feature_sogg_40cs2_1;
    struct_csfeatures3_1.(dd)=time_extraction_featurecs3_1;
end
save('features3_1cs.mat','-struct','struct_csfeatures3_1')
load('features3_1cs.mat')

for sub=1:25
        g=sprintf('feature_sogg_40cs_%d',sub);
        y=sprintf('feature_sogg_60cs_%d',sub);
        k=sprintf('labels_train_40_60cs_%d',sub);
        j=sprintf('labels_test_40_60cs_%d',sub);

        g_g=eval(g);
        y_y=eval(y);
        k_k=eval(k);
        j_j=eval(j);

        tic;
        mod=fitcsvm(g_g,k_k,'Standardize',true,'KernelFunction','polynomial','OptimizeHyperparameters','auto');
        time_train_40_60cs3_1(sub,1)=toc;
        tic;
        predizione=predict(mod,y_y);
        time_predict_40_60cs3_1(sub,1)=toc;
        accuracy_40_60cs3_1(sub,1)=sum(predizione==j_j)/length(j_j)*100;
        positive_class = 1;

TP=sum((predizione==positive_class) & (j_j==positive_class));
FP=sum((predizione==positive_class) & (j_j~=positive_class));
FN=sum((predizione~=positive_class) & (j_j==positive_class));
    precision1_40_60cs3_1(sub,ses,1)=(TP)/(TP+FP);
    recall1_40_60cs3_1(sub,ses,1)=(TP)/(TP+FN);

    positive_class = 2;

TP2=sum((predizione==positive_class) & (j_j==positive_class));
FP2=sum((predizione==positive_class) & (j_j~=positive_class));
FN2=sum((predizione~=positive_class) & (j_j==positive_class));
    precision2_40_60cs3_1(sub,ses,1)=(TP2)/(TP2+FP2);
    recall2_40_60cs3_1(sub,ses,1)=(TP2)/(TP2+FN2);
end

for sub=1:25
        g=sprintf('feature_sogg_40cs_%d',sub);
        y=sprintf('feature_sogg_60cs_%d',sub);
        k=sprintf('labels_train_40_60cs_%d',sub);
        j=sprintf('labels_test_40_60cs_%d',sub);

        g_g=eval(g);
        y_y=eval(y);
        k_k=eval(k);
        j_j=eval(j);

        g_g=normalize(g_g,'zscore');
        y_y=normalize(y_y,'zscore');

        tic;
        mod=fitcnet(g_g,k_k,'OptimizeHyperparameters','auto');
        time_train_40_60cs3_1(sub,2)=toc;
        tic;
        predizione=predict(mod,y_y);
        time_predict_40_60cs3_1(sub,2)=toc;
        accuracy_40_60cs3_1(sub,2)=sum(predizione==j_j)/length(j_j)*100;
        positive_class = 1;

TP=sum((predizione==positive_class) & (j_j==positive_class));
FP=sum((predizione==positive_class) & (j_j~=positive_class));
FN=sum((predizione~=positive_class) & (j_j==positive_class));
    precision1_40_60cs3_1(sub,ses,1)=(TP)/(TP+FP);
    recall1_40_60cs3_1(sub,ses,1)=(TP)/(TP+FN);

    positive_class = 2;

TP2=sum((predizione==positive_class) & (j_j==positive_class));
FP2=sum((predizione==positive_class) & (j_j~=positive_class));
FN2=sum((predizione~=positive_class) & (j_j==positive_class));
    precision2_40_60cs3_1(sub,ses,1)=(TP2)/(TP2+FP2);
    recall2_40_60cs3_1(sub,ses,1)=(TP2)/(TP2+FN2);
end
for sub=1:25
        g=sprintf('feature_sogg_40cs_%d',sub);
        y=sprintf('feature_sogg_60cs_%d',sub);
        k=sprintf('labels_train_40_60cs_%d',sub);
        j=sprintf('labels_test_40_60cs_%d',sub);

        g_g=eval(g);
        y_y=eval(y);
        k_k=eval(k);
        j_j=eval(j);

        g_g=normalize(g_g,'zscore');
        y_y=normalize(y_y,'zscore');

        tic;
        mod=fitcensemble(g_g,k_k,'OptimizeHyperparameters','auto');
        time_train_40_60cs3_1(sub,3)=toc;
        tic;
        predizione=predict(mod,y_y);
        time_predict_40_60cs3_1(sub,3)=toc;
        accuracy_40_60cs3_1(sub,3)=sum(predizione==j_j)/length(j_j)*100;
        positive_class = 1;

TP=sum((predizione==positive_class) & (j_j==positive_class));
FP=sum((predizione==positive_class) & (j_j~=positive_class));
FN=sum((predizione~=positive_class) & (j_j==positive_class));
    precision1_40_60cs3_1(sub,ses,1)=(TP)/(TP+FP);
    recall1_40_60cs3_1(sub,ses,1)=(TP)/(TP+FN);

    positive_class = 2;

TP2=sum((predizione==positive_class) & (j_j==positive_class));
FP2=sum((predizione==positive_class) & (j_j~=positive_class));
FN2=sum((predizione~=positive_class) & (j_j==positive_class));
    precision2_40_60cs3_1(sub,ses,1)=(TP2)/(TP2+FP2);
    recall2_40_60cs3_1(sub,ses,1)=(TP2)/(TP2+FN2);
end