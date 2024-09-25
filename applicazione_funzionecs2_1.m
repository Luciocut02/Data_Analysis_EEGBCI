load('cs2.mat')
struct_csfeatures2_2=struct();
for sub=1:25
    w=sprintf('matrix_train_60_40cs_%d',sub);
    m=sprintf('matrix_test_60_40cs_%d',sub);
    n=sprintf('labels_train_60_40cs_%d',sub);
    u=sprintf('labels_test_60_40cs_%d',sub);
    ww=eval(w);
    mm=eval(m);
    nn=eval(n);
    uu=eval(u);
    [feature_sogg_60cs2_1,feature_sogg_40cs2_1,time_extraction_featurecs2_1]=feature_freq_time_domaincs2_1(ww,mm,nn);
    pp=sprintf('feature_sogg_60cs_%d',sub);
    oo=sprintf('feature_sogg_40cs_%d',sub);
    dd=sprintf('time_extraction_featurecs2_%d',sub);
    struct_csfeatures2_2.(pp)=feature_sogg_60cs2_1;
    struct_csfeatures2_2.(oo)=feature_sogg_40cs2_1;
    struct_csfeatures2_2.(dd)=time_extraction_featurecs2_1;
end
save('features2_2cs.mat','-struct','struct_csfeatures2_2')
load('features2_2cs.mat')

for sub=1:25
        g=sprintf('feature_sogg_60cs_%d',sub);
        y=sprintf('feature_sogg_40cs_%d',sub);
        k=sprintf('labels_train_60_40cs_%d',sub);
        j=sprintf('labels_test_60_40cs_%d',sub);

        g_g=eval(g);
        y_y=eval(y);
        k_k=eval(k);
        j_j=eval(j);

         g_g=normalize(g_g,'zscore');
        y_y=normalize(y_y,'zscore');

        tic;
        mod=fitcsvm(g_g,k_k,'KernelFunction','polynomial','OptimizeHyperparameters','auto');
        close all
        time_train_60_40cs2_1(sub,1)=toc;
        tic;
        predizione=predict(mod,y_y);
        time_predict_60_40cs2_1(sub,1)=toc;
        accuracy_60_40cs2_1(sub,1)=sum(predizione==j_j)/length(j_j)*100;
        positive_class = 1;

TP=sum((predizione==positive_class) & (j_j==positive_class));
FP=sum((predizione==positive_class) & (j_j~=positive_class));
FN=sum((predizione~=positive_class) & (j_j==positive_class));
    precision1_60_40cs2_1(sub,1)=(TP)/(TP+FP);
    recall1_60_40cs2_1(sub,1)=(TP)/(TP+FN);

    positive_class = 2;

TP2=sum((predizione==positive_class) & (j_j==positive_class));
FP2=sum((predizione==positive_class) & (j_j~=positive_class));
FN2=sum((predizione~=positive_class) & (j_j==positive_class));
    precision2_60_40cs2_1(sub,1)=(TP2)/(TP2+FP2);
    recall2_60_40cs2_1(sub,1)=(TP2)/(TP2+FN2);
end

for sub=1:25
        g=sprintf('feature_sogg_60cs_%d',sub);
        y=sprintf('feature_sogg_40cs_%d',sub);
        k=sprintf('labels_train_60_40cs_%d',sub);
        j=sprintf('labels_test_60_40cs_%d',sub);

        g_g=eval(g);
        y_y=eval(y);
        k_k=eval(k);
        j_j=eval(j);

        g_g=normalize(g_g,'zscore');
        y_y=normalize(y_y,'zscore');

        tic;
        mod=fitcnet(g_g,k_k,'OptimizeHyperparameters','auto');
        close all
        time_train_60_40cs2_1(sub,2)=toc;
        tic;
        predizione=predict(mod,y_y);
        time_predict_60_40cs2_1(sub,2)=toc;
        accuracy_60_40cs2_1(sub,2)=sum(predizione==j_j)/length(j_j)*100;
        positive_class = 1;

TP=sum((predizione==positive_class) & (j_j==positive_class));
FP=sum((predizione==positive_class) & (j_j~=positive_class));
FN=sum((predizione~=positive_class) & (j_j==positive_class));
    precision1_60_40cs2_1(sub,2)=(TP)/(TP+FP);
    recall1_60_40cs2_1(sub,2)=(TP)/(TP+FN);

    positive_class = 2;

TP2=sum((predizione==positive_class) & (j_j==positive_class));
FP2=sum((predizione==positive_class) & (j_j~=positive_class));
FN2=sum((predizione~=positive_class) & (j_j==positive_class));
    precision2_60_40cs2_1(sub,2)=(TP2)/(TP2+FP2);
    recall2_60_40cs2_1(sub,2)=(TP2)/(TP2+FN2);
end
for sub=1:25
        g=sprintf('feature_sogg_60cs_%d',sub);
        y=sprintf('feature_sogg_40cs_%d',sub);
        k=sprintf('labels_train_60_40cs_%d',sub);
        j=sprintf('labels_test_60_40cs_%d',sub);

        g_g=eval(g);
        y_y=eval(y);
        k_k=eval(k);
        j_j=eval(j);


        tic;
        mod=fitcensemble(g_g,k_k,'OptimizeHyperparameters','auto');
        close all
        time_train_60_40cs2_1(sub,3)=toc;
        tic;
        predizione=predict(mod,y_y);
        time_predict_60_40cs2_1(sub,3)=toc;
        accuracy_60_40cs2_1(sub,3)=sum(predizione==j_j)/length(j_j)*100;
        positive_class = 1;

TP=sum((predizione==positive_class) & (j_j==positive_class));
FP=sum((predizione==positive_class) & (j_j~=positive_class));
FN=sum((predizione~=positive_class) & (j_j==positive_class));
    precision1_60_40cs2_1(sub,3)=(TP)/(TP+FP);
    recall1_60_40cs2_1(sub,3)=(TP)/(TP+FN);

    positive_class = 2;

TP2=sum((predizione==positive_class) & (j_j==positive_class));
FP2=sum((predizione==positive_class) & (j_j~=positive_class));
FN2=sum((predizione~=positive_class) & (j_j==positive_class));
    precision2_60_40cs2_1(sub,3)=(TP2)/(TP2+FP2);
    recall2_60_40cs2_1(sub,3)=(TP2)/(TP2+FN2);
end
save('results2cs.mat','accuracy_60_40cs2_1','precision1_60_40cs2_1','precision2_60_40cs2_1','recall1_60_40cs2_1','recall2_60_40cs2_1')
save('timecs2.mat','time_train_60_40cs2_1','time_predict_60_40cs2_1')