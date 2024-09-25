% add gaussian noise with 1% of standard deviation
fatt_molt=zeros(25,5);
for sub=1:25
    for sess=1:5
        if sub<10
        x=sprintf('sub-00%d_ses-0%d_task_motorimagery_eeg.mat',sub,sess);
        else
        x=sprintf('sub-0%d_ses-0%d_task_motorimagery_eeg.mat',sub,sess);
        end
        cd  C:\Users\lucio\OneDrive\Desktop\dati+rumore1\
        f=load(x,'data');
        dati=f.data;
        data_mean=zeros(size(dati,1),32);
        data_mean2=zeros(size(dati,1),1);
        for tr=1:size(dati,1)
            for chn=1:32
        data_mean(tr,chn)=median(dati(tr,chn,:),3);
        data_mean2(tr,1)=median(data_mean(tr,:));
            end
        end
        fatt_molt(sub,sess)=median(data_mean2(:,1));
    end
end
for sub=1:25
    for sess=1:5
        data_r(sub,sess,1:32,1:1000)=fatt_molt(sub,sess)*0.01*randn(32,1000);
        data_r2(sub,sess,1:32,1:1000)=fatt_molt(sub,sess)*0.02*randn(32,1000);
        data_r3(sub,sess,1:32,1:1000)=fatt_molt(sub,sess)*0.05*randn(32,1000);
    end
end
data_noise=struct();
for sub=1:25
    for sess=1:5
        if sub<10
        x=sprintf('sub-00%d_ses-0%d_task_motorimagery_eeg.mat',sub,sess);
        else
        x=sprintf('sub-0%d_ses-0%d_task_motorimagery_eeg.mat',sub,sess);
        end
        cd  C:\Users\lucio\OneDrive\Desktop\dati+rumore1\
        f=load(x,'data');
        dati=f.data;
        data_rum=zeros(size(dati,1),32,1000);
        for tr=1:size(data_rum,1)
            for chn=1:32
                for camp=1:1000
        data_rum(tr,chn,camp)=dati(tr,chn,camp)+data_r(sub,sess,chn,camp);
                end
            end
        end
        s=sprintf('sub_%d_%d_noise',sub,sess);
        data_noise.(s)=data_rum;
    end
end
save('noise_data.mat','-struct','data_noise')
% add gaussian noise with 2% of standard deviation
data_noise2=struct();
for sub=1:25
    for sess=1:5
        if sub<10
        x=sprintf('sub-00%d_ses-0%d_task_motorimagery_eeg.mat',sub,sess);
        else
        x=sprintf('sub-0%d_ses-0%d_task_motorimagery_eeg.mat',sub,sess);
        end
        cd  C:\Users\lucio\OneDrive\Desktop\dati+rumore1\
        f=load(x,'data');
        dati=f.data;
        data_rum2=zeros(size(dati,1),32,1000);
        for tr=1:size(data_rum2,1)
            for chn=1:32
                for camp=1:1000
        data_rum2(tr,chn,camp)=dati(tr,chn,camp)+data_r2(sub,sess,chn,camp);
                end
            end
        end
        s=sprintf('sub_%d_%d_noise2',sub,sess);
        data_noise2.(s)=data_rum2;
    end
end
save('noise_data2.mat','-struct','data_noise2')
% add gaussian noise with 5% of standard deviation
data_noise3=struct();
for sub=1:25
    for sess=1:5
        if sub<10
        x=sprintf('sub-00%d_ses-0%d_task_motorimagery_eeg.mat',sub,sess);
        else
        x=sprintf('sub-0%d_ses-0%d_task_motorimagery_eeg.mat',sub,sess);
        end
        cd  C:\Users\lucio\OneDrive\Desktop\dati+rumore1\
        f=load(x,'data');
        dati=f.data;
        data_rum3=zeros(size(dati,1),32,1000);
        for tr=1:size(data_rum3,1)
            for chn=1:32
                for camp=1:1000
        data_rum3(tr,chn,camp)=dati(tr,chn,camp)+data_r3(sub,sess,chn,camp);
                end
            end
        end
        s=sprintf('sub_%d_%d_noise3',sub,sess);
        data_noise3.(s)=data_rum3;
    end
end
save('noise_data3.mat','-struct','data_noise3')