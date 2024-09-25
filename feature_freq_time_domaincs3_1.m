function [feature_sogg_40cs3_1,feature_sogg_60cs3_1,time_extraction_featurecs3_1] = feature_freq_time_domaincs3_1(matrix_train_40_60cs,matrix_test_40_60cs,labels_train_40_60cs)
tic;   
fc=250; %Hz
vv=size(matrix_train_40_60cs,1);
rr=size(matrix_test_40_60cs,1);

%extraction statistic feature
data1=matrix_train_40_60cs(1:vv,:,:).^2;
data1_sum=sum(data1,3);
data1_m=mean(data1_sum,3);
data1_std(1:vv,1:32)=std(matrix_train_40_60cs(1:vv,:,:),0,3);
data_var(1:vv,1:32)=var(matrix_train_40_60cs(1:vv,:,:),0,3);

data2=matrix_test_40_60cs(1:rr,:,:).^2;
data2_sum=sum(data2,3);
data2_m=mean(data2_sum,3);
data2_std(1:rr,1:32)=std(matrix_test_40_60cs(1:rr,:,:),0,3);
data_2var(1:rr,1:32)=var(matrix_test_40_60cs(1:rr,:,:),0,3);

%feature extraction-->time domain
%Peak latency
for s=1:vv
    for n=1:32
         [f,r]=max(matrix_train_40_60cs(s,n,:),[],3);
          picco(s,n)=r;
        
    end
end
picco=picco./fc;


for s=1:rr
    for n=1:32
         [f,r]=max(matrix_test_40_60cs(s,n,:),[],3);
          picco2(s,n)=r;
    end
end
picco2=picco2./fc;
% picco3=mean(picco2,3);
% picco4=mean(picco3,2);
% picco3(1:round(0.2*rr),:)=picco2(vv+1:rr,:);
%RMS
for o=1:vv
    for j=1:32
        rms_data(o,j)=rms(matrix_train_40_60cs(o,j,:),"all");
    end
end
% rms_data=mean(rms_data,2);
for o=1:rr
    for j=1:32
        rms_data2(o,j)=rms(matrix_test_40_60cs(o,j,:),"all");
    end
end
% rms_data2=mean(rms_data2,2);
% rms_data3(1:round(0.2*rr),:)=rms_data2(vv+1:rr,:);
%peak-to-peak amplitude
for o=1:vv
    for j=1:32
        peaktopeak(o,j)=max(squeeze(matrix_train_40_60cs(o,j,:)))-min(squeeze(matrix_train_40_60cs(o,j,:)));
    end
end
% peaktopeak=mean(peaktopeak,2);

for o=1:rr
    for j=1:32
        peaktopeak2(o,j)=max(squeeze(matrix_test_40_60cs(o,j,:)))-min(squeeze(matrix_test_40_60cs(o,j,:)));
    end
end
% peaktopeak2=mean(peaktopeak2,2);
% peaktopeak3(1:round(0.2*rr),:)=peaktopeak2(vv+1:rr,:);

%feature extraction-->Frequency domain
% extraction PSD
bands = [8 13; 14 30];  % Definizione delle bande di frequenza (mu e beta)
num_bands = size(bands, 1);  % Numero di bande  % Frequenza di campionamento % Numero di trial
PSD = zeros(vv, 32 * num_bands);  % Matrice per salvare le potenze su bande

% Ciclo su ogni trial
for s = 1:vv
    % Ciclo su ogni canale
    for n = 1:32
        % Calcolo della PSD usando il metodo di Welch
        [pxx, f] = pwelch(squeeze(matrix_train_40_60cs(s, n, :)), [], [], [], fc);
        
        % Ciclo su ogni banda di frequenza
        for b = 1:num_bands
            % Estrazione degli indici della frequenza per la banda corrente
            band_idx = find(f >= bands(b, 1) & f <= bands(b, 2));
            
            % Calcolo della potenza nella banda corrente
            power_band=bandpower(pxx, fc, bands(b, :));
            
            % Salvo la potenza nella banda b per il canale n
            PSD(s, (n-1)*num_bands + b)=power_band;
        end
    end
end

% len_fft=length(data);
% [pxx2, f] = pwelch(squeeze(data(1, 1, 1:1000)), [], [], [], fc);
% psd_size2= length(pxx2);
% PSD3 = zeros(round(0.2*rr), 32, psd_size2);
 PSD2=zeros(rr,32*num_bands);
for s=1:rr
    for n=1:32
        [pxx2, f2]=pwelch(squeeze(matrix_test_40_60cs(s,n,:)),[],[],[],fc);
        for b = 1:num_bands
            % Estrazione degli indici della frequenza per la banda corrente
            band_idx2= find(f2 >= bands(b, 1) & f2 <= bands(b, 2));
            % Calcolo della potenza nella banda corrente
            power_band2=bandpower(pxx2, fc, bands(b, :));
            % Salvo la potenza nella banda b per il canale n
            PSD2(s,(n-1)*num_bands + b)=power_band2;
        end
    end
end
% PSD4=mean(PSD3,3);
% PSD5=mean(PSD4,2);




%STFT
for trial=1:vv
    for chn=1:32
        signal_STFT(trial,chn,:,:)=abs(stft(squeeze(matrix_train_40_60cs(trial,chn,:)),fc));
    end
end
 signal_STFT=mean(signal_STFT,4);
 signal_STFT=mean(signal_STFT,3);
 % signal_STFT=mean(signal_STFT,2);

for trial=1:rr
    for chn=1:32
        signal_STFT2(trial,chn,:,:)=abs(stft(squeeze(matrix_test_40_60cs(trial,chn,:)),fc));
    end
end
 signal_STFT2=mean(signal_STFT2,4);
 signal_STFT2=mean(signal_STFT2,3);
 % signal_STFT3(1:round(0.2*rr),:)=signal_STFT2(1:rr,:);
 % signal_STFT2=mean(signal_STFT2,2);
%Kurtosis
for trial=1:vv
    for chn=1:32
        k(trial,chn)=kurtosis(matrix_train_40_60cs(trial,chn,:));
    end
end
 % k=mean(k,2);

 for trial=1:rr
    for chn=1:32
        k2(trial,chn)=kurtosis(matrix_test_40_60cs(trial,chn,:));
    end
end
%  k2=mean(k2,2);
 % k3(1:round(0.2*rr),:)=k2(vv+1:rr,:)

%wavelet transformate
scales=1:32;
for trial = 1:vv
    for channel = 1:32
        signal = squeeze(matrix_train_40_60cs(trial, channel, :));
        % Calcolo la CWT
        cwt_coeffs = cwt(signal, 'amor',250);
        % Calcolo l'energia delle wavelet
        wavelet_energy(trial, channel) = sum(abs(cwt_coeffs(:)).^2);
    end
end
% en_wav_trs=mean(wavelet_energy,2);

scales=1:32;
for trial =1:rr
    for channel = 1:32
        signal2= squeeze(matrix_test_40_60cs(trial, channel, :));
        % Calcolo la CWT
        cwt_coeffs2 = cwt(signal2, 'amor',250);
        % Calcolo l'energia delle wavelet
        wavelet_energy2(trial, channel) = sum(abs(cwt_coeffs2(:)).^2);
    end
end
% wavelet_energy3(1:round(0.2*rr),:)=wavelet_energy2(vv+1:rr,:);
% en_wav_trs2=mean(wavelet_energy2,2);
%skewnees
skew_data=skewness(matrix_train_40_60cs(1:vv,1:32,:),0,3);
skew_data2=skewness(matrix_test_40_60cs(1:rr,1:32,:),0,3);

%Command spatial pattern
% data_r1_1=zeros(100,32,1000)
% data_l1_1=zeros(100,32,1000)
% l=zeros(1,100)
% r=zeros(1,100)
% for i=1:vv
%      if(labels(1,i)==1)
%         l(1,i)=i;
%      else
%         r(1,i)=i;
%      end
% end
% for j=1:vv
%     if l(1,j)~=0
%     data_l1_1(j,:,:)=data(j,:,:);
%     else
%     data_r1_1(j,:,:)=data(j,:,:);
%     end
% end
% 
% data_l1_1=pagetranspose(data_l1_1);
% data_r1_1=pagetranspose(data_r1_1);
% 
% for s=1:vv
%     for r=1:32
% csppattern(r,s,:)=csp_function(squeeze(data_l1_1(r,s,:)),squeeze(data_r1_1(r,s,:)))
%     end
% end
% csppattern=mean(csppattern,3)
% csppattern=mean(csppattern,1)
% csppattern=transpose(csppattern)
filter_bands = [8 13; 14 30; 31 50];
num_trials =vv;
num_channels = 32;
num_samples = 1000;
csp_filters = cell(size(filter_bands, 1), 1);
for i = 1:size(filter_bands, 1)

    class1_data = matrix_train_40_60cs(labels_train_40_60cs(1:num_trials,1) == 1, :, :);
    class2_data = matrix_train_40_60cs(labels_train_40_60cs(1:num_trials,1) == 2, :, :);
    cov_class1 = zeros(num_channels, num_channels);
    cov_class2 = zeros(num_channels, num_channels);

    for trial = 1:size(class1_data, 1)
        cov_class1 = cov_class1 + (squeeze(class1_data(trial, :, :)) * squeeze(class1_data(trial, :, :))') / num_samples;
    end
    cov_class1 = cov_class1 / size(class1_data, 1);

    for trial = 1:size(class2_data, 1)
        cov_class2 = cov_class2 + (squeeze(class2_data(trial, :, :)) * squeeze(class2_data(trial, :, :))') / num_samples;
    end
    cov_class2 = cov_class2 / size(class2_data, 1);


    [V, ~] = eig(cov_class1, cov_class1 + cov_class2);
    csp_filters{i} = V(:, [1:3, end-2:end]); 
end
num_trials2=vv;
features = zeros(num_trials2, size(filter_bands, 1) * size(csp_filters{1}, 2));
for i = 1:size(filter_bands, 1)
    for trial = 1:num_trials2
        csp_projection = squeeze(matrix_train_40_60cs(trial, :, :))' * csp_filters{i};
        features(trial, (i-1)*size(csp_filters{i}, 2) + 1:i*size(csp_filters{i}, 2)) = var(csp_projection);
    end
end
features3= zeros(rr, size(filter_bands, 1) * size(csp_filters{1}, 2));
for i = 1:size(filter_bands, 1)
    for trial=1:rr
        csp_projection = squeeze(matrix_test_40_60cs(trial, :, :))' * csp_filters{i};
        features3(trial, (i-1)*size(csp_filters{i}, 2) + 1:i*size(csp_filters{i}, 2)) = var(csp_projection);
    end
end
% features4(1:round(0.2*rr),:)=features3(vv+1:rr,:);
%calcolo ERD/ERS
event=1000/2;
eeg_pre=event-200;
eeg_post=event+300;
epochs=matrix_train_40_60cs(1:vv,:,eeg_pre:eeg_post-1);
epochs_power=epochs.^2;
baseline_power = mean(epochs_power( :, :,1:eeg_pre), 3);

for trial = 1:vv
    for chn=1:32
    event_power = mean(epochs_power(trial, chn, eeg_pre+1:end), 3);
    erd_ers_per_trial(trial,chn) = mean((event_power - baseline_power(trial,chn)) ./ baseline_power(trial,chn)) * 100;
    end
end


epochs2=matrix_test_40_60cs(1:rr,:,eeg_pre:eeg_post-1);
epochs_power2=epochs2.^2;
baseline_power2=mean(epochs_power2(:, :,1:eeg_pre), 3);

for trial=1:rr
    for chn=1:32
    event_power2=mean(epochs_power2(trial, chn, eeg_pre+1:end), 3);
    erd_ers_per_trial2(trial,chn) = mean((event_power2 - baseline_power2(trial,chn)) ./ baseline_power2(trial,chn)) * 100;
    end
end

%autoencoder
XTrain=zeros(vv,32,1000);
XTrain(1:vv,:,:)=matrix_train_40_60cs(1:vv,:,:);
eegdata1=mean(XTrain,3);
hiddenSize=vv; 
autoenc1=trainAutoencoder(eegdata1, hiddenSize, ...
        'MaxEpochs', 50, ...
        'L2WeightRegularization', 0.004, ...
        'SparsityRegularization', 4, ...
        'SparsityProportion', 0.15);
    features_encode=zeros(rr,32);
    features_encode=encode(autoenc1,eegdata1);

XTrain2=zeros(vv,32,1000);
XTrain2(1:rr,:,:)=matrix_test_40_60cs(1:rr,:,:);
eegdata2=mean(XTrain2,3);
hiddenSize=rr; 
autoenc2=trainAutoencoder(eegdata2, hiddenSize, ...
        'MaxEpochs', 50, ...
        'L2WeightRegularization', 0.004, ...
        'SparsityRegularization', 4, ...
        'SparsityProportion', 0.15);
    features_encode2=zeros(rr,32);
    features_encode2=encode(autoenc2,eegdata2);
%coherence
features_chc=extractCoherenceFeatures(matrix_train_40_60cs,fc);
features_chc2=extractCoherenceFeatures(matrix_test_40_60cs,fc);
% for trial = 1:vv
%         for ch = 1:32
%             signal = squeeze(data(trial, ch, :));
% 
%             % Calcola la Higuchi's Fractal Dimension
%             features_hig(trial, ch) = higuchi_fractal_dimension(signal,3,20); % 10 è la lunghezza massima dei sottosegmenti
%         end
% end
% 
% features_hig2=zeros(round(0.2*rr),32);
%   for trial = vv+1:rr
%         for ch = 1:32
%             signal=squeeze(data(trial, ch, :));
% 
%             % Calcola la Higuchi's Fractal Dimension
%             features_hig2(trial, ch) = higuchi_fractal_dimension(signal,3,20); % 10 è la lunghezza massima dei sottosegmenti
%         end
%   end
% features_hig3(1:round(0.2*rr),:)=features_hig2(vv+1:rr,:);



feature_sogg_40cs3_1(1:vv,1:32,1)=data1_m(1:vv,1:32);
feature_sogg_40cs3_1(1:vv,33:64,1)=data1_std(1:vv,1:32);
feature_sogg_40cs3_1(1:vv,65:96,1)=picco(1:vv,1:32);
feature_sogg_40cs3_1(1:vv,97:128,1)=rms_data(1:vv,1:32);
feature_sogg_40cs3_1(1:vv,129:160,1)=peaktopeak(1:vv,1:32);
feature_sogg_40cs3_1(1:vv,161:192,1)=signal_STFT(1:vv,1:32);
feature_sogg_40cs3_1(1:vv,193:224,1)=k(1:vv,1:32);
feature_sogg_40cs3_1(1:vv,225:288,1)=PSD(1:vv,1:64);
feature_sogg_40cs3_1(1:vv,289:320,1)=wavelet_energy(1:vv,1:32);
feature_sogg_40cs3_1(1:vv,321:352,1)=skew_data(1:vv,1:32);
feature_sogg_40cs3_1(1:vv,353:384,1)=data_var(1:vv,1:32);
feature_sogg_40cs3_1(1:vv,385:416,1)=erd_ers_per_trial(1:vv,1:32);
feature_sogg_40cs3_1(1:vv,417:434,1)=features(1:vv,1:18);
feature_sogg_40cs3_1(1:vv,435:466,1)=features_encode(1:vv,1:32);
feature_sogg_40cs3_1(1:vv,467:466+size(features_chc,2),1)=features_chc(1:vv,1:size(features_chc,2));
% feature_sogg_80pc(1:vv,962:993,1)=features_hig(1:vv,1:32);
% 
% 


feature_sogg_60cs3_1(1:rr,1:32,1)=data2_m(1:rr,1:32);
feature_sogg_60cs3_1(1:rr,33:64,1)=data2_std(1:rr,1:32);
feature_sogg_60cs3_1(1:rr,65:96,1)=picco2(1:rr,1:32);
feature_sogg_60cs3_1(1:rr,97:128,1)=rms_data2(1:rr,1:32);
feature_sogg_60cs3_1(1:rr,129:160,1)=peaktopeak2(1:rr,1:32);
feature_sogg_60cs3_1(1:rr,161:192,1)=signal_STFT2(1:rr,1:32);
feature_sogg_60cs3_1(1:rr,193:224,1)=k2(1:rr,1:32);
feature_sogg_60cs3_1(1:rr,225:288,1)=PSD2(1:rr,1:64);
feature_sogg_60cs3_1(1:rr,289:320,1)=wavelet_energy2(1:rr,1:32);
feature_sogg_60cs3_1(1:rr,321:352,1)=skew_data2(1:rr,1:32);
feature_sogg_60cs3_1(1:rr,353:384,1)=data_2var(1:rr,1:32);
feature_sogg_60cs3_1(1:rr,385:416,1)=erd_ers_per_trial2(1:rr,1:32);
feature_sogg_60cs3_1(1:rr,417:434,1)=features3(1:rr,1:18);
feature_sogg_60cs3_1(1:rr,435:466,1)=features_encode2(1:rr,1:32);
feature_sogg_60cs3_1(1:rr,467:466+size(features_chc2,2),1)=features_chc2(1:rr,1:size(features_chc2,2));
% feature_sogg_20pc(1:round(0.2*rr),962:993,1)=features_hig3(1:round(0.2*rr),1:32);

time_extraction_featurecs3_1=toc;
end