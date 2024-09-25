% function features = extractCoherenceFeatures(data, fs)
%     [numTrials, numChannels, numSamples]=size(data);
%     features = zeros(numTrials, numChannels * (numChannels - 1) / 2); 
%     for trial = 1:numTrials
%         for ch1 = 1:numChannels-1
%             for ch2 = ch1+1:numChannels
%                 signal1 = squeeze(data(trial, ch1,:));
%                 signal2 = squeeze(data(trial, ch2,:));
%                 [Cxy, F] = mscohere(signal1, signal2, [], [], [], fs);
%                 features(trial, (ch1-1)*numChannels + ch2 - ch1) = mean(Cxy);
%             end
%         end
%     end
% end

function features = extractCoherenceFeatures(data, fs)
    [numTrials, numChannels, numSamples] = size(data);
    
    % Pre-alloca il vettore delle feature per la coerenza tra ogni coppia di canali
    numPairs = nchoosek(numChannels, 2); % Numero di combinazioni di coppie di canali
    features = zeros(numTrials, numPairs); % Preallocazione per features (coerenza per ogni trial)
    
    % Per ogni trial, calcola la coerenza tra tutte le coppie di canali
    for trial = 1:numTrials
        featureIndex = 1; % Indice per la matrice delle feature
        for ch1 = 1:numChannels-1
            for ch2 = ch1+1:numChannels
                signal1 = squeeze(data(trial, ch1, :));
                signal2 = squeeze(data(trial, ch2, :));
                
                % Calcola la coerenza tra le due serie temporali
                [Cxy, ~] = mscohere(signal1, signal2, [], [], [], fs);
                
                % Salva la media della coerenza nel vettore delle feature
                features(trial, featureIndex) = mean(Cxy);
                featureIndex = featureIndex + 1; % Aggiorna l'indice delle feature
            end
        end
    end
end