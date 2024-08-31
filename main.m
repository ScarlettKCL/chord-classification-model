data_folder = fullfile("/users/scarlettluzac/chord-classification-model/chord-audio-data/");
dataset = audioDatastore(data_folder, IncludeSubfolders=true, FileExtensions='.wav', LabelSource='foldernames');
dataset_length = length(dataset.Files);
files_list = dataset.Files;
[train_ds, val_ds] = splitEachLabel(dataset, 0.8, 'randomized');
train_x = cellfun(@(x) extractMFCC(x), train_ds.Files, UniformOutput=false);
val_x = cellfun(@(x) extractMFCC(x), val_ds.Files, UniformOutput=false);
target_length = 228;
pad_sequence = @(seq) padarray(seq, [target_length - size(seq,1), 0], 'post');
train_x = cellfun(@(x) pad_sequence(x), train_x, UniformOutput=false);
val_x = cellfun(@(x) pad_sequence(x), val_x, UniformOutput=false);
train_y = train_ds.Labels;
val_y = val_ds.Labels;
net_layers = [sequenceInputLayer(target_length)
    lstmLayer(100,OutputMode="last")
    fullyConnectedLayer(length((unique(train_y))))
    softmaxLayer
    classificationLayer];
options = trainingOptions("adam", MiniBatchSize=32, MaxEpochs=50, InitialLearnRate=0.001, Shuffle='every-epoch', Plots="training-progress", Verbose=false);
neural_net = trainNetwork(train_x,train_y, net_layers,options);
predicted_lang = classify(neural_net, val_x, MiniBatchSize=64);
accuracy = sum(predicted_lang == val_y) / numel(val_y);
disp("Test accuracy: " + accuracy);
confusion_matrix = confusionchart(val_y, predicted_lang);

function mfcc_features = extractMFCC(file_path)
 
    [audio_data, sample_rate] = audioread(file_path);
    num_coeffs = 14; 
    window_length = round(0.025 * sample_rate);
    overlap_length = round(0.015 * sample_rate); 
    window = hamming(window_length, 'periodic');
    mfcc_features = mfcc(audio_data, sample_rate, NumCoeffs=num_coeffs, Window=window, OverlapLength=overlap_length);
end