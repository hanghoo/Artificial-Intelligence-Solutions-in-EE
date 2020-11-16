%--------------------------------------------------------------------------
%| Project 8 -  Optical Character Recognition using CNN                   |
%--------------------------------------------------------------------------
%
% Instructor            : Prof. Uyar
%
% Student 1 Name        : James Bond
% Student 1 CCNY email  : jbond007
% Student 1 Log In Name : ee6530_007
% Student 2 Name        : 
% Student 2 CCNY email  :
% Student 2 Log In Name : 
% Student 3 Name        :
% Student 2 CCNY email  :
% Student 3 Log In Name :
% --------------------------------------------------------------------------
% | I UNDERSTAND THAT COPYING PROGRAMS FROM OTHERS WILL BE DEALT           |
% | WITH DISCIPLINARY RULES OF CCNY.                                       |
% --------------------------------------------------------------------------
%
% Train a CNN to translate the following phrases from Korean to Chinese:
% 
% 1. good morning
% 2. thank you
% 3. how much?
% 4. too expensive
% 5. grocery store
% 6. airport
% 7. taxi stand
% 8. restaurant
% 9. bus stop
% 10. good bye
%
%
% In the cnn_train() function, use the following architecture for the CNN:
%     - CONV LAYER(size = 5x5, noof filters = 10)
%     - BATCH NORMALIZATION
%     - RELU
%     - POOL(size = 2x2, stride = 2)
%     - CONV LAYER(size = 5x5, noof filters = 25)
%     - BATCH NORMALIZATION
%     - RELU
%     - POOL(size = 2x2, stride = 2)
%     - CONV LAYER(size = 5x5, noof filters = 40)
%     - BATCH NORMALIZATION
%     - RELU
%     - POOL(size = 2x2, stride = 2)
%     - CONV LAYER(size = 5x5, noof filters = 55)
%     - BATCH NORMALIZATION
%     - RELU
%     - POOL(size = 1x13, stride = 1)
%     - FULLY CONNECTED ANN
%
% DO NOT MODIFY ANYTHING IN THIS FILE EXCEPT FOR THE FUNCTION cnn_train()
% WHICH CAN BE FOUND AT THE BOTTOM OF THIS FILE

%% Speech Recognition and Translator System using CNNs
clc;
clear;
% select training directory
train_dir = 'TrainingSet'; 
% select reference directory
ref_dir = 'ReferenceSet';

if exist('functions','dir') == 7
    % get access to all files in the 'functions' directory
    addpath('functions');
else
    error("Error: Unable to locate 'functions' directory")
end

message = sprintf('WELCOME TO UNIVERSAL TRANSLATOR. MAKE YOUR SELECTION:');
disp(message);
disp('Enter 1 to train a new CNN');
disp('Enter 2 to load an existing CNN');
user_entry = input('Enter your choice: ');
    
if (user_entry == 1)
    % select input language
    directories = dir(train_dir);
    languages = {};
    for i = 1:length(directories)
        directory = directories(i);
        if directory.isdir && ~strcmp(directory.name,...
                '.') && ~strcmp(directory.name,'..')
            % append languages with directory.name
            languages{length(languages)+1} = directory.name;
            fprintf('Enter %d for %s\n', length(languages), directory.name);
        end
    end
    
    try
        % select translation input language
        user_input = input('Select an input language: ');
        input_lang = languages{user_input};
        fprintf('You selected %s as the input language.\n', input_lang);
    catch
        fprintf('Your selection is out of bounds... Leaving... Bye\n');
        return;
    end
    full_train_path = fullfile(train_dir, input_lang);
    
    % train a new CNN:
    disp('Be patient until statistics are displayed...');
    cnn_trained = CNN_Train(full_train_path);
    input('CNN training is complete. Press enter to start using CNN.\n');
elseif (user_entry == 2)
    % use an existing CNN:
    cnn_file = input('Enter CNN name (dont forget .mat extension): ', 's');
    cnn_path = pwd;
    disp(cnn_file);
    try
        % load the CNN and assign it to cnn_trained
        load_data = load(fullfile(cnn_path,cnn_file));
        cnn_trained = load_data.cnn_trained;
    catch
        disp('Enter a valid file name next time... Leaving... Bye');
        return;
    end
else
    disp('Enter a valid choice next time... Leaving... Bye');
    return;
end

% Now there is a trained CNN loaded (either a new one or an existing one).
% Use trained CNN to make predictions:
input('PRESS ENTER AND IMMEDIATELY SAY A WORD INTO YOUR MICROPHONE');
disp('Recording...');

% setup recording parameters for microphone
sample_rate = 16000;
bits_per_sample = 8;
noof_channels = 1;
recording_duration = 2;

% create the object that performs recording:
recObj = audiorecorder(sample_rate, bits_per_sample, noof_channels);
% recording now:
recordblocking(recObj, recording_duration);

% playback the voice recorded:
disp('Replaying recording...');
play(recObj);
recording_duration = recObj.TotalSamples / recObj.SampleRate;
pause(recording_duration);

% read languages from reference directory and store them into a list
directories = dir(ref_dir);
languages = {};
for i = 1:length(directories)
    directory = directories(i);
    if directory.isdir && ~strcmp(directory.name,...
            '.') && ~strcmp(directory.name,'..')
        % append languages with directory.name
        languages{length(languages)+1} = directory.name;
        fprintf('Enter %d for %s\n', length(languages), directory.name);
    end
end
try
    % select translation output language
    user_input = input('Select an output language: ');
    output_lang = languages{user_input};
    fprintf('You selected %s as the output language.\n', output_lang);
catch
    fprintf('Your selection is out of bounds... Leaving... Bye\n');
    return;
end
%% CNN predicts the translation
cnn_prediction = CNN_Predict(recObj, cnn_trained);
full_ref_path = fullfile(ref_dir, output_lang, char(cnn_prediction));
try  
    % ads: audio data store is an object that holds audio files
    ref_ads = audioDatastore(full_ref_path); 
    PlayBack_Translation(recObj, ref_ads);
catch
    fprintf('translation for %s is not available in %s\n',output_lang, ...
        char(cnn_prediction));
end
user_entry = input('Do you want to save CNN? (y/n) ','s');
if(user_entry == 'y')
    file_name = input('Enter file name (do not forget .mat extension): ',...
        's');
    save(file_name, 'cnn_trained');
else   
end
disp('Have a predictably nice day!');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% END OF PROGRAM %%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Prediction] = CNN_Predict(recObj, net)
%CNN_Predict: Use CNN to predict the word 
    
    y = getaudiodata(recObj);
    Fs = recObj.SampleRate;
    audiowrite('temp.wav',y,Fs);
    adsTest = audioDatastore('temp.wav');
    
    % set audio parameters
    fs = 16000;
    segmentDuration = length(audioread(adsTest.Files{1})) / fs;
    frameDuration = 0.025;
    hopDuration = 0.010;
    numBands = 40;
    epsil = 1e-6;
    
    % compute speech spectrograms, they will be the inputs
    % to the CNN
    XTest = speechSpectrograms(adsTest,segmentDuration,frameDuration,...
        hopDuration,numBands,fs);
    XTest = log10(XTest + epsil);
    
    [Prediction,~] = classify(net, XTest);
    
    % delete temporary file
    if(isfile('temp.wav')) 
        delete temp.wav;
    end 
end

%% MAKE MODIFICATIONS BELOW

function net = CNN_Train(train_dir)
% Train a new CNN using information from TrainingSet directory
% Then return trained CNN

    % load the training audio dataset into ads_train.
    % train_dir contains the path to the dataset:
    
    
    
    
    % set audio parameters
    fs = 16000;
    segmentDuration = length(audioread(adsTrain.Files{1})) / fs;
    frameDuration = 0.025;
    hopDuration = 0.010;
    numBands = 40;
    epsil = 1e-6;
    
    % convert the audio samples into images by generating their
    % spectrograms:
    
    
    
    % create input and expected output matrices to train on:
    
    
    
    % get the dimensions of XTrain:
    
    % set specSize to the height and width of a single spectrogram:
    
    % append a depth of 1 to specsize to get the input size of the CNN:
    
    % get the total number of unique labels:
    
  
    % Define the architecture of the CNN
    layers = [
        % set the size of the input image as the size of a single
        % spectrogram:
        
        
        % Add the following layers to the CNN:
        %     - CONV LAYER(size = 5x5, noof filters = 10)
        %     - BATCH NORMALIZATION
        %     - RELU
        %     - POOL(size = 2x2, stride = 2)
        %     - CONV LAYER(size = 5x5, noof filters = 25)
        %     - BATCH NORMALIZATION
        %     - RELU
        %     - POOL(size = 2x2, stride = 2)
        %     - CONV LAYER(size = 5x5, noof filters = 40)
        %     - BATCH NORMALIZATION
        %     - RELU
        %     - POOL(size = 2x2, stride = 2)
        %     - CONV LAYER(size = 5x5, noof filters = 55)
        %     - BATCH NORMALIZATION
        %     - RELU
        %     - POOL(size = 1x13, stride = 1)
        
        
        
        % create the fully connected ANN ouput layer
        % the number of neurons in the output layer is equal to the number
        % of unique labels:
        
        
        % use a softmax layer to convert the outputs to a set of 
        % probalities where each output represents the probability that the
        % image is a corresponding label:
        
        
        % add a classification layer to make each output mutually
        % exclusive, in other words there's only a single correct label for 
        % each image:
        
        ];

    % specify training parameters:
    
    
    % train the CNN:
    
end
