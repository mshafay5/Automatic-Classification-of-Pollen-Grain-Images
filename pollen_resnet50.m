
net = resnet50;

%For Training on VGG19, uncomment the lines with 4 '%' marks
% % % % net = vgg19;

inputSize = net.Layers(1).InputSize;
imds = imageDatastore('E:\train'...
   ,'IncludeSubfolders',true,'LabelSource','foldernames');
[imdsTrain,imdsTest] = splitEachLabel(imds,0.7,'randomized');
%Augmentation of the dataset since dataset is small
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);

%prevention check to see if the network is series or not. We have a series
%network
net.Layers;
if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 

[learnableLayer,classLayer] = findLayersToReplace(lgraph);
[learnableLayer,classLayer]; 


%total number of classes = 4
numClasses = numel(categories(imdsTrain.Labels));

%changing the last fully connected layer of the RESNET50 
%to only 4 neurons
if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);
lgraph.Layers


layers = lgraph.Layers;
connections = lgraph.Connections;

%freezing all the layer weights except for the last fully connected layer
layers(1:174) = freezeWeights(layers(1:174));
lgraph = createLgraphUsingConnections(layers,connections);
% % % % Comment the upper 2 lines before running the code for vgg19
% % % % layers(1:45) = freezeWeights(layers(1:45));
% % % % lgraph = createLgraphUsingConnections(layers,connections);

%Training parameters
miniBatchSize = 30;
valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',20, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');
net = trainNetwork(augimdsTrain,lgraph,options);

%Predicting the output and calcuating accuracy and Confusion Matrix
[YPred,probs] = classify(net,augimdsTest);
accuracy = mean(YPred == imdsTest.Labels)
ground_labels = imdsTest.Labels;
C = confusionmat(ground_labels,YPred)
confusionchart(C)

countEachLabel(imdsTest)

% Run for testing afterwards

% folder = 'L:\train\for test\';
% img = imread(fullfile(folder,'three (3).png'));
% img = imresize(img,[224 224]);
% [label,score] = classify(net,img);
% imshow(img)
% hold on
% title({['Prediction by the model: ' char(label)], ['Prediction score: ' num2str(max(score),2)], ['Actual Label: 3' ]})

