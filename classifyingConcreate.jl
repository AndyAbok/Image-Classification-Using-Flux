using FileIO
using Images
using Random 
using MLDataPattern
using Flux
using Base.Iterators: partition
using Flux:onehotbatch,@epochs
using Flux.Data: DataLoader
using Printf
using Statistics
using BSON: @save

# function load_images(path::String)
#     img_paths = readdir(path)
#     imgs = []
#     for img_path in img_paths
#         push!(imgs, load(string(path,img_path)))
#     end
#     return imgs
# end

#= 
1.)
- We start by defining a function to convert the images into grayscale.
- we save this in a new directory to preserve the original data set state.
- The images will be scaled to a width and height of 90
=#

function resize_and_grayify(directory, im_name, width::Int64, height::Int64)
    resized_gray_img = Gray.(load(directory * "/" * im_name)) |> (x -> imresize(x, width, height))
    try
        save("preprocessed_" * directory * "/" * im_name, resized_gray_img)
    catch e
        if isa(e, SystemError)
            mkdir("preprocessed_" * directory)
            save("preprocessed_" * directory * "/" * im_name, resized_gray_img)
        end
    end
end

#Wrapper function for the proccsing to enable handling of multiple  images at once.

function processImages(directory, width::Int64, height::Int64)
    files_list = readdir(directory)    
    map(x -> resize_and_grayify(directory, x, width, height),files_list)
end

#= 
2.)
- We begin processing the images using the above functions.
- The resolution we are going to use for both the width and height 
- The second bit converts the images to gray scale and the defined scale and saves them in a new directory.
- The third step reads the new directory of the images and finally they are loaded into the environment.(the proccessed images)
=#

resolutionScale = 90

processImages("W/CW", resolutionScale, resolutionScale)
processImages("W/UW", resolutionScale, resolutionScale)

crackedWallDir = readdir("preprocessed_W/CW")
uncrackedWallDir = readdir("preprocessed_W/UW")

crackedWall = load.("preprocessed_W/CW/" .* crackedWallDir)
uncrackedWall = load.("preprocessed_W/UW/" .* uncrackedWallDir)	

classificationData = vcat(crackedWall, uncrackedWall)

#= 
3.) Data Processing 
- we begin by giving the two categorises labels the is o and 1.
- we Procced to dividing the data into train and test set 
=#

begin
    labels = vcat([0 for _ in 1:length(crackedWall)], [1 for _ in 1:length(uncrackedWall)])
    (x_train, y_train), (x_test, y_test) = splitobs(shuffleobs((classificationData, labels)), at = 0.7)
end

# This function creates minibatches of the data.Each minibatch is a tuple of (data, labels).
function createMinibatch(X, Y, idxs)    
    X_batch = Array{Float32}(undef, size(X[1])..., 1, length(idxs))
    for i in 1:length(idxs)
        X_batch[:, :, :, i] = Float32.(X[idxs[i]])       
    end
    Y_batch = onehotbatch(Y[idxs], 0:1)
    return (X_batch, Y_batch)
end

begin
    # here we define the train and test sets.
    batchsize = 128
    mb_idxs = partition(1:length(x_train), batchsize)
    train_set = [createMinibatch(x_train, y_train, i) for i in mb_idxs]
    test_set = createMinibatch(x_test, y_test, 1:length(x_test))
end

train_data = DataLoader((x_train, y_train), batchsize=batchsize , shuffle=true)
test_data = DataLoader((x_test, y_test), batchsize=  batchsize)

#=
4.) Modell  building
- Here the CNN learner is defined
=#

Learner =
    Chain(
        Conv((3, 3), 1=>32, pad=(1,1), relu),
        MaxPool((2,2)),
        Conv((3, 3), 32=>64, pad=(1,1), relu),
        MaxPool((2,2)),
        Conv((3, 3), 64=>128, pad=(1,1), relu),
        MaxPool((2,2)),
        flatten,
        Dense(15488, 2),
        softmax)

begin
    train_loss = Float64[]
    test_loss = Float64[]
    acc = Float64[]
    ps = Flux.params(Learner)
    opt = ADAM()
    L(x, y) = Flux.crossentropy(Learner(x), y)
    L((x,y)) = Flux.crossentropy(Learner(x), y)
    accuracy(x, y, f) = mean(Flux.onecold(f(x)) .== Flux.onecold(y))
    
    function update_loss!()
        push!(train_loss, mean(L.(train_set)))
        push!(test_loss, mean(L(test_set)))
        push!(acc, accuracy(test_set..., Learner))
        @printf("train loss = %.2f, test loss = %.2f, accuracy = %.2f\n", train_loss[end], test_loss[end], acc[end])
    end
end

# Here we train our model for n epochs times.
@epochs 10 Flux.train!(L, ps, train_set, opt;
               cb = Flux.throttle(update_loss!, 8))

plot(train_loss, xlabel="Iterations", title="Model Training", label="Train loss", lw=2, alpha=0.9)
plot!(test_loss, label="Test loss", lw=2, alpha=0.9)
plot!(acc, label="Accuracy", lw=2, alpha=0.9)

@show accuracy(test_set...,Learner)
@save "wallClassifier.bson" Learner

# We define the prediction prediction Engine.
using BSON: @load

@load "wallClassifier.bson" Learner

function predictionEngine(imageName::String)

    @assert occursin(".jpg",imageName) == true || throw(error("The image Must be of type .jpg but was $imageName"))

    imagewidth = 90
    imageheight = 90
    imgToPredict = Gray.(load("newimages/" * imageName ))  |> (x -> imresize(x, imagewidth, imageheight))
    predIndex = Learner(reshape(imgToPredict, size(imgToPredict)...,1,1)) |> vec |> findmax

    if predIndex[2] == 1 
        println("Cracked Wall : Probability = $(predIndex[1])")
    else 
        println("Uncracked Wall : Probability = $(predIndex[1])")
    end
    
    return ifelse(predIndex[2] == 1,("Cracked Wall",predIndex[1]),("UnCracked Wall",predIndex[1]))
end

predictionEngine("7080-154.jpg")





