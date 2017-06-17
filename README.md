# **Behavioral Cloning**

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

### Model Architecture and Training Strategy

#### 1. Solution Design Approach
When I have analyzed `driving_log.csv`, I saw a lot of 0-like steers records. Much more then recodrs with valuable steers.
So I removed ~60% of them using
```
    if float(line[3]) < .03 and randint(0,9) > 5:
        continue
```

Then of all, I decided to optimize input images using:
`x / 255.0 - 0.5` formula

Then I cropped region of interest using (70, 25) options.

#### Model Architecture

As a starting point used model from nVidia team.
It contains of:
 1. Lambda layer
 2. Croping layer
 3. Convolutional layer (24, 5,5, sample(2,2), followed by Dropout(0.2)
 4. Convolutional layer (36, 5,5, sample(2,2), followed by Dropout(0.2)
 5. Convolutional layer (48, 5,5, sample(2,2), followed by Dropout(0.2)
 6. Convolutional layer (64, 5,5, sample(2,2), followed by Dropout(0.2)
 7. Convolutional layer (64, 5,5, sample(2,2), followed by Dropout(0.2)
 8. Flatten layer
 9. Dense(100)
 10. Dense(50)
 11. Dense(10)
 12. Dense(1)

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.
I have added Dropout layers after each convolutional layers and then overfitting became lower.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to get back to center of road.

To augment the data sat, I also flipped images and angles thinking that this would increase model accuracy.

After the collection process, I had `17492` number of data points.
I then preprocessed this data by `11232` data points.


I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by.
I used an adam optimizer so that manually training the learning rate wasn't necessary.

I found that best performance I got using `10` epochs.
My project loss - `3%`, validation loss - `6%`