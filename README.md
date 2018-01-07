# Easy Real time gender age prediction from webcam video with Keras
This is complementary source code for my [blog post](https://www.dlology.com/blog/easy-real-time-gender-age-prediction-from-webcam-video-with-keras/).

 Here is the demo

![alt text](https://gitcdn.xyz/cdn/Tony607/blog_statics/master/images/face/age_gender_demo.gif "age gender demo")



## Dependencies
- Python3.5
- numpy 1.13.3+mkl
- Keras 2.0.8+
- TensorFlow 1.4.0
- opencv 1.0.1+
- opencv-python 3.3.0+contrib

Tested on:
- Windows 10 with Tensorflow 1.4.0 GPU

### install requirements
```
pip3 install -r requirements.txt
```

## Run the demo
```
python realtime_demo.py
```

When you use it for the first time , weights are downloaded and stored in **./pretrained_models** folder.
Or you can download it directly from
```
https://github.com/Tony607/Keras_age_gender/releases/download/V1.0/weights.18-4.06.hdf5
```