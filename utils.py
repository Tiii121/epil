# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions to display the pose detection results."""

import cv2
import numpy as np
from tflite_support.task import processor

import RPi.GPIO as GPIO
GPIO.setwarnings(False)
import time

left_motor_pin_pwm = 20  # GPIO pin for controlling left motor speed
right_motor_pin_pwm = 21  # GPIO pin for controlling right motor speed

# Set up direction pins for motor control
left_motor_pin_dir1 = 8  # GPIO pin for left motor direction 1
left_motor_pin_dir2 = 11  # GPIO pin for left motor direction 2
right_motor_pin_dir1 = 18   # GPIO pin for right motor direction 1
right_motor_pin_dir2 = 17   # GPIO pin for right motor direction 2
GPIO.setmode(GPIO.BCM)
GPIO.setup(left_motor_pin_pwm, GPIO.OUT)
GPIO.setup(right_motor_pin_pwm, GPIO.OUT)
GPIO.setup(left_motor_pin_dir1, GPIO.OUT)
GPIO.setup(left_motor_pin_dir2, GPIO.OUT)
GPIO.setup(right_motor_pin_dir1, GPIO.OUT)
GPIO.setup(right_motor_pin_dir2, GPIO.OUT)

# Initialize PWM for motor speed control
left_motor_pwm = GPIO.PWM(left_motor_pin_pwm, 1000)  # 1000 Hz frequency
right_motor_pwm = GPIO.PWM(right_motor_pin_pwm, 1000)

left_motor_pwm.start(0)  # Start with 0% duty cycle (stopped)
right_motor_pwm.start(0)


left_motor_pwm.ChangeDutyCycle(0) #  Speed Motor
right_motor_pwm.ChangeDutyCycle(0) #  Speed Motor
GPIO.output(left_motor_pin_dir1, GPIO.LOW)
GPIO.output(left_motor_pin_dir2, GPIO.LOW)
GPIO.output(right_motor_pin_dir1, GPIO.LOW)
GPIO.output(right_motor_pin_dir2, GPIO.LOW)


# Function to control the car based on voice commands
def forward():
            GPIO.output(left_motor_pin_dir2, GPIO.LOW)
            GPIO.output(left_motor_pin_dir1, GPIO.HIGH)
            GPIO.output(right_motor_pin_dir2, GPIO.LOW)
            GPIO.output(right_motor_pin_dir1, GPIO.HIGH)
            left_motor_pwm.ChangeDutyCycle(50)  # 40% duty cycle for forward motion Speed Motor
            right_motor_pwm.ChangeDutyCycle(50)
            


def right():
            GPIO.output(left_motor_pin_dir2, GPIO.HIGH)
            GPIO.output(left_motor_pin_dir1, GPIO.LOW)
            GPIO.output(right_motor_pin_dir2, GPIO.LOW)
            GPIO.output(right_motor_pin_dir1, GPIO.HIGH)
            left_motor_pwm.ChangeDutyCycle(35)  # 40% duty cycle for forward motion Speed Motor
            right_motor_pwm.ChangeDutyCycle(35)
def left():
            GPIO.output(left_motor_pin_dir2, GPIO.LOW)
            GPIO.output(left_motor_pin_dir1, GPIO.HIGH)
            GPIO.output(right_motor_pin_dir2, GPIO.HIGH)
            GPIO.output(right_motor_pin_dir1, GPIO.LOW)
            left_motor_pwm.ChangeDutyCycle(35)  # 40% duty cycle for forward motion Speed Motor
            right_motor_pwm.ChangeDutyCycle(35)            


# Function to control the car based on voice commands
def forward2():
            GPIO.output(left_motor_pin_dir2, GPIO.LOW)
            GPIO.output(left_motor_pin_dir1, GPIO.HIGH)
            GPIO.output(right_motor_pin_dir2, GPIO.LOW)
            GPIO.output(right_motor_pin_dir1, GPIO.HIGH)
            left_motor_pwm.ChangeDutyCycle(35)  # 40% duty cycle for forward motion Speed Motor
            right_motor_pwm.ChangeDutyCycle(35)
            


def right2():
            GPIO.output(left_motor_pin_dir2, GPIO.HIGH)
            GPIO.output(left_motor_pin_dir1, GPIO.LOW)
            GPIO.output(right_motor_pin_dir2, GPIO.LOW)
            GPIO.output(right_motor_pin_dir1, GPIO.HIGH)
            left_motor_pwm.ChangeDutyCycle(35)  # 40% duty cycle for forward motion Speed Motor
            right_motor_pwm.ChangeDutyCycle(35)
def left2():
            GPIO.output(left_motor_pin_dir2, GPIO.LOW)
            GPIO.output(left_motor_pin_dir1, GPIO.HIGH)
            GPIO.output(right_motor_pin_dir2, GPIO.HIGH)
            GPIO.output(right_motor_pin_dir1, GPIO.LOW)
            left_motor_pwm.ChangeDutyCycle(35)  # 40% duty cycle for forward motion Speed Motor
            right_motor_pwm.ChangeDutyCycle(35)  

def forward3():
            GPIO.output(left_motor_pin_dir2, GPIO.LOW)
            GPIO.output(left_motor_pin_dir1, GPIO.HIGH)
            GPIO.output(right_motor_pin_dir2, GPIO.LOW)
            GPIO.output(right_motor_pin_dir1, GPIO.HIGH)
            left_motor_pwm.ChangeDutyCycle(30)  # 40% duty cycle for forward motion Speed Motor
            right_motor_pwm.ChangeDutyCycle(30)

def Stopping():
            GPIO.output(left_motor_pin_dir1, GPIO.LOW)
            GPIO.output(left_motor_pin_dir2, GPIO.LOW)
            GPIO.output(right_motor_pin_dir1, GPIO.LOW)
            GPIO.output(right_motor_pin_dir2, GPIO.LOW)
            left_motor_pwm.ChangeDutyCycle(0)  # 40% duty cycle for backward motion Speed Motor
            right_motor_pwm.ChangeDutyCycle(0)
            
def actionmove():
    
            forward()
            time.sleep(1)

            right()
            time.sleep(0.2)
            left()
            time.sleep(0.2)

            forward()
            time.sleep(1)

            right2()
            time.sleep(0.2)
            left2()
            time.sleep(0.2)

            forward()
            time.sleep(1)

            right2()
            time.sleep(0.2)
            left2()
            time.sleep(0.2)

_MARGIN = 10  # pixels
_ROW_SIZE = 10  # pixels
_FONT_SIZE = 1
_FONT_THICKNESS = 2
_TEXT_COLOR = (0, 0, 255)  # red


def visualize(
    image: np.ndarray,
    detection_result: processor.DetectionResult,
) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.

  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.

  Returns:
    Image with bounding boxes.
  """
  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    #cv2.rectangle(image, start_point, end_point, _TEXT_COLOR, 3)
    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (_MARGIN + bbox.origin_x,  _MARGIN + _ROW_SIZE + bbox.origin_y)
    if category_name!='Normal':
         cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                _FONT_SIZE, (255, 0, 0), _FONT_THICKNESS)
         
         cv2.rectangle(image, start_point, end_point, _TEXT_COLOR, 3)
         actionmove()
         Stopping()
    else:
        
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                _FONT_SIZE, (0, 255, 0), _FONT_THICKNESS)
        cv2.rectangle(image, start_point, end_point, (0,255,0), 3)
        Stopping()

  return image
