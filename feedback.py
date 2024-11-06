import argparse
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from eval import ToTensor, Normalize
from model import EventDetector
import numpy as np
import torch.nn.functional as F
from ultralytics import YOLO
import cv2

event_names = {
    0: 'Address',
    1: 'Toe-up',
    2: 'Mid-backswing (arm parallel)',
    3: 'Top',
    4: 'Mid-downswing (arm parallel)',
    5: 'Impact',
    6: 'Mid-follow-through (shaft parallel)',
    7: 'Finish'
}

class SampleVideo(Dataset):
    def __init__(self, path, input_size=160, transform=None):
        self.path = path
        self.input_size = input_size
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.path)
        frame_size = [cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)]
        ratio = self.input_size / max(frame_size)
        new_size = tuple([int(x * ratio) for x in frame_size])
        delta_w = self.input_size - new_size[1]
        delta_h = self.input_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        images = []
        for pos in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            ret, img = cap.read()
            if not ret:
                break
            resized = cv2.resize(img, (new_size[1], new_size[0]))
            b_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                       value=[0.406 * 255, 0.456 * 255, 0.485 * 255])
            b_img_rgb = cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB)
            images.append(b_img_rgb)
        cap.release()
        labels = np.zeros(len(images))
        sample = {'images': np.asarray(images), 'labels': np.asarray(labels)}
        if self.transform:
            sample = self.transform(sample)
        return sample

def analyze_video(path, seq_length=64):
    ds = SampleVideo(path, transform=transforms.Compose([ToTensor(),
                            Normalize([0.485, 0.456, 0.406],
                                      [0.229, 0.224, 0.225])]))

    dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)

    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)

    try:
        save_dict = torch.load('./SwingNet Pretrained.tar', weights_only=True)
    except FileNotFoundError:
        print("Model weights not found. Download model weights and place in 'models' folder.")
        return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(save_dict['model_state_dict'])
    model.to(device)
    model.eval()

    for sample in dl:
        images = sample['images']
        batch = 0
        while batch * seq_length < images.shape[1]:
            if (batch + 1) * seq_length > images.shape[1]:
                image_batch = images[:, batch * seq_length:, :, :, :]
            else:
                image_batch = images[:, batch * seq_length:(batch + 1) * seq_length, :, :, :]
            logits = model(image_batch.cuda())
            if batch == 0:
                probs = F.softmax(logits.data, dim=1).cpu().numpy()
            else:
                probs = np.append(probs, F.softmax(logits.data, dim=1).cpu().numpy(), 0)
            batch += 1

    events = np.argmax(probs, axis=0)[:-1]
    confidence = [probs[e, i] for i, e in enumerate(events)]
    result = {event_names[i]: {'frame': e, 'confidence': np.round(confidence[i], 3)} for i, e in enumerate(events)}
    
    return result

# Load a model
model = YOLO("YOLO/yolo11x-pose.pt",verbose=False)

def get_feedback(path):
    swing_feedbacks = {}
    
    for key,value in analyze_video(path).items():
        frame = value['frame']
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        _, img = cap.read()
        cap.release()
        results = model(img)
        for result in results:
            feedback = pose_analysis(key, result.keypoints.xy[0].cpu().numpy())
            if feedback:
                swing_feedbacks[key] = feedback
            annotated_frame = result.plot(labels=False, boxes=False, masks=False, kpt_radius=10) 
            annotated_frame = cv2.resize(annotated_frame, (int(annotated_frame.shape[1] * 0.8), int(annotated_frame.shape[0] * 0.8)))
            cv2.imshow('Annotated Frame', annotated_frame)
            cv2.waitKey(0)
    cv2.destroyAllWindows()
    return swing_feedbacks


import math

def pose_analysis(position, keypoints):
    feedbacks = []
    # yolo_keypoints = {
    #     0: "Nose",
    #     1: "Left Eye",
    #     2: "Right Eye",
    #     3: "Left Ear",
    #     4: "Right Ear",
    #     5: "Left Shoulder",
    #     6: "Right Shoulder",
    #     7: "Left Elbow",
    #     8: "Right Elbow",
    #     9: "Left Wrist",
    #     10: "Right Wrist",
    #     11: "Left Hip",
    #     12: "Right Hip",
    #     13: "Left Knee",
    #     14: "Right Knee",
    #     15: "Left Ankle",
    #     16: "Right Ankle"
    # }
    
    def calculate_angle(pointA, pointB, pointC):
        # Calculate angle ABC where B is the vertex
        AB = math.sqrt((pointB[0] - pointA[0]) ** 2 + (pointB[1] - pointA[1]) ** 2)
        BC = math.sqrt((pointC[0] - pointB[0]) ** 2 + (pointC[1] - pointB[1]) ** 2)
        AC = math.sqrt((pointC[0] - pointA[0]) ** 2 + (pointC[1] - pointA[1]) ** 2)
        
        # Using the law of cosines to calculate the angle
        angle = math.degrees(math.acos((AB**2 + BC**2 - AC**2) / (2 * AB * BC)))
        return angle

    if position == 'Address':
        # Check spine angle
        if all(k is not None and len(k) > 0 for k in [keypoints[0], keypoints[11], keypoints[12]]):
            nose = keypoints[0]
            left_hip = keypoints[11]
            right_hip = keypoints[12]
            
            spine_angle = calculate_angle(nose, left_hip, right_hip)
            is_spine_correct = 0 <= abs(90 - spine_angle) <= 15
            # print(f"Spine Angle: {spine_angle:.2f} degrees")
            # print("Spine angle is within the acceptable range." if is_spine_correct else "Spine angle is outside the acceptable range.")
            feedbacks.append(f"Spine angle is {spine_angle:.2f} degrees. {'Great posture!' if is_spine_correct else 'Try to keep a more upright spine.'}")
        else:
            # print("Missing keypoints for spine angle analysis.")
            pass
        
        # Check foot width relative to shoulder width
        if all(k is not None and len(k) > 0 for k in [keypoints[5], keypoints[6], keypoints[15], keypoints[16]]):
            left_shoulder = keypoints[5]
            right_shoulder = keypoints[6]
            left_ankle = keypoints[15]
            right_ankle = keypoints[16]
            
            shoulder_width = math.dist(left_shoulder, right_shoulder)
            foot_width = math.dist(left_ankle, right_ankle)
            
            is_foot_width_correct = foot_width >= shoulder_width
            # print(f"Shoulder Width: {shoulder_width:.2f}")
            # print(f"Foot Width: {foot_width:.2f}")
            # print("Foot width is acceptable." if is_foot_width_correct else "Foot width is too narrow.")
            feedbacks.append(f"Foot width is wider than shoulder by {foot_width - shoulder_width:.2f} units. {'Good stance width for stability.' if is_foot_width_correct else 'Consider widening your stance for better balance.'}")
        else:
            # print("Missing keypoints for foot width analysis.")
            pass
        
        # Check if left arm is straight
        if all(k is not None and len(k) > 0 for k in [keypoints[5], keypoints[7], keypoints[9]]):
            left_shoulder = keypoints[5]
            left_elbow = keypoints[7]
            left_wrist = keypoints[9]
            
            left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            is_left_arm_straight = 140 <= left_arm_angle <= 180  # Acceptable range for a straight arm
            
            # print(f"Left Arm Angle: {left_arm_angle:.2f} degrees")
            # print("Left arm is straight." if is_left_arm_straight else "Left arm is bent.")
            feedbacks.append(f"Left arm angle is {left_arm_angle:.2f} degrees. {'Nice and straight!' if is_left_arm_straight else 'Try to keep your left arm straighter for better control.'}")
        else:
            # print("Missing keypoints for left arm analysis.")
            pass
        
    elif position == 'Toe-up':
        # Check if left arm is straight
        if all(k is not None and len(k) > 0 for k in [keypoints[5], keypoints[7], keypoints[9]]):
            left_shoulder = keypoints[5]
            left_elbow = keypoints[7]
            left_wrist = keypoints[9]
            
            left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            is_left_arm_straight = 140 <= left_arm_angle <= 180
            
            # print(f"Left Arm Angle: {left_arm_angle:.2f} degrees")
            # print("Left arm is straight." if is_left_arm_straight else "Left arm is bent.")
            feedbacks.append(f"Left arm angle is {left_arm_angle:.2f} degrees. {'Nice and straight!' if is_left_arm_straight else 'Try to keep your left arm straighter for better control.'}")
        else:
            # print("Missing keypoints for left arm analysis.")
            pass
        
        # Check head stability relative to shoulders
        if all(k is not None and len(k) > 0 for k in [keypoints[0], keypoints[15], keypoints[16]]):
            nose = keypoints[0]
            left_ankle = keypoints[15]
            right_ankle = keypoints[16]
            
            # Calculate the midpoint between shoulders
            feet_midpoint = np.array([(left_ankle[0] + right_ankle[0]) / 2,
                                        (left_ankle[1] + right_ankle[1]) / 2])
            
            x_offset = nose[0] - feet_midpoint[0]
            
            # Define a threshold for acceptable head stability (in pixels)
            threshold = 15  # Adjust based on resolution and acceptable movement

            # Check if deviation is within threshold
            if x_offset > threshold:
                feedbacks.append(
                    f"Head moved {x_offset:.2f} pixels from center. "
                    "Try to maintain head stability in the Toe-up position."
                )
            else:
                feedbacks.append("Head is stable in the Toe-up position. Good job!")
        else:
            # print("Missing keypoints for head stability analysis.")
            pass
    elif position == 'Mid-backswing (arm parallel)':
        # Check if left arm is straight
        if all(k is not None and len(k) > 0 for k in [keypoints[5], keypoints[7], keypoints[9]]):
            left_shoulder = keypoints[5]
            left_elbow = keypoints[7]
            left_wrist = keypoints[9]
            
            left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            is_left_arm_straight = 140 <= left_arm_angle <= 180
            
            # print(f"Left Arm Angle: {left_arm_angle:.2f} degrees")
            # print("Left arm is straight." if is_left_arm_straight else "Left arm is bent.")
            feedbacks.append(f"Left arm angle is {left_arm_angle:.2f} degrees. {'Nice and straight!' if is_left_arm_straight else 'Try to keep your left arm straighter for better control.'}")
        
        # Check head stability relative to shoulders
        if all(k is not None and len(k) > 0 for k in [keypoints[0], keypoints[15], keypoints[16]]):
            nose = keypoints[0]
            left_ankle = keypoints[15]
            right_ankle = keypoints[16]
            
            # Calculate the midpoint between shoulders
            feet_midpoint = np.array([(left_ankle[0] + right_ankle[0]) / 2,
                                        (left_ankle[1] + right_ankle[1]) / 2])
            
            x_offset = nose[0] - feet_midpoint[0]
            
            # Define a threshold for acceptable head stability (in pixels)
            threshold = 15  # Adjust based on resolution and acceptable movement

            # Check if deviation is within threshold
            if x_offset > threshold:
                feedbacks.append(
                    f"Head moved {x_offset:.2f} pixels from center. "
                    "Try to maintain head stability in the Mid-backswing position."
                )
            else:
                feedbacks.append("Head is stable in the Mid-backswing position. Good job!")
        else:
            # print("Missing keypoints for head stability analysis.")
            pass
        
    elif position == 'Top':
        # Check if left arm is straight
        if all(k is not None and len(k) > 0 for k in [keypoints[5], keypoints[7], keypoints[9]]):
            left_shoulder = keypoints[5]
            left_elbow = keypoints[7]
            left_wrist = keypoints[9]
            
            left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            is_left_arm_straight = 140 <= left_arm_angle <= 180
            
            # print(f"Left Arm Angle: {left_arm_angle:.2f} degrees")
            # print("Left arm is straight." if is_left_arm_straight else "Left arm is bent.")
            feedbacks.append(f"Left arm angle is {left_arm_angle:.2f} degrees. {'Nice and straight!' if is_left_arm_straight else 'Try to keep your left arm straighter for better control.'}")
        
    print('Position:',position,feedbacks)
    return feedbacks




if __name__ == '__main__':
    # Example usage
    path = 'punn.mp4'
    # path = './vids/19.mp4'
    print(get_feedback(path))
