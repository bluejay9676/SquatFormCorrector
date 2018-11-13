
import argparse
import cv2
import time
import numpy as np

"""
MPI:
1 - head
2, 5 - shoulder
14 - back
8, 9, 10 - groin, knee, feet
11, 12, 13 - groin, knee, feet

COCO:
1 - head
2, 5 - shoulder
8, 9, 10 - groin, knee, feet
11, 12, 13 - groin, knee, feet

pose_pairs - pair of points. Used for drawing skeletal structure.
crucial_points - points used for calculating squat posture.
"""

inWidth = 368
inHeight = 368
threshold = 0.1

class SquatFormCorrector:
    def __init__(self, input_source=0, mode="MPI", write_flag=False, output_file_name="output.avi"):
        # Set config.
        self.write_flag = write_flag
        if mode is "MPI" :
            protoFile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
            weightsFile = "pose/mpi/pose_iter_160000.caffemodel"
            self.pose_pairs = [ [1,14], [1,2], [1,5], [14,8], [14,11], [8,9], [9,10], [11,12], [12,13] ]
            self.crucial_points = [1, 2, 5, 14, 8, 9, 10, 11, 12, 13]
        else: # COCO
            protoFile = "pose/coco/pose_deploy_linevec.prototxt"
            weightsFile = "pose/coco/pose_iter_440000.caffemodel"
            self.pose_pairs = [ [1,8], [1,2], [1,5], [1,11], [8,9], [9,10], [11,12], [12,13] ]
            self.crucial_points = [1, 2, 5, 8, 9, 10, 11, 12, 13]

        # Init video source.
        self.cap = cv2.VideoCapture(input_source)
        if (self.cap.isOpened()== False): 
            print("Error opening video stream or file")

        # Init model.
        self.net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile) 
        self.vid_writer = None

        if write_flag:
            hasFrame, frame = cap.read()
            self.vid_writer = cv2.VideoWriter(output_file_name, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame.shape[1],frame.shape[0]))

        self.point_position = {} # k : point  v : (int, int) normalized location on the video.

    
    def _is_wrong_form(self, is_side):
        """
        Detect wrong squat form.
        If side view:
            1. Neck -> Back -> Groin == linear?
            2. Knee farther than Feet?
            3. Hip under Knee?
            4. Bar path (or neck path) == linear?
        If front view:
            1. Shoulder width == stance width.
            2. Are knees forming 11? (it should form M like shape)
        """
        raise NotImplementedError # TODO implement

    def process(self, is_side, delay_rate=200, draw_skeleton=True):
        while(self.cap.isOpened()):
            t = time.time()
            hasFrame, frame = self.cap.read()
            if not hasFrame:
                break

            frameWidth = frame.shape[1]
            frameHeight = frame.shape[0]

            inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                    (0, 0, 0), swapRB=False, crop=False)
            self.net.setInput(inpBlob)
            output = self.net.forward()

            H = output.shape[2]
            W = output.shape[3]

            for i in self.crucial_points:
                # confidence map of corresponding body's part.
                probMap = output[0, i, :, :]

                # Find global maxima of the probMap.
                minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
                
                # Scale the point to fit on the original image
                x = (frameWidth * point[0]) / W
                y = (frameHeight * point[1]) / H

                if prob > threshold : 
                    # Add the point to the list if the probability is greater than the threshold
                    self.point_position[i] = (int(x), int(y))
                    cv2.circle(frame, (int(x), int(y)), 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                else :
                    self.point_position[i] = None

            if draw_skeleton:
                for pair in self.pose_pairs:
                    partA = pair[0]
                    partB = pair[1]
                    if self.point_position[partA] and self.point_position[partB]:
                        cv2.line(frame, self.point_position[partA], self.point_position[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)

            reason = self._is_wrong_form(is_side=is_side)
            if reason:
                cv2.putText(frame, reason, (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
            else:
                cv2.putText(frame, "Good form!", (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)

            if self.write_flag:
                self.vid_writer.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if self.write_flag:
            self.vid_writer.release()
        self.cap.release()
        cv2.destroyAllWindows()


parser = argparse.ArgumentParser(description='SquatFormCorrector')
# TODO fix args
parser.add_argument('--train',
    action='store_true',
    help='Train flag' )
parser.add_argument('--test',
    action='store_true',
    help='Test flag' )
parser.add_argument('--p',
    action='store_true',
    help='Test run the entire pipeline. Train->store->load->predict' )
parser.add_argument('--gs',
    action='store_true',
    help='Grid search' )

# {input_name} {output_name} {mode} {write_result_to_avi} {side_view} {delay_rate} {draw_skeleton}

if __name__ == "__main__":
    args = parser.parse_args()
    # TODO use args
    corrector = SquatFormCorrector(input_source="sample_video.mp4", output_file_name="test.avi")
    corrector.process(is_side=False, delay_rate=200, draw_skeleton=False)