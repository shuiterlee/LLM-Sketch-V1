import numpy as np
import sys
import torch
import math
import config
import models
import utils
from PIL import Image
from prompts.success_detection_prompt import SUCCESS_DETECTION_PROMPT
from config import OK, PROGRESS, FAIL, ENDC
from config import CAPTURE_IMAGES, ADD_BOUNDING_CUBES, ADD_TRAJECTORY_POINTS, EXECUTE_TRAJECTORY, OPEN_GRIPPER, CLOSE_GRIPPER, TASK_COMPLETED, RESET_ENVIRONMENT

class API:

    def __init__(self, args, main_connection, logger, client, langsam_model, xmem_model=None, device=None):

        self.args = args
        self.main_connection = main_connection
        self.logger = logger
        self.client = client
        self.langsam_model = langsam_model
        self.xmem_model = xmem_model
        self.device = device
        self.segmentation_texts = []
        self.segmentation_count = 0
        self.trajectory_length = 0
        self.attempted_task = False
        self.completed_task = False
        self.failed_task = False
        self.head_camera_position = None
        self.head_camera_orientation_q = None
        self.wrist_camera_position = None
        self.wrist_camera_orientation_q = None
        self.command = None



    def detect_object(self, segmentation_text):

        self.logger.info(PROGRESS + "Capturing head and wrist camera images..." + ENDC)
        self.main_connection.send([CAPTURE_IMAGES])
        [head_camera_position, head_camera_orientation_q, wrist_camera_position, wrist_camera_orientation_q, env_connection_message] = self.main_connection.recv()
        self.logger.info(env_connection_message)

        self.head_camera_position = head_camera_position
        self.head_camera_orientation_q = head_camera_orientation_q
        self.wrist_camera_position = wrist_camera_position
        self.wrist_camera_orientation_q = wrist_camera_orientation_q

        rgb_image_head = Image.open(config.rgb_image_head_path).convert("RGB")
        depth_image_head = Image.open(config.depth_image_head_path).convert("L")
        depth_array = np.array(depth_image_head) / 255.

        if self.segmentation_count == 0:
            xmem_image = Image.fromarray(np.zeros_like(depth_array)).convert("L")
            xmem_image.save(config.xmem_input_path)

        segmentation_texts = [segmentation_text]

        self.logger.info(PROGRESS + "Segmenting head camera image..." + ENDC)
        model_predictions, boxes, segmentation_texts = models.get_langsam_output(rgb_image_head, self.langsam_model, segmentation_texts, self.segmentation_count)
        self.logger.info(OK + "Finished segmenting head camera image!" + ENDC)

        masks = utils.get_segmentation_mask(model_predictions, config.segmentation_threshold)

        bounding_cubes_world_coordinates, bounding_cubes_orientations = utils.get_bounding_cube_from_point_cloud(rgb_image_head, masks, depth_array, self.head_camera_position, self.head_camera_orientation_q, self.segmentation_count)

        utils.save_xmem_image(masks)

        self.segmentation_texts.extend(segmentation_texts)

        self.logger.info(PROGRESS + "Adding bounding cubes to the environment..." + ENDC)
        self.main_connection.send([ADD_BOUNDING_CUBES, bounding_cubes_world_coordinates])
        [env_connection_message] = self.main_connection.recv()
        self.logger.info(env_connection_message)

        for i, bounding_cube_world_coordinates in enumerate(bounding_cubes_world_coordinates):

            bounding_cube_world_coordinates[4][2] -= config.bounding_cube_depth_offset

            object_width = np.around(np.linalg.norm(bounding_cube_world_coordinates[1] - bounding_cube_world_coordinates[0]), 3)
            object_length = np.around(np.linalg.norm(bounding_cube_world_coordinates[2] - bounding_cube_world_coordinates[1]), 3)
            object_height = np.around(np.linalg.norm(bounding_cube_world_coordinates[5] - bounding_cube_world_coordinates[0]), 3)

            print("Position of " + segmentation_texts[i] + ":", list(np.around(bounding_cube_world_coordinates[4], 3)))

            print("Dimensions:")
            print("Width:", object_width)
            print("Length:", object_length)
            print("Height:", object_height)

            if object_width < object_length:
                print("Orientation along shorter side (width):", np.around(bounding_cubes_orientations[i][0], 3))
                print("Orientation along longer side (length):", np.around(bounding_cubes_orientations[i][1], 3), "\n")
            else:
                print("Orientation along shorter side (length):", np.around(bounding_cubes_orientations[i][1], 3))
                print("Orientation along longer side (width):", np.around(bounding_cubes_orientations[i][0], 3), "\n")

        self.segmentation_count += 1



    def execute_trajectory(self, trajectory):

        self.logger.info(PROGRESS + "Adding trajectory points to the environment..." + ENDC)
        self.main_connection.send([ADD_TRAJECTORY_POINTS, trajectory])

        self.logger.info(PROGRESS + "Executing generated trajectory..." + ENDC)
        self.main_connection.send([EXECUTE_TRAJECTORY, trajectory])

        self.trajectory_length += len(trajectory)



    def open_gripper(self):

        self.logger.info(PROGRESS + "Opening gripper..." + ENDC)
        self.main_connection.send([OPEN_GRIPPER])



    def close_gripper(self):

        self.logger.info(PROGRESS + "Closing gripper..." + ENDC)
        self.main_connection.send([CLOSE_GRIPPER])



    def task_completed(self):

        if self.attempted_task:

            self.completed_task = True

        else:

            self.logger.info(PROGRESS + "Waiting to execute all generated trajectories..." + ENDC)
            self.main_connection.send([TASK_COMPLETED])
            [env_connection_message] = self.main_connection.recv()
            self.logger.info(env_connection_message)

            self.logger.info(PROGRESS + "Processing object positions..." + ENDC)
            
            num_objects = len(self.segmentation_texts)
            
            new_prompt = SUCCESS_DETECTION_PROMPT.replace("[INSERT TASK]", self.command)
            new_prompt += "\n"

            self.logger.info(PROGRESS + "Using simple position estimation..." + ENDC)

            for object_idx, object_name in enumerate(self.segmentation_texts):
                object_positions = [[0, 0, 0]]
                object_orientations = [0.0]
                
                new_prompt += object_name + " trajectory positions and orientations:\n"
                new_prompt += "Positions:\n"
                new_prompt += str(np.around(object_positions, 3)) + "\n"
                new_prompt += "Orientations:\n"
                new_prompt += str(np.around(object_orientations, 3)) + "\n"
                new_prompt += "\n"

            self.logger.info(OK + "Finished position estimation!" + ENDC)

            self.attempted_task = True

            messages = []

            self.logger.info(PROGRESS + "Generating ChatGPT output..." + ENDC)
            messages = models.get_chatgpt_output(self.client, self.args.language_model, new_prompt, messages, "system", file=sys.stderr)
            self.logger.info(OK + "Finished generating ChatGPT output!" + ENDC)

            code_block = messages[-1]["content"].split("```python")

            task_completed = self.task_completed
            task_failed = self.task_failed

            for block in code_block:
                if len(block.split("```")) > 1:
                    code = block.split("```")[0]
                    exec(code)



    def task_failed(self):

        self.failed_task = True

        self.logger.info(PROGRESS + "Resetting environment..." + ENDC)
        self.main_connection.send([RESET_ENVIRONMENT])
        [env_connection_message] = self.main_connection.recv()
        self.logger.info(env_connection_message)

        self.segmentation_count = 0
        self.trajectory_length = 0
        self.segmentation_texts = []
        self.attempted_task = False
