"""
get_bag_data.py

This script includes the get_bag_data function, which processes specific topics (CompressedImage, Odometry, NavSatFix, and PointCloud2) from ROS2 bag files (.mcap or .db3). 
It extracts the data and saves it in accessible formats (.png, .csv, .pcd) within an organized folder structure.

Author: João Campanhã, INESC TEC
Date: 08/11/2024

Requirements:
    - OpenCV
    - MCAP plugin (https://mcap.dev/guides/getting-started/ros-2)
    - Open 3D (https://www.open3d.org/docs/release/getting_started.html)
    - ROS2 dependencies (rosbag2_py, rclpy, rosidl_runtime_py)

Usage:

The topics that are desired to be extracted, can be specified in a .yaml file of the following format:
    namespace: /robotx  #Optional
    topics:
    CompressedImage:
        - /my/topic/compressed
    PointCloud2:
        - /my/topic/point_cloud
    Odometry:
        - /my/topic/odom
    NavSatFix:
        - /my/topic/global

The program will automatically search for any .yaml file in the folder where the script is located. 
If a .yaml file is found, you can run the program as follows:
    python get_bag_data.py {path/to/bag_file}

Alternatively, if you want to specify a custom .yaml file, use the --topic flag:
    python get_bag_data.py {path/to/bag_file} --topic {path/to/file.yaml}

You can also manually specify a topic directly via the command line like this:
    python get_bag_data.py {path/to/bag_file} --topic {/specific/topic}

"""

import os
import argparse
import numpy as np
import glob
from datetime import datetime
import csv
import yaml

from sensor_msgs.msg import CompressedImage, NavSatFix, PointCloud2
from sensor_msgs_py import point_cloud2
from nav_msgs.msg import Odometry 
from tf2_msgs.msg import TFMessage

import cv2
import open3d as o3d
import open3d.core as o3c

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py


# Topic headers that are stored in a CSV 
topic_headers = {
    "nav_msgs/msg/Odometry": ["Timestamp", "Frame_ID", "Child_Frame_ID", "Position_X", "Position_Y", "Position_Z", "Orientation_X", "Orientation_Y", "Orientation_Z", "Orientation_W", "Pose_Covariance", "Vel_linear_X", "Vel_linear_Y", "Vel_linear_Z", "Vel_angular_X", "Vel_angular_Y", "Vel_angular_Z",  "Velocity_Covariance",],
    "sensor_msgs/msg/NavSatFix": ["Timestamp", "Latitude", "Longitude", "Altitude", "Position_Covariance", "Position_Covariance_Type"],
    "tf2_msgs/msg/TFMessage": ["Timestamp", "Frame_ID", "Child_Frame_ID", "Position_X", "Position_Y", "Position_Z", "Orientation_X", "Orientation_Y", "Orientation_Z", "Orientation_W"]
}

# This are the allowed message types that the program can process
allowed_types = {
        "nav_msgs/msg/Odometry": "Odometry messages",
        "sensor_msgs/msg/NavSatFix": "NavSatFix messages",
        "sensor_msgs/msg/CompressedImage": "Compressed Images messages",
        "sensor_msgs/msg/PointCloud2": "Point Cloud Messages",
        "tf2_msgs/msg/TFMessage": "TF Messages"
    }


def load_topics_from_yaml(file_path):
    """
    Loads topics configuration from a YAML file.
    
    Args:
        file_path (str): Path to the YAML configuration fil ̣e.
        
    Returns:
        dict: A dictionary where keys are message types (e.g. 'CompressedImage') 
              and values are lists of topics associated with those message types.
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)

        # Extract the namespace, defaulting to empty string if not found
        namespace = config.get("namespace", "")

        # Check if config has a 'topics' dictionary with the expected structure
        if isinstance(config, dict) and isinstance(config.get("topics"), dict):
            topics = config["topics"]

            if namespace:
                # Apply namespace to each topic in the topics dictionary
                for topic_type, topic_list in topics.items():
                    # Prepend the namespace to each topic if it doesn't already start with it
                    topics[topic_type] = [
                        namespace + topic if not topic.startswith(namespace) else topic
                        for topic in topic_list
                    ]
            else:
                print(f"\nNo namespace provided. Using topics as they are.")

            print(f"\nUsing topics from YAML file '{file_path}': {topics}")

        else:
            topics = {}
            raise ValueError("YAML file '{file_path}' should contain a 'topics' dictionary with message types.")
            

    return topics, namespace


def get_topic_message_type(bag_path, topic_name, data_type):
    """
    Extracts the message type for a specific topic in a ROS2 bag file.

    Args:
        bag_path (str): Path to the ROS2 bag file.
        topic_name (str): The topic name to retrieve the message type for.
        data_type (str): The format of the bag file (e.g., 'mcap', 'sqlite3').

    Returns:
        str: The message type of the specified topic, or None if the topic is not found.
    """
    # Initialize the reader and open the bag file
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id=data_type)
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    # Retrieve metadata to get topic information
    metadata = reader.get_metadata()
    topics_info = metadata.topics_with_message_count

    # Search for the specified topic in the metadata
    for topic_info in topics_info:
        if topic_info.topic_metadata.name == topic_name:
            # Close the reader before returning
            del reader
            return topic_info.topic_metadata.type

    # Close the reader if topic is not found
    del reader
    return None


def get_all_message_types(bag_path: str, data_type):
    """
    Extracts message types for each topic in a ROS2 bag file.

    Args:
        bag_path (str): Path to the ROS2 bag file.
        data_type (str): The format of the bag file (e.g., 'mcap', 'sqlite3').

    Returns:
        dict: A dictionary where keys are topic names and values are message types.
    
    Example:
        message_types = get_all_message_types('/path/to/bag', 'mcap')
    """

    # Initialize the reader and open the bag file
    reader = rosbag2_py.SequentialReader()

    # reader.open(storage_options, converter_options)
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id=data_type),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )

    # Retrieve and parse metadata to get topic information
    metadata = reader.get_metadata()
    topics_info = metadata.topics_with_message_count

    # Build a dictionary of topic names and their message types
    # topic_message_types = {topic_info.name: topic_info.type for topic_info in topics_info}
    topic_message_types = {
        topic_info.topic_metadata.name: topic_info.topic_metadata.type
        for topic_info in topics_info
    }
    
    # Close the reader
    del reader

    return topic_message_types



def read_messages(input_bag: str, storage_id, filtered_topics):
    """
    Reads messages from a ROS2 bag file, yielding only the specified topics, message and timestamp.

    Args:
        input_bag (str): Path to the ROS2 bag file to read messages from.
        storage_id (str): Storage ID format of the ROS2 bag file ('mcap' or 'sqlite3').
        filtered_topics (str or list): Specific topic(s) to extract messages from; use a string for a single topic or a list for multiple topics.

    Yields:
        tuple: (topic (str), msg (Message object), timestamp (int)) - Topic name, deserialized message, and timestamp of each message.

    Description:
        - Opens the ROS2 bag file and reads all available topics and their types.
        - Filters messages based on `filtered_topics`, attempts to deserialize, and yields them.
        - Skips messages that cannot be deserialized, printing an error message when this occurs.
    """

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=input_bag, storage_id=storage_id),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )

    topic_types = reader.get_all_topics_and_types()

    def typename(topic_name):
        for topic_type in topic_types:
            if topic_type.name == topic_name:
                return topic_type.type
        raise ValueError(f"topic {topic_name} not in bag")


    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        
        if topic in filtered_topics:
            try:
                msg_type = get_message(typename(topic))
                msg = deserialize_message(data, msg_type)
            except Exception as e:
                # Skip topics or messages that cannot be deserialized
                print(f"Error deserializing message on topic {topic}: {e}")
                continue

            yield topic, msg, msg_type, timestamp

    del reader



def create_directory_and_csv(topic, topic_type, OUTPUT_MAIN):
    """
    Creates a directory and a CSV file for a specific topic if required.

    Args:
        topic (str): The name of the topic for which a directory and CSV file need to be created.
        topic_type (str): The message type of the topic, used to check if CSV creation is needed.
        OUTPUT_MAIN (str): The main output directory path where topic subdirectories will be created.

    Returns:
        subdirectory (str): Path of the created or existing directory for the topic.
        csv_path (str): Path of the created or existing CSV file for the topic, empty if no CSV was required.

    Description:
        - Creates a subdirectory for the topic inside the main output directory.
        - Creates a CSV file if required (based on `topic_headers`), or manages existing files:
            - Creates a new CSV file if it doesn’t exist.
            - Clears any data beyond the header if the CSV file already exists.
    """

    # Creates subdirectory for the topic
    subdirectory = os.path.join(OUTPUT_MAIN, topic.strip("/").replace("/", "_"))

    # Checks if the directory already exists
    if os.path.exists(subdirectory):
        print(f"\nDirectory already exists: {subdirectory}")
    else:
        # Creates the directory only if it doesn't exist
        os.makedirs(subdirectory, exist_ok=True)
        print(f"Created directory: {subdirectory}")

    csv_path = ""
    print("TOPCISSSC TYPEEEE")
    print(topic_type)
    # Only create CSV file if a header is defined for the topic
    if topic_type in topic_headers:

        # If it has a header, a csv file needs to be created
        csv_path = os.path.join(subdirectory, f"{topic.strip('/').replace('/', '_')}.csv")

        # Create file if it doesnt exist
        if not os.path.exists(csv_path):
            with open(csv_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(topic_headers[topic_type])  # Write header
                print(f"Created CSV file with header: {csv_path}")
        else:
            print(f"CSV file already exists: {csv_path}")

            # Open the file in read mode to read the contents
            with open(csv_path, mode='r', newline='') as file:
                lines = file.readlines()
            
            # Check if the file contains more than just the header
            if len(lines) > 1:
                # Open the file again in write mode, to overwrite it
                with open(csv_path, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    # Write only the header back to the file
                    writer.writerow(topic_headers[topic_type])
                    print(f"Cleared all lines in the CSV file except the header: {csv_path}")


    return subdirectory, csv_path



def validate_single_topic_type(input_bag, inserted_topics, topics, expected_types, data_type):
    """
    Validates the message type of given topics in a ROS2 bag and adds valid topics to the inserted_topics list.

    Args:
        input_bag (str): Path to the ROS2 bag file.
        inserted_topics (list): List to store validated topics with their types.
        topics (list or str): List of topic names or a single topic to validate.
        expected_types (str or list): Expected message type(s) for the topics.
        data_type (str): ROS2 bag file type ('mcap' or 'sqlite3').

    Returns:
        list: The `inserted_topics` list with validated topics.

    Raises:
        ValueError: If a topic's type does not match the expected type(s).

    Description:
        Checks if each topic matches the expected message type(s). Invalid topics are skipped or raise an error.
        Valid topics are added to `inserted_topics`.
    """

    # Ensure topics is a list to simplify iteration
    topics = [topics] if isinstance(topics, str) else topics

    for topic in topics:
        topic_type = get_topic_message_type(input_bag, topic, data_type)

        if topic_type is None:
            print(f"\n[WARNING]: The '{topic}' does not exist!\n")
            continue

        if isinstance(expected_types, list):  # When expected_types is a list
            if topic_type not in expected_types:
                raise ValueError(f"Invalid type for topic '{topic}': expected one of {expected_types}, got '{topic_type}'")
            
        else:  # When expected_types is a single string
            if topic_type not in [key for key, _ in expected_types]:  # Extracting only the allowed types
                raise ValueError(f"Invalid type for topic '{topic}': expected '{expected_types}', got '{topic_type}'")
            

        inserted_topics.append({"topic": topic, "type": topic_type})


    # If all topics pass the checks
    return inserted_topics




def validate_topic_type(input_bag, inserted_topics, topics, expected_types, data_type):
    """
    Validates the message types of given topics in a ROS2 bag and adds valid topics to the inserted_topics list.

    Args:
        input_bag (str): Path to the ROS2 bag file.
        inserted_topics (list): List to store validated topics with their types.
        topics (list or str): List of topics or a single topic to validate.
        expected_types (str or list): Expected message type(s) for the topics.
        data_type (str): The ROS2 bag file type ('mcap' or 'sqlite3').

    Returns:
        list: The `inserted_topics` list with validated topics.

    Raises:
        ValueError: If a topic's type doesn't match the expected type(s).

    Description:
        This function checks if each topic matches the expected message type. If a topic does not exist or 
        has an invalid type, it is skipped or raises an error. Valid topics are added to `inserted_topics`.
    """

    for topic in topics:
        topic_type = get_topic_message_type(input_bag, topic, data_type)
        
        # Check if topic_type is None first and handle it with a warning
        if topic_type is None:
            print(f"[WARNING]: The '{topic}' does not exist!\n")
            continue  # Skip further processing for this topic
        
        # Only proceed to type check if topic_type is not None
        if topic_type != expected_types:
            raise ValueError(f"Invalid type for topic '{topic}': expected '{expected_types}', got '{topic_type}'")
        
        # If all checks pass, add topic to inserted_topics
        inserted_topics.append({"topic": topic, "type": topic_type})
        

    # If all topics pass the checks
    return inserted_topics


def setup_environment(args, group_of_topics, single_topic, namespace):
    """
    Sets up the environment for processing ROS2 bag data by validating topics, creating directories, 
    and initializing CSV files.

    Args:
        args (argparse.Namespace): Command-line arguments, including:
            - `args.input` (str): Path to the input ROS2 bag file (.mcap or .db3).
        group_of_topics (dict): Topics grouped by message types (e.g., "CompressedImage").
        single_topic (str): A single topic to process, used if `group_of_topics` is not provided.
        namespace (str): Optional namespace string for the output directory.

    Returns:
        tuple:
            - created_subdirectories (dict): Topic names and paths to their subdirectories.
            - created_CSVs (dict): Topic names and paths to their initialized CSV files.
            - data_type (str): The bag file type ("mcap" or "sqlite3").
            - inserted_topics (list): Validated topics to process.

    Raises:
        ValueError: If the input file is invalid or no topics are specified.

    Description:
        - **Determines File Type**: Extracts file extension to identify .mcap or .db3 file types.
        - **Validates Topics**: Checks message types in the bag file and validates topics based on input.
        - **Sets Up Directories and CSVs**: Creates subdirectories and initializes CSV files with headers for each validated topic.
        - **Returns Setup Details**: Returns dictionaries with subdirectory and CSV paths, file type, and validated topics.

    Notes:
        - The output directory is based on the input file and optionally the namespace.
        - Only supported topics are processed.
    """

    ##################### Get the file data type and create main dir #####################
    # Obtain file extension and name
    file_extension = os.path.splitext(args.input)[1].lower()
    file_name = os.path.splitext(os.path.basename(args.input))[0]

    file_path = os.path.splitext(args.input)[0]
    print(file_path)

    absolute_path = os.path.abspath(file_path)
    # Extract the parent folder path, which is 'RECEIVE_DIR'
    parent_folder_path = os.path.dirname(absolute_path)


    # Define the main output directory based on the file name
    if namespace:
        namespace = namespace.replace("/", "")
        OUTPUT_MAIN = f"{parent_folder_path}/DATA_{namespace}_{file_name}"
    else:
        OUTPUT_MAIN = f"{parent_folder_path}/DATA_{file_name}"
        
    os.makedirs(OUTPUT_MAIN, exist_ok=True)
    # Determine data type

    if file_extension == ".mcap":
        data_type = "mcap"
    elif file_extension == ".db3":
        data_type = "sqlite3"
    else:
        raise ValueError("Not a valid ROS2 bag file!")

    ##################### Get message types and checks the desired topic #####################

    # Get all the message types:
    topic_message_types = get_all_message_types(args.input, data_type)

    print("\nLooking at the supported message types in the bag...")
    for msg_type, msg_name in allowed_types.items():
        if msg_type in topic_message_types.values():
            print(f"  {msg_name} are present in the bag.")
        else:
            print(f"  [WARNING]: No {msg_name} found in the bag!")

    print("\n")
    inserted_topics = [] #This will store the topics that we want to process, in order to create the files

    # Process topics based on the argument input
    if not group_of_topics:
        if single_topic:
            inserted_topics = validate_single_topic_type(args.input, inserted_topics, single_topic, allowed_types.items(), data_type)
        else:
            raise ValueError("No topic is specified!")
    else:
        # Validate message types for each group of topics
        if group_of_topics.get("CompressedImage"):
            inserted_topics = validate_topic_type(args.input, inserted_topics, group_of_topics["CompressedImage"], 'sensor_msgs/msg/CompressedImage', data_type)

        if group_of_topics.get("PointCloud2"):
            inserted_topics = validate_topic_type(args.input, inserted_topics, group_of_topics["PointCloud2"], 'sensor_msgs/msg/PointCloud2', data_type)

        if group_of_topics.get("NavSatFix"):
            inserted_topics = validate_topic_type(args.input, inserted_topics, group_of_topics["NavSatFix"], 'sensor_msgs/msg/NavSatFix', data_type)

        if group_of_topics.get("Odometry"):
            inserted_topics = validate_topic_type(args.input, inserted_topics, group_of_topics["Odometry"], 'nav_msgs/msg/Odometry', data_type)    

        if group_of_topics.get("TFMessage"):
            inserted_topics = validate_topic_type(args.input, inserted_topics, group_of_topics["TFMessage"], 'tf2_msgs/msg/TFMessage', data_type)   

    # ##################### Creating subdirectories and respective CSV Files if needed #####################
    created_subdirectories = {}
    created_CSVs = {}

    for entry in inserted_topics:
        topic = entry["topic"]
        topic_type = entry["type"]

        # Create subdirectory and CSV for the specific topic
        subdirectory, csv_path = create_directory_and_csv(topic, topic_type, OUTPUT_MAIN)
        created_subdirectories[topic] = subdirectory
        created_CSVs[topic] = csv_path


    return created_subdirectories, created_CSVs, data_type, inserted_topics


def extract_numpy_pcl(msg):
    """
    Extracts position and intensity data from a ROS2 PointCloud2 message as NumPy arrays.

    Args:
        msg (PointCloud2): A ROS2 PointCloud2 message containing point cloud data. 
                           The message should have fields like 'x', 'y', 'z', and optionally 'intensity'.

    Returns:
        tuple:
            - positions (numpy.ndarray): A NumPy array of shape (N, 3) containing the x, y, z coordinates of points.
            - intensities (numpy.ndarray or None): A NumPy array of shape (N, 1) containing the intensity values of 
                                                   points, if available. Returns `None` if the intensity field is not present.

    Notes:
        - This function uses `sensor_msgs.point_cloud2.read_points_list`, which ensures flexibility in handling 
          various field names and skips invalid points (e.g., NaNs).
        - The output `positions` is always a 2D array of 3D positions (x, y, z).
        - If the intensity field is not present in the input message, `intensities` will be `None`.
    """
    
    # Read points using read_points (flexible with different data types)
    points = list(point_cloud2.read_points_list(msg, field_names=['x', 'y', 'z', 'intensity'], skip_nans=True))

    # Converting to matrix format
    points_array = np.array(points)

    # Separate position (x, y, z) and intensity (if available)
    positions = points_array[:, :3]  # x, y, z
    intensities = points_array[:, 3].reshape(-1, 1) if points_array.shape[1] > 3 else None  # intensity column if present

    return positions, intensities


def save_point_cloud_data(output_folder, topic, timestamp, msg):
    """
    Saves PointCloud2 data from a ROS2 message as a .pcd file.

    Args:
        output_folder (str): The directory to save the point cloud data.
        topic (str): The topic name from which the PointCloud2 message was received.
        timestamp (int): The timestamp of the message (in nanoseconds).
        msg (sensor_msgs.msg.PointCloud2): The ROS2 PointCloud2 message to save.

    Returns:
        None: The function performs I/O operations but does not return any value.

    Description:
        - Converts PointCloud2 data to numpy arrays (position, intensity).
        - Maps the data to Open3D tensors and creates a point cloud object.
        - Saves the point cloud as a .pcd file in the specified output folder.
    """

    #Unix to readable date
    ts = timestamp/1000000000
    ts_string = datetime.fromtimestamp(ts).strftime('%Y-%m-%d_(%H:%M:%S.%f)')

    position, intensity = extract_numpy_pcl(msg)
    if position is None:
        print(f"Failed to convert PointCloud2 message for topic {topic} at timestamp {ts_string}")
        return
    
    # Mapping to sensors
    map_to_tensors = {}

    map_to_tensors["positions"] = o3d.core.Tensor(position, o3c.float32)  # XYZ data
    
    # If intensity is available, store it as a separate attribute
    if intensity is not None:
        map_to_tensors["intensities"] = o3d.core.Tensor(intensity, o3c.float32)  # Intensity data

    # Adding the mapping to the Open3D tensor-based point cloud object
    pcd = o3d.t.geometry.PointCloud(map_to_tensors)
    
    # Save the point cloud as a .pcd file
    filename = os.path.join(output_folder, f"{topic.lstrip('/').replace('/', '_')}_{ts_string}.pcd")
    o3d.t.io.write_point_cloud(filename, pcd)
    print(f"Saved point cloud data from {topic} to {filename}")


def save_visual_data(output_folder, topic, timestamp, msg):
    """
    Saves visual data (images) to disk if the file does not already exist.

    Args:
        output_folder (dict): A dictionary mapping topic names to their corresponding subdirectory paths 
                              for saving images. For example, {'/camera': '/path/to/camera_data'}.
        topic (str): The topic name from which the message was received. Typically, a string 
                     starting with '/' (e.g., '/camera/image_raw').
        timestamp (int): The timestamp of the message in nanoseconds. Used for naming the image file.
        msg (Message): A ROS2 message object, expected to be of type `CompressedImage`. 
                       The `msg.data` field should contain image data.

    Returns:
        None: The function performs file operations (I/O) but does not return a value.

    Notes:
        - The image is saved in PNG format with a filename derived from the topic name and 
          timestamp. The format is: `<topic>_<YYYY-MM-DD_(HH:MM:SS.microseconds)>.png`.
        - The function currently supports only `CompressedImage` messages. 
          Other message types are ignored with a warning.
        - If the file already exists, no action is performed.
    """

    ts = timestamp/1000000000
    ts_string = datetime.fromtimestamp(ts).strftime('%Y-%m-%d_(%H:%M:%S.%f)')
    
    visual_filename = os.path.join(output_folder, f"{topic.lstrip('/').replace('/', '_')}_{ts_string}.png")
    #Verify if image doesnt exist
    if not os.path.exists(visual_filename):

        if isinstance(msg, CompressedImage):
            # Convert compressed image to OpenCV format
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        else:
            print(f"Unsupported message type on topic {topic}")
            return

        #Escrever imagem para o path
        cv2.imwrite(visual_filename, cv_image)
        print(f"Saved {topic} image to {visual_filename}")



def save_odom_data(created_CSVs, topic, timestamp, msg):
    """
    Appends odometry data to a specific CSV file.

    Args:
        created_CSVs (dict): Dictionary where keys are topics, and values are the corresponding 
                             CSV file paths for saving data.
        topic (str): The topic name from which the message was received.
        timestamp (int): Timestamp of the message.
        msg (Message): The ROS2 message object, expected to be of type `Odometry`.

    Returns:
        None: The function performs I/O operations but does not return any value.
    """

    odom_pose_covariance_matrix = (
        f"[[{msg.pose.covariance[0]}, {msg.pose.covariance[1]}, {msg.pose.covariance[2]}, {msg.pose.covariance[3]}, {msg.pose.covariance[4]}, {msg.pose.covariance[5]}], "
        f"[{msg.pose.covariance[6]}, {msg.pose.covariance[7]}, {msg.pose.covariance[8]}, {msg.pose.covariance[9]}, {msg.pose.covariance[10]}, {msg.pose.covariance[11]}], "
        f"[{msg.pose.covariance[12]}, {msg.pose.covariance[13]}, {msg.pose.covariance[14]}, {msg.pose.covariance[15]}, {msg.pose.covariance[16]}, {msg.pose.covariance[17]}], "
        f"[{msg.pose.covariance[18]}, {msg.pose.covariance[19]}, {msg.pose.covariance[20]}, {msg.pose.covariance[21]}, {msg.pose.covariance[22]}, {msg.pose.covariance[23]}], "
        f"[{msg.pose.covariance[24]}, {msg.pose.covariance[25]}, {msg.pose.covariance[26]}, {msg.pose.covariance[27]}, {msg.pose.covariance[28]}, {msg.pose.covariance[29]}], "
        f"[{msg.pose.covariance[30]}, {msg.pose.covariance[31]}, {msg.pose.covariance[32]}, {msg.pose.covariance[33]}, {msg.pose.covariance[34]}, {msg.pose.covariance[35]}]]"
    )   

    odom_twist_covariance_matrix = (
        f"[[{msg.twist.covariance[0]}, {msg.twist.covariance[1]}, {msg.twist.covariance[2]}, {msg.twist.covariance[3]}, {msg.twist.covariance[4]}, {msg.twist.covariance[5]}], "
        f"[{msg.twist.covariance[6]}, {msg.twist.covariance[7]}, {msg.twist.covariance[8]}, {msg.twist.covariance[9]}, {msg.twist.covariance[10]}, {msg.twist.covariance[11]}], "
        f"[{msg.twist.covariance[12]}, {msg.twist.covariance[13]}, {msg.twist.covariance[14]}, {msg.twist.covariance[15]}, {msg.twist.covariance[16]}, {msg.twist.covariance[17]}], "
        f"[{msg.twist.covariance[18]}, {msg.twist.covariance[19]}, {msg.twist.covariance[20]}, {msg.twist.covariance[21]}, {msg.twist.covariance[22]}, {msg.twist.covariance[23]}], "
        f"[{msg.twist.covariance[24]}, {msg.twist.covariance[25]}, {msg.twist.covariance[26]}, {msg.twist.covariance[27]}, {msg.twist.covariance[28]}, {msg.twist.covariance[29]}], "
        f"[{msg.twist.covariance[30]}, {msg.twist.covariance[31]}, {msg.twist.covariance[32]}, {msg.twist.covariance[33]}, {msg.twist.covariance[34]}, {msg.twist.covariance[35]}]]"
    )

    data = [
        timestamp,
        msg.header.frame_id,
        msg.child_frame_id,
        msg.pose.pose.position.x,
        msg.pose.pose.position.y,
        msg.pose.pose.position.z,
        msg.pose.pose.orientation.x,
        msg.pose.pose.orientation.y,
        msg.pose.pose.orientation.z,
        msg.pose.pose.orientation.w,
        odom_pose_covariance_matrix,
        msg.twist.twist.linear.x,
        msg.twist.twist.linear.y,
        msg.twist.twist.linear.z,
        msg.twist.twist.angular.x,
        msg.twist.twist.angular.y,
        msg.twist.twist.angular.z,
        odom_twist_covariance_matrix
    ]

    with open(created_CSVs[topic], mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

    print(f"Appended odometry data to {created_CSVs[topic]}")



def save_global_data(created_CSVs, topic, timestamp, msg):
    """
    Appends global position data to a specific CSV file.

    Args:
        created_CSVs (dict): Dictionary where keys are topics, and values are the corresponding 
                             CSV file paths for saving data.
        topic (str): The topic name from which the message was received.
        timestamp (int): Timestamp of the message.
        msg (Message): The ROS2 message object, expected to be of type `NavSatFix`.

    Returns:
        None: The function performs I/O operations but does not return any value.
    """
    
    position_covariance_matrix = (
        f"[[{msg.position_covariance[0]}, {msg.position_covariance[1]}, {msg.position_covariance[2]}], "
        f"[{msg.position_covariance[3]}, {msg.position_covariance[4]}, {msg.position_covariance[5]}], "
        f"[{msg.position_covariance[6]}, {msg.position_covariance[7]}, {msg.position_covariance[8]}]]"
    )

    data = [
        timestamp,
        msg.latitude,
        msg.longitude,
        msg.altitude,
        position_covariance_matrix,  # Unpack the position covariance list into separate columns
        msg.position_covariance_type
    ]
            
    with open(created_CSVs[topic], mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

    print(f"Appended global position data to {created_CSVs[topic]}")


def save_tf_data(created_CSVs, topic, timestamp, msg):
    """
    Appends transformation data to a specific CSV file.

    Args:
        created_CSVs (dict): Dictionary where keys are topics, and values are the corresponding 
                            CSV file paths for saving data.
        topic (str): The topic name from which the message was received.
        timestamp (int): Timestamp of the message.
        msg (TFMessage): The ROS2 message object, expected to be of type `TFMessage`.

    Returns:
        None: The function performs I/O operations but does not return any value.
    """
    
    for transform in msg.transforms:
        # Extract details from TransformStamped
        frame_id = transform.header.frame_id
        child_frame_id = transform.child_frame_id
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        
        # Prepare data to be saved
        data = [
            timestamp,
            frame_id,
            child_frame_id,
            translation.x,
            translation.y,
            translation.z,
            rotation.x,
            rotation.y,
            rotation.z,
            rotation.w
        ]
        
        # Append data to the CSV file corresponding to the topic
        with open(created_CSVs[topic], mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
        
    print(f"Appended TF data to {created_CSVs[topic]}")



def main():

    ##################### Get the arguments from the command line #####################
    parser = argparse.ArgumentParser(description="Extract and display images from a specific ROS topic")
    parser.add_argument(
        "input", help="Select a valid ROS2 bag (.mcap or .db3) path to extract data",
    )
    parser.add_argument(
        "--topic", 
        help="Specify a topic to extract data from or provide a YAML file with topics"
    )
    args = parser.parse_args()

    single_topic = {}
    group_of_topics = {}

    ##################### Determine topics based on the --topic argument or default behavior #####################
    if args.topic:
        # If --topic is specified, check if it's a file or a topic
        if args.topic.endswith(".yaml"):
            if os.path.isfile(args.topic):
                # Load topics from the YAML file
                group_of_topics, namespace = load_topics_from_yaml(args.topic)

            else:
                raise ValueError(f"The specified YAML was not found!")
        else:
            # Treat as a single topic
            single_topic = [args.topic]
            namespace = None
            print(f"\nExtracting data from specified topic: {args.topic}")
    else:
        # If no --topic is provided, look for a YAML file in the current directory
        yaml_files = glob.glob("*.yaml")
        if yaml_files:
            # Load topics from the first YAML file found
            yaml_path = yaml_files[0]
            group_of_topics, namespace = load_topics_from_yaml(yaml_path)

        else:
            # Warn if no topic or YAML file is found
            raise ValueError(f"No topic specified and no YAML file found in the current directory.")
 

    ##################### Setup environment and obtain required paths and topics#####################
    print("\nChecking and creating the environment... \n")
    created_subdirectories, created_CSVs, data_type, desired_topics = setup_environment(args, group_of_topics, single_topic, namespace)
            
    print("\n[BAG INFO]:")
    print(f"  File path: {args.input}")
    filtered_topics = [entry['topic'] for entry in desired_topics]
    print(f"  Topic(s) that will be processed: {filtered_topics}")
    print(f"  Data type: {data_type}\n")

    #TODO: Maybe add threading

    for topic, msg, msg_type, timestamp in read_messages(args.input, data_type, filtered_topics):
        #SAVE VISUAL DATA
        if msg_type is CompressedImage:
            save_visual_data(created_subdirectories[topic], topic, timestamp, msg)
        #SAVE POINT CLOUD DATA
        if msg_type is PointCloud2:
            save_point_cloud_data(created_subdirectories[topic], topic, timestamp, msg)

        #SAVE ODOMETRY DATA
        if msg_type is Odometry:
            save_odom_data(created_CSVs, topic, timestamp, msg)
            
        #SAVE GLOBAL DATA
        if msg_type is NavSatFix:
            save_global_data(created_CSVs, topic, timestamp, msg)
        
        if msg_type is TFMessage:
            save_tf_data(created_CSVs, topic, timestamp, msg)
    
    print("\nData extraction completed.")


if __name__ == "__main__":
    main()

