# Finland-Fireman-Dataset

## **Introduction**
This project aims to develop a robust fire, smoke, person, lake, vehicles, and building detection system using the YOLOv8 (You Only Look Once) object detection model. The model is trained on the **Finland Fireman Project Dataset**, which includes annotated images featuring fire incidents, rescue operations, and various environmental scenarios. The goal is to leverage this dataset to create an efficient detection system that can identify critical objects in emergency situations, particularly for fire and smoke detection.

This project can be run on your local machine for testing purposes or scaled to a high-performance computing (HPC) environment like **Lehmus** for training with larger datasets and faster performance.

---

## **Part 1: Running the Project on Your Local Machine**

### **Prerequisites**
Before running the project on your local machine, ensure you have the following tools and libraries installed:

1. **Python 3.8+**
   - You can download Python from the official [Python website](https://www.python.org/downloads/).

2. **Conda (Recommended)**
   - Conda is a package and environment manager. Download and install Conda from the official [Miniconda or Anaconda page](https://docs.conda.io/en/latest/miniconda.html).

3. **Ultralytics YOLOv8**
   - YOLOv8 is developed by Ultralytics. You can install it directly from PyPI using pip.
   ```bash
   pip install ultralytics
   ```

4. **GPU support (Optional but recommended)**
   - For training, a machine with an Nvidia GPU and CUDA installed is recommended for better performance.

### **Project Setup**

1. **Clone the Repository**
   Clone this repository to your local machine:
   ```bash
   git clone https://github.com/yourusername/fire-smoke-detection-yolo.git
   cd fire-smoke-detection-yolo
   ```

2. **Prepare the Dataset**
   - The dataset should be in YOLO format with images and labels in `data/images/` and `data/labels/`.
   - Ensure your `data.yaml` file is correctly configured with the number of classes (`nc`) and the paths to your training and validation images:
     ```yaml
     train: ./data/images/train  # Training images folder
     val: ./data/images/val      # Validation images folder

     nc: 6  # number of classes
     names: ['fire', 'smoke', 'person', 'lake', 'vehicles', 'building']
     ```

3. **Create and Activate a Conda Environment**
   If you're using Conda, create a virtual environment:
   ```bash
   conda create -n yolov8_env python=3.8
   conda activate yolov8_env
   ```

4. **Install Dependencies**
   Install the required Python packages, including Ultralytics YOLOv8:
   ```bash
   pip install ultralytics
   ```

5. **Run YOLOv8 Training**
   Run the YOLOv8 model for training on the dataset:
   ```bash
   yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=100 imgsz=640
   ```

   This will start training on your local machine. Results, including trained weights, will be saved in the `runs/train/` directory.

### **Inference**
Once training is complete, you can use the trained model to run inference on new images:
```bash
yolo task=detect mode=predict model=runs/train/weights/best.pt source=/path/to/images
```

---

## **Part 2: Running the Project on Lehmus (HPC Cluster)**

### **Step-by-Step Instructions for Lehmus**

1. **Access Lehmus**
   First, connect to the Lehmus HPC system. Use `ssh` to login:
   ```bash
   ssh lehmus-login1.oulu.fi -l your_username
   ```

2. **Transfer Data to Lehmus**
   Use `scp` to upload your dataset from your local machine to your Lehmus home directory or research partition.
   ```bash
   scp -r /path/to/local/data your_username@lehmus-login1.oulu.fi:/homedir06/your_username/modeltrain/
   ```

3. **Create a Conda Environment on Lehmus**
   On Lehmus, load the Conda module, create, and activate the environment:
   ```bash
   module load conda
   conda create -n yolov8_env python=3.12
   conda activate yolov8_env
   ```

4. **Install YOLOv8 and Dependencies**
   Install the Ultralytics YOLOv8 package within your environment:
   ```bash
   pip install ultralytics
   ```

5. **SLURM Job Submission**
   On Lehmus, you will need to use SLURM to submit a job for training the YOLOv8 model on the GPU. The following SLURM script `my_yolo_job.sh` is configured to request 1 Nvidia A30 GPU, 24GB of memory, and 16 CPU cores:

   **SLURM Job Script (`my_yolo_job.sh`):**
   ```bash
   #!/bin/bash
   #SBATCH --gres=gpu:a30:1   # Request 1 Nvidia A30 GPU
   #SBATCH --mem=24G          # Memory required
   #SBATCH --cpus-per-task=16  # CPU cores required
   #SBATCH --time=01-00:00:00  # Max time (1 day)
   #SBATCH --mail-user=your_email@domain.com  # Notifications
   #SBATCH --mail-type=BEGIN,END,FAIL  # Email on start, end, or fail

   # Load Conda and activate the environment
   module load conda
   conda activate yolov8_env

   # Copy dataset to fast storage
   cp -r /homedir06/your_username/modeltrain/firemandataset/ $LOCAL_SCRATCH
   cd $LOCAL_SCRATCH/firemandataset

   # Run YOLOv8 model training
   yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=100 imgsz=1024

   # Move results back to home directory
   mkdir -p $HOME/results/$SLURM_JOB_ID
   mv $LOCAL_SCRATCH/firemandataset/runs/detect/train/ $HOME/results/$SLURM_JOB_ID
   ```

   Save this script to the `scripts/` directory.

6. **Submit the SLURM Job**
   Once the script is ready, submit the job using the `sbatch` command:
   ```bash
   sbatch scripts/my_yolo_job.sh
   ```

7. **Monitor the Job**
   You can monitor the job status using the `squeue` command:
   ```bash
   squeue -u your_username
   ```

   You can also check the output log once the job completes:
   ```bash
   cat slurm-<jobid>.out
   ```

8. **Retrieve Results**
   Once the job completes, the results (including the trained model) will be available in your home directory under `results/<SLURM_JOB_ID>/`. You can download them back to your local machine using `scp`:
   ```bash
   scp -r your_username@lehmus-login1.oulu.fi:/homedir06/your_username/results/<SLURM_JOB_ID> /path/to/local/directory
   ```

### **Additional Notes**
- If you encounter any errors with the script, ensure paths in the SLURM script are correct and that your dataset is properly structured in YOLO format.
- Modify the `yolo` command in the script to change parameters like number of epochs, batch size, or image size.

---

## **Conclusion**
This project enables efficient fire, smoke, and object detection using the YOLOv8 model trained on the Inland Fireman Project dataset. It supports both local execution and scalable training on the Lehmus HPC cluster for high-performance computing environments. Follow the instructions above to run the project based on your setup.

For any issues or further questions, feel free to open an issue or contribute by submitting a pull request.
