# Finland-Fireman-Dataset

## **Introduction**
This project aims to develop a robust fire, smoke, person, lake, vehicles, and building detection system using the YOLOv8 (You Only Look Once) object detection model. The model is trained on the **Finland Fireman Project Dataset**, which includes annotated images featuring fire incidents, rescue operations, and various environmental scenarios. The goal is to leverage this dataset to create an efficient detection system that can identify critical objects in emergency situations. This project will also try to encounter class imblance issue for object detection and how we can make a robust object detection system for edge device.

This project can be run on your local machine for testing purposes or scaled to a high-performance computing (HPC) environment like **Lehmus** for training with larger datasets and faster performance.

---

## **Accessing the Dataset**

The dataset used in this project is derived from the **Finland Fireman Project**, which focuses on fire incident and rescue operation data. This dataset is part of ongoing research and is associated with a published paper, which provides further details on the data collection and annotation processes.

You can access the paper and learn more about the dataset through the following link:  
[**Finland Fireman Project Dataset - Paper Link**](<https://doi.org/10.5281/zenodo.13732947>).

For research collaboration or if you'd like to contribute to the project, we encourage you to reach out. Since the dataset is not publicly available, you can send us an email at **taahamed23@student.oulu.fi** to request access, including details about your intended use or potential collaboration. Each request will be considered based on its relevance to ongoing research.

---

## **Part 1: Running the Project on Your Local Machine**

### **Requirements**
To run this project locally, you will need the following:

- Python 3.8+
- PIP (Python package installer)
- Ultralytics YOLO library
- CUDA (optional, for GPU acceleration)
- PyTorch (for deep learning training)

### **1. Installation**

#### **Step 1: Install the Required Python Libraries**
1. Install PyTorch:
   - If you are using a machine with a GPU:
     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     ```
   - If you are using a machine without a GPU:
     ```bash
     pip install torch torchvision torchaudio
     ```

2. Install the Ultralytics YOLO library:
   ```bash
   pip install ultralytics
   ```

#### **Step 2: Set Up the Dataset**
Place your dataset in a directory on your local machine. Ensure that the folder structure looks like this:

```
firemandataset/
├── train/
├── val/
├── test/
└── data.yaml
```

### **2. Training the YOLOv8 Model**

#### **Step 1: Python Script for Training**
You can use the following Python script to train the YOLOv9 model on your local machine. Update the paths to match your local directory structure:

```python
import os
import torch
from ultralytics import YOLO

# Set up directory paths
base_path = r'.\firemandataset'
train_path = os.path.join(base_path, 'train')
test_path = os.path.join(base_path, 'test')
val_path = os.path.join(base_path, 'val')
config_path = os.path.join(base_path, 'data.yaml')

# Write configuration to the YAML file
with open(config_path, "w") as f:
    f.write(f"train: {train_path}\n")
    f.write(f"val: {val_path}\n")
    f.write(f"test: {test_path}\n")
    f.write("nc: 6\n")  # Number of classes in the dataset
    f.write("names: ['fire', 'smoke', 'person', 'lake', 'vehicle','building']\n")  # Class names

# Initialize the YOLO model
model = YOLO('yolov8n.pt')

# Check for GPU availability and set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Train the model
results = model.train(
    data=config_path,
    epochs=100,  # Number of epochs
    imgsz=1024,  # Image size
    plots=True,
    patience=10,  # Early stopping patience
)
```

#### **Step 2: Running the Code**
1. Save the Python script as `train_yolov8.py`.
2. Run the script from your terminal or command prompt:
    ```bash
    python train_yolov8.py
    ```

### **3. Understanding the Script**
- **base_path**: This is the location where your dataset is stored. Make sure to update it with the correct path to your dataset.
- **model = YOLO('yolov8n.pt')**: This line loads the YOLOv8 model pre-trained weights.
- **epochs**: This parameter specifies how many times the model will train over the entire dataset.
- **imgsz**: Defines the image resolution. You can increase or decrease it based on your machine's capabilities.
---

## **Part 2: Running the Project on Lehmus (HPC Cluster)**

### **Step-by-Step Instructions for Lehmus**
If you have access of Lehmus you can run this but before that you have increased the size of Lehmus drive from 10 GB to minimum 100 GB. If can do it by sending a mail to ICT Oulu.
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
