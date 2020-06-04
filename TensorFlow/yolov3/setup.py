import subprocess
import os
import urllib.request
import tarfile

# Set working directory to the directory containing this script.
os.chdir(os.path.dirname(os.path.realpath(__file__)))

def prepare_weights(name):
    checkpoint_index = os.path.join("checkpoints", f"{name}.tf.index")
    if not os.path.exists(checkpoint_index):
        weights_file = os.path.join("data", f"{name}.weights")
        if not os.path.exists(weights_file):
            print(f"Downloading pre-trained '{name}' weights...")
            url = f"https://pjreddie.com/media/files/{name}.weights"
            urllib.request.urlretrieve(url, weights_file)

        checkpoint_file = os.path.join("checkpoints", f"{name}.tf")
        is_tiny = name == "yolov3-tiny"
        print(f"Converting pre-trained '{name}' weights...")
        cl = " ".join((
            f"python convert.py",
            f"--weights {weights_file}",
            f"--output {checkpoint_file}",
            f"--tiny={is_tiny}",
        ))
        subprocess.run(cl, shell=True, check=True)

# Download and convert original pre-trained YOLO V3 weights.
prepare_weights("yolov3")
# prepare_weights("yolov3-tiny")

# Download and extract the VOC 2012 dataset.
voc_dataset_dir = os.path.join("data", "voc2012_raw")
if not os.path.exists(voc_dataset_dir):
    voc_dataset_file = os.path.join("data", "voc2012_raw.tar")
    if not os.path.exists(voc_dataset_file):
        print("Downloading VOC 2012 dataset...")
        url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
        urllib.request.urlretrieve(url, voc_dataset_file)

    print(f"Extracting {voc_dataset_file}")
    with tarfile.TarFile(voc_dataset_file) as f:
        f.extractall(voc_dataset_dir)

# Split VOC 2012 dataset into training and validation TFRecords.
def voc_convert_to_record(name):
    out_path = os.path.join("data", f"voc2012_{name}.tfrecord")
    if not os.path.exists(out_path):
        print(f"Creating TFRecord for '{name}' split...")
        voc_2012_dir = os.path.join(voc_dataset_dir, "VOCdevkit", "VOC2012")
        voc_2012_py = os.path.join("tools", "voc2012.py")
        cl = " ".join((
            f"python {voc_2012_py}",
            f"--data_dir {voc_2012_dir}",
            f"--split {name}",
            f"--output_file {out_path}",
        ))
        subprocess.run(cl, shell=True, check=True)

voc_convert_to_record("train")
voc_convert_to_record("val")