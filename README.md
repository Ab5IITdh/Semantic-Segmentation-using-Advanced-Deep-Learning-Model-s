# Semantic-Segmentation-using-Advanced-Deep-Learning-Model-s
Semantic segmentation is a fundamental task in computer vision ,In this repository we will explore through Advanced model Architectures

# 1. DeepLabv3_MobileNet-Pytorch

Pretrained DeepLabv3 for Kitto Road Datset.

## Quick Start 

### 1. Available Architectures
| DeepLabV3    |  DeepLabV3+        |
| :---: | :---:     |
|deeplabv3_mobilenet|deeplabv3plus_mobilenet ||



please refer to [network/modeling.py](https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/master/network/modeling.py) for all model entries.

Download pretrained models: [Dropbox](https://www.dropbox.com/sh/w3z9z8lqpi8b2w7/AAB0vkl4F5vy6HdIhmRCTKHSa?dl=0)


### 2. Visualize segmentation outputs:
```python
def visualize_samples(dataset, num_samples=3):
    """Display a few samples from the dataset with an overlay of the segmentation mask."""
    for i in range(num_samples):
        image, mask = dataset[i]
        
        # Convert image tensor (C, H, W) to a NumPy array (H, W, C) and de-normalize.
        image_np = image.numpy().transpose(1, 2, 0)
        image_np = image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        image_np = np.clip(image_np, 0, 1)
        
        # Convert mask tensor to NumPy array.
        mask_np = mask.numpy()
        
        # Create an overlay: highlight road pixels (mask == 1) in red.
        overlay = image_np.copy()
        # Define the color for road (red) and the transparency factor.
        road_color = np.array([1, 0, 0])  # Red in normalized RGB.
        alpha = 0.5  # Transparency factor.
        # For each pixel where mask==1, blend the original image with the red color.
        overlay[mask_np == 1] = (1 - alpha) * overlay[mask_np == 1] + alpha * road_color
        
        # Plot the original image, the binary mask (with a colormap), and the overlay.
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(image_np)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        # Using a colormap like 'viridis' to better highlight differences.
        plt.imshow(mask_np, cmap='viridis')
        plt.title('Segmentation Mask')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(overlay)
        plt.title('Overlay')
        plt.axis('off')
        
        plt.show()

# Visualize a few training samples.
visualize_samples(train_dataset, num_samples=3)
```

Image folder:
```bash
python predict.py --input datasets/data/kitti_Road_Dataset/leftImg8bit/train/bremen  --dataset Kitti_Road --model deeplabv3plus_mobilenet --ckpt checkpoints/best_deeplabv3plus_mobilenet_Kitti_Road_os16.pth --save_val_results_to test_results
```

### 3. New backbones

Please refer to [this commit (Xception)](https://github.com/VainF/DeepLabV3Plus-Pytorch/commit/c4b51e435e32b0deba5fc7c8ff106293df90590d) for more details about how to add new backbones.

### 4. New datasets

You can train deeplab models on your own datasets. Your ``torch.utils.data.Dataset`` should provide a decoding method that transforms your predictions to colorized images:
```python
#  Augmentations
transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2(),
],is_check_shapes=False)

#  Load Dataset
train_dataset = KittiDataset(
    image_dir="/kaggle/input/kittiroadsegmentation/training/image_2",
    mask_dir="/kaggle/input/kittiroadsegmentation/training/gt_image_2",
    transform=transform
)
print(train_dataset.__len__())
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
```


## Results

### 1. Performance on Pascal VOC2012 Aug (21 classes, 513 x 513)

Training: 513x513 random crop  
validation: 513x513 center crop

|  Model          | Batch Size  | FLOPs  | train/val OS   |  mIoU        | Dropbox  | Tencent Weiyun  | 
| :--------        | :-------------: | :----:   | :-----------: | :--------: | :--------: | :----:   |
| DeepLabV3-MobileNet       | 16      |  6.0G      |   16/16  |  0.701     |    [Download](https://www.dropbox.com/s/uhksxwfcim3nkpo/best_deeplabv3_mobilenet_voc_os16.pth?dl=0)       | [Download](https://share.weiyun.com/A4ubD1DD) |
### 2. Performance on Cityscapes (19 classes, 1024 x 2048)

Training: 768x768 random crop  
validation: 1024x2048

|  Model          | Batch Size  | FLOPs  | train/val OS   |  mIoU        | Dropbox  |  Tencent Weiyun  |
| :--------        | :-------------: | :----:   | :-----------: | :--------: | :--------: |  :----:   |
| DeepLabV3Plus-MobileNet   | 16      |  135G      |  16/16   |  0.721  |    [Download](https://www.dropbox.com/s/753ojyvsh3vdjol/best_deeplabv3plus_mobilenet_cityscapes_os16.pth?dl=0) | [Download](https://share.weiyun.com/aSKjdpbL) 

#### Segmentation Results on Kitti Road Dataset (DeepLabv3Plus-MobileNet)
![image](https://github.com/user-attachments/assets/8c299488-60ff-4b74-959b-cd87fec84f7e)


### 2. Prepare Datasets


#### 2.2  Kitti Road Segmentation (Recommended!!)

The dataset comprises two main parts: a training set with annotated ground truth and a testing set for performance evaluation (https://www.cvlibs.net/download.php?file=data_road.zip).

#### 1. **Data Composition** 

The dataset comprises two main parts: a training set with annotated ground truth and a testing set for performance evaluation.

- **Training Set:**
  - **Images:**  
    The training images are stored in a folder typically named image_2 within the training directory. These images capture a variety of urban road scenes under different conditions.
  
  - **Ground Truth Annotations:**  
    Accompanying the images, the ground truth masks are provided in the gt_image_2 folder. These annotations are manually created and mark the road regions (and in some cases, individual lanes) with specific labels. The ground truth masks are often binary (or multi-class in more detailed annotations) where pixels corresponding to the road are labeled with a distinct value (for example, 1 or 255) and the background is labeled as 0.

- **Testing Set:**
  - The testing data is organized in a similar fashion but typically only includes the raw images (without ground truth annotations). This enables an unbiased evaluation of segmentation models on unseen data.

#### 3. **Image Categories**

The dataset contains images that represent diverse urban road conditions. The naming conventions of the files hint at different scene types:

- **Urban Unmarked (um):**  
  Images with the prefix um_ (e.g., um_000000.png to um_000094.png) represent road scenes where lanes are not marked with explicit line segments.

- **Urban Marked (uu):**  
  Images with the prefix uu_ (e.g., uu_000000.png to uu_000097.png) depict scenes where roads have clear lane markings.

- **Urban Multiple Marked Lanes (umm):**  
  Images with the prefix umm_ (e.g., umm_000000.png to umm_000095.png) contain more complex scenarios with multiple lane markings. These scenes challenge segmentation algorithms to correctly identify several drivable lanes.

#### 4. **File Naming Conventions and Structure**

The careful naming of files in the dataset supports easy pairing between images and their corresponding annotations:

- **Training Folder Structure:**
  - **Images (training/image_2):**  
    The images are sequentially numbered and grouped by type. For example:
    - um_000000.png to um_000094.png (urban unmarked)
    - umm_000000.png to umm_000095.png (urban multiple marked lanes)
    - uu_000000.png to uu_000097.png (urban marked)
  
  - **Annotations (training/gt_image_2):**  
    The ground truth masks follow a similar naming pattern but with identifiers that indicate the type of annotation:
    - Files such as um_lane_000000.png to um_lane_000094.png may provide lane-specific annotations for urban unmarked scenes.
    - Similarly, files like umm_road_000000.png to umm_road_000095.png and uu_road_000000.png to uu_000097.png are used for the other respective categories.
    
    This naming convention ensures that each image from image_2 has a corresponding mask in gt_image_2, thereby simplifying the process of dataset creation and model training.

- **Testing Folder Structure:**
  - **Images (testing/image_2/image_2):**  
    Test images are organized in a nested image_2 folder and follow a sequential naming similar to the training set:
    - For instance, um_000000.png to um_000095.png, umm_000000.png to umm_000093.png, and uu_000000.png to uu_000099.png.
  
  The testing folder is used during model evaluation, where the segmentation algorithm’s output can be compared against the ground truth (if available in a validation subset) or visually inspected.

### 2. Train your model on Kitti Road Segmentation

```bash
python main.py --model deeplabv3plus_mobilenet --dataset kitti_Road --enable_vis --vis_port 28333 --gpu_id 0  --lr 0.1  --crop_size 768 --batch_size 16 --output_stride 16 --data_root ./datasets/data/kitti_road 
```

## Reference

[1] [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)

[2] [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)
