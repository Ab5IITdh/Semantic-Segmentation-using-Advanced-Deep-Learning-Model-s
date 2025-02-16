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
### 2. Performance on Kitti-Road Datset
| Metric               | Value  |
|----------------------|--------|
| Class 0 IoU         | 0.5848 |
| Class 1 IoU         | 0.9935 |
| Mean IoU            | 0.8545 |
| Pixel Accuracy      | 0.9937 |
| Mean Pixel Accuracy | 0.8943 |



|  Model          | Batch Size  | train/val OS   |  mIoU        | Dropbox  |  Tencent Weiyun  |
| :--------        | :-------------: | :-----------: | :--------: | :--------: |  :----:   |
| DeepLabV3Plus-MobileNet   | 16      |  16/16   |  0.8545  |    [Download](https://www.dropbox.com/s/753ojyvsh3vdjol/best_deeplabv3plus_mobilenet_cityscapes_os16.pth?dl=0) | [Download](https://share.weiyun.com/aSKjdpbL) 

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

# 1. SWIN-Transformer_UNet Model Architecture-Pytorch  \
![image](https://github.com/user-attachments/assets/68bad3a1-26e5-456c-b97e-e0e6feb84837)
```python
###############################################################################
# Step 2: Load the Pretrained Swin-U-Net Model Architecture and Fine Tune
###############################################################################
# Note: The following is a simplified version of the Swin-U-Net architecture.
# For a complete implementation, please refer to the provided notebook link.
# The model below is adapted for segmentation on the KITTI Road Dataset.

class SwinUnet(nn.Module):
    def __init__(self, img_size=256, in_chans=3, num_classes=2, embed_dim=96):
        super(SwinUnet, self).__init__()
        # A simplified encoder (mimicking a patch embedding + convolutional encoder).
        self.encoder = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # Downsample by 2.
        )
        # A simple bottleneck block.
        self.bottleneck = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim*2),
            nn.ReLU(inplace=True)
        )
        # A simplified decoder using transpose convolution for upsampling.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim*2, embed_dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        # Final segmentation head.
        self.seg_head = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, x):
        enc = self.encoder(x)      # [B, embed_dim, H, W]
        bottleneck = self.bottleneck(enc)  # [B, embed_dim*2, H, W]
        dec = self.decoder(bottleneck)     # [B, embed_dim, 2*H, 2*W] (should match input resolution)
        out = self.seg_head(dec)           # [B, num_classes, H, W]
        return out

# Instantiate the model.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SwinUnet(img_size=256, in_chans=3, num_classes=2, embed_dim=96).to(device)

# Optionally load pretrained weights here if available.
# For example:
# model.load_state_dict(torch.load('path_to_pretrained_weights.pth'))

# Set up optimizer and loss criterion.
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()  # Suitable for segmentation (expects targets as LongTensor)
```

# Instantiate the model.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SwinUnet(img_size=256, in_chans=3, num_classes=2, embed_dim=96).to(device)

# Optionally load pretrained weights here if available.
# For example:
# model.load_state_dict(torch.load('path_to_pretrained_weights.pth'))

# Set up optimizer and loss criterion.
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()  # Suitable for segmentation (expects targets as LongTensor)

# Swin-Unet
[ECCVW2022] The codes for the work "Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation"(https://arxiv.org/abs/2105.05537). Our paper has been accepted by ECCV 2022 MEDICAL COMPUTER VISION WORKSHOP (https://mcv-workshop.github.io/). We updated the Reproducibility. I hope this will help you to reproduce the results.\
![image](https://github.com/user-attachments/assets/2c0ee201-603f-4084-a2a1-3711a1d163ef)

## 1. Download pre-trained swin transformer model (Swin-T)
* [Get pre-trained model in this link] (https://drive.google.com/drive/folders/1UC3XOoezeum0uck4KBVGa8osahs6rKUY?usp=sharing): Put pretrained Swin-T into folder "pretrained_ckpt/"

## 3. Environment

- Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

## 4. Train/Test

- Run the train script on Kitti Road Semantic Segmentation dataset. The batch size we used is 24. If you do not have enough GPU memory, the bacth size can be reduced to 12 or 6 to save memory.

- Train

```bash
sh train.sh or python train.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --root_path your DATA_DIR --max_epochs 150 --output_dir your OUT_DIR  --img_size 224 --base_lr 0.05 --batch_size 24
```

- Test 

```bash
sh test.sh or python test.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path your DATA_DIR --output_dir your OUT_DIR --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24
```
# Evaluate the model on the training set (as an example).
```python
model.eval()
all_ious = []
all_mean_iou = []
all_pixel_acc = []
all_mean_pixel_acc = []

with torch.no_grad():
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        for pred, mask in zip(preds, masks):
            ious, mean_iou, pix_acc, mean_pix_acc = compute_metrics(pred.cpu(), mask.cpu(), num_classes=2)
            all_ious.append(ious)
            all_mean_iou.append(mean_iou)
            all_pixel_acc.append(pix_acc)
            all_mean_pixel_acc.append(mean_pix_acc)

avg_ious = np.nanmean(all_ious, axis=0)
avg_mean_iou = np.nanmean(all_mean_iou)
avg_pixel_acc = np.mean(all_pixel_acc)
avg_mean_pixel_acc = np.nanmean(all_mean_pixel_acc)

print("\nEvaluation Metrics on Training Set:")
for cls in range(2):
    print(f"Class {cls} IoU: {avg_ious[cls]:.4f}")
print(f"Mean IoU: {avg_mean_iou:.4f}")
print(f"Pixel Accuracy: {avg_pixel_acc:.4f}")
print(f"Mean Pixel Accuracy: {avg_mean_pixel_acc:.4f}")
```
![image](https://github.com/user-attachments/assets/7d4123cb-efe7-4e99-98f0-ab674294c2ef)

## Reference

[1] [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)

[2] [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)

[3] [TransUnet](https://github.com/Beckschen/TransUNet)

[4] [SwinTransformer](https://github.com/microsoft/Swin-Transformer)
