import os
import matplotlib.pyplot as plt
import albumentations as A


class Augment():
    
    def __init__(self, dataset_dir, export_dir):
       self.dataset_dir = dataset_dir
       self.export_dir = export_dir
       self.original_images = os.listdir(dataset_dir)
       self.predicted = [txt.replace("images", "results") for txt in self.original_images]

        
    def augment(self, x):
        transform = A.Compose([
                A.CLAHE(),
                A.RandomRotate90(),
                A.Transpose(),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
                A.Blur(blur_limit=3),
                A.OpticalDistortion(),
                A.GridDistortion(),
                A.HueSaturationValue(),
            ])
    

    def show_1_image(self,image, title="Image"):
        plt.imshow(image)
        plt.title(title)
        plt.show()

    def show_before_after_images(self,image1, image2, title1="before", title2="after"):
        plt.subplot(1, 2, 1)
        plt.imshow(image1)
        plt.title(title1)
        plt.subplot(1, 2, 2)
        plt.imshow(image2)
        plt.title(title2)
        plt.show()
    
    def show_2_images(self):
     for i,j in zip(aug_obj.original_images, aug_obj.predicted):    
        image1 = plt.imread(os.path.join(aug_obj.dataset_dir, i))
        image2 = plt.imread(os.path.join(aug_obj.export_dir, j))
        self.show_before_after_images(image1, image2)    



if __name__ == '__main__':
    img_pth_origin = "images"
    img_pth_predicted = "results"
    aug_obj=Augment(img_pth_origin, img_pth_predicted)
    aug_obj.show_2_images()
   
        


