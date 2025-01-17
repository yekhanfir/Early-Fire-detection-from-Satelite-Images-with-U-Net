from segmentation_models_pytorch import Unet

def get_unet_model():
    """
    Define and return the UNet model.
    """
    return Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )
