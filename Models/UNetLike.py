import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn


import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetLike2(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, init_features=32):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            init_features (int): Number of features in the first layer.
        """
        super(UNetLike2, self).__init__()

        # Encoder (Downsampling Path)
        self.encoder1 = self._block(in_channels, init_features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self._block(init_features, init_features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self._block(init_features * 2, init_features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self._block(init_features * 4, init_features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = self._block(init_features * 8, init_features * 16, name="bottleneck")

        # Decoder (Upsampling Path)
        self.upconv4 = nn.ConvTranspose2d(init_features * 16, init_features * 8, kernel_size=2, stride=2)
        self.decoder4 = self._block(init_features * 16, init_features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(init_features * 8, init_features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block(init_features * 8, init_features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(init_features * 4, init_features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block(init_features * 4, init_features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(init_features * 2, init_features, kernel_size=2, stride=2)
        self.decoder1 = self._block(init_features * 2, init_features, name="dec1")

        # Final layer
        self.final_conv = nn.Conv2d(init_features, out_channels, kernel_size=1)

    def forward(self, x):
        original_shape = x.shape[2:]  # Save input spatial dimensions

        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder with skip connections and cropping
        dec4 = self.upconv4(bottleneck)
        enc4_cropped = self.crop_to_match(enc4, dec4)
        dec4 = torch.cat((dec4, enc4_cropped), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        enc3_cropped = self.crop_to_match(enc3, dec3)
        dec3 = torch.cat((dec3, enc3_cropped), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        enc2_cropped = self.crop_to_match(enc2, dec2)
        dec2 = torch.cat((dec2, enc2_cropped), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        enc1_cropped = self.crop_to_match(enc1, dec1)
        dec1 = torch.cat((dec1, enc1_cropped), dim=1)
        dec1 = self.decoder1(dec1)

        out = self.final_conv(dec1)

        # Final interpolation to ensure output matches input shape exactly.
        out = F.interpolate(out, size=original_shape, mode='bilinear', align_corners=False)
        return out

    def crop_to_match(self, enc_tensor, dec_tensor):
        """
        Crop the encoder tensor to match the spatial dimensions of the decoder tensor.
        """
        _, _, h_dec, w_dec = dec_tensor.size()
        _, _, h_enc, w_enc = enc_tensor.size()
        delta_h = h_enc - h_dec
        delta_w = w_enc - w_dec
        crop_top = delta_h // 2
        crop_left = delta_w // 2
        return enc_tensor[:, :, crop_top: h_enc - (delta_h - crop_top),
                          crop_left: w_enc - (delta_w - crop_left)]

    def _block(self, in_channels, features, name):
        """
        Helper function to create a convolutional block.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )


class UNetLike(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, init_features=32):
        """
        Args:
            in_channels (int): Number of input channels (default: 1 for single-channel images).
            out_channels (int): Number of output channels (customizable).
            init_features (int): Number of features in the first layer (default: 32).
        """
        super(UNetLike, self).__init__()

        # Encoder (Downsampling Path)
        self.encoder1 = self._block(in_channels, init_features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self._block(init_features, init_features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self._block(init_features * 2, init_features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self._block(init_features * 4, init_features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = self._block(init_features * 8, init_features * 16, name="bottleneck")

        # Decoder (Upsampling Path)
        self.upconv4 = nn.ConvTranspose2d(init_features * 16, init_features * 8, kernel_size=2, stride=2)
        self.decoder4 = self._block(init_features * 16, init_features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(init_features * 8, init_features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block(init_features * 8, init_features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(init_features * 4, init_features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block(init_features * 4, init_features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(init_features * 2, init_features, kernel_size=2, stride=2)
        self.decoder1 = self._block(init_features * 2, init_features, name="dec1")

        # Final layer
        self.final_conv = nn.Conv2d(init_features, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        print("dec4 shape:", dec4.shape)
        print("enc4 shape:", enc4.shape)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        # Final layer
        out = self.final_conv(dec1)
        return out

    def _block(self, in_channels, features, name):
        """
        Helper function to create a convolutional block.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )
        

if __name__ == "__main__":
    # Initialize the model
    model = UNetLike2(in_channels=1, out_channels=3, init_features=32)
    
    x = torch.randn((2, 1, 62, 80))
    out = model(x)
    print("Output shape:", out.shape)
    
    # # Generate some random input
    x = torch.randn((2, 1, 150, 200))
    out = model(x)
    print("Output shape:", out.shape)

    # # Generate some random input
    x = torch.randn((2, 1, 120, 160))
    out = model(x)
    print("Output shape:", out.shape)
    