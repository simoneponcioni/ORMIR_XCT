from pathlib import Path
from struct import unpack

import numpy as np
import SimpleITK as sitk


class ImageReader:
    def __init__(self, image_path):
        self.image_path = image_path
        
    def get_AIM_ints(self, f):
        """Function by Glen L. Niebur, University of Notre Dame (2010)
        reads the integer data of an AIM file to find its header length
        
        Args:
            f: File object opened in binary mode
            
        Returns:
            tuple: Header integers from the AIM file
        """
        nheaderints = 32
        f.seek(0)
        binints = f.read(nheaderints * 4)
        header_int = unpack("=32i", binints)
        return header_int

    def read_image(self):
        """
        Reads an image file and returns it as a SimpleITK image along with scaling parameters.
        
        Returns:
            tuple: (sitk.Image, scaling, slope, intercept)
                - sitk.Image: The loaded and processed image
                - scaling (float): Scaling factor, None if not available
                - slope (float): Slope value for intensity transformation, None if not available
                - intercept (float): Intercept value for intensity transformation, None if not available
        """
        if '.AIM' in Path(self.image_path).suffix.upper():
            return self._read_aim()
        else:
            # Handle other image formats using SimpleITK
            image = sitk.ReadImage(self.image_path)
            return image, None, None, None
    
    def _read_aim(self):
        """
        Internal method to read AIM format files.
        
        Returns:
            tuple: (sitk.Image, scaling, slope, intercept)
        """
        with open(self.image_path, "rb") as f:
            header_ints = self.get_AIM_ints(f)
            
            # Determine AIM version and format
            if int(header_ints[5]) == 16:  # Version 020
                version = "020"
                format_code = header_ints[10]
                header_size = header_ints[2]
                dimensions = (header_ints[14], header_ints[15], header_ints[16])
            else:  # Version 030
                version = "030"
                format_code = header_ints[17]
                header_size = header_ints[8]
                dimensions = (header_ints[24], header_ints[26], header_ints[28])
            
            # Read format
            if format_code == 131074:
                dtype = np.int16
            elif format_code == 65537:
                dtype = np.int8
            else:
                raise ValueError(f"Unsupported AIM format code: {format_code}")
            
            # Read header information
            header = f.read(header_size)
            header_str = header.decode('ascii', errors='ignore')
            header_lines = header_str.split('\n')
            
            # Parse header for metadata
            scaling = None
            slope = None
            intercept = None
            # TODO: generalize this for other XCT resolutions
            spacing = [0.061, 0.061, 0.061]  # Default spacing for XCTII
            
            for line in header_lines:
                if "Scaled by factor" in line:
                    scaling = float(line.split()[-1])
                elif "Density: slope" in line:
                    slope = float(line.split()[-1])
                elif "Density: intercept" in line:
                    intercept = float(line.split()[-1])
                elif "Orig-ISQ-Dim-um" in line or "Orig-GOBJ-Dim-um" in line:
                    try:
                        dims_um = [float(s) for s in line.split() if s.replace('.','',1).isdigit()]
                        if len(dims_um) == 3:
                            spacing = [d/1000.0 for d in dims_um]  # Convert to mm
                    except (ValueError, IndexError):
                        pass
            
            # Read image data
            if version == "020":
                header_offset = 160 + header_size
            else:  # version == "030"
                header_offset = 280 + header_size
            
            f.seek(header_offset)
            data = np.frombuffer(f.read(), dtype=dtype)
            data = data.reshape(dimensions, order='F')
            
            # Convert to SimpleITK image
            image = sitk.GetImageFromArray(data)
            image.SetSpacing(spacing)
            
            # Apply padding and flipping as in the original code
            image = self._pad_and_flip_image(image)
            
            return image, scaling, slope, intercept
    
    def _pad_and_flip_image(self, image, pad_size=10):
        """
        Applies padding and flipping operations to maintain consistency with the original code.
        
        Args:
            image (sitk.Image): Input image
            pad_size (int): Size of padding to apply
            
        Returns:
            sitk.Image: Processed image
        """
        # Pad image
        pad_filter = sitk.ConstantPadImageFilter()
        pad_filter.SetConstant(0)
        pad_size = [pad_size] * image.GetDimension()
        pad_filter.SetPadLowerBound(pad_size)
        pad_filter.SetPadUpperBound(pad_size)
        padded_image = pad_filter.Execute(image)
        
        # Flip image
        flipped_image = sitk.Flip(padded_image, [True, False, False])
        
        return flipped_image
    
