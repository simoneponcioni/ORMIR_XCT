from struct import unpack
import numpy as np
import SimpleITK as sitk
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy  # type: ignore


class ImageReader:
    def __init__(self, image_path):
        """
        Initialize the ImageReader with a path to an image file.

        Args:
            image_path (str): Path to the image file
        """
        self.image_path = image_path
        self.image_path_str = str(image_path.resolve())

    def get_AIM_ints(self, f):
        """Function by Glen L. Niebur, University of Notre Dame (2010)
        reads the integer data of an AIM file to find its header length"""
        nheaderints = 32
        f.seek(0)
        binints = f.read(nheaderints * 4)
        header_int = unpack("=32i", binints)
        return header_int

    def read_image(self):
        """
        Read the image file and return the image and associated metadata.

        Returns:
            tuple: (sitk.Image, scaling, mu_water, rescale_slope, rescale_intercept)
        """
        return self._read_aim()

    def _read_aim(self):
        """
        Reads an AIM file and provides the corresponding VTK image along with spacing, calibration data, and header information.
        """
        # read header
        print(self.image_path.name)
        with open(self.image_path, "rb") as f:
            AIM_ints = self.get_AIM_ints(f)
            # check AIM version
            if int(AIM_ints[5]) == 16:
                print("     -> version 020")
                if int(AIM_ints[10]) == 131074:
                    format = "short"
                    print("     -> format " + format)
                elif int(AIM_ints[10]) == 65537:
                    format = "char"
                    print("     -> format " + format)
                elif int(AIM_ints[10]) == 1376257:
                    format = "bin compressed"
                    print("     -> format " + format + " not supported! Exiting!")
                    exit(1)
                else:
                    format = "unknown"
                    print("     -> format " + format + "! Exiting!")
                    exit(1)
                header = f.read(AIM_ints[2])
                header_len = len(header) + 160
                extents = (
                    0,
                    AIM_ints[14] - 1,
                    0,
                    AIM_ints[15] - 1,
                    0,
                    AIM_ints[16] - 1,
                )
            else:
                print("     -> version 030")
                if int(AIM_ints[17]) == 131074:
                    format = "short"
                    print("     -> format " + format)
                elif int(AIM_ints[17]) == 65537:
                    format = "char"
                    print("     -> format " + format)
                elif int(AIM_ints[17]) == 1376257:
                    format = "bin compressed"
                    print("     -> format " + format + " not supported! Exiting!")
                    exit(1)
                else:
                    format = "unknown"
                    print("     -> format " + format + "! Exiting!")
                    exit(1)
                header = f.read(AIM_ints[8])
                header_len = len(header) + 280
                extents = (
                    0,
                    AIM_ints[24] - 1,
                    0,
                    AIM_ints[26] - 1,
                    0,
                    AIM_ints[28] - 1,
                )

        # collect data from header if existing
        # header = re.sub('(?i) +', ' ', header)
        header = header.split(b"\n")
        header.pop(0)
        header.pop(0)
        header.pop(0)
        header.pop(0)
        scaling = None
        slope = None
        intercept = None
        mu_water = None
        IPLPostScanScaling = 1
        for line in header:
            if line.find(b"Orig-ISQ-Dim-p") > -1:
                origdimp = [int(s) for s in line.split(b" ") if s.isdigit()]

            if line.find("Orig-ISQ-Dim-um".encode()) > -1:
                origdimum = [int(s) for s in line.split(b" ") if s.isdigit()]

            if line.find("Orig-GOBJ-Dim-p".encode()) > -1:
                origdimp = [int(s) for s in line.split(b" ") if s.isdigit()]

            if line.find("Orig-GOBJ-Dim-um".encode()) > -1:
                origdimum = [int(s) for s in line.split(b" ") if s.isdigit()]

            if line.find("Scaled by factor".encode()) > -1:
                scaling = float(line.split(" ".encode())[-1])
            if line.find("Density: intercept".encode()) > -1:
                intercept = float(line.split(" ".encode())[-1])
            if line.find("Density: slope".encode()) > -1:
                slope = float(line.split(" ".encode())[-1])
                # if el_size scale was applied, the above still takes the original
                # voxel size. This function works only if an isotropic scaling
                # is applied!
            if line.find("HU: mu water".encode()) > -1:
                mu_water = float(line.split(" ".encode())[-1])
            if line.find("downscaled".encode()) > -1:
                # avoids that the filename 'downscaled' is interpreted as
                # a scaling factor
                pass
            elif line.find("scale".encode()) > -1:
                IPLPostScanScaling = float(line.split(" ".encode())[-1])
        # Spacing is calculated from Original Dimensions. This is wrong, when
        # the images were coarsened and the voxel size is not anymore
        # corresponding to the original scanning resolution!

        try:
            spacing = IPLPostScanScaling * (
                np.around(np.asarray(origdimum) / np.asarray(origdimp) / 1000, 5)
            )
        except UnboundLocalError:
            pass

        # read AIM
        reader = vtk.vtkImageReader2()
        reader.SetFileName(self.image_path_str)
        reader.SetDataByteOrderToLittleEndian()
        reader.SetFileDimensionality(3)
        reader.SetDataExtent(extents)
        reader.SetHeaderSize(header_len)
        if format == "short":
            reader.SetDataScalarTypeToShort()
        elif format == "char":
            reader.SetDataScalarTypeToChar()
        reader.SetDataSpacing(spacing)
        reader.Update()
        imvtk = reader.GetOutput()

        # Convert VTK to SimpleITK
        vtk_array = imvtk.GetPointData().GetScalars()
        np_array = vtk_to_numpy(vtk_array)
        dimensions = imvtk.GetDimensions()
        np_array = np_array.reshape(dimensions, order="F")
        image = sitk.GetImageFromArray(np_array)
        image.SetSpacing(spacing)
        image = sitk.PermuteAxes(image, [2, 1, 0])
        return image, scaling, mu_water, slope, intercept
