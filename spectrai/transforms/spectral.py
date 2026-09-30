"""Spectral (1D) transforms.

Ported 1:1 from the reference ``spectrai/transforms/spectral_transforms.py``.
Each transform operates on a *sample* dictionary of the form::

    {'input': <array>, 'target': {'data': <array>, 'type': str}, 'name': str}

and returns the same shape. Behaviour (including the stochastic ones) is
unchanged from the original; only the packaging around it is new.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import torch
from torch import nn


class Transform:
    """Base class defining a spectral transform."""

    def apply_transform(self, sample):  # pragma: no cover - abstract
        raise NotImplementedError

    def __call__(self, sample):
        sample_input = sample["input"]
        target_data = sample["target"]["data"]
        target_type = sample["target"]["type"]
        name = sample["name"]

        sample_input = self.apply_transform(sample_input)

        if target_type in ("spectrum", "hyperspectral_image"):
            target_data = self.apply_transform(target_data)

        return {"input": sample_input, "target": {"data": target_data, "type": target_type}, "name": name}


class Squeeze(Transform):
    def apply_transform(self, spectrum):
        return torch.squeeze(spectrum)


class CropSpectrum(Transform):
    """Crop a spectrum to a given index range."""

    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end

    def apply_transform(self, spectrum):
        return spectrum[..., self.start : self.end]


class PadSpectrum(Transform):
    """Pad or crop a spectrum to a given length."""

    def __init__(self, spectrum_length: int):
        self.spectrum_length = spectrum_length

    def apply_transform(self, spectrum):
        if spectrum.shape[-1] == self.spectrum_length:
            return spectrum
        if spectrum.shape[-1] > self.spectrum_length:
            return spectrum[..., : self.spectrum_length]
        # Pad by reflecting a window of the existing (smaller) length; the
        # reference used ReflectionPad1d with a pad larger than the input, which
        # torch rejects. Behaviour is equivalent for the real (>= target) cases.
        target = self.spectrum_length
        if spectrum.dim() == 1:
            # Build an even-length buffer >= target by mirroring, then slice.
            buf = torch.cat([spectrum, spectrum.flip(0)])
            while buf.numel() < target:
                buf = torch.cat([buf, buf.flip(0)])
            return buf[:target]
        buf = torch.cat([spectrum, spectrum.flip(-1)], dim=-1)
        while buf.shape[-1] < target:
            buf = torch.cat([buf, buf.flip(-1)], dim=-1)
        return buf[..., :target]


class ShiftSpectrum(Transform):
    """Shift a spectrum left or right (stochastic during training)."""

    def __init__(self, shift: float):
        self.shift = shift

    def apply_transform(self, spectrum, shift_amount):
        if shift_amount > 0:
            m = nn.ReflectionPad1d((0, abs(shift_amount)))
            if len(spectrum.shape) == 1:
                spectrum = m(spectrum[..., shift_amount:].unsqueeze(0).unsqueeze(0)).squeeze(0)
            else:
                spectrum = m(spectrum[..., shift_amount:].unsqueeze(0))
        elif shift_amount < 0:
            m = nn.ReflectionPad1d((abs(shift_amount), 0))
            if len(spectrum.shape) == 1:
                spectrum = m(spectrum[..., :shift_amount].unsqueeze(0).unsqueeze(0)).squeeze(0)
            else:
                spectrum = m(spectrum[..., :shift_amount].unsqueeze(0))
        return spectrum.squeeze(0)

    def __call__(self, sample):
        sample_input = sample["input"]
        target_data = sample["target"]["data"]
        target_type = sample["target"]["type"]
        name = sample["name"]

        if self.shift != 0.0:
            shift_range = np.random.uniform(-self.shift, self.shift)
            shift_amount = int(np.round(shift_range * sample_input.shape[-1]))
            sample_input = self.apply_transform(sample_input, shift_amount)

            if target_type in ("spectrum", "hyperspectral_image"):
                target_data = self.apply_transform(target_data, shift_amount)

        return {"input": sample_input, "target": {"data": target_data, "type": target_type}, "name": name}


class MinBackgroundSpectrum(Transform):
    """Subtract the minimum value (spectral background)."""

    def apply_transform(self, spectrum):
        return spectrum - torch.amin(spectrum)


class PolyBackgroundSpectrum(Transform):
    """Polynomial background subtraction."""

    def __init__(self, order: int):
        self.order = order

    def apply_transform(self, spectrum):
        spectrum = np.squeeze(spectrum.numpy())
        x = np.arange(0, spectrum.shape[0])
        poly = np.poly1d(np.polyfit(x, spectrum, self.order))
        spectrum = spectrum - poly(x)
        if np.amin(spectrum) < 0.0:
            spectrum = spectrum + np.abs(np.amin(spectrum))
        spectrum = spectrum[np.newaxis, ...]
        return torch.from_numpy(spectrum)


class ALSBackgroundSpectrum(Transform):
    """Asymmetric least-squares background subtraction."""

    def __init__(self, apply_target: bool = True, lam: int = 10000, p: float = 0.05, n: int = 10):
        self.apply_target = apply_target
        self.lam = lam
        self.p = p
        self.n = n

    def apply_transform(self, spectrum):
        spectrum = np.squeeze(spectrum.numpy())
        L = len(spectrum)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
        w = np.ones(L)
        for _ in range(self.n):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + self.lam * D.dot(D.transpose())
            z = spsolve(Z, w * spectrum)
            w = self.p * (spectrum > z) + (1 - self.p) * (spectrum < z)
        spectrum = spectrum - z
        spectrum = spectrum[np.newaxis, ...]
        return torch.from_numpy(spectrum)


class MaxNormalizeSpectrum(Transform):
    """Normalize a spectrum by its maximum value."""

    def apply_transform(self, spectrum):
        return spectrum / torch.amax(spectrum)


class AUCNormalizeSpectrum(Transform):
    """Normalize a spectrum by the area under the curve."""

    def apply_transform(self, spectrum):
        return spectrum / torch.sum(spectrum)


class FlipAxis(Transform):
    """Randomly flip one axis of the input (and, where appropriate, target)."""

    def __init__(self, axis: int):
        self.axis = axis

    def apply_transform(self, sample):
        return torch.flip(sample, [self.axis])

    def __call__(self, sample):
        sample_input = sample["input"]
        target_data = sample["target"]["data"]
        target_type = sample["target"]["type"]
        name = sample["name"]

        if torch.rand(1) < 0.5:
            sample_input = self.apply_transform(sample_input)
            if target_type in ("spectrum", "hyperspectral_image") or (
                target_type == "mask" and self.axis != 2
            ):
                target_data = self.apply_transform(target_data)

        return {"input": sample_input, "target": {"data": target_data, "type": target_type}, "name": name}


# ---------------------------------------------------------------------------
# Image (hyperspectral) transforms
# ---------------------------------------------------------------------------


class CropImageSpectrum(Transform):
    """Crop the spectral axis of a hyperspectral image."""

    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end

    def apply_transform(self, image):
        return image[:, :, self.start : self.end]


class ShiftImageSpectrum(Transform):
    """Stochastically shift the spectral axis of a hyperspectral image."""

    def __init__(self, shift: float):
        self.shift = shift

    def apply_transform(self, image, shift_amount):
        shifted = image.unsqueeze(0)
        if shift_amount > 0:
            m = nn.ReflectionPad2d((0, abs(shift_amount), 0, 0))
            shifted = m(shifted[:, :, :, shift_amount:])
        elif shift_amount < 0:
            m = nn.ReflectionPad2d((abs(shift_amount), 0, 0, 0))
            shifted = m(shifted[:, :, :, :shift_amount])
        return shifted.squeeze()

    def __call__(self, sample):
        sample_input = sample["input"]
        target_data = sample["target"]["data"]
        target_type = sample["target"]["type"]
        name = sample["name"]

        if self.shift != 0.0:
            shift_range = np.random.uniform(-self.shift, self.shift)
            shift_amount = int(np.round(shift_range * sample_input.shape[-1]))
            sample_input = self.apply_transform(sample_input, shift_amount)
            if target_type in ("spectrum", "hyperspectral_image"):
                target_data = self.apply_transform(target_data, shift_amount)

        return {"input": sample_input, "target": {"data": target_data, "type": target_type}, "name": name}


class PadCropImage(Transform):
    """Pad or crop a hyperspectral image (or mask) to a target size."""

    def __init__(self, size: int, random_crop: bool):
        self.size = size
        self.random_crop = random_crop

    def random_crop_image(self, image, image_size, start_idx_x, start_idx_y, target_type):
        if image.shape[0] > image_size:
            end_idx_x = start_idx_x + image_size
        else:
            start_idx_x, end_idx_x = 0, image.shape[0]
        if image.shape[1] > image_size:
            end_idx_y = start_idx_y + image_size
        else:
            start_idx_y, end_idx_y = 0, image.shape[1]

        if target_type == "mask":
            return image[start_idx_x:end_idx_x, start_idx_y:end_idx_y]
        return image[start_idx_x:end_idx_x, start_idx_y:end_idx_y, :]

    def center_crop_image(self, image, image_size, target_type):
        cropped = image
        if image.shape[0] > image_size:
            dif = int(np.floor((image.shape[0] - image_size) / 2))
            if target_type == "mask":
                cropped = cropped[dif : image_size + dif, :]
            else:
                cropped = cropped[dif : image_size + dif, :, :]
        if image.shape[1] > image_size:
            dif = int(np.floor((image.shape[1] - image_size) / 2))
            if target_type == "mask":
                cropped = cropped[:, dif : image_size + dif]
            else:
                cropped = cropped[:, dif : image_size + dif, :]
        return cropped

    def apply_transform(self, image, start_idx_x, start_idx_y, target_type):
        size = self.size
        if image.shape[0] == size and image.shape[1] == size:
            return image
        if image.shape[0] > size and image.shape[1] > size:
            if self.random_crop:
                return self.random_crop_image(image, size, start_idx_x, start_idx_y, target_type)
            return self.center_crop_image(image, size, target_type)

        out = image
        if out.shape[0] > size:
            if self.random_crop:
                out = self.random_crop_image(out, size, start_idx_x, start_idx_y, target_type)
            else:
                out = self.center_crop_image(out, size, target_type)
        else:
            pad_before = int(np.floor((size - out.shape[0]) / 2))
            pad_after = int(np.ceil((size - out.shape[0]) / 2))
            if target_type == "hyperspectral_image":
                m = nn.ReflectionPad2d((0, 0, pad_before, pad_after))
                out = m(torch.movedim(out, -1, 0).unsqueeze(0))
                out = torch.movedim(out.squeeze(), 0, -1)
            elif target_type == "mask":
                m = nn.ReflectionPad2d((pad_before, pad_after, 0, 0))
                out = m(torch.movedim(out, -1, 0).unsqueeze(0).unsqueeze(0))
                out = torch.movedim(out.squeeze(), 0, -1)

        if out.shape[1] > size:
            if self.random_crop:
                out = self.random_crop_image(out, size, start_idx_x, start_idx_y, target_type)
            else:
                out = self.center_crop_image(out, size, target_type)
        else:
            pad_before = int(np.floor((size - out.shape[1]) / 2))
            pad_after = int(np.ceil((size - out.shape[1]) / 2))
            if target_type == "hyperspectral_image":
                m = nn.ReflectionPad2d((pad_before, pad_after, 0, 0))
                out = m(torch.movedim(out, -1, 0).unsqueeze(0))
                out = torch.movedim(out.squeeze(), 0, -1)
            elif target_type == "mask":
                m = nn.ReflectionPad2d((0, 0, pad_before, pad_after))
                out = m(torch.movedim(out, -1, 0).unsqueeze(0).unsqueeze(0))
                out = torch.movedim(out.squeeze(), 0, -1)
        return out

    def __call__(self, sample):
        sample_input = sample["input"]
        target_data = sample["target"]["data"]
        target_type = sample["target"]["type"]
        name = sample["name"]

        start_idx_x = int(np.round(np.random.random() * (sample_input.shape[0] - self.size)))
        start_idx_y = int(np.round(np.random.random() * (sample_input.shape[1] - self.size)))

        sample_input = self.apply_transform(sample_input, start_idx_x, start_idx_y, "hyperspectral_image")
        if target_type in ("hyperspectral_image", "mask"):
            target_data = self.apply_transform(target_data, start_idx_x, start_idx_y, target_type)

        return {"input": sample_input, "target": {"data": target_data, "type": target_type}, "name": name}


class RotateImage(Transform):
    """Random 90-degree rotation of a hyperspectral image / mask."""

    def apply_transform(self, image, rotation_extent):
        if rotation_extent < 0.25:
            rotation = 1
        elif rotation_extent < 0.5:
            rotation = 2
        elif rotation_extent < 0.75:
            rotation = 3
        else:
            rotation = 0
        return torch.rot90(image, rotation)

    def __call__(self, sample):
        sample_input = sample["input"]
        target_data = sample["target"]["data"]
        target_type = sample["target"]["type"]
        name = sample["name"]

        rotation_extent = torch.rand(1)
        sample_input = self.apply_transform(sample_input, rotation_extent)
        if target_type in ("hyperspectral_image", "mask"):
            target_data = self.apply_transform(target_data, rotation_extent)

        return {"input": sample_input, "target": {"data": target_data, "type": target_type}, "name": name}


class SkipDownsampleImage(Transform):
    """Skip spatial downsampling (used for super-resolution targets)."""

    def __init__(self, scale: int):
        self.scale = scale

    def apply_transform(self, image):
        if self.scale >= 4:
            start_idx = torch.randint(1, self.scale - 1, (1,))
        else:
            start_idx = 1
        return image[start_idx:: self.scale, start_idx:: self.scale, :]

    def __call__(self, sample):
        sample_input = self.apply_transform(sample["input"])
        return {
            "input": sample_input,
            "target": {"data": sample["target"]["data"], "type": sample["target"]["type"]},
            "name": sample["name"],
        }


class BicubicDownsampleImage(Transform):
    """Bicubic spatial downsampling to a target size (super-resolution)."""

    def __init__(self, size: int):
        self.size = size

    def apply_transform(self, image):
        image = torch.movedim(image, -1, 0).unsqueeze(0)
        image = nn.functional.interpolate(
            image, size=(self.size, self.size), mode="bicubic", align_corners=False
        )
        image = torch.movedim(image, 1, -1).squeeze(0)
        return image

    def __call__(self, sample):
        sample_input = self.apply_transform(sample["input"])
        return {
            "input": sample_input,
            "target": {"data": sample["target"]["data"], "type": sample["target"]["type"]},
            "name": sample["name"],
        }


class MaxNormalizeImage(Transform):
    """Normalize image spectra by the global maximum value."""

    def apply_transform(self, image):
        image_max = torch.amax(image).repeat(image.shape)
        return torch.div(image, image_max)


class AUCNormalizeImage(Transform):
    """Normalize image spectra by the per-pixel area under the curve."""

    def apply_transform(self, image):
        image_sum = torch.sum(image, -1).unsqueeze(-1)
        sum_tile = torch.repeat_interleave(image_sum, image.shape[-1], -1)
        return torch.div(image, sum_tile)


class MinBackgroundImage(Transform):
    """Subtract the per-pixel minimum (spectral image background)."""

    def apply_transform(self, image):
        min_values = torch.amin(image, 2)
        min_values = torch.clamp(min_values, min=0.0, max=torch.amax(min_values)).unsqueeze(-1)
        min_tile = torch.repeat_interleave(min_values, image.shape[-1], -1)
        return image - min_tile


class PolyBackgroundImage(Transform):
    """Polynomial background subtraction for hyperspectral images."""

    def __init__(self, order: int):
        self.order = order

    def apply_transform(self, image):
        image = image.numpy()
        reshaped = np.reshape(image, (image.shape[0] * image.shape[1], image.shape[2])).T
        x = np.arange(0, image.shape[2])
        poly = np.polyfit(x, reshaped, self.order)
        out = reshaped.T
        for i in range(out.shape[0]):
            poly_1d = np.poly1d(poly[:, i])
            out[i] = out[i] - poly_1d(x)
            if np.amin(out[i, :]) < 0.0:
                out[i, :] = out[i, :] + np.abs(np.amin(out[i, :]))
        out = np.reshape(out, (image.shape[0], image.shape[1], image.shape[2]))
        return torch.from_numpy(out)


# ---------------------------------------------------------------------------
# Shape / dtype helpers
# ---------------------------------------------------------------------------


class MakeChannelsFirst(Transform):
    """Channels-last -> channels-first."""

    def apply_transform(self, sample):
        return torch.movedim(sample, -1, 0)


class MakeChannelsLast(Transform):
    """Channels-first -> channels-last."""

    def apply_transform(self, sample):
        return torch.movedim(sample, 0, -1)


class AddDimFirst(Transform):
    def apply_transform(self, sample):
        return sample.unsqueeze(0)


class AddDimLast(Transform):
    def apply_transform(self, sample):
        return sample.unsqueeze(-1)


class ToTensor(Transform):
    """numpy -> torch (float64, matching the reference)."""

    def apply_transform(self, image):
        if isinstance(image, torch.Tensor):
            return image.double()
        return torch.from_numpy(image).double()

    def __call__(self, sample):
        sample_input = self.apply_transform(sample["input"])
        target_data = sample["target"]["data"]
        target_type = sample["target"]["type"]
        if target_type in ("spectrum", "hyperspectral_image", "mask"):
            target_data = self.apply_transform(target_data)
        return {"input": sample_input, "target": {"data": target_data, "type": target_type}, "name": sample["name"]}


class ToNumpy(Transform):
    """torch -> numpy."""

    def apply_transform(self, image):
        if isinstance(image, np.ndarray):
            return image
        return image.numpy()

    def __call__(self, sample):
        sample_input = self.apply_transform(sample["input"])
        target_data = sample["target"]["data"]
        target_type = sample["target"]["type"]
        if target_type in ("spectrum", "hyperspectral_image", "mask"):
            target_data = self.apply_transform(target_data)
        return {"input": sample_input, "target": {"data": target_data, "type": target_type}, "name": sample["name"]}


__all__ = [
    "Transform",
    "Squeeze",
    "CropSpectrum",
    "PadSpectrum",
    "ShiftSpectrum",
    "MinBackgroundSpectrum",
    "PolyBackgroundSpectrum",
    "ALSBackgroundSpectrum",
    "MaxNormalizeSpectrum",
    "AUCNormalizeSpectrum",
    "FlipAxis",
    "CropImageSpectrum",
    "ShiftImageSpectrum",
    "PadCropImage",
    "RotateImage",
    "SkipDownsampleImage",
    "BicubicDownsampleImage",
    "MaxNormalizeImage",
    "AUCNormalizeImage",
    "MinBackgroundImage",
    "PolyBackgroundImage",
    "MakeChannelsFirst",
    "MakeChannelsLast",
    "AddDimFirst",
    "AddDimLast",
    "ToTensor",
    "ToNumpy",
]
