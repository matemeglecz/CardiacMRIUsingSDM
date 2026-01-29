from . import mapping_utils
import numpy as np
from skimage.transform import resize
import torchvision.transforms as transforms
import random
import torch
from torch.utils.data import Dataset
from collections import namedtuple
import os

DATASET_MEAN = ..
DATASET_STD = ..

DATASET_MEAN_T1_mapping = ..
DATASET_STD_T1_mapping = ..

DATASET_MEAN_T2_mapping = ..
DATASET_STD_T2_mapping = ..

TEST_PATIENTS = [7, 21, 30, 33, 34, 37, 41, 58, 86, 110, 123, 135, 145, 148, 155, 163, 164, 172, 177, 183, 190, 191, 207, 212, 220]
VAL_PATIENTS  = [3, 4, 12, 14, 19, 23, 28, 35, 40, 46, 50, 55, 98, 107, 130, 137, 156, 162, 176, 182, 185, 197, 209, 213, 219]

SYN_PATIENTS = [181, 186, 84, 1748, 1812, 1821, 62, 1665, 196, 1978, 95, 57, 1750, 1708, 1773, 1622, 153, 1575, 1629, 109, 99, 1830, 1833, 1601, 1737, 1984, 127, 1584, 1860, 47, 18, 1814, 111, 1805, 221, 1613, 1603, 1587, 65, 1585, 1661, 1900, 1621, 120, 1699, 1614, 1963, 1715, 1726, 165, 171, 119, 1938, 1691, 170, 60, 5, 202, 1924, 1880, 1962, 1712, 1572, 143, 1969, 1616, 1793, 93, 1632, 1914, 1850, 1672, 64, 1677, 1829, 1893, 1878, 1945, 1890, 1970, 1656, 24, 73, 192, 1756, 1867, 105, 1954, 1999, 9, 158, 1883, 124, 1692, 1711, 1927, 1671, 1820, 1961, 1733, 1780, 1611, 1943, 206, 203, 1940, 1932, 38, 63, 1740, 1992, 1874, 173, 1871, 1903, 1816, 61, 1591, 1991, 76, 1818, 1885, 1669, 13, 115, 1869, 1845, 201, 1828, 1875, 72, 194, 1879, 1610, 1837, 174, 1870, 1925, 1653, 1789, 1735, 141, 184, 1634, 1644, 1888, 1997, 1567, 53, 1785, 178, 1695, 1721, 1668, 1645, 1619, 1933, 1976, 1977, 79, 1929, 42, 1660, 118, 218, 1559, 161, 1864, 97, 1817, 166, 1889, 1823, 1971, 133, 1965, 1784, 1956, 1998, 1792, 1936, 51, 142, 179, 195, 1844, 136, 112, 1982, 52, 1895, 1898, 1836, 1771, 1946, 39, 1994, 15]
SYN_ROUND2_PATIENTS = [1750, 1587, 103, 1918, 147, 66, 1647, 1657, 1791, 121, 1672, 1807, 1621, 1616, 125, 1640, 1635, 113, 129, 198, 1839, 59, 1995, 1565, 1667, 1656, 1632, 1909, 1786, 1980, 1915, 1711, 140, 1796, 1558, 1824, 1659, 1908, 8, 1603, 211, 1674, 116, 1599, 1843, 1780, 1803, 1793, 1592, 1792, 31, 1931, 1790, 1809, 1695, 1794, 83, 160, 1876, 49, 1739, 1990, 1951, 1746, 150, 1775, 1701, 117, 1586, 1694, 167, 67, 1618, 1863, 152, 1594, 1920, 1610, 1646, 1905, 1593, 1832, 108, 1953, 1851, 1958, 189, 1669, 1804, 1685, 1772, 1649, 1736, 1770, 1882, 1678, 1650, 1948, 1569, 32, 20, 1812, 77, 1620, 100, 1584, 1815, 1708, 1725, 169, 1767, 214, 1789, 2]#, 1922, 96, 154, 1740, 1666, 1993, 1859, 1638, 1765, 1857, 1645, 1641, 222, 144, 1588, 88, 1825, 1636, 1847, 27, 6, 1712, 217, 1887, 215, 1781, 1853, 1967, 1614, 1892, 1622, 1625, 1709, 1576, 138]
SYN_PATIENTS = SYN_PATIENTS + SYN_ROUND2_PATIENTS

SYN_PATIENTS = [id for id in SYN_PATIENTS if id not in [1736, 1770, 1678, 1650, 1569, 1812, 1620, 1584, 1708, 1725, 1767, 1789, 1740, 1666, 1638, 1765, 1645, 1641, 1588, 1636, 1712, 1781, 1614, 1622, 1625, 1709, 1576]]


TRAIN_PATIENTS = [id for id in range(1, 222+1) if id not in TEST_PATIENTS and id not in VAL_PATIENTS and id not in SYN_PATIENTS]

DIAG_SEG_23_08_TRAIN = [1584, 1731, 1712, 1639, 1714, 1625, 1758, 1612, 1661, 1700, 1668, 1788, 1784, 1793, 1649, 1735, 1654, 1603, 1671, 1613, 1657, 1568, 1705, 1642, 1767, 1794, 1771, 1592, 1707, 1647, 1804, 1739, 1650, 1799, 1640, 1748, 1616, 1696, 1627, 1695, 1681, 1609, 1738, 1744, 1672, 1809, 1567, 1734, 1798, 1750, 1772, 1646, 1608, 1765, 1723, 1756, 1593, 1641, 1716, 1770, 1620, 1599, 1789, 1774, 1753, 1763, 1591, 1638, 1601, 1619, 1733, 1644, 1787, 1699, 1722, 1812, 1569, 1571, 1682, 1736, 1785, 1648, 1690, 1803, 1651, 1791, 1691, 1576, 1718, 1615, 1807, 1623, 1725, 1667, 1602, 1761, 1796, 1617, 1587, 1800, 1562, 1678, 1780, 1776, 1610, 1680, 1595, 1709, 1762, 1683, 1577, 1582, 1653, 1792, 1583, 1692, 1754, 1730, 1594, 1811, 1775, 1559, 1560, 1614, 1656, 1708, 1808, 1740, 1563, 1726, 1635, 1677, 1564, 1636, 1589, 1628, 1729, 1660, 1757, 1766, 1674, 1605, 1606, 1783, 1732, 1751, 1578, 1622, 1694, 1795, 1659, 1728, 1769, 1773, 1720, 1581, 1621, 1645, 1721, 1611, 1643, 1713, 1777, 1597, 1669, 1565, 1684, 1666, 1598, 1588, 1633, 1790, 1737, 1686, 1711, 1604, 1685, 1801, 1629, 1632, 1760, 1806, 1586, 1805, 1558, 1585, 1746, 1580, 1575, 1786, 1600, 1752, 1572, 1781, 1665, 1634, 1715, 1673, 1573, 1579, 1693, 1701, 1618, 1727]
DIAG_SEG_23_08_VAL = [1652, 1561, 1719, 1557, 1566, 1658, 1630, 1698, 1688, 1675, 1687, 1706, 1607, 1782, 1574, 1702, 1747, 1717, 1743, 1810, 1778, 1697, 1664, 1745, 1663, 1689, 1703, 1704, 1802, 1624, 1797, 1590, 1631, 1779, 1768, 1676, 1596, 1742, 1570, 1655, 1759, 1741, 1662, 1749, 1724, 1637, 1626, 1755, 1764, 1710, 1670, 1679]

DIAG_SEG_23_08_TRAIN = [id for id in DIAG_SEG_23_08_TRAIN if id not in SYN_PATIENTS]

TRAIN_FROM_23 = [id for id in range(1813, 2000) if id not in SYN_PATIENTS]
TEST_FROM_23 = [id for id in range(2000, 2084+1)]

# The patients below are corrected in 2023_08_15, therefore they are ignored in 2023_07_09
CORRECTED = []

IGNORED_SAMPLES = [] + CORRECTED


class SeDataset(Dataset):

    def __init__(self,
            data_root,
            resolution,
            classes=3,
            random_crop=False,
            random_flip=False,
            split='train',
            type_labeling=False,
            resize=False):
        super().__init__()


        self.transforms = None#transforms
        self.split = split
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.resolution = resolution
        self.type_labeling = type_labeling
        self.num_classes = classes
        self.resize = resize

        contours_filename = f"Contours.json"

        self.all_samples = mapping_utils.construct_samples_list(
            data_root, contours_filename
        )
        mapping_utils.print_diagnostics(data_root, self.all_samples)

        partitions = {
            "train": TRAIN_PATIENTS + DIAG_SEG_23_08_TRAIN + TRAIN_FROM_23,
            "val": VAL_PATIENTS + DIAG_SEG_23_08_VAL,
            "test": TEST_PATIENTS + TEST_FROM_23,
            "syn": SYN_PATIENTS,
        }

        print(len(self.all_samples))

        self.all_samples = self.remove_ignored(self.all_samples, IGNORED_SAMPLES, data_root)


        self.samples = mapping_utils.split_samples_list(
            self.all_samples, partitions[self.split]
        )

        self.to_mapping_only()

        print(f"Number of samples: {len(self.samples)}")


    # we only need the mapping images
    def to_mapping_only(self):
        self.samples = [(x, t) for x, t in self.samples if "_Mapping_" in x]

    def __getitem__(self, index: int) -> dict:
        """
        Args:
            index (int): Index
        Returns:
            dict: (sample, mask)
        """
        path, mask_path = self.samples[index]
        sample = mapping_utils.load_dicom(path, mode=None, use_modality_lut=False)
        mask_contours = mapping_utils.load_contours(mask_path)
        mask = mapping_utils.contours_to_masks(mask_contours, sample.shape)

        size_ori = sample.shape

        if self.transforms is not None:           
            transformed = self.transforms(image=sample, mask=mask)
            sample, mask = transformed["image"], transformed["mask"]

        if 'T2' in path and self.type_labeling:            
            mask = mask + self.num_classes
        
        sample = sample / 4095
        if self.resize:
            # Convert images to channels_first mode, from albumentations' 2d grayscale images
            sample = resize(sample, (self.resolution, self.resolution), anti_aliasing=True)
            
            mask = resize(mask, (self.resolution, self.resolution), anti_aliasing=False, mode='edge', preserve_range=True, order=0)
        else:            
            # fill the image with zeros to make it resolution x resolution
            sample = np.pad(sample, ((0, self.resolution - sample.shape[0]), (0, self.resolution - sample.shape[1])), mode='constant', constant_values=0)
            mask += 1
            mask = np.pad(mask, ((0, self.resolution - mask.shape[0]), (0, self.resolution - mask.shape[1])), mode='constant', constant_values=0)

        sample = np.expand_dims(sample, 0)

        mask = np.expand_dims(mask, 0)

        sample = sample.astype(np.float32)
        mask = mask.astype(np.float32)
        
        if self.random_crop and random.random() < 0.33:
            opt = {
                'preprocess': 'resize_and_crop',      
                'crop_size': int(self.resolution),
                'load_size': int(self.resolution * 1.5),
                'flip': self.random_flip,
            }
            
            transform_params = get_params(opt, sample.shape[1:])
            B_transform = get_transform(opt, transform_params, method=transforms.InterpolationMode.NEAREST, grayscale=True)
            A_transform = get_transform(opt, transform_params, grayscale=True)
                      
            sample = torch.from_numpy(sample)
            mask = torch.from_numpy(mask)
            
            sample = A_transform(sample)
            mask = B_transform(mask)

            sample = sample.numpy()
            mask = mask.numpy()
        
        mask = np.squeeze(mask, axis=0)
  
        if 'T1' in path:
            sample = (sample - DATASET_MEAN_T1_mapping) / (DATASET_STD_T1_mapping)
        elif 'T2' in path:
            sample = (sample - DATASET_MEAN_T2_mapping) / (DATASET_STD_T2_mapping)


        # normalize the sample to -1, 1
        sample = (sample - np.min(sample.flatten())) / (np.max(sample.flatten()) - np.min(sample.flatten()))

        # mask to torch tensor
        mask = torch.from_numpy(mask).long()
        sample = torch.from_numpy(sample).float()

        out_dict = {}
        out_dict['path'] = path
        out_dict['label_ori'] = mask.detach().clone()
        out_dict['label'] = mask[None,]
        out_dict['size_ori'] = size_ori
        out_dict['mask_path'] = mask_path


        return sample, out_dict

    def __len__(self) -> int:
        return len(self.samples)


    @staticmethod
    def remove_ignored(samples, ignore_list, dataset_root, verbose=False):
        samples_numpy = np.array(samples)
        remove_count = 0
        for sample in ignore_list:
            sample = os.path.join(dataset_root, sample)  
            # sample is like     "2022_11_22/Patient (5)/T1_Mapping_/Apex.dcm" 
            # map sample is like "2022_11_22/Patient (5)/T1_Mapping_/_map_apex/"
            # also sample can be a Patient folder, e.g.: "2022_11_22/Patient (5)"
            map_sample = sample.replace("_Mapping_/Apex.dcm", "_map_apex")
            map_sample = map_sample.replace("_Mapping_/Mid.dcm", "_map_mid_")
            map_sample = map_sample.replace("_Mapping_/Base.dcm", "_map_base")
            for s in [sample, map_sample]:
                samples_to_drop = np.char.startswith(samples_numpy[:,0], s)
                if samples_to_drop.any():
                    samples_numpy = samples_numpy[~samples_to_drop]
                    if verbose:
                        print(colored(f"Removed {s} from the dataset based on the IGNORED_SAMPLES list", "red"))
                    remove_count += np.sum(samples_to_drop)
        print(f"Removed {remove_count} samples from the dataset based on the IGNORED_SAMPLES list")
        return samples_numpy.tolist()




def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt['preprocess'] == 'resize_and_crop':
        new_h = new_w = opt['load_size']
    elif opt['preprocess'] == 'scale_width_and_crop':
        new_w = opt['load_size']
        new_h = opt['load_size'] * h // w

    x = random.randint(0, np.maximum(0, new_w - opt['crop_size']))
    y = random.randint(0, np.maximum(0, new_h - opt['crop_size']))


    return {'crop_pos': (x, y)}


def get_transform(opt, params=None, grayscale=False, method=transforms.InterpolationMode.BICUBIC):
    transform_list = []
    #if grayscale:
    #    transform_list.append(transforms.Grayscale(1))
    if 'resize' in opt['preprocess']:
        osize = [opt['load_size'], opt['load_size']]
        transform_list.append(transforms.Resize(osize, method))

    if 'crop' in opt['preprocess']:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt['crop_size'])))

    if opt['flip']:
        transform_list.append(transforms.RandomHorizontalFlip())

    return transforms.Compose(transform_list)


def __crop(img, pos, size):
    ow, oh = img.shape[1:]
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img[:, y1:y1 + th, x1:x1 + tw]
    return img

