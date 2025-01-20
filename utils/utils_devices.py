import os
import torch


class DeviceManager:
    _instance = None
    _device = None
    _visible_devices = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DeviceManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def _initialize_device(cls, visible_devices="0", device_id=0):
        """
        Initialize the device with proper CUDA settings
        Args:
            visible_devices: str or list, e.g. "0" or "0,1" or ["0", "1"]
            device_id: int, which GPU to use among visible devices
        """
        if torch.cuda.is_available():
            # Handle visible devices configuration
            if visible_devices is not None:
                if isinstance(visible_devices, (list, tuple)):
                    visible_devices = ",".join(map(str, visible_devices))
                os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_devices)
                cls._visible_devices = visible_devices

            # Get available device count after setting CUDA_VISIBLE_DEVICES
            available_devices = torch.cuda.device_count()

            if available_devices > 0:
                # Ensure device_id is valid
                device_id = min(device_id, available_devices - 1)
                cls._device = torch.device(f"cuda:{device_id}")
                torch.cuda.empty_cache()
                torch.cuda.set_device(cls._device)
                print(f"Using GPU {device_id}: {torch.cuda.get_device_name(device_id)}")
                print(f"Available GPUs: {available_devices}")
                print(f"Visible devices: {os.environ.get('CUDA_VISIBLE_DEVICES', 'All')}")
            else:
                print("No CUDA devices available after filtering. Falling back to CPU.")
                cls._device = torch.device("cpu")
        else:
            cls._device = torch.device("cpu")
            print("CUDA is not available. Using CPU.")

    @classmethod
    def initialize(cls, visible_devices="0", device_id=0):
        """
        Public method to initialize device settings
        Args:
            visible_devices: str or list, e.g. "0,1" or ["0", "1"]
            device_id: int, which GPU to use among visible devices
        """
        cls._initialize_device(visible_devices, device_id)
        return cls.get_device()

    @classmethod
    def get_device(cls):
        """Get the singleton device instance"""
        if cls._device is None:
            cls._initialize_device()
        return cls._device

    @classmethod
    def setup_model(cls, model):
        """Helper method to move model to correct device"""
        if cls._device is None:
            cls._initialize_device()
        return model.to(cls._device)

    @classmethod
    def get_visible_devices(cls):
        """Get currently set visible devices"""
        return cls._visible_devices


def initialize_device(visible_devices="0", device_id=0):
    """Convenience function to initialize device settings"""
    return DeviceManager.initialize(visible_devices, device_id)


def get_device():
    """Convenience function to get current device"""
    return DeviceManager.get_device()


def setup_model(model):
    """Convenience function to setup model on correct device"""
    return DeviceManager.setup_model(model)