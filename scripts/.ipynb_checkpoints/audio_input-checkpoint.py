import sounddevice as sd
import numpy as np

def list_input_devices():
    print("Available audio input devices:\n##########")
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            print(f"[{idx}] {dev['name']} (Input Channels: {dev['max_input_channels']})")
    print("##########")

def get_input_device():
    list_input_devices()
    try:
        device_index = int(input("Select input device index: ").strip())
        device_info = sd.query_devices(device_index)
        if device_info['max_input_channels'] == 0:
            raise ValueError("Selected device has no input channels.")
        print(f"Selected device: {device_info['name']}")
        return device_index
    except Exception as e:
        print(f"Error selecting device: {e}")
        return None

#list_input_devices()
get_input_device()