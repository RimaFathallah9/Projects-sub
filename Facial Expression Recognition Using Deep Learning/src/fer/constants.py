"""Dataset-wide constants. Class order follows the original FER-2013 label ids 0..6."""

CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
NUM_CLASSES = len(CLASSES)
IMG_SIZE = 48
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
