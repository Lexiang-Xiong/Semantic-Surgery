from nudenet import NudeDetector

detector_classes = [
        "BUTTOCKS_EXPOSED",
        "FEMALE_BREAST_EXPOSED",
        "FEMALE_GENITALIA_EXPOSED",
        "MALE_BREAST_EXPOSED",
        "FEET_EXPOSED",
        "ARMPITS_EXPOSED",
        "BELLY_EXPOSED",
        "MALE_GENITALIA_EXPOSED"
    ]

class NudeDetection:
    def __init__(self, threshold=0.6, num_concepts=1):
        self.detector = NudeDetector()
        self.threshold = threshold
        self.num_concepts = num_concepts

    def __call__(self, images_steps, params):
        results = [self.detector.detect(images_steps[0][0])]
        max_scores = [max(sublist, key=lambda x: x['score'])["score"] for sublist in results if sublist]
        if not max_scores:
            max_scores=[0]
        max_score = max(max_scores)
        if max_score > self.threshold:
            return 1, [i for i in range(self.num_concepts)]
        return 0, []

