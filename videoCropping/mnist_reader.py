# mnist_reader.py
import numpy as np
import onnxruntime as ort
import cv2

class MNISTReader:
    """
    Minimal: classify the ENTIRE crop as one digit (0–9).
    readtext(image_bgr) -> [ (bbox, text, conf) ]
    """
    def __init__(self, model_path: str):
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name  = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name
        self.tgt_size    = (28, 28)

    def _prep28(self, img_bgr):
        g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # De-pixelate: upsample a lot, light blur, then downsample
        h, w = g.shape[:2]
        scale = 6 if max(h, w) < 40 else 3
        g = cv2.resize(g, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
        g = cv2.GaussianBlur(g, (3,3), 0)

        # Adaptive binarize and make "ink" white (MNIST polarity)
        th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
        if np.mean(th) > 127:
            th = 255 - th

        # Small close to keep thin strokes contiguous
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8), iterations=1)

        # Tight bounding box around ink; if nothing, return zeros
        ys, xs = np.where(th > 0)
        if len(xs) == 0:
            canvas = np.zeros((28, 28), np.uint8)
        else:
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
            crop = th[y0:y1+1, x0:x1+1]

            # Aspect-preserving resize: longest side -> 20 px, then pad to 28×28
            ch, cw = crop.shape[:2]
            s = 20.0 / max(ch, cw)
            nw, nh = max(1, int(round(cw*s))), max(1, int(round(ch*s)))
            resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)

            canvas = np.zeros((28, 28), np.uint8)
            y_off = (28 - nh) // 2
            x_off = (28 - nw) // 2
            canvas[y_off:y_off+nh, x_off:x_off+nw] = resized

            # Center by center-of-mass
            ys, xs = np.where(canvas > 0)
            if len(xs) and len(ys):
                cy, cx = int(np.mean(ys)), int(np.mean(xs))
                M = np.float32([[1, 0, 14 - cx], [0, 1, 14 - cy]])
                canvas = cv2.warpAffine(canvas, M, (28, 28),
                                        flags=cv2.INTER_NEAREST, borderValue=0)

        # Normalize like MNIST
        g = canvas.astype(np.float32) / 255.0
        g = (g - 0.1307) / 0.3081
        x = g[None, None, :, :]  # NCHW float32
        return x

    def readtext(self, image_bgr):
        x = self._prep28(image_bgr)
        logits = self.sess.run([self.output_name], {self.input_name: x})[0][0]  # (10,)
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        cls = int(np.argmax(probs))
        conf = float(probs[cls])

        h, w = image_bgr.shape[:2]
        return [((0, 0, w, h), str(cls), conf)]
