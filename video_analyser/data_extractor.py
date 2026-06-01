import cv2
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize


VIDEO_PATH = "videos/output_video.mp4"
OUTPUT_DIR = "sampled_frames"
SAMPLE_INTERVAL = 0.1  # seconds


class VideoAnalyserDataExtractor:

    def extract_data(self):
        # Placeholder for data extraction logic
        # This method should be implemented to extract relevant data from the video analyser
        pass

    def detect_cells_side(self, img):
        # Placeholder for cell side detection logic
        # This method should be implemented to determine the side of the cells in the video analyser
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=150,
            minLineLength=200,
            maxLineGap=10
        )

        # separate vertical and horizontal lines
        vertical_lines = []
        horizontal_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            if abs(x1 - x2) < 5:  # vertical
                vertical_lines.append(x1)
            elif abs(y1 - y2) < 5:  # horizontal
                horizontal_lines.append(y1)

        vertical_lines = sorted(vertical_lines)
        horizontal_lines = sorted(horizontal_lines)

        # Remove duplicates (lines detected multiple times)
        def remove_close(values, threshold=10):
            filtered = [values[0]]
            for v in values[1:]:
                if abs(v - filtered[-1]) > threshold:
                    filtered.append(v)
            return filtered

        vertical_lines = remove_close(vertical_lines)
        horizontal_lines = remove_close(horizontal_lines)

        # Compute cell size
        cell_widths = np.diff(vertical_lines)
        cell_heights = np.diff(horizontal_lines)

        cell_size_x = int(np.median(cell_widths))
        cell_size_y = int(np.median(cell_heights))

        cell_size = int((cell_size_x + cell_size_y) / 2)

        print("Detected cell size:", cell_size)

        return cell_size




    def extract_objects_ids(self, first_frame_path: str):
        # Placeholder for object ID extraction logic
        # This method should be implemented to extract object IDs from the video analyser


        img = cv2.imread(first_frame_path)

        # Convert to HSV (better for color separation)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Create mask for non-background pixels
        # Since background is very light/white-ish
        mask = cv2.inRange(hsv, (0, 40, 40), (180, 255, 255))

        # remove small noise and fill holes
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find connected components (objects)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

        objects = []
        object_id = 1

        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            # Extract region
            roi = img[y:y+h, x:x+w]

            # Compute average color (BGR)
            avg_color = cv2.mean(roi, mask=mask[y:y+h, x:x+w])[:3]

            objects.append({
                "id": object_id,
                "bbox": (x, y, w, h),
                "color_bgr": tuple(map(int, avg_color)),
                "width_pixels": w,
                "height_pixels": h
            })

            object_id += 1

        cell_size = self.detect_cells_side(img)
        for obj in objects:
            obj["width_cells"] = round(obj["width_pixels"] / cell_size)
            obj["height_cells"] = round(obj["height_pixels"] / cell_size)

        return objects
    
    def extract_objects_ids_2(self, first_frame_path: str):
        # Placeholder for alternative object ID extraction logic
        # Convert to grayscale
        image = cv2.imread(first_frame_path)
        image_copy = image.copy()
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)  # anything >0 is part of a box

        # Noise removal
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # Sure background
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Distance transform
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)

        # Unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown==255] = 0

        # Apply watershed
        markers = cv2.watershed(image, markers)

        detected_boxes = []
        id_counter = 1

        for marker_id in range(2, markers.max()+1):  # skip background
            mask = np.uint8(markers == marker_id) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                box_crop = image[y:y+h, x:x+w]
                avg_color = cv2.mean(box_crop)[:3]
                avg_color_hex = '#%02x%02x%02x' % (int(avg_color[2]), int(avg_color[1]), int(avg_color[0]))
                
                detected_boxes.append({
                    "id": id_counter,
                    "position": (x, y),
                    "size": (w, h),
                    "color": avg_color_hex
                })
                id_counter += 1

                cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Separated Boxes", image_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def detect_boxes_faint_lines(self, first_frame_path, blur_ksize=11, canny_thresh1=30, canny_thresh2=100, dilate_iter=1):
        """
        Detects boxes in an image separated by faint thin dark lines.

        Parameters:
            img (numpy.ndarray): Input BGR image.
            blur_ksize (int): Kernel size for Gaussian blur to emphasize lines.
            canny_thresh1 (int): Lower threshold for Canny edge detection.
            canny_thresh2 (int): Upper threshold for Canny edge detection.
            dilate_iter (int): Number of iterations for dilating edges.

        Returns:
            boxes (list of tuples): List of bounding boxes (x, y, w, h).
            edge_img (numpy.ndarray): Image showing detected edges.
            result_img (numpy.ndarray): Original image with bounding boxes drawn.
        """
        img = cv2.imread(first_frame_path)
        black_mask = np.all(img < 50, axis=2)   # near-black
        white_mask = np.all(img > 210, axis=2)   # near-white
        keep_mask = ~(black_mask | white_mask).astype(np.uint8)  # 1 for pixels we keep

        # Apply mask to color image
        masked_color = cv2.bitwise_and(img, img, mask=keep_mask*255)

        # Convert masked image to grayscale for processing
        gray_masked = cv2.cvtColor(masked_color, cv2.COLOR_BGR2GRAY)
        
        # Smooth image to detect faint lines
        smooth = cv2.GaussianBlur(gray_masked, (blur_ksize, blur_ksize), 0)
        diff = cv2.subtract(smooth, gray_masked)

        # Threshold to get prominent lines
        _, thresh = cv2.threshold(diff, 5, 255, cv2.THRESH_BINARY)

        # Edge detection
        edges = cv2.Canny(thresh, canny_thresh1, canny_thresh2)

        # Dilate edges to make contours more solid
        kernel = np.ones((2,2), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=dilate_iter)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        result_img = img.copy()
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h))
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow("Gray masked", gray_masked)
        cv2.imshow("Blurred", smooth)
        cv2.imshow("Thresholded", thresh)
        cv2.imshow("Dilated", dilated)
        cv2.imshow("Edges", edges)
        cv2.imshow("Detected Boxes", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return boxes, edges, result_img

    def extract_color_mask(self, first_frame_path, black_thresh=64, white_thresh=160,canny_thresh1=50,
                                 canny_thresh2=150,):
        img = cv2.imread(first_frame_path)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        h, s, v = cv2.split(hsv)
        sat_thresh = 40  
        color_mask = cv2.inRange(s, sat_thresh, 255)
        print("Color mask created with saturation threshold:", color_mask[0,:])
        cv2.imshow("Masked Color", color_mask)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return color_mask
    
    def detect_boxes(self, first_frame_path):
        color_mask = self.extract_color_mask(first_frame_path)
        _, item_edges = self.clean_grid_and_extract_edges(first_frame_path)
        edges_colored = cv2.bitwise_and(item_edges, item_edges, mask=color_mask)
        kernel = np.ones((3,3), np.uint8)
        closed = cv2.morphologyEx(edges_colored, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mask_objects = np.zeros_like(closed)
        cv2.drawContours(mask_objects, contours, -1, 255, thickness=cv2.FILLED)
        boxes = []
        result_img = cv2.imread(first_frame_path)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200:   # remove noise
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h))
            cv2.rectangle(result_img, (x,y), (x+w,y+h), (0,255,0), 2)

        cv2.imshow("Edges colores", edges_colored)
        cv2.imshow("Closed Edges", closed)
        cv2.imshow("Mask Objects", mask_objects)
        cv2.imshow("Detected Boxes", result_img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def detect_boxes_faint_lines_hsv(self, first_frame_path,
                                 blur_ksize=5,
                                 canny_thresh1=50,
                                 canny_thresh2=150,
                                 dilate_iter=1,
                                 black_thresh=64,
                                 white_thresh=150):
        # Read the image
        img = cv2.imread(first_frame_path)
        gray = cv2.imread(first_frame_path, cv2.IMREAD_GRAYSCALE)


        # # Ensure blur_ksize is odd
        # blur_ksize = max(1, blur_ksize | 1)
        # # Convert masked color to grayscale for line detection
        # gray = cv2.cvtColor(masked_color, cv2.COLOR_BGR2GRAY)

        # # Convert to HSV
        # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # h, s, v = cv2.split(hsv)

        # # # Mask: keep pixels with V between black and white thresholds
        # keep_mask = cv2.inRange(v, v_black_thresh, v_white_thresh)

        
        # Create mask for pixels that are NOT black or white
        mask = cv2.inRange(gray, black_thresh, white_thresh)  # keeps pixels between thresholds

        # Apply mask
        masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
        # Apply mask to original color image
        masked_color = cv2.bitwise_and(img, img, mask=mask)
        median_val = np.median(gray)

        lower = int(max(0, 0.66 * median_val))
        upper = int(min(255, 1.33 * median_val))
        canny_thresh1 = lower
        canny_thresh2 = upper
        edges1 = cv2.Canny(masked_color, canny_thresh1, canny_thresh2)
        edges2 = cv2.Canny(masked_gray, canny_thresh1, canny_thresh2)
        combined_edges = cv2.bitwise_or(edges1, edges2)
        self.detect_items(img, edges2)

        

        # # Smooth to detect faint lines
        # smooth = cv2.GaussianBlur(gray_masked, (blur_ksize, blur_ksize), 0)
        # diff = cv2.subtract(smooth, gray_masked)

        # # Threshold to get prominent lines
        # _, thresh = cv2.threshold(diff, 5, 255, cv2.THRESH_BINARY)

        # # Edge detection
        # edges = cv2.Canny(thresh, canny_thresh1, canny_thresh2)

        # # Dilate edges to make contours more solid
        # kernel = np.ones((2,2), np.uint8)
        # dilated = cv2.dilate(edges, kernel, iterations=dilate_iter)

        # # Find contours
        # contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # boxes = []
        # result_img = img.copy()
        # for cnt in contours:
        #     x, y, w, h = cv2.boundingRect(cnt)
        #     boxes.append((x, y, w, h))
        #     cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Show images
        cv2.imshow("Original", img)
        # Convertir a grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Calcular el histograma
        hist = cv2.calcHist([gray], [0], None, [256], [0,256])

        # Dibujar el histograma
        plt.figure(figsize=(10,4))
        plt.plot(hist, color='black')
        plt.title("Histograma de Grayscale")
        plt.xlabel("Intensidad de píxel")
        plt.ylabel("Número de píxeles")
        plt.xlim([0, 255])
        plt.show()
        cv2.imshow("Masked Color", masked_color)
        # cv2.imshow("Gray Masked", gray_masked)
        # cv2.imshow("Blurred", smooth)
        # cv2.imshow("Thresholded", thresh)
        # cv2.imshow("Dilated", dilated)
        cv2.imshow("Edges 1", edges1)
        cv2.imshow("Edges 2", edges2)
        cv2.imshow("Edges combined", combined_edges)
        # cv2.imshow("Detected Boxes", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        boxes = 0 
        result_img = 0
        return boxes, edges, result_img

    
    def detect_items(self, img, combined_edges):
        
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = hsv[:,:,1]

        plt.hist(s.ravel(), bins=256, range=(0,256))
        plt.title("Saturation Histogram")
        plt.xlabel("S value")
        plt.ylabel("Pixel count")
        plt.show()

        # Threshold saturation to keep only colored areas
        sat_thresh = 127   # adjust if needed
        color_mask = cv2.threshold(s, sat_thresh, 255, cv2.THRESH_BINARY)[1]


        edges_colored = cv2.bitwise_and(combined_edges, combined_edges, mask=color_mask)
        # Find contours on edges_colored
        contours, _ = cv2.findContours(edges_colored, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result_img = img.copy()
        boxes = []

        for cnt in contours:
            # Approximate polygon to reduce noise
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            # Only consider quadrilaterals (parallelograms/rectangles)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                boxes.append((x, y, w, h))
                cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw green box

        cv2.imshow("Colored Mask", color_mask)
        cv2.imshow("Color edges", edges_colored)
        cv2.imshow("Detected Colored Boxes", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def show_steps(self, images, titles):
        """Función auxiliar para visualizar múltiples pasos del proceso."""
        n = len(images)
        plt.figure(figsize=(n * 4, 4))
        for i in range(n):
            plt.subplot(1, n, i + 1)
            plt.imshow(images[i], cmap='gray' if len(images[i].shape) == 2 else None)
            plt.title(titles[i])
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    def clean_grid_and_extract_edges(self, path):
        """
        Aísla los cuadrados del fondo, elimina la cuadrícula y extrae bordes finos.

        """
        img_bgr = cv2.imread(path)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)

        _, mask = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY)
        
        img_masked = cv2.bitwise_and(img_gray, img_gray, mask=mask)

        filtered = cv2.medianBlur(img_masked, 11)

        edges = cv2.Canny(filtered, 50, 150, apertureSize=5, L2gradient=True)

        # Dilatamos para que los bordes externos/internos se toquen y formen una sola masa
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        skeleton = skeletonize(dilated > 0)
        edges_cleaned = (skeleton * 255).astype(np.uint8)

        self.show_steps(
            [img_masked, filtered, edges, edges_cleaned],
            ["Fondo Negro", "Median filter", "Canny Raw", "Bordes Finales"]
        )

        return filtered, edges_cleaned


    def extract_frames(self): 
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            raise RuntimeError("Failed to open video")

        # Try to get FPS, fallback to 30 if OpenCV fails
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            print("Warning: FPS detected as 0. Using default FPS = 30")
            fps = 30

        metadata = []
        frame_number = 0
        frame_count = 0
        next_save_time = 0.0
        i = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # end of video

            # Compute timestamp of current frame
            timestamp = frame_count / fps

            # Save frame if we reached the next interval
            if timestamp >= next_save_time:
                filename = f"frame_{frame_number:04d}.png"
                filepath = os.path.join(OUTPUT_DIR, filename)
                cv2.imwrite(filepath, frame[200:864, 113:1670]) # images are 1920 x 1024 pixeles
                
                metadata.append({
                    "frame_number": frame_number,
                    "timestamp_seconds": round(timestamp, 3),
                    "file": filename
                })

                frame_number += 1
                next_save_time += SAMPLE_INTERVAL

            frame_count += 1

        cap.release()

        # Save metadata JSON
        with open("frames_metadata.json", "w") as f:
            json.dump({
                "video_fps": fps,
                "sample_interval_seconds": SAMPLE_INTERVAL,
                "total_extracted_frames": frame_number,
                "frames": metadata
            }, f, indent=4)

        print(f"Extracted {frame_number} frames and saved metadata.")


if __name__ == "__main__":
    video_analyser = None  # Placeholder for the actual video analyser instance
    extractor = VideoAnalyserDataExtractor()
    path_img = "sampled_frames/frame_0000.png"
    # extractor.extract_frames()
        
    # objects = extractor.extract_objects_ids_2("sampled_frames/frame_0000.png")
    # _, _, _ =extractor.detect_boxes_faint_lines("sampled_frames/frame_0000.png")
    # _, _, _ = extractor.detect_boxes_faint_lines_hsv("sampled_frames/frame_0000.png")
    # processed_img, edges_cleaned = extractor.clean_grid_and_extract_edges(path_img)
    extractor.detect_boxes(path_img)

    # print("Extracted objects:", objects)

