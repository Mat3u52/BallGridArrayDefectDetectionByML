import os
import sys
import random
import cv2
import shutil
import csv
import numpy as np
import matplotlib.pyplot as plt
from zipfile import ZipFile
from typing import List
from PIL import Image

def print_zip_summary(zip_path: str) -> None:
    """
    Prints a summary of the contents of a ZIP file.

    Given the path to a ZIP file, this function opens the file in read-only mode,
    retrieves a list of all the files contained within it, and prints a summary of the ZIP file.
    Specifically, it displays:
    - The base name of the ZIP file.
    - The file size of the ZIP archive in bytes.
    - The total number of files contained within the archive.
    - A list of the last 5 files in the archive, if the archive contains that many.
      If the ZIP archive has fewer than 5 files, it will print all of them.

    :param zip_path: str, the path to the ZIP file to summarize.
    :return: None
    """
    with ZipFile(zip_path, 'r') as zip_file:
        # Get the list of files
        files: List[str] = zip_file.namelist()

        print(f"File name: {os.path.basename(zip_path)}")
        print(f"File size: {os.path.getsize(zip_path)}")

        # Print the total number of files
        print(f"Total files in ZIP: {len(files)}")

        # Show only the last 5 files
        print("Last 5 files in ZIP:")

        for file_name in files[-5:]:
            print(file_name)

def unzip_precess(zip_path: str, zip_path_dist: str) -> None:
    """
    Extracts files from a zip archive to a specified directory, showing extraction progress.

    :param zip_path: str, The path to the zip file to extract.
    :param zip_path_dist: str, The directory path where the contents of the zip file will be extracted.
    :return: None
    """
    if os.path.isfile(zip_path) and not os.path.isdir(zip_path_dist):
        with ZipFile(zip_path) as zip_file:
            files_list: List[str] = zip_file.namelist()
            for idx, file in enumerate(files_list):
                percent = round((idx / len(files_list)) * 100)
                sys.stdout.write(f"\runzip process: {percent}%")
                sys.stdout.flush()
                zip_file.extract(file, zip_path_dist)
            zip_file.close()
            sys.stdout.write("\runzip process: 100%\n")
            sys.stdout.flush()

def tree(directory: str, file_range: int = -5, indent: int = 0) -> None:
    """
    Recursively lists the contents of a directory in a structured, hierarchical format.

    :param directory: str, path to directory.
    :param file_range: int, amount of samples.
    :param indent: int, space iterator.
    :return: None.
    """
    try:
        entries: list[str] = os.listdir(directory)

        dir_entries: list[str] = [e for e in entries if os.path.isdir(os.path.join(directory, e))]
        file_entries: list[str] = [e for e in entries if os.path.isfile(os.path.join(directory, e))][file_range:]

        for d in dir_entries:
            print("    " * indent + f"[Directory] {d}/")
            # Recursively call tree to print the contents of each subdirectory
            tree(os.path.join(directory, d), file_range, indent + 1)

        for f in file_entries:
            if f[-4:] != '.png':
                print("    " * indent + f"[other file] - {f} [note]The file should be removed {os.path.join(directory, f)}")
            else:
                print("    " * indent + f"[png file] - {f}")

    except FileNotFoundError:
        print(f"Error: Directory '{directory}' not found.")
    except PermissionError:
        print(f"Error: Permission denied for directory '{directory}'.")

def show_samples(image_directory: str) -> None:
    """
    Randomly selects and displays five .png images from a given directory. Each image is resized to
    300x300 pixels and shown in a pop-up window for 5 seconds.

    :param image_directory: str, The path to the directory containing image files.
    :return: None
    """
    all_files: list[str] = os.listdir(image_directory)
    image_files: list[str] = [f for f in all_files if f.endswith('png')]
    random_images: list[str] = random.sample(image_files, 5)

    for i, image_file in enumerate(random_images):
        image_path: str = os.path.join(image_directory, image_file)
        img: np.ndarray = cv2.imread(image_path)

        if img is not None:
            img = cv2.resize(img, (300, 300))
            cv2.imshow(f'Image {i + 1}', img)
            cv2.waitKey(1000)

    cv2.destroyAllWindows()

    good_sample_image_path: str = 'examples/single_good_ball.png'
    good_img: np.ndarray = cv2.imread(good_sample_image_path)
    not_good_sample_image_path: str = 'examples/single_not_good_ball.png'
    not_good_img: np.ndarray = cv2.imread(not_good_sample_image_path)

    bga_good_sample_image_path: str = 'examples/good_image.jpg'
    bga_good_img: np.ndarray = cv2.imread(bga_good_sample_image_path)
    bga_not_good_sample_image_path: str = 'examples/not_good_image.jpg'
    bga_not_good_img: np.ndarray = cv2.imread(bga_not_good_sample_image_path)

    bga_good_img = cv2.resize(bga_good_img, (600, 800))
    cv2.imshow(f'BGA - Golden sample', bga_good_img)
    # cv2.waitKey(5000)

    bga_not_good_img = cv2.resize(bga_not_good_img, (600, 800))
    cv2.imshow(f'BGA - Fail', bga_not_good_img)
    cv2.waitKey(5000)

    good_img = cv2.resize(good_img, (300, 300))
    cv2.imshow(f'Golden sample', good_img)
    # cv2.waitKey(5000)

    not_good_img = cv2.resize(not_good_img, (300, 300))
    cv2.imshow(f'Fail', not_good_img)
    cv2.waitKey(5000)

def recognizer(image: str, image_id: int, margin: int = 7) -> None:
    """
    Processes an image to identify the largest circular object, calculates its diameter and area,
    and highlights void areas within a margin around the detected contour.

    :param image: str, Path to the input image file.
    :param image_id: int, Identifier for naming output files.
    :param margin: int, Margin size for detection (default is 3 pixels).
    """
    threshold = 110
    image_path = image

    # Open the image
    img = Image.open(image)

    # Convert image to RGB if not already in RGB mode
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Binarize the image
    pixels = list(img.getdata())
    new_pixels = [(0, 0, 0) if (p[0] if isinstance(p, tuple) else p) <= threshold else (255, 255, 255) for p in pixels]
    new_img = Image.new('RGB', img.size)
    new_img.putdata(new_pixels)
    new_img.save(f'labeled/bitmapDiagnostics/{image_id}_bitmap.jpg')

    # Load the binarized image
    binary_image = cv2.imread(f'labeled/bitmapDiagnostics/{image_id}_bitmap.jpg', cv2.IMREAD_GRAYSCALE)

    # Perform edge detection and find contours
    edges = cv2.Canny(binary_image, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour closest to the image center
    image_center = (binary_image.shape[1] // 2, binary_image.shape[0] // 2)
    best_contour = min(
        contours, key=lambda contour: cv2.pointPolygonTest(contour, image_center, True) ** 2, default=None
    )

    # Initialize variables for circle properties
    mask = np.zeros_like(binary_image)
    diameter = 0

    if best_contour is not None:
        # Fit a minimum enclosing circle
        ((x, y), radius) = cv2.minEnclosingCircle(best_contour)
        diameter = 2 * radius

        # Draw the detected circle and create a mask
        cv2.circle(mask, (int(x), int(y)), int(radius), 255, thickness=-1)

        # Add inward margin by applying erosion
        kernel = np.ones((margin, margin), np.uint8)
        mask_with_margin = cv2.erode(mask, kernel, iterations=1)

        # Draw the blue contour on the output image
        output_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(output_image, [best_contour], -1, (255, 0, 0), 2)

    else:
        output_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

    # Detect white pixels inside the margin-adjusted circle
    white_pixel_mask = (binary_image >= 200) & (mask_with_margin == 255) if best_contour is not None else None
    white_pixel_count = np.count_nonzero(white_pixel_mask) if white_pixel_mask is not None else 0

    # Highlight detected voids in the output image
    if white_pixel_mask is not None:
        output_image[white_pixel_mask] = [0, 255, 255]  # Highlight voids as yellow

    # Plot 1: Detected Circle with Blue Contour
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title(f'Detected Object with Diameter: {diameter:.2f} pixels')
    plt.axis('off')
    mask_plot_path = f"labeled/bitmapDiagnostics/{image_id}_edge_plot.png"
    plt.savefig(mask_plot_path, bbox_inches='tight', dpi=300)
    # plt.show()
    plt.close('all')

    # Plot 2: Void Area Highlighted
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title(f'Void Area Highlighted (White Pixels: {white_pixel_count} px)')
    plt.axis('off')
    plt.text(
        10, 30, f'Void Area: {white_pixel_count} px', color='red', fontsize=14,
        bbox=dict(facecolor='white', alpha=0.8)
    )
    void_plot_path = f"labeled/bitmapDiagnostics/{image_id}_void_plot.png"
    plt.savefig(void_plot_path, bbox_inches='tight', dpi=300)
    # plt.show()
    plt.close('all')

    # Print stats
    circle_area = np.pi * (radius ** 2) if best_contour is not None else 0
    print(f'Diameter: {diameter:.2f} px')
    print(f'Circle Area: {circle_area:.2f} px²')
    print(f'Void Area (White Pixels): {white_pixel_count} px²')

    # Classify the image based on criteria
    if white_pixel_count > 15:
        shutil.copyfile(image_path, f'labeled/fail/{image_id}_ball.png')
    else:
        shutil.copyfile(image_path, f'labeled/pass/{image_id}_ball.png')

    # save data to the .csv file
    with open("SolderBallsSize.csv", mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([diameter, circle_area, white_pixel_count])


if __name__ == "__main__":
    # general information about .zip file
    print_zip_summary('Dataset.zip')
    # unzip the file
    unzip_precess('Dataset.zip', 'UnzippedDataset')
    # directory and file tree
    tree('UnzippedDataset')
    # Show samples
    show_samples('UnzippedDataset/Dataset/inspection/2024-10-31_15-09-15-335')

    # labeling the images and collect diagnostics
    i: int = 0
    for root, _, files in os.walk('UnzippedDataset\\Dataset\\inspection'):
        print(root)
        for file in files:
            if file.endswith('_48_48_4.png'):
                print(os.path.join(root, file))
                print(i)
                recognizer(os.path.join(root, file), i)
            i += 1