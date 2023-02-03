import math
import random
import time
import scipy


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import pygame

"""
Classes for beamforming
"""


class Antenna():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def read_sig(self, wave):
        pass


class Source():
    def __init__(self, freq, speed, source_pos):
        self.freq = freq
        self.speed = speed
        self.source_pos = source_pos

    def sig_at_time_and_location(self, pos, time):
        x1, y1, z1 = pos
        x0, y0, z0 = self.source_pos
        d = np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2 + (z0 - z1) ** 2)
        pi2f = 2 * np.pi * self.freq
        wavelength = self.speed / self.freq
        sig = np.sin((2 * np.pi * d) / wavelength + 2 * np.pi * self.freq * time) / d
        return sig


# Generate thin rectangles for the PCBs and the other rectangles for the SMP connectors





def save_array_to_csv(array, filename):
    x_values = [array[i][0] for i in range(len(array))]
    y_values = [array[i][1] for i in range(len(array))]
    array_df = pd.DataFrame()
    array_df["x"] = x_values
    array_df["y"] = y_values

    array_df.to_csv(f"{filename}", index=False)


def load_array(filename):
    array_df = pd.read_csv(filename)
    array = [(array_df["x"][i], array_df["y"][i]) for i in range(len(array_df))]
    return array


def load_DCs(filename):
    DCs_df = pd.read_csv(filename, names=["DC_Orien"])
    DCs = DCs_df["DC_Orien"]
    return DCs


def save_DCs(DCs, filename):
    DCs = list(DCs)

    np.savetxt(filename,
               DCs,
               delimiter=", ",
               fmt='% s')


def load_in_array(filename):
    array_radius = 9.75 / 100
    global freq, speed
    df = pd.read_csv(filename)
    array = []
    for r in range(len(df)):
        array.append(Antenna(df["x"][r] / 100 - array_radius, df["y"][r] / 100 - array_radius, 0))
    # Make each antenna a source

    # Source properties:
    speed = 3e8

    # They all lie on Z=0
    sources = [Source(freq, speed, (array[i].x, array[i].y, 0)) for i in range(len(array))]
    return array, sources




def calc_gain(Im):
    center_image, outside_image = cut_out_center(Im, circle_gain_crop_radius)

    gain = max(center_image) / max(outside_image)
    return gain


def score_gain_fast(array):
    filename = "temp_array.csv"
    save_array_to_csv(array, filename)

    I = create_image(filename)

    gain = calc_gain(I)

    return gain



def create_AP(filename):
    df = pd.read_csv(filename)
    x = df["x"] / 100 - array_radius / 100
    y = df["y"] / 100 - array_radius / 100
    z = np.zeros(len(x))

    AP = []
    for i in range(len(x)):
        position_of_single_antenna = (x[i], y[i], z[i])
        AP.append(position_of_single_antenna)
    AP = np.array(AP)

    return AP


def create_planar_surface(width, Z_depth, res):
    X = np.linspace(-width/2, width/2, res)
    Y = np.linspace(-width/2, width/2, res)
    X_full = []
    Y_full = []
    Z_full = []
    for x in X:
        for y in Y:
            X_full.append(x)
            Y_full.append(y)
            Z_full.append(Z_depth)

    PP = []
    for i in range(len(X_full)):
        position_of_single_antenna = (X_full[i], Y_full[i], Z_full[i])
        PP.append(position_of_single_antenna)
    PP = np.array(PP)
    surface = (X_full, Y_full, X_full)
    return PP, surface


def create_interference_pattern(D):
    # number of time steps
    num_steps = 3

    # time step size
    dt = T / num_steps

    # initialize array to store intensity at each time step
    I = [np.sum(((1 / D) * np.cos(k * D - omega * i * dt)), axis=0) ** 2 for i in range(num_steps)]

    # calculate mean intensity over all time steps
    I = np.mean(I, axis=0)

    image = I.reshape(res, res)

    return image


def cut_out_center(image, radius):
    # Get the shape of the image
    rows, cols = image.shape
    # Create a mask of all False values with the same shape as the image
    mask = np.zeros_like(image, dtype=bool)
    # Create a center point for the circular shape
    center = (rows // 2, cols // 2)
    # Use NumPy's meshgrid function to create a grid of x and y values
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    # Use the euclidean distance formula to calculate the distance from each point in the grid to the center point
    dist = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    # Create a mask where all points within the specified radius of the center point are True
    mask_inside = dist <= radius
    mask_outside = dist >= radius
    # Use the mask to extract the center of the image
    center_image = image[mask_inside]
    outside_image = image[mask_outside]
    return center_image, outside_image


def create_image(filename):
    AP = create_AP(filename)

    # PP is the matrix that contains nformation about the screen pixel positions. (X,Y,Z) are the coordinates of the screen pixels
    PP, (X, Y, Z) = create_planar_surface(width, Z_depth, res)

    # D is the matrix that contains information about the antenna-pixel separations and therfore the phase differences
    D = np.sqrt(np.sum((AP[:, np.newaxis, :] - PP[np.newaxis, :, :]) ** 2, axis=-1))

    I = np.transpose(create_interference_pattern(D))

    return I


# Set the global variables
array_radius = 12  # This is the outer circle
inner_array_radius = 10


# No Constraints:
ant_radius = 0

freq = 20e9
speed = 3e8
omega = 2 * np.pi * freq

k = omega / speed


# Setting up the viewing screen
res = 300
width = 2
Z_depth = 1
circle_gain_crop_radius_m = 90e-3
circle_gain_crop_radius = circle_gain_crop_radius_m * (res / (width * 2))  # pixels
T = 1 / freq
x_axis = np.linspace(-width, width, res)
y_axis = np.linspace(-width, width, res)


def create_image_png(filename):
    im_start = create_image(filename)
    # normalise Img
    im_start_norm = im_start / im_start.max()
    im_start_norm_log = np.log10(im_start_norm)

    plt.figure(figsize=(10, 8), dpi=120)
    plt.title(f"{filename}: $Gain: {round(calc_gain(im_start_norm), 3)}$", fontsize=20)
    plt.imshow(im_start_norm, cmap="magma")
    # plt.imshow(im_start_norm_log,cmap="magma")
    plt.colorbar(label=("Intensity"))
    plt.xlabel("x (m)", fontsize=20)
    plt.ylabel("y (m)", fontsize=20)

    # set the x and y axis ticks and labels
    divisions = 14

    # set the x and y axis ticks and labels
    plt.gca().set_xticks(np.arange(0, res, step=res // divisions))
    plt.gca().set_yticks(np.arange(0, res, step=res // divisions))
    plt.gca().set_xticklabels([round(x, 2) for x in x_axis[::res // divisions]])
    plt.gca().set_yticklabels([round(y, 2) for y in y_axis[::res // divisions]])

    # Add a circle around the center of the image/plot
    center_x = res / 2
    center_y = res / 2
    radius = circle_gain_crop_radius
    circle = plt.Circle((center_x, center_y), radius, color='red', fill=False)
    #plt.gca().add_artist(circle)

    plt.savefig('figure_interference_pattern.png')


create_image_png("20Ant_GainImage.csv")


# Initialize PyGame
pygame.init()


# Set up the screen
screen = pygame.display.set_mode((0,0), pygame.FULLSCREEN)

screen_width = screen.get_width()
screen_height = screen.get_height()



circle = pygame.image.load("Circle.png")
arrayRadius = inner_array_radius  # cm
circle_width, circle_height = circle.get_size()
# Pixel width in cm:
px_width = circle_width / (arrayRadius * 2)
px_height = circle_height / (arrayRadius * 2)


# Set the title of the screen
pygame.display.set_caption('Interference Pattern Generator')

# List to store the mouse coordinates
coords = []
x_values = []
y_values = []
coords_df = pd.DataFrame(columns=["x", "y"])
# Flag to track if the user has left clicked
left_clicked = False


# Create a font for the text
font = pygame.font.Font(None, 36)
# Render the text
text = font.render("Welcome to the (Very Simple) Interference Pattern Generator", True, (255, 255, 255))


# Create a surface for the text box
text_box = pygame.Surface((text.get_width() + 10, text.get_height() + 10))
# Fill the text box with a color
text_box.fill((0, 0, 0))
# Blit the text on the text box
text_box.blit(text, (5, 5))
# Position the text box on the screen
text_box_rect = text_box.get_rect()
text_box_rect.center = (circle_width+300, 20)

screen.fill((255,255,255))
screen.blit(circle, (0, 50))
# Main game loop
while True:



    # Check for events
    for event in pygame.event.get():
        # If the user quits, end the game
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

        # If the user left clicks, set the flag to True
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            left_clicked = True
            # If the user left clicks, set the flag to True
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            left_clicked = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                coords = []
                coords_df = pd.DataFrame(columns=["x", "y"])
                coords_df.to_csv("coordinates.csv", index=False)
                x_values = []
                y_values = []
                screen.fill((255, 255, 255))
                screen.blit(circle, (0, 50))
                text_box.blit(text, (5, 5))
                
    # Get the mouse coordinates
    x, y = pygame.mouse.get_pos()

    # If the user has left clicked, add the coordinates to the list
    if left_clicked:
        # Convert x and y into a distance
        Y = -(((screen_height - y)-circle_height/2)/px_height)+array_radius
        X = ((x - (circle_width / 2)) / px_width)+array_radius

        coords.append((x, y))

        x_values.append(X)
        y_values.append(Y)
        coords_df = pd.DataFrame(columns=["x", "y"])
        coords_df["x"] = y_values
        coords_df["y"] = x_values
        coords_df.to_csv("coordinates.csv", index=False)

        print(f"{coords_df}")

        #time.sleep(2e-1)
        create_image_png("coordinates.csv")

        interference_pattern = pygame.image.load("figure_interference_pattern.png")

    try:
        screen.blit(interference_pattern, (circle_width, 50))
    except:
        print("")
    pygame.display.update()


    # Draw the coordinates on the screen
    for coord in coords:
        pygame.draw.circle(screen, (0, 0, 0), coord, 4)

    # Draw the text box on the screen
    screen.blit(text_box, text_box_rect)

    # Update the display
    pygame.display.flip()
    # Update the screen
    pygame.display.update()
